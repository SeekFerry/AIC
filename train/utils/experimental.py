# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""checkpoint 加载(train11111new 比赛版,从 models/experimental.py 精简)。

保留训练/验证需要的 Ensemble 与 attempt_load,模型类改为从 models11111new 导入。
"""

import torch
from torch import nn
from ultralytics.utils.patches import torch_load

from .downloads import attempt_download
from .general import LOGGER


# 多模型集成封装:对同一输入求各模型输出并拼接(仅单模型时等价于透传)
class Ensemble(nn.ModuleList):
    """Combines outputs from multiple models to improve inference results."""

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Applies ensemble of models on input `x`, with options for augmentation, profiling, and visualization,
        returning inference outputs.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


# 从 checkpoint 加载单模型或模型集成(缺失文件自动下载,可选 fuse 加速)
def attempt_load(weights, device=None, inplace=True, fuse=True):
    """Load a single model or an ensemble of models from one or more checkpoints.

    Args:
        weights (str | list[str]): Path or list of paths to model checkpoint(s); missing files are auto-downloaded.
        device (torch.device, optional): Device to load the model onto.
        inplace (bool): Use inplace ops (e.g. slice assignment) for compatibility across torch versions.
        fuse (bool): Fuse Conv2d and BatchNorm2d layers for faster inference.

    Returns:
        (torch.nn.Module): The loaded model, or an `Ensemble` if multiple weights are provided.
    """
    from models import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch_load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                m.anchor_grid = [torch.zeros(1)] * m.nl
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model
