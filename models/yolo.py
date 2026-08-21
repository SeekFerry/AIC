# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLOv3 模型主类与配置文件解析器（从 models/yolo.py 精简而来）。

用法：
    $ python yolo.py --cfg yolov3-spp.yaml
"""

import contextlib
import logging
import math
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .common import (
    Bottleneck,
    BottleneckCSP,
    C3,
    C3Ghost,
    C3SPP,
    C3TR,
    C3x,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    MixConv2d,
    SPP,
    SPPF,
    MultiModalFusion,
)

# -----------------------------------------------------------------------------------
# 以下辅助函数为精简自 utils/general.py、utils/torch_utils.py、utils/autoanchor.py
# 以及 ultralytics 依赖包的等价实现，保证本文件可独立复制使用。
# -----------------------------------------------------------------------------------

# 日志记录器（等价于 yolov3 中从 ultralytics 复用的 LOGGER）
LOGGER = logging.getLogger("yolov3")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


# 给字符串添加 ANSI 颜色（源自 ultralytics.utils.colorstr）
def colorstr(*input):
    """Color a string using ANSI escape codes; e.g. colorstr('blue', 'bold', 'hello')."""
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


# 返回不小于 x 且能被 divisor 整除的最小值（源自 ultralytics.utils.ops.make_divisible）
def make_divisible(x, divisor):
    """Return the smallest number >= x that is divisible by the given divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


# 检查当前版本是否满足 required 的 pip 风格版本要求（精简自 ultralytics.utils.checks.check_version）
def check_version(current="0.0.0", required="0.0.0", name="version", hard=False, msg=""):
    """Check current version against the required version or range (pip-style, e.g. '>=1.10.0')."""

    def parse_version(version):
        """Parse a version string into a comparable tuple of integers, ignoring prefixes/suffixes (e.g. '2.13.0+cpu')."""
        try:
            nums = [int(x) for x in re.search(r"\d+(?:\.\d+)*", version).group(0).split(".")]
            return tuple(nums + [0] * (3 - len(nums)))
        except Exception:
            return (0, 0, 0)

    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        n = max(len(c), len(v))  # pad to equal length for exact comparison
        cn, vn = c + (0,) * (n - len(c)), v + (0,) * (n - len(v))
        if (
            (op == "==" and cn != vn)
            or (op == "!=" and cn == vn)
            or (op == ">=" and not (cn >= vn))
            or (op == "<=" and not (cn <= vn))
            or (op == ">" and not (cn > vn))
            or (op == "<" and not (cn < vn))
        ):
            result = False
    if not result and hard:
        raise ModuleNotFoundError(f"{name}{required} is required, but {name}=={current} is currently installed {msg}")
    return result


# 将 Conv2d 与 BatchNorm2d 融合为单个卷积（源自 ultralytics.utils.torch_utils.fuse_conv_and_bn）
def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d and BatchNorm2d layers into a single convolution for faster inference."""
    # Compute fused weights: Conv2d weight is [out_channels, in_channels // groups, kH, kW], scale along axis 0
    bn_scale = bn.weight.div(torch.sqrt(bn.eps + bn.running_var))
    conv.weight.data = conv.weight * bn_scale.view(-1, 1, 1, 1)

    # Compute fused bias
    b_conv = (
        torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = bn_scale * b_conv + b_bn

    if conv.bias is None:
        conv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        conv.bias.data = fused_bias

    return conv.requires_grad_(False)


# 初始化模型权重/偏置与模块默认设置（源自 ultralytics.utils.torch_utils.initialize_weights）
def initialize_weights(model):
    """Initialize model weights, biases, and module settings to default values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


# 打印模型摘要信息（精简自 ultralytics.utils.torch_utils.model_info，不计算 FLOPs）
def model_info(model, detailed=False, verbose=True, imgsz=640):
    """Print and return model layer/parameter/gradient counts; `detailed` prints per-layer info."""
    if not verbose:
        return
    n_p = sum(p.numel() for p in model.parameters())  # number of parameters
    n_g = sum(p.numel() for p in model.parameters() if p.requires_grad)  # number of gradients
    layers = OrderedDict((n, m) for n, m in model.named_modules() if len(m._modules) == 0)
    n_l = len(layers)  # number of layers
    if detailed:
        LOGGER.info(f"{'layer':>5}{'name':>40}{'type':>20}{'gradient':>10}{'parameters':>12}{'shape':>20}")
        for i, (mn, m) in enumerate(layers.items()):
            mn = mn.replace("module_list.", "")
            mt = m.__class__.__name__
            if len(m._parameters):
                for pn, p in m.named_parameters():
                    LOGGER.info(
                        f"{i:>5g}{f'{mn}.{pn}':>40}{mt:>20}{p.requires_grad!r:>10}{p.numel():>12g}{list(p.shape)!s:>20}"
                    )
            else:  # layers with no learnable params
                LOGGER.info(f"{i:>5g}{mn:>40}{mt:>20}{False!r:>10}{0:>12g}{[]!s:>20}")
    LOGGER.info(f"Model summary: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients")
    return n_l, n_p, n_g, 0


# 缩放并填充图像张量（源自 ultralytics.utils.torch_utils.scale_img，用于增强推理 TTA）
def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scale and pad an image tensor, optionally maintaining aspect ratio and padding to gs multiple."""
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


PREFIX = colorstr("AutoAnchor: ")


# 检查并纠正 Detect 模块中的锚框顺序，使其与 stride 顺序一致（源自 utils/autoanchor.py）
def check_anchor_order(m):
    """Checks and corrects anchor order in YOLOv3's Detect() module if mismatched with stride order."""
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # anchor order does not match stride order
        LOGGER.info(f"{PREFIX}Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)

"""YOLOv3 检测头：将特征图转换为预测，并在推理时生成网格与锚框网格用于解码。"""
class Detect(nn.Module):

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """Initializes YOLOv3 detection layer with class count, anchors, channels, and operation modes."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through convolutional layers, reshaping output for detection.

        Expects x as list of tensors with shape(bs, C, H, W).
        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):  # noqa: B008
        """Generates a coordinate grid and anchor grid of shape `(1, na, ny, nx, 2)` for decoding Detect outputs."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = (
            torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)
        )  # torch>=1.10 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


"""YOLOv3 模型基类：定义前向传播、层融合、摘要打印与设备迁移等通用能力。"""
class BaseModel(nn.Module):
    def forward(self, x, profile=False, visualize=False):
        """Performs a single-scale inference or training step on input `x`（profile/visualize 为兼容保留参数）。"""
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """Executes a single inference or training step over the assembled layer sequence `self.model`."""
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """Fuses Conv2d() and BatchNorm2d() layers in the model to optimize inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        """Prints model information; `verbose` for detailed, `img_size` for input image size (default 640)."""
        model_info(self, detailed=verbose, imgsz=img_size)

    def _apply(self, fn):
        """Applies `to()`, `cpu()`, `cuda()`, `half()` to model tensors, excluding parameters or registered buffers."""
        super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


"""YOLOv3 检测模型：读取 yaml 配置构建网络，并初始化 stride、锚框与权重。"""
class DetectionModel(BaseModel):
    def __init__(self, cfg="yolov3-spp.yaml", ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """Initializes YOLOv3 detection model with configurable YAML, input channels, classes, and anchors."""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Processes input through the model, with options for augmentation, profiling, and visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference by scaling and flipping input images, returning concatenated predictions."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """Rescales predictions after augmentation by adjusting scales and flips based on image dimensions."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """Clips augmented inference tails from YOLOv3 predictions, affecting the first and last detection layers."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        """Initializes biases for objectness and classes in Detect() module; optionally uses class frequency `cf`."""
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv3 'Model' class for backwards compatibility


"""Parse a YOLOv3 model dict into an `nn.Sequential` module, scaling depth and width per the config.

    Args:
        d (dict): Model configuration with `backbone`/`head` layer lists plus `anchors`, `nc`, `depth_multiple`, and
            `width_multiple` keys.
        ch (list[int]): Input channels, typically `[3]` for RGB images.

    Returns:
        model (torch.nn.Sequential): Assembled model layers.
        save (list[int]): Sorted indices of layers whose outputs are retained for later use (skip connections).
    """
def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"], d.get("activation")
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{f!s:>18}{n_:>3}{np:10.0f}  {t:<40}{args!s:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

class MultiModalBackbone(nn.Module):
    """
    三流共享权重的骨干网络
    三个模态使用相同的 backbone 结构，但不共享权重（各自独立）
    """
    def __init__(self, cfg, ch=(3, 3, 3)):
        super().__init__()
        self.num_modalities = len(ch)
        
        # 为每个模态创建独立的 backbone
        self.backbones = nn.ModuleList()
        for c in ch:
            model = DetectionModel(cfg=cfg, ch=c)
            self.backbones.append(model)
        
        # 记录每个模态输出的特征图索引（需要根据 YAML 调整）
        self.p3_idx = 6
        self.p4_idx = 8
        self.p5_idx = 10
    
    def forward(self, x_rgb, x_depth, x_ir):
        """
        输入：三个模态的图像 [B, 3, H, W]
        输出：三个模态的 P3, P4, P5 特征图
        """
        outs_rgb = []
        outs_depth = []
        outs_ir = []
        
        for i, (x, backbone) in enumerate(zip([x_rgb, x_depth, x_ir], self.backbones)):
            y = []
            x_i = x
            for m in backbone.model:
                if m.f != -1:
                    x_i = y[m.f] if isinstance(m.f, int) else [x_i if j == -1 else y[j] for j in m.f]
                x_i = m(x_i)
                y.append(x_i if m.i in backbone.save else None)
            
            p3 = y[self.p3_idx]
            p4 = y[self.p4_idx]
            p5 = y[self.p5_idx]
            
            if i == 0:   # RGB
                outs_rgb.extend([p3, p4, p5])
            elif i == 1: # Depth
                outs_depth.extend([p3, p4, p5])
            elif i == 2: # IR
                outs_ir.extend([p3, p4, p5])
        
        # 返回三个模态的特征图，格式为 [模态数][尺度数]
        return {
            'rgb': [outs_rgb[0], outs_rgb[1], outs_rgb[2]],
            'depth': [outs_depth[0], outs_depth[1], outs_depth[2]],
            'ir': [outs_ir[0], outs_ir[1], outs_ir[2]],
        }


class MultiModalDetectionModel(BaseModel):
    """
    三模态检测模型：三条独立流 → 融合 → 检测头
    """
    def __init__(self, cfg='yolov3-spp.yaml', ch=(3, 3, 3), nc=None, anchors=None):
        super().__init__()
        self.num_modalities = len(ch)
        self.names = [str(i) for i in range(nc)] if nc else []
        
        import yaml
        with open(cfg, 'r') as f:
            self.yaml = yaml.safe_load(f)
        
        # 构建三个独立 backbone
        self.backbones = nn.ModuleList()
        for c in ch:
            temp_cfg = deepcopy(self.yaml)
            temp_cfg['ch'] = c
            temp_model, save = parse_model(
                {'backbone': temp_cfg['backbone'],
                 'head': temp_cfg['head'], 
                 'nc': temp_cfg['nc'], 
                 'depth_multiple': temp_cfg['depth_multiple'], 
                 'width_multiple': temp_cfg['width_multiple'],
                 'anchors': temp_cfg['anchors']}, 
                ch=[c]
            )
            backbone_layers = nn.Sequential(*[temp_model[i] for i in range(11)])
            backbone_wrapper = BaseModel()
            backbone_wrapper.model = backbone_layers
            backbone_wrapper.save = save
            self.backbones.append(backbone_wrapper)
            del temp_model
        
        # 定义融合模块（在 P3, P4, P5 三个尺度分别融合）
        self.fusion_p3 = MultiModalFusion(c1=256, c2=256, hide_channel=8)
        self.fusion_p4 = MultiModalFusion(c1=512, c2=512, hide_channel=8)
        self.fusion_p5 = MultiModalFusion(c1=1024, c2=1024, hide_channel=8)
        
        # 构建 head（从 YAML 中解析 head 部分）
                
        full_model, full_save = parse_model(
            {
                'backbone': self.yaml['backbone'],
                'head': self.yaml['head'],
                'nc': self.yaml['nc'],
                'depth_multiple': self.yaml['depth_multiple'],
                'width_multiple': self.yaml['width_multiple'],
                'anchors': self.yaml['anchors']
            },
            ch=[3]  # 单流输入，只是为了解析 head 结构
        )
                
        head_layers = []
        for i in range(11, len(full_model)):
            m = full_model[i]
            # 修改 m.f 适配输入
            if m.f != -1:
                if isinstance(m.f, int):
                    if m.f in [6, 8, 10]:  # P3, P4, P5
                        m.f = [0, 1, 2][[6, 8, 10].index(m.f)]
                elif isinstance(m.f, list):
                    new_f = []
                    for f in m.f:
                        if f in [6, 8, 10]:
                            new_f.append([0, 1, 2][[6, 8, 10].index(f)])
                        elif f == -1:
                            new_f.append(-1)
                        else:
                            new_f.append(f)
                    m.f = new_f
            head_layers.append(m)

        self.head = nn.ModuleList(head_layers)
        
        # 初始化检测头
        m = self.head[-1]
        if isinstance(m, Detect):
            s = 256
            dummy = torch.zeros(1, 3, s, s)
            m.stride = torch.tensor([s / x.shape[-2] for x in self._forward_dummy(dummy)])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride

    def _forward_dummy(self, x):
        return self.forward(x, x, x)
    
    def forward(self, x_rgb, x_depth, x_ir, augment=False, profile=False, visualize=False):
        """
        输入：三个模态的图像
        """
        # 1. 分别通过三个 backbone
        feat_rgb = self._forward_backbone(self.backbones[0], x_rgb)
        feat_depth = self._forward_backbone(self.backbones[1], x_depth)
        feat_ir = self._forward_backbone(self.backbones[2], x_ir)
        
        # 分别融合
        p3_fused = self.fusion_p3(feat_rgb[0], feat_depth[0], feat_ir[0])
        p4_fused = self.fusion_p4(feat_rgb[1], feat_depth[1], feat_ir[1])
        p5_fused = self.fusion_p5(feat_rgb[2], feat_depth[2], feat_ir[2])
        
        x = [p3_fused, p4_fused, p5_fused]
        for m in self.head:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = [x[m.f]]
                else:
                    x = [x[j] if isinstance(j, int) else [x[k] for k in j] for j in m.f]
            x = m(x)
        
        return x
    
    def _forward_backbone(self, backbone, x):
        """获取 backbone 的三个尺度输出"""
        y = []
        for m in backbone.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in backbone.save else None)
        
        # 需要根据实际 YAML 调整这些索引
        p3_idx, p4_idx, p5_idx = 6, 8, 10 
        return [y[p3_idx], y[p4_idx], y[p5_idx]]