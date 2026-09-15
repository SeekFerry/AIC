# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
三模态测试集推理脚本(比赛提交格式)。

功能:
  1. 读取测试集根目录(内含 visible / infrared(或 infared) / depth 三个模态文件夹,无 labels)。
  2. 自动识别模型类型:
       - 三流模型 MultiModalDetectionModel -> 三个模态分开输入 model(visible, infrared, depth)
       - 单流早融合模型(9 通道)          -> 三模态拼成 9 通道一次输入
  3. 每张图生成同名预测 TXT,每行格式:
         class_id norm_center_x norm_center_y norm_w norm_h confidence
     未检测到目标时生成空 TXT(不允许缺失)。
  4. 每张图最多保留 --max-det 个框(按置信度截断)。
  5. 全部 TXT 打包为一个 zip 压缩包。

用法:
    python output.py --weights runs/train/exp/weights/best.pt
    (训练集/测试集路径统一在 data/tri.yaml 中填写:path 为训练集,test 为测试集)
"""

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
REPO_ROOT = FILE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data import letterbox  # noqa: E402
from train.utils.experimental import attempt_load  # noqa: E402
from train.utils.general import (  # noqa: E402
    LOGGER,
    TQDM,
    check_img_size,
    colorstr,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
    yaml_load,
)
from train.utils.torch_utils import select_device  # noqa: E402

IMG_FORMATS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm")


# 从 data/tri.yaml 读取测试集路径(训练集路径为 path,测试集路径为 test)。
def dataset_test_path():
    """Read the test dataset path from data/tri.yaml; fall back to a placeholder."""
    try:
        d = yaml_load(str(REPO_ROOT / "data" / "tri.yaml"))
    except Exception:
        d = {}
    test = d.get("test")
    if not test:
        return "D:/pytorch/AIC_sample_test"
    p = Path(test)
    if p.is_absolute():
        return str(p)
    # test 与 path 一样,都是相对代码根目录(REPO_ROOT)填写的,如 ../AIC2026_PHASE_1_1000
    return str((REPO_ROOT / p).resolve())


# 解析某个模态的实际目录,兼容 infared/infrared 的拼写差异。
def resolve_modality_dir(root, name):
    """Return the actual subfolder for modality `name`, tolerating the infared/infrared typo."""
    root = Path(root)
    if (root / name).is_dir():
        return root / name
    if name in ("infared", "infrared"):
        alt = "infrared" if name == "infared" else "infared"
        if (root / alt).is_dir():
            return root / alt
    return None


# 判断模型是否要求三个模态分别输入(三流 MultiModalDetectionModel)。
# 单流模型(9 通道早融合)只需一个输入张量,三流模型需要 (visible, infrared, depth) 三个张量。
def is_multi_stream(model):
    """Return True if the model expects three separate modality tensors (three-stream model)."""
    if hasattr(model, "backbones"):
        return True
    try:
        import inspect

        required = [
            p
            for p in inspect.signature(model.forward).parameters.values()
            if p.default is inspect.Parameter.empty and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        return len(required) > 1
    except Exception:
        return False


# 读取三个模态 -> 共享同一套 letterbox -> 各自转成 (1,3,H,W) 张量。
# 返回: [visible, infrared, depth] 三个张量, 原图尺寸 (h0, w0), letterbox 参数 (ratio, pad)
def load_multimodal_tensor(files, imgsz, stride):
    """Load visible/infrared/depth images with one shared letterbox and return per-modality tensors."""
    ims = []
    h0 = w0 = None
    for f in files:
        im = cv2.imread(str(f))  # BGR
        assert im is not None, f"Image Not Found {f}"
        if h0 is None:
            h0, w0 = im.shape[:2]
        ims.append(im)
    im = np.concatenate(ims, axis=2)  # HxWx(3*num_modalities),三模态共享同一套几何变换
    im, ratio, pad = letterbox(im, imgsz, stride=stride, auto=False)

    # 拆回三个模态,各自 HWC->CHW 且 BGR->RGB
    mods = np.split(im, len(ims), axis=2)
    tensors = [
        torch.from_numpy(np.ascontiguousarray(m.transpose(2, 0, 1)[::-1])).float().unsqueeze(0) / 255
        for m in mods
    ]
    return tensors, (h0, w0), (ratio, pad)


# 将一张图的检测结果按比赛格式写入 TXT(未检测到则写入空文件)。
def save_one_txt(predn, h0, w0, file):
    """Write normalized [class_id, cx, cy, w, h, conf] lines; empty file if no detections."""
    with open(file, "w") as f:
        for *xyxy, conf, cls in predn.tolist():
            xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
            cx, cy, w, h = xywh[0] / w0, xywh[1] / h0, xywh[2] / w0, xywh[3] / h0
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")


def run(
    weights,
    source,
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=100,
    device="",
    half=True,
    project=REPO_ROOT / "runs/predict",
    name="exp",
    exist_ok=False,
):
    """Run tri-modal inference over a test folder and pack per-image TXT predictions into a zip."""
    device = select_device(device, batch_size=1)

    # 加载模型
    model = attempt_load(weights, device=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    names = model.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    half &= device.type != "cpu"
    model.half() if half else model.float()
    model.eval()

    # 输出目录
    save_dir = Path(project) / name
    if not exist_ok:
        save_dir = _increment_path(save_dir)
    labels_dir = save_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 扫描测试集(以 visible 为锚点,三模态同名一一对应)
    root = Path(source)
    assert root.is_dir(), f"Test dataset folder not found: {root}"
    vis_dir = resolve_modality_dir(root, "visible")
    assert vis_dir is not None, f"Missing visible folder in {root}"
    ir_dir = resolve_modality_dir(root, "infrared")
    assert ir_dir is not None, f"Missing infrared/infared folder in {root}"
    dep_dir = resolve_modality_dir(root, "depth")
    assert dep_dir is not None, f"Missing depth folder in {root}"

    vis_files = sorted(
        x for x in vis_dir.rglob("*.*") if x.suffix.lower().lstrip(".") in IMG_FORMATS
    )
    assert vis_files, f"No images found in {vis_dir}"
    LOGGER.info(f"Found {len(vis_files)} test images in {root}")

    multi_stream = is_multi_stream(model)  # True: 三流模型(分开输入) / False: 9 通道早融合
    LOGGER.info(f"Model type: {'three-stream (visible/infrared/depth)' if multi_stream else 'single-stream (9-channel early fusion)'}")

    for vis_f in TQDM(vis_files, desc="Predicting"):
        rel = vis_f.relative_to(vis_dir)
        files = [vis_f, ir_dir / rel, dep_dir / rel]
        for f in files[1:]:
            assert f.is_file(), f"Missing paired image {f}"

        tensors, (h0, w0), ratio_pad = load_multimodal_tensor(files, imgsz, stride)
        tensors = [t.to(device, non_blocking=True) for t in tensors]
        tensors = [t.half() if half else t.float() for t in tensors]

        with torch.no_grad():
            if multi_stream:
                # 顺序必须与训练一致:model(visible, infrared, depth)
                pred = model(tensors[0], tensors[1], tensors[2])[0]
            else:
                pred = model(torch.cat(tensors, dim=1))[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)[0]

        # 预测框从推理尺寸映射回原图(像素坐标 xyxy)
        predn = pred.clone()
        scale_boxes(tensors[0].shape[2:], predn[:, :4], (h0, w0), ratio_pad=ratio_pad)

        save_one_txt(predn, h0, w0, labels_dir / f"{vis_f.stem}.txt")

    # 打包所有 TXT 为 zip,直接输出到 result 目录
    txt_files = sorted(labels_dir.glob("*.txt"))
    result_dir = REPO_ROOT / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    zip_path = result_dir / "predictions.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for txt in txt_files:
            z.write(txt, arcname=txt.name)
    LOGGER.info(f"{len(txt_files)} TXT files saved to {labels_dir}")
    LOGGER.info(f"Packed into {colorstr('bold', zip_path)}")


# 简单的目录自增命名(exp -> exp2 -> exp3 ...),避免覆盖已有结果。
def _increment_path(path):
    """Increment path like runs/predict/exp, runs/predict/exp2, ... if it already exists."""
    path = Path(path)
    if not path.exists():
        return path
    for n in range(2, 9999):
        p = path.with_name(path.name + str(n))
        if not p.exists():
            return p
    return path


def parse_opt():
    """Parse command-line arguments for the predict script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(REPO_ROOT / "runs/train/exp/weights/best.pt"), help="trained model path")
    parser.add_argument("--source", type=str, default=dataset_test_path(), help="test dataset folder (contains visible/infrared/depth)")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=100, help="maximum detections per image (task limit: 100)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs/predict"), help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="overwrite existing project/name")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
