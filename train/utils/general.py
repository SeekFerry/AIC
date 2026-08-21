# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""通用工具函数集合(train11111new 比赛版,从 utils/general.py 精简)。

保留 train.py / val.py 实际依赖的函数,删除比赛用不到的:
日志器(Loggers)、超参进化(evolve)、环境探测(colab/jupyter/kaggle/docker)、
分类器后处理(apply_classifier)、图像显示检查(check_imshow)等。
"""

from __future__ import annotations

import glob
import inspect
import os
import platform
import random
import re
import subprocess
import sys
import time
import urllib
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from ultralytics.data.converter import coco80_to_coco91_class  # noqa: F401
from ultralytics.utils import LOGGER, TQDM, colorstr  # noqa: F401
from ultralytics.utils.checks import is_ascii, print_args  # noqa: F401
from ultralytics.utils.checks import check_version as check_version_ultralytics
from ultralytics.utils.files import WorkingDirectory, file_date, get_latest_run  # noqa: F401
from ultralytics.utils.files import increment_path as increment_path_ultralytics
from ultralytics.utils.git import GitRepo
from ultralytics.utils.ops import (  # noqa: F401
    Profile,
    clip_boxes,
    make_divisible,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywh,
    xyxy2xywhn,
)
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import autocast, intersect_dicts, one_cycle  # noqa: F401

from . import TryExcept, emojis
from .downloads import curl_download
from .metrics import box_iou

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 仓库根目录(data11111new / models11111new / losses11111new 所在层级)

# 全局设置
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # YOLOv3 多进程线程数
DATASETS_DIR = Path(os.getenv("YOLOv3_DATASETS_DIR", ROOT.parent / "datasets"))  # 数据集缓存目录
AUTOINSTALL = str(os.getenv("YOLOv3_AUTOINSTALL", "true")).lower() == "true"  # 自动安装开关
FONT = "Arial.ttf"  # 绘图字体下载地址对应文件名

torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # 格式化输出
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # 关闭 OpenCV 多线程(与 PyTorch DataLoader 不兼容)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr 最大线程数
os.environ["OMP_NUM_THREADS"] = "1" if platform.system() == "Darwin" else str(NUM_THREADS)  # OpenMP 线程数


# 判断目录是否可写(可选通过实际写文件验证);用于定位用户配置目录
def is_writeable(dir, test=False):
    """Determines if a directory is writeable, optionally tests by writing a file if `test=True`."""
    if not test:
        return os.access(dir, os.W_OK)  # possible issues on Windows
    file = Path(dir) / "tmp.txt"
    try:
        with open(file, "w"):  # open file with write permissions
            pass
        file.unlink()  # remove file
        return True
    except OSError:
        return False


# 返回用户配置目录路径(优先环境变量,否则按操作系统),必要时创建
def user_config_dir(dir="Ultralytics", env_var="YOLOV3_CONFIG_DIR"):
    """Returns user configuration directory path, prefers `env_var` if set, else uses OS-specific path, creates
    directory if needed.
    """
    if env := os.getenv(env_var):
        path = Path(env)  # use environment variable
    else:
        cfg = {"Windows": "AppData/Roaming", "Linux": ".config", "Darwin": "Library/Application Support"}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), "")  # OS-specific config dir
        path = (path if is_writeable(path) else Path("/tmp")) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


CONFIG_DIR = user_config_dir()  # Ultralytics settings dir


# 初始化随机种子以保证可复现;deterministic=True 时开启确定性算法
# Keep local (do not dedup): stricter deterministic semantics than ultralytics init_seeds (no warn_only fallback)
def init_seeds(seed=0, deterministic=False):
    """Initializes RNG seeds for reproducibility; `seed`: RNG seed, `deterministic`: enforces deterministic behavior if
    True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    if deterministic and check_version(torch.__version__, "1.12.0"):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


# 检测网络连通性(两次尝试连接 1.1.1.1:443)
def check_online():
    """Checks internet connectivity by attempting to connect to "1.1.1.1" on port 443 twice; returns True if successful."""
    import socket

    def run_once():
        """Attempts a single internet connectivity check to '1.1.1.1' on port 443 and returns True if successful."""
        try:
            socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
            return True
        except OSError:
            return False

    return run_once() or run_once()  # check twice to increase robustness to intermittent connectivity issues


# 返回目录的 git 描述字符串(非 git 仓库则返回空)
def git_describe(path=ROOT):  # path must be a directory
    """Returns human-readable git description of a directory if it's a git repository, otherwise an empty string."""
    try:
        assert (Path(path) / ".git").exists()  # dir in a clone, file in a worktree or submodule
        return check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
    except Exception:
        return ""


# 检查 git 信息,返回 {remote, branch, commit} 字典(写进 checkpoint)
def check_git_info(path=ROOT):
    """Checks YOLOv3 git info, returning a dict with remote URL, branch name, and commit hash."""
    repo = GitRepo(path)
    if repo.root != Path(path):  # GitRepo searches parents, ignore a repo that merely contains YOLOv3
        return {"remote": None, "branch": None, "commit": None}
    remote = repo.origin.replace(".git", "") if repo.origin else None  # i.e. 'https://github.com/ultralytics/yolov3'
    return {"remote": remote, "branch": repo.branch, "commit": repo.commit}  # branch is None on detached HEAD


# 版本检查封装(current 需满足 minimum,pinned=True 时精确匹配)
def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False):
    """Check `current` against a `minimum` version (exact match if `pinned`) with the installed Ultralytics checker."""
    return check_version_ultralytics(current, f"=={minimum}" if pinned else f">={minimum}", name, hard, verbose)


# 将图像尺寸调整为 stride 的整数倍并保证不小于 floor
def check_img_size(imgsz, s=32, floor=0):
    """Adjusts image size to be divisible by `s`, ensuring it's above `floor`; returns int for single dim or list for
    dims.
    """
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f"--img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}")
    return new_size


# 校验文件后缀是否符合允许列表
def check_suffix(file="yolov3-tiny.pt", suffix=(".pt",), msg=""):
    """Checks for acceptable file suffixes, supports batch checking for lists or tuples of filenames."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


# 查找/下载 YAML 文件并返回路径,强制 .yaml/.yml 后缀
def check_yaml(file, suffix=(".yaml", ".yml")):
    """Searches/downloads a YAML file and returns its path, ensuring it has a .yaml or .yml suffix."""
    return check_file(file, suffix)


# 查找文件:本地存在/URL 下载/在 data、data11111new、models11111new、utils 中搜索
def check_file(file, suffix=""):
    """Checks for file's existence locally, downloads if a URL, and enforces optional suffix."""
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file
    elif file.startswith(("http:/", "https:/")):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split("?")[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            LOGGER.info(f"Found {url} locally at {file}")  # file already exists
        else:
            LOGGER.info(f"Downloading {url} to {file}...")
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f"File download failed: {url}"  # check
        return file
    else:  # search
        files = []
        for d in "data", "data11111new", "models11111new", "utils":  # search directories
            files.extend(glob.glob(str(ROOT / d / "**" / file), recursive=True))  # find file
        assert len(files), f"File not found: {file}"  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


# 检查并下载绘图字体到 CONFIG_DIR
def check_font(font=FONT, progress=False):
    """Checks and downloads the specified font to CONFIG_DIR if not present, with optional download progress."""
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{font.name}"
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=progress)


# 校验数据集配置:自动下载、解析 yaml、补齐 nc 与绝对路径、按需下载数据集
def check_dataset(data, autodownload=True):
    """Verifies and prepares dataset by downloading if absent, checking, and unzipping; supports auto-downloading."""
    # Download (optional)
    extract_dir = ""
    if isinstance(data, (str, Path)) and (is_zipfile(data) or is_tarfile(data)):
        download(data, dir=f"{DATASETS_DIR}/{Path(data).stem}", unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob("*.yaml"))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # dictionary

    # Checks
    for k in "train", "val", "names":
        assert k in data, emojis(f"data.yaml '{k}:' field missing ❌")
    if isinstance(data["names"], (list, tuple)):  # old array format
        data["names"] = dict(enumerate(data["names"]))  # convert to dict
    assert all(isinstance(k, int) for k in data["names"]), "data.yaml names keys must be integers, i.e. 2: car"
    data["nc"] = len(data["names"])

    # Resolve paths
    path = Path(extract_dir or data.get("path") or "")  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data["path"] = path  # download scripts
    for k in "train", "val", "test":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse yaml
    _train, val, _test, s = (data.get(x) for x in ("train", "val", "test", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            LOGGER.info("\nDataset not found ⚠️, missing paths %s" % [str(x) for x in val if not x.exists()])
            if not s or not autodownload:
                raise RuntimeError("Dataset not found ❌")
            t = time.time()
            if s.startswith("http") and s.endswith(".zip"):  # URL
                download(s, dir=DATASETS_DIR, curl=True)
                r = 0  # success
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = subprocess.run(s, shell=True, check=False).returncode
            else:  # python script
                exec(s, {"yaml": data})  # noqa: S102
                r = 0  # exec returns None, treat completion without exception as success
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r == 0 else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf", progress=True)  # download fonts
    return data  # dictionary


# 检查 AMP 是否可用(比赛版:仅用传入模型比较 FP32/AMP 输出,不依赖 DetectMultiBackend/AutoShape)
def check_amp(model):
    """Checks PyTorch AMP functionality with the given model, returning True if AMP operates correctly."""
    device = next(model.parameters()).device  # get model device
    if device.type in ("cpu", "mps"):
        return False  # AMP only used on CUDA devices

    def amp_allclose(model, im):
        """Compares FP32 and AMP inference results for a model and image, ensuring outputs are within 10% tolerance."""
        with torch.no_grad():
            a = model(im)[0]  # FP32 inference
        with autocast(enabled=True), torch.no_grad():
            b = model(im)[0]  # AMP inference
        return a.shape == b.shape and torch.allclose(a, b, atol=0.1)  # close to 10% absolute tolerance

    prefix = colorstr("AMP: ")
    f = ROOT / "data" / "images" / "bus.jpg"  # image to check
    # 输入通道数需与模型一致(三模态早融合为 9 通道)
    in_ch = 3
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            in_ch = m.in_channels
            break
    im = torch.zeros((1, in_ch, 640, 640), device=device)  # fallback dummy image
    if f.exists():
        try:
            from data import LoadImages

            sample = next(iter(LoadImages(f, img_size=640, stride=32)))
            im = torch.from_numpy(sample[1]).float().unsqueeze(0).to(device)
        except Exception:
            pass
    try:
        assert amp_allclose(deepcopy(model).eval(), im)
        LOGGER.info(f"{prefix}checks passed ✅")
        return True
    except Exception:
        help_url = "https://docs.ultralytics.com/guides/yolo-common-issues/"
        LOGGER.warning(f"{prefix}checks failed, disabling Automatic Mixed Precision. See {help_url}")
        return False


# 安全加载 YAML 文件为字典(忽略文件编码错误)
def yaml_load(file="data.yaml"):
    """Safely loads a YAML file, ignoring file errors; default file is 'data.yaml'."""
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)


# 将字典保存为 YAML(Path 对象先转 str)
def yaml_save(file="data.yaml", data=None):
    """Safely saves data to a YAML file, converting `Path` objects to strings; defaults to 'data.yaml'."""
    if data is None:
        data = {}
    with open(file, "w") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


# 解压 zip 文件(默认到文件所在目录,排除隐藏文件)
def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX")):
    """Unzips '*.zip' to `path` (default: file's parent), excluding files matching `exclude` (`('.DS_Store',
    '__MACOSX')`).
    """
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)


# 下载 URL(可多线程、可解压、可删除压缩包),支持 gz/tar/zip
def download(url, dir=".", unzip=True, delete=True, curl=False, threads=1, retry=3):
    """Downloads files from URLs into a specified directory, optionally unzips, and supports multithreading and retries."""

    def download_one(url, dir):
        """Downloads a file from a URL into the specified directory, supporting retries and using curl or torch methods."""
        success = True
        if os.path.isfile(url):
            f = Path(url)  # filename
        else:  # does not exist
            f = dir / Path(url).name
            LOGGER.info(f"Downloading {url} to {f}...")
            for i in range(retry + 1):
                if curl:
                    success = curl_download(url, f, silent=(threads > 1))
                else:
                    torch.hub.download_url_to_file(url, f, progress=threads == 1)  # torch download
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f"Download failure, retrying {i + 1}/{retry} {url}...")
                else:
                    LOGGER.warning(f"Failed to download {url}...")

        if unzip and success and (f.suffix == ".gz" or is_zipfile(f) or is_tarfile(f)):
            LOGGER.info(f"Unzipping {f}...")
            if is_zipfile(f):
                unzip_file(f, dir)  # unzip
            elif is_tarfile(f):
                subprocess.run(["tar", "xf", f, "--directory", f.parent], check=True)  # unzip
            elif f.suffix == ".gz":
                subprocess.run(["tar", "xfz", f, "--directory", f.parent], check=True)  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


# 根据标签统计各类别权重,用于缓解类别不平衡
def labels_to_class_weights(labels, nc=80):
    """Calculates class weights from labels to counteract dataset imbalance; `labels` is a list of numpy arrays with
    shape `(n, 5)`.
    """
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float()


# 根据标签与类别权重计算每张图的采样权重,用于平衡采样
def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):  # noqa: B008
    """Calculates image weights from labels using class weights, for balanced sampling."""
    # Usage: index = random.choices(range(n), weights=image_weights, k=1)  # weighted image sample
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)


# 将推理尺寸下的预测框映射回原图尺寸(后处理必需)
# Keep local (do not dedup): ultralytics scale_boxes sub-pixel rounding differs and shifts mAP
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescales bounding boxes from one image shape to another, optionally with ratio and padding adjustments."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


# 非极大值抑制(NMS):过滤重叠框,保留每个目标的最优检测
# Keep local (do not dedup): anchor-based YOLOv3 head has an objectness channel; ultralytics NMS is anchor-free
def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """Run Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Args:
        prediction (torch.Tensor | list | tuple): Model output; the inference tensor is used if a tuple/list is passed.
        conf_thres (float): Confidence threshold in [0, 1]; boxes below it are discarded.
        iou_thres (float): IoU threshold in [0, 1] for the NMS overlap test.
        classes (list | None): If set, keep only detections whose class is in this list.
        agnostic (bool): If True, perform class-agnostic NMS (boxes not offset by class).
        multi_label (bool): If True, allow multiple labels per box.
        labels (tuple): Optional apriori labels per image for autolabelling.
        max_det (int): Maximum number of detections to keep per image.
        nm (int): Number of mask coefficients (segmentation); 0 for detection.

    Returns:
        (list[torch.Tensor]): One (n, 6 + nm) tensor per image, each row [xyxy, conf, cls, mask...].
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv3 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


# 从 checkpoint 中剔除优化器等冗余状态,压缩模型体积用于部署
# Keep local (do not dedup): tied to YOLOv3 checkpoint format
def strip_optimizer(f="best.pt", s=""):  # from utils.general import *; strip_optimizer()
    """Strips optimizer from a checkpoint file 'f', optionally saving as 's', to finalize training."""
    x = torch_load(f, map_location=torch.device("cpu"))
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with ema
    for k in "optimizer", "best_fitness", "ema", "updates":  # keys
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


# 路径自增封装(保持 YOLOv3 默认 sep='' 即 runs/exp2 命名风格)
def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """Increment a path with the installed Ultralytics helper, keeping the YOLOv3 default `sep=''` (runs/exp2)."""
    return increment_path_ultralytics(path, exist_ok=exist_ok, sep=sep, mkdir=mkdir)


# OpenCV 多语言路径友好函数
# ------------------------------------------------------------------------------------
imshow_ = cv2.imshow  # copy to avoid recursion errors


# 读取图像(支持中文/多语言路径)
def imread(filename, flags=cv2.IMREAD_COLOR):
    """Reads an image from a file, supporting multilanguage paths, and returns it in the specified color scheme."""
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


# 写图像(支持中文/多语言路径)
def imwrite(filename, img):
    """Write an image to a file, supporting multilanguage paths; returns True on success, False on failure.

    Args:
        filename (str): Destination file path.
        img (np.ndarray): Image array to write.

    Returns:
        (bool): True if the image was written successfully, False otherwise.
    """
    try:
        cv2.imencode(Path(filename).suffix, img)[1].tofile(filename)
        return True
    except Exception:
        return False


# 显示图像(窗口名编码以支持非 ASCII)
def imshow(path, im):
    """Display image `im` (ndarray) in an OpenCV window named `path`, encoding the name for non-ASCII support."""
    imshow_(path.encode("unicode_escape").decode(), im)


if Path(inspect.stack()[0].filename).parent.parent.as_posix() in inspect.stack()[-1].filename:
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow  # redefine
