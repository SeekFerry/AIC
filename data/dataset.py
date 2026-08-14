# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""数据集核心:扫路径、读图、返回原始像素,以及 DataLoader 封装。

原样提取自 ``utils/dataloaders.py``(``LoadImagesAndLabels`` / ``LoadImages`` /
``create_dataloader`` / ``InfiniteDataLoader`` / ``verify_image_label`` 等),
删除了比赛用不上的流媒体/截屏加载器(``LoadStreams`` / ``LoadScreenshots``)。
``torch_distributed_zero_first`` 原样提取自 ``utils/torch_utils.py``。
"""

import contextlib
import glob
import math
import os
import random
from contextlib import contextmanager
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import torch.distributed as dist
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from ultralytics.data.build import seed_worker
from ultralytics.data.utils import get_hash, img2label_paths
from ultralytics.utils import LOGGER, TQDM

from .collate import collate_fn, collate_fn3, collate_fn4
from .transforms import Albumentations, augment_hsv, copy_paste, mixup, random_perspective
from .utils import letterbox, segments2boxes, xyn2xy, xywhn2xyxy, xyxy2xywhn

# Parameters
HELP_URL = "See https://docs.ultralytics.com/datasets/ for dataset formatting guidance"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "-1"))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", "-1"))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", "true")).lower() == "true"  # global pin_memory for dataloaders
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv3 multiprocessing threads

# Get orientation exif tag
for orientation in ExifTags.TAGS:
    if ExifTags.TAGS[orientation] == "Orientation":
        break

# 三模态比赛数据集:根目录下的模态文件夹(以第一个为锚点,同名文件一一对应)。
MODALITY_DIRS = ("visible", "infared", "depth")
LABEL_DIR_NAMES = ("labels",)


# 解析某个模态的实际目录,兼容 infared/infrared 的拼写差异。
def _resolve_modality_dir(root, name):
    """Return the actual subfolder for modality `name`, tolerating the infared/infrared typo."""
    root = Path(root)
    if (root / name).is_dir():
        return root / name
    if name in ("infared", "infrared"):
        alt = "infrared" if name == "infared" else "infared"
        if (root / alt).is_dir():
            return root / alt
    return None


# 查找 labels 标注目录。
def _resolve_label_dir(root, names=LABEL_DIR_NAMES):
    """Return the labels subfolder under `root`, or None if absent."""
    root = Path(root)
    for name in names:
        if (root / name).is_dir():
            return root / name
    return None


# 由锚点图像路径 + labels 目录得到同名 txt 标注路径。
def _modal_to_label(im_file, labels_dir):
    """Map a modality image path to its label txt path (same stem, .txt suffix)."""
    return str(Path(labels_dir) / f"{Path(im_file).stem}.txt")


# 判断目录是否为三模态比赛数据集,返回模态数量(普通单模态返回 1)。
def dataset_num_modalities(path, modalities=MODALITY_DIRS):
    """Return the number of modality folders under `path` (1 if the layout is not detected)."""
    root = Path(path)
    if root.is_dir() and _resolve_modality_dir(root, modalities[0]) is not None:
        return len(modalities)
    return 1


# 考虑 EXIF 旋转元数据,返回修正后的图像宽高 (w, h)。
def exif_size(img):
    """Returns corrected image size (width, height) considering EXIF rotation metadata."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


# DDP 下保证每个 rank 只初始化一次缓存(非主 rank 先等主 rank 完成)。
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Context manager ensuring ordered execution in distributed training by synchronizing local masters first."""
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


# 训练入口:构造 LoadImagesAndLabels 并返回 (DataLoader, dataset),处理 worker 数/采样器/随机种子。
def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    shuffle=False,
    seed=0,
    modalities=None,
):
    """Creates a DataLoader for training, with options for augmentation, caching, and parallelization."""
    if rect and shuffle:
        LOGGER.warning("--rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            modalities=modalities,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    collate = collate_fn3 if getattr(dataset, "multimodal", False) else (collate_fn4 if quad else collate_fn)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset

# 无限重复采样器的 DataLoader:训练时避免每个 epoch 重建 worker,提速。
class InfiniteDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax and a repeating
        sampler.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Iterates over the dataset indefinitely, yielding batches from the batch_sampler."""
        for _ in range(len(self)):
            yield next(self.iterator)

# 无限重复的采样器,配合 InfiniteDataLoader 使用。
class _RepeatSampler:
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """Initializes an infinitely repeating sampler with a provided `sampler` object."""
        self.sampler = sampler

    def __iter__(self):
        """Provides an iterator that infinitely repeats over a given `sampler` object."""
        while True:
            yield from iter(self.sampler)

# 推理/验证加载器:扫目录/图片/视频,逐张返回 (path, 预处理后的图, 原图, cap, 信息串)。
class LoadImages:
    """Loads images and videos for YOLOv3 from various sources, including directories and '*.txt' path lists."""

    # 初始化:支持图片/视频/目录/文本列表输入,设置缩放尺寸、stride、可选 transforms 与视频抽帧步长。
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initializes the data loader for YOLOv3, supporting image, video, directory, and '*.txt' path lists with
        customizable image sizing.
        """
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = [x for x in Path(path).read_text().splitlines() if x.strip()]
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        """Initializes the iterator by resetting count to zero and returning the iterator instance itself."""
        self.count = 0
        return self

    # 取下一帧/下一张图,读到末尾抛 StopIteration;图片做 letterbox 并转 CHW、BGR->RGB。
    def __next__(self):
        """Advances to the next file in the dataset, raising StopIteration when all files are processed."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f"Image Not Found {path}"
            s = f"image {self.count}/{self.nf} {path}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

# （删）打开一个新的视频文件,统计总帧数与旋转方向元数据。
    def _new_video(self, path):
        """Initializes a video capture object with frame counting and orientation from a given path."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
# （删）按视频元数据里的旋转角度旋转图像(用于视频帧)。
    
    def _cv2_rotate(self, im):
        """Rotates a cv2 image based on the video's metadata orientation; returns the rotated image."""
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        """Returns the number of files in the dataset."""
        return self.nf  # number of files

# 训练 Dataset:扫路径、校验并缓存标签(*.cache)、矩形批、RAM/磁盘缓存;__getitem__ 完成 Mosaic/MixUp/letterbox/翻转/HSV 全流程。
class LoadImagesAndLabels(Dataset):
    """Loads images and labels for YOLOv3 training and validation with support for augmentations and caching."""

    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        min_items=0,
        prefix="",
        modalities=None,
    ):
        """Initializes a dataset with images and labels for YOLOv3 training and validation."""
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.modalities = tuple(modalities or MODALITY_DIRS)
        self.multimodal = False
        self.albumentations = Albumentations(size=img_size) if augment else None

        try:
            f = []  # image files
            modal_files = None  # [visible_files, ir_files, depth_files] when multimodal
            # 三模态比赛布局:path 为根目录,内含 visible/infared(infrared)/depth/labels 四个文件夹。
            root = Path(path) if not isinstance(path, list) else None
            anchor_dir = _resolve_modality_dir(root, self.modalities[0]) if root is not None else None
            if root is not None and root.is_dir() and anchor_dir is not None:
                self.multimodal = True
                cache_images = False  # 多模态暂不做图像缓存,避免与拼接通道数不一致
                self.albumentations = None  # 颜色类增强只对 RGB 有意义,三模态共享管线时关闭

                anchor_files = sorted(
                    x.replace("/", os.sep)
                    for x in glob.glob(str(anchor_dir / "**" / "*.*"), recursive=True)
                    if x.split(".")[-1].lower() in IMG_FORMATS
                )
                assert anchor_files, f"{prefix}No images found in {anchor_dir}"

                dirs = [_resolve_modality_dir(root, m) for m in self.modalities]
                missing = [m for m, d in zip(self.modalities, dirs) if d is None]
                assert not missing, f"{prefix}Missing modality folders {missing} in {root}"
                labels_dir = _resolve_label_dir(root)
                assert labels_dir is not None, f"{prefix}Missing labels folder in {root}"
                self.label_root = labels_dir

                modal_files = []
                for d in dirs:
                    files = []
                    for af in anchor_files:
                        rel = os.path.relpath(af, anchor_dir)
                        mf = os.path.normpath(os.path.join(d, rel))
                        assert os.path.isfile(mf), f"{prefix}Missing paired image {mf}"
                        files.append(mf)
                    modal_files.append(files)
                f = anchor_files
            else:
                for p in path if isinstance(path, list) else [path]:
                    p = Path(p)  # os-agnostic
                    if p.is_dir():  # dir
                        f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                        # f = list(p.rglob('*.*'))  # pathlib
                    elif p.is_file():  # file
                        with open(p) as t:
                            t = t.read().strip().splitlines()
                            parent = str(p.parent) + os.sep
                            f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # to global path
                            # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                    else:
                        raise FileNotFoundError(f"{prefix}{p} does not exist")
            self.im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f"{prefix}No images found"
            if modal_files is not None:
                self.modal_im_files = modal_files  # 与 self.im_files 顺序严格一致
        except Exception as e:
            raise RuntimeError(f"{prefix}Error loading data from {path}: {e}\n{HELP_URL}") from e

        # Check cache
        if self.multimodal:
            self.label_files = [_modal_to_label(f, self.label_root) for f in self.im_files]  # labels
            cache_path = self.label_root.with_suffix(".cache")
        else:
            self.label_files = img2label_paths(self.im_files)  # labels
            cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")
        # 缓存哈希需覆盖所有模态文件,避免某个模态更新后仍命中旧缓存。
        self.cache_hash_files = (
            self.label_files + [mf for files in self.modal_im_files for mf in files]
            if self.multimodal
            else self.label_files + self.im_files
        )
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.cache_hash_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            LOGGER.info(prefix + d)  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0 or not augment, f"{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f"{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}"
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())  # update
        if self.multimodal:
            self.label_files = [_modal_to_label(f, self.label_root) for f in self.im_files]  # update
        else:
            self.label_files = img2label_paths(cache.keys())  # update

        # Filter images
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f"{prefix}{n - len(include)}/{n} images filtered from dataset")
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh
            if self.multimodal:
                self.modal_im_files = [[mfs[i] for i in include] for mfs in self.modal_im_files]

        # Create indices
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]
            if self.multimodal:
                self.modal_im_files = [[mfs[i] for i in irect] for mfs in self.modal_im_files]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into RAM/disk for faster training
        if cache_images == "ram" and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == "disk" else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = TQDM(enumerate(results), total=n, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{prefix}Caching images ({b / gb:.1f}GB {cache_images})"
            pbar.close()

# 估算整库缓存所需 RAM,不够则返回 False 以关闭缓存。
    def check_cache_ram(self, safety_margin=0.1, prefix=""):
        """Evaluates if there's enough RAM to cache dataset images, considering a safety margin."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(
                f"{prefix}{mem_required / gb:.1f}GB RAM required, "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )
        return cache
    
# 多进程扫描所有图+标签,校验合法性并写 *.cache,加速后续加载。
    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """Caches dataset labels, checks image existence and readability, and records image shapes and segments."""
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."
        with Pool(NUM_THREADS) as pool:
            pbar = TQDM(
                pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                desc=desc,
                total=len(self.im_files),
            )
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.cache_hash_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x
    
# 返回数据集图像数量。
    def __len__(self):
        """Returns the number of image files in the dataset."""
        return len(self.im_files)

# 取第 index 个样本:走 Mosaic/MixUp 或 letterbox+透视,再做 HSV/翻转,最后转 CHW、BGR->RGB 并返回张量。    
    def __getitem__(self, index):
        """Fetches dataset item at `index` after applying indexing via `self.indices`, supporting
        linear/shuffled/image_weights modes.
        """
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        if self.mosaic and random.random() < hyp["mosaic"]:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

        if self.augment:
            if not self.multimodal:
                # Albumentations
                img, labels = self.albumentations(img, labels)
                nl = len(labels)  # update after albumentations

                # HSV color-space
                augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down(几何增强对三个模态一致应用)
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert:单模态 HWC->CHW + BGR->RGB;多模态先按通道拆成三组,再分别转 CHW。
        if self.multimodal:
            ims = np.split(img, len(self.modalities), axis=2)  # list of HxWx3
            ims_out = [torch.from_numpy(np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])) for im in ims]
            # 返回 (visible, infared, depth, labels, path, shapes) 三组图像 + 共享标签。
            return (*ims_out, labels_out, self.im_files[index], shapes)

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

    # 读单张图(支持 .npy 缓存)并缩放到 img_size 尺度;返回 (im, 原始hw, 缩放后hw)。多模态改造点。
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def load_image(self, i):
        """Loads image(s) by index, returning the image, its original dimensions, and resized dimensions."""
        # 多模态:同时读取三个模态并按通道拼接成 HxWx(3*n),保证后续几何增强完全一致。
        if self.multimodal:
            ims = []
            h0 = w0 = None
            for files in self.modal_im_files:
                im = cv2.imread(files[i])  # BGR
                assert im is not None, f"Image Not Found {files[i]}"
                if h0 is None:
                    h0, w0 = im.shape[:2]  # orig hw(三个模态对应同一张照片,尺寸一致)
                    r = self.img_size / max(h0, w0)  # ratio
                    interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                    new_hw = (math.ceil(w0 * r), math.ceil(h0 * r)) if r != 1 else (w0, h0)
                if im.shape[:2] != new_hw:
                    im = cv2.resize(im, new_hw, interpolation=interp)
                ims.append(im)
            im = np.concatenate(ims, axis=2)  # HxWx(3*num_modalities)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

        im, f, fn = (
            self.ims[i],
            self.im_files[i],
            self.npy_files[i],
        )
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                # NOTE(competition): 此处读取单张 RGB 图。多模态比赛需在此处同时读取
                # 深度图 / 红外图并拼接或融合,并同步调整 load_mosaic / letterbox 的通道假设。
                assert im is not None, f"Image Not Found {f}"
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
    # 把单张图缓存为 .npy,加速后续读取。
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        """Saves an image to disk as an *.npy file for faster future loading."""
        f = self.npy_files[i]
    # 4 图拼 Mosaic,并同步变换标签;随后做 copy_paste 与随机透视,返回 (img4, labels4)。
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index):
        """Build a 4-image mosaic from the image at `index` plus 3 random images, returning the mosaic and labels."""
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index, *random.choices(self.indices, k=3)]  # 3 additional image indices
        random.shuffle(indices)
        for i, mosaic_index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(mosaic_index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[mosaic_index].copy(), self.segments[mosaic_index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img4, labels4

# 校验单张图+标签对:修复损坏 JPEG、剔除非法坐标与重复行,返回规范化标签。

# Ancillary functions --------------------------------------------------------------------------------------------------

def verify_image_label(args):
    """Checks and verifies one image-label pair, fixing common issues and reporting anomalies."""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if lb.ndim == 2 and lb.shape[1] > 5:
                    lb = lb[:, :5]  # 比赛格式允许末尾带 confidence,训练只需 class_id + xywh
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]
