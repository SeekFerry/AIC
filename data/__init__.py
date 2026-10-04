# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOv3 数据加载与预处理包(比赛用,自包含)。

train.py 只从这里 import,例如::

    from data11111new import LoadImagesAndLabels, create_dataloader

内容原样提取自 YOLOv3 源码:
  - ``utils/dataloaders.py``   -> dataset.py / collate.py
  - ``utils/augmentations.py`` -> transforms.py
  - ``utils/general.py``       -> utils.py(坐标转换)
  - ``utils/torch_utils.py``   -> dataset.py(torch_distributed_zero_first)

依赖: ``numpy``, ``opencv-python``, ``torch``, ``pillow``, ``psutil`` 以及 ``ultralytics`` 包。
"""

from .collate import collate_fn, collate_fn3, collate_fn4
from .dataset import (
    IMG_FORMATS,
    MODALITY_DIRS,
    VID_FORMATS,
    InfiniteDataLoader,
    LoadImages,
    LoadImagesAndLabels,
    create_dataloader,
    dataset_num_modalities,
    exif_size,
    load_depth_image,
    verify_image_label,
)
from .transforms import (
    Albumentations,
    augment_hsv,
    box_candidates,
    copy_paste,
    cutout,
    hist_equalize,
    mixup,
    random_perspective,
    replicate,
)
from .utils import (
    letterbox,
    resample_segments,
    scale_boxes,
    segment2box,
    segments2boxes,
    xyn2xy,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywh,
    xyxy2xywhn,
)

__all__ = [
    # dataset.py
    "IMG_FORMATS",
    "MODALITY_DIRS",
    "VID_FORMATS",
    "InfiniteDataLoader",
    "LoadImages",
    "LoadImagesAndLabels",
    "create_dataloader",
    "dataset_num_modalities",
    "exif_size",
    "load_depth_image",
    "verify_image_label",
    # transforms.py
    "Albumentations",
    "augment_hsv",
    "box_candidates",
    "copy_paste",
    "cutout",
    "hist_equalize",
    "mixup",
    "random_perspective",
    "replicate",
    # collate.py
    "collate_fn",
    "collate_fn3",
    "collate_fn4",
    # utils.py
    "letterbox",
    "resample_segments",
    "scale_boxes",
    "segment2box",
    "segments2boxes",
    "xyn2xy",
    "xywh2xyxy",
    "xywhn2xyxy",
    "xyxy2xywh",
    "xyxy2xywhn",
]
