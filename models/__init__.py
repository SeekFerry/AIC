# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""对外接口：train.py 只需 `from models11111new import DetectionModel`（或别名 `Model`）。"""

from .common import (  # noqa: F401
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
    Bottleneck,
    BottleneckCSP,
)
from .yolo import (  # noqa: F401
    BaseModel,
    Detect,
    DetectionModel,
    Model,
    MultiModalDetectionModel,
    parse_model,
)
