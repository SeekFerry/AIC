# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""数据预处理辅助函数:letterbox(缩放填充)与坐标转换。

来源说明(均原样提取自 YOLOv3 源码,未改逻辑):
  - ``letterbox``        来自 ``utils/augmentations.py``
  - ``xyn2xy`` 等坐标函数 来自 ``utils/general.py``(标注 "Keep local (do not dedup)")
  - ``xyxy2xywh`` / ``xywh2xyxy`` / ``xywhn2xyxy`` / ``xyxy2xywhn`` / ``clip_boxes``
    与 ``utils/general.py`` 一样,从 ``ultralytics`` 包再导出。
"""

import cv2
import numpy as np
import torch
from ultralytics.utils.ops import (  # 与 utils/general.py 相同的再导出
    clip_boxes,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywh,
    xyxy2xywhn,
)


# 等比缩放并填充图像到目标尺寸,返回 (img, ratio, pad);推理与训练统一输入尺寸的核心。
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads an image to a new shape with optional scaling, filling, and stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 多模态拼接图(如 HxWx9)需用四元等值标量填充色,OpenCV 要求 borderValue 为 4 个相等分量。
    if im.ndim == 3 and isinstance(color, (tuple, list)) and len(color) != im.shape[2]:
        color = (color[0], color[0], color[0], color[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


# 归一化坐标 -> 像素坐标(用于分割点/segment),并叠加宽度、高度与 padding 偏移。
# Keep local (do not dedup): no ultralytics equivalent
def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """Converts normalized segments to pixel segments, shape (n,2), adjusting for width `w`, height `h`, and padding."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y

# 单个多边形点集 -> 外接框 (xyxy),并保证坐标在图像范围内。
def segment2box(segment, width=640, height=640):
    """Converts a single segment to a bounding box using image dimensions, output shape (4,), ensuring coordinates stay
    within image boundaries.
    """
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    (
        x,
        y,
    ) = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy

# 一组多边形 -> 一组框 (xywh);把分割标签转成检测框标签。
def segments2boxes(segments):
    """Converts segmentation labels to bounding box labels in format (xywh) from (xy1, xy2, ...)."""
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # xyxy
    return xyxy2xywh(np.array(boxes))  # xywh

# 把多边形重采样为固定 n 个点(透视变换前上采样,保证变换平滑)。
def resample_segments(segments, n=1000):
    """Resamples segments to a fixed number of points (n), returning up-sampled (n,2) segment arrays."""
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments

# 把预测框从推理尺寸映射回原图尺寸(后处理必需,含去 padding 与缩放)。
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
