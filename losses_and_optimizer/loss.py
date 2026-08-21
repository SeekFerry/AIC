# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOv3 损失函数(比赛用,自包含)。

内容原样提取自 YOLOv3 源码:
  - ``utils/loss.py``             -> ComputeLoss / FocalLoss / QFocalLoss / BCEBlurWithLogitsLoss
  - ``utils/torch_utils.py``      -> is_parallel / de_parallel
  - ``ultralytics.utils.metrics`` -> bbox_iou / smooth_bce(utils/loss.py 正是从这里 import 它们)

依赖: ``torch``。
"""

from __future__ import annotations

import math

import torch
from torch import nn


# 计算两框之间的 IoU/GIoU/DIoU/CIoU(utils/loss.py 从 ultralytics.utils.metrics import bbox_iou)
def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4. For instance, you
    may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4). Internally, the code will split the last
    dimension into (x, y, w, h) if `xywh=True`, or (x1, y1, x2, y2) if `xywh=False`.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


# 计算标签平滑后的正/负样本 BCE 目标值(utils/loss.py 从 ultralytics.utils.metrics import smooth_bce)
def smooth_bce(eps: float = 0.1) -> tuple[float, float]:
    """Compute smoothed positive and negative Binary Cross-Entropy targets."""
    return 1.0 - 0.5 * eps, 0.5 * eps


# DSSM 框损失:针对微小目标,综合"中心距离 + 宽高差 + 形状差"并按外接框对角线归一化,
# 用于弥补传统 IoU 对微小目标位置偏移过于敏感的问题。
def dssm_box_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    alpha: float = 1.0,
    K: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute DSSM box loss for tiny objects.

    Args:
        pred_boxes (Tensor): (n, 4) predicted boxes in (x, y, w, h) center format.
        target_boxes (Tensor): (n, 4) target boxes in (x, y, w, h) center format.
        alpha (float): weight of the shape (aspect-ratio) difference term.
        K (float): exponential balancing constant for the composite distance.

    Returns:
        Tensor: (n, 1) per-box scalar losses in [0, 1).
    """
    px, py, pw, ph = pred_boxes.chunk(4, -1)
    tx, ty, tw, th = target_boxes.chunk(4, -1)

    # 1. 中心点距离平方
    d2_center = (px - tx).pow(2) + (py - ty).pow(2)

    # 2. 宽高差异平方
    d2_wh = (pw - tw).pow(2) + (ph - th).pow(2)

    # 3. 形状差异:宽高向量的余弦相似度 -> 角度损失
    r_cosine = (pw * tw + ph * th) / (
        torch.sqrt(pw.pow(2) + ph.pow(2)) * torch.sqrt(tw.pow(2) + th.pow(2)) + eps
    )
    d_angle = alpha * (1.0 - r_cosine)

    # 4. 外接矩形对角线平方(归一化因子)
    d2_diag = (
        torch.max(px + pw / 2, tx + tw / 2) - torch.min(px - pw / 2, tx - tw / 2)
    ).pow(2) + (
        torch.max(py + ph / 2, ty + th / 2) - torch.min(py - ph / 2, ty - th / 2)
    ).pow(2) + eps

    # 5. 组合距离度量(越小表示框越匹配)
    r_composite = (d2_center + d2_wh + d_angle.pow(2)) / d2_diag

    # 6. 论文相似度得分 e^{-K·R} ∈ (0, 1];损失取其补,与 1-IoU 同方向(匹配越好损失越小)
    return 1.0 - torch.exp(-K * r_composite)


# 判断模型是否使用了 DataParallel/DistributedDataParallel 多卡封装(用于 de_parallel 前处理)
def is_parallel(model):
    """Checks if a model is using DataParallel (DP) or DistributedDataParallel (DDP)."""
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


# 若模型被 DP/DDP 封装,则返回其内部的单卡模型,用于访问 Detect 模块属性
def de_parallel(model):
    """Returns a single-GPU model if input model is using DataParallel (DP) or DistributedDataParallel (DDP)."""
    return model.module if is_parallel(model) else model


# 带 alpha 模糊的 BCEWithLogitsLoss,用于减轻缺失标签(missing label)带来的副作用
class BCEBlurWithLogitsLoss(nn.Module):
    """Implements BCEWithLogitsLoss with adjustments to mitigate missing label effects using an alpha parameter."""

    def __init__(self, alpha=0.05):
        """Initializes BCEBlurWithLogitsLoss with alpha to reduce missing label effects; default alpha is 0.05."""
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Compute mean BCE loss between `pred` logits and `true` labels, downweighting probable missing labels."""
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


# Focal Loss,通过按预测置信度调制损失来缓解类别不平衡
class FocalLoss(nn.Module):
    """Implements Focal Loss to address class imbalance by modulating the loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha for enhanced training on imbalanced
        datasets.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Compute focal loss between `pred` and `true`, scaling by alpha- and gamma-weighted modulating factors."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


# Quality Focal Loss,用 |true - pred_prob| 作为调制因子处理类别不平衡
class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to handle class imbalance with a modulating factor and alpha."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes QFocalLoss with specified loss function, gamma, and alpha for element-wise focal loss
        application.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes focal loss between predictions and true labels using configured loss function, `gamma`, and `alpha`."""
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


# Keep local (do not dedup): anchor-based YOLOv3 loss has no ultralytics equivalent
# YOLOv3 总损失计算类:汇总分类损失、框回归损失与目标置信度损失
class ComputeLoss:
    """Computes the total loss for YOLO models by aggregating classification, box regression, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model's device and hyperparameters, and sets autobalance."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_bce(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        # 归一化面积占比阈值: 目标 w*h < 该值(1920x1080 下即面积 < 0.2% 整图)判为 tiny object
        self.tiny_area_ratio = h.get("tiny_area_ratio", 0.002)

    # 前向计算:输入预测 p 与标签 targets,返回总损失与 (lbox, lobj, lcls) 分项损失
    def __call__(self, p, targets):  # predictions, targets
        """Computes loss given predictions and targets, returning class, box, and object loss as tensors."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors, tiny_flags = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target), 仍用于 objectness
                tiny_i = tiny_flags[i]  # (n,) bool: 该目标是否为 tiny object
                if tiny_i.any():
                    # 非 tiny 目标沿用原 CIoU 损失;tiny 目标改用 DSSM 框损失
                    box_loss = torch.zeros_like(iou)
                    nt_mask = ~tiny_i
                    if nt_mask.any():
                        box_loss[nt_mask] = 1.0 - iou[nt_mask]
                    box_loss[tiny_i] = dssm_box_loss(pbox[tiny_i], tbox[i][tiny_i]).squeeze()
                    lbox += box_loss.mean()
                else:
                    lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    # 将 targets(image, class, x, y, w, h) 按层匹配到锚框,返回分类/框/索引/锚框
    def build_targets(self, p, targets):
        """Match `targets` (image, class, x, y, w, h) to anchors per layer, returning tcls, tbox, indices, anchors, tiny_flags."""
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tiny_flags = [], [], [], [], []
        gain = torch.ones(8, device=self.device)  # normalized to gridspace gain(多一列 tiny 标志,不参与缩放)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # tiny object 标志:归一化框面积占比 w*h < 阈值(1920x1080 下即面积 < 0.2% 整图)
        tiny = (targets[:, 3] * targets[:, 4]) < self.tiny_area_ratio  # (nt,)
        tiny = tiny.float().view(1, nt, 1).repeat(na, 1, 1)  # (na, nt, 1)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None], tiny), 2)  # append anchor indices + tiny flag

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                left, bottom = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, left, bottom))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, at = t.chunk(4, 1)  # (image, class), grid xy, grid wh, (anchors, tiny)
            a, tiny_i = at[:, 0], at[:, 1]  # anchors, tiny flag
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            tiny_i = tiny_i.bool()
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tiny_flags.append(tiny_i)  # tiny object 标志

        return tcls, tbox, indices, anch, tiny_flags
