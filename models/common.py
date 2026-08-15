# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""公共基础网络层（从 models/common.py 精简而来，仅保留网络结构构建所需的层）。"""

import math
import warnings

import numpy as np
import torch
from torch import nn


# 自动计算“same”卷积所需的填充量，并支持空洞卷积（dilation）下的实际卷积核尺寸
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Automatically calculates same shape padding for convolutional layers, optionally adjusts for dilation."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# 标准卷积 + 批归一化 + 激活函数的基本组合模块（YOLOv3 网络的最小编码单元）
class Conv(nn.Module):
    """A standard Conv2D layer with batch normalization and optional activation for neural networks."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard Conv2D layer with batch normalization and optional activation; args are channel_in,
        channel_out, kernel_size, stride, padding, groups, dilation, and activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies convolution, batch normalization, and activation to input `x`; `x` shape: [N, C_in, H, W] -> [N,
        C_out, H_out, W_out].
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies fused convolution and activation to input `x`; input shape: [N, C_in, H, W] -> [N, C_out, H_out,
        W_out].
        """
        return self.act(self.conv(x))


# 深度可分离卷积：分组数取输入/输出通道的最大公约数，降低参数量
class DWConv(Conv):
    """Implements depth-wise convolution for efficient spatial feature extraction in neural networks."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initializes depth-wise convolution with optional activation; parameters are channel in/out, kernel, stride,
        dilation.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# 深度可分离转置卷积：用于上采样等场景，分组数取输入/输出通道的最大公约数
class DWConvTranspose2d(nn.ConvTranspose2d):
    """Implements a depth-wise transpose convolution layer with specified channels, kernel size, stride, and padding."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initializes a depth-wise transpose convolution layer with specified in/out channels, kernel size, stride, and
        input/output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


# 单层 Transformer 编码块：多头注意力 + 前馈网络，省略 LayerNorm
class TransformerLayer(nn.Module):
    """Transformer layer with multi-head attention and feed-forward network, optimized by removing LayerNorm."""

    def __init__(self, c, num_heads):
        """Initializes a Transformer layer as per https://arxiv.org/abs/2010.11929, sans LayerNorm, with specified
        embedding dimension and number of heads.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Performs forward pass with multi-head attention and residual connections on input tensor 'x' [batch, seq_len,
        features].
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


# Vision Transformer 块：由多层 TransformerLayer 组成，可选前置卷积做通道对齐
class TransformerBlock(nn.Module):
    """Implements a Vision Transformer block with transformer layers; https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initializes a Transformer block with optional convolution, linear, and transformer layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Applies an optional convolution, transforms features, and reshapes output matching input dimensions."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


# Bottleneck 残差块：1x1 降维 + 3x3 卷积，可选残差连接（Darknet53 的核心块）
class Bottleneck(nn.Module):
    """Implements a bottleneck layer with optional shortcut for efficient feature extraction in neural networks."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        """Initializes a standard bottleneck layer with optional shortcut; args: input channels (c1), output channels
        (c2), shortcut (bool), groups (g), expansion factor (e).
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Executes forward pass, performing convolutional ops and optional shortcut addition; expects input tensor x."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# CSP 结构的 Bottleneck 层：将特征一分为二后融合（Cross Stage Partial）
class BottleneckCSP(nn.Module):
    """Implements a CSP Bottleneck layer for feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """Initializes CSP Bottleneck with channel in/out, optional shortcut, groups, expansion; see
        https://github.com/WongKinYiu/CrossStagePartialNetworks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Processes input through layers, combining outputs with activation and normalization for feature extraction."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


# 交叉卷积下采样模块：用 1D 与 2D 卷积组合替代单个大卷积核
class CrossConv(nn.Module):
    """Implements Cross Convolution Downsample with 1D and 2D convolutions and optional shortcut."""

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """Initializes CrossConv with downsample options, combining 1D and 2D convolutions, optional shortcut if
        input/output channels match.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Performs forward pass using sequential 1D and 2D convolutions with optional shortcut addition."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# C3 模块：CSP 结构 + 3 个卷积 + n 个 Bottleneck（YOLOv3 骨干的常用组合块）
class C3(nn.Module):
    """Implements a CSP Bottleneck with 3 convolutions, optional shortcuts, group convolutions, and expansion factor."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """Initializes CSP Bottleneck with 3 convolutions, optional shortcuts, group convolutions, and expansion factor."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Processes input tensor `x` through convolutions and bottlenecks, returning the concatenated output tensor."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# C3x 模块：C3 的变体，内部 Bottleneck 替换为交叉卷积（CrossConv）
class C3x(C3):
    """Extends the C3 module with cross-convolutions for enhanced feature extraction and flexibility."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3x module with cross-convolutions, extending the C3 module with customizable parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


# C3TR 模块：C3 的变体，内部替换为 TransformerBlock，引入注意力机制
class C3TR(C3):
    """C3 module with TransformerBlock for integrating attention mechanisms in CNNs."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with TransformerBlock, extending C3 for attention mechanisms."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


# C3SPP 模块：C3 的变体，内部替换为空间金字塔池化（SPP）
class C3SPP(C3):
    """Extends C3 with Spatial Pyramid Pooling (SPP) for enhanced feature extraction in CNNs."""

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3SPP module, extending C3 with Spatial Pyramid Pooling for enhanced feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


# C3Ghost 模块：C3 的变体，内部替换为 Ghost Bottleneck，更轻量
class C3Ghost(C3):
    """Implements a C3 module with Ghost Bottlenecks for efficient feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3Ghost module with Ghost Bottlenecks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


# 空间金字塔池化（SPP）：多尺度池化后拼接，增大感受野（YOLOv3-SPP 的核心模块）
class SPP(nn.Module):
    """Implements Spatial Pyramid Pooling (SPP) for enhanced feature extraction; see https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with specified channels and kernels.

        More at https://arxiv.org/abs/1406.4729
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Applies convolution and max pooling layers to the input tensor `x`, concatenates results for feature
        extraction.

        `x` is a tensor of shape [N, C, H, W]. See https://arxiv.org/abs/1406.4729 for more details.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


# 快速空间金字塔池化（SPPF）：串联多个相同池化核，等价于 SPP(k=(5,9,13)) 但更快
class SPPF(nn.Module):
    """Implements a fast Spatial Pyramid Pooling (SPPF) layer for efficient feature extraction in YOLOv3 models."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        """Initializes the SPPF layer with specified input/output channels and kernel size for YOLOv3."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Performs forward pass combining convolutions and max pooling on input `x` of shape [N, C, H, W] to produce
        feature map.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# Focus 模块：把宽高信息切分后堆叠到通道维度，实现无信息损失的下采样
class Focus(nn.Module):
    """Focuses spatial information into channel space using configurable convolution."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        """Initializes Focus module to focus width and height information into channel space with configurable
        convolution parameters.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        """Applies focused downsampling to input tensor, returning a convolved output with increased channel depth."""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


# Ghost 卷积：先生成一部分特征，再通过廉价操作生成其余特征，降低计算量
class GhostConv(nn.Module):
    """Implements Ghost Convolution for efficient feature extraction; see github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        """Initializes GhostConv with in/out channels, kernel size, stride, groups; see
        https://github.com/huawei-noah/ghostnet.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Executes forward pass, applying convolutions and concatenating results; input `x` is a tensor."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


# Ghost Bottleneck：GhostNet 的轻量残差块，由 GhostConv 与深度可分离卷积组成
class GhostBottleneck(nn.Module):
    """Implements a Ghost Bottleneck layer for efficient feature extraction from GhostNet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        """Initializes GhostBottleneck module with in/out channels, kernel size, and stride; see
        https://github.com/huawei-noah/ghostnet.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Performs a forward pass through the network, returning the sum of convolution and shortcut outputs."""
        return self.conv(x) + self.shortcut(x)


# Contract 模块：把空间尺寸折叠进通道，例如 (1,64,80,80) -> (1,256,40,40)
class Contract(nn.Module):
    """Contracts spatial dimensions into channels, e.g., (1,64,80,80) to (1,256,40,40) with a specified gain."""

    def __init__(self, gain=2):
        """Initializes Contract module to refine input dimensions, e.g., from (1,64,80,80) to (1,256,40,40) with a
        default gain of 2.
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor (b,c,h,w) to contracted shape (b,c*s^2,h/s,w/s) with default gain s=2, e.g.,
        (1,64,80,80) to (1,256,40,40).
        """
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


# Expand 模块：把通道维度展开到空间维度，实现上采样，例如 (1,64,80,80) -> (1,16,160,160)
class Expand(nn.Module):
    """Expands spatial dimensions of input tensor by a factor while reducing channels correspondingly."""

    def __init__(self, gain=2):
        """Initializes Expand module to increase spatial dimensions by factor `gain` while reducing channels
        correspondingly.
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Expands spatial dimensions of input tensor `x` by factor `gain` while reducing channels, transforming shape
        `(B,C,H,W)` to `(B,C/gain^2,H*gain,W*gain)`.
        """
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


# Concat 模块：沿指定维度拼接多个张量（用于 FPN 特征融合）
class Concat(nn.Module):
    """Concatenates a list of tensors along a specified dimension for efficient feature aggregation."""

    def __init__(self, dimension=1):
        """Initializes a module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; x is a list of tensors to concatenate, dimension
        defaults to 1.
        """
        return torch.cat(x, self.d)


# MixConv2d：混合深度卷积，在一层内用多个不同卷积核并行卷积后拼接（源自 models/experimental.py）
class MixConv2d(nn.Module):
    """Implements mixed depth-wise convolutions for efficient neural networks; see https://arxiv.org/abs/1907.09595."""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        """Initialize MixConv2d with mixed depth-wise convolution layers; see https://arxiv.org/abs/1907.09595."""
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """Applies a series of convolutions, batch normalization, and SiLU activation to input tensor `x`."""
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

"""深度与红外特征融合（正交投影约束）"""
class DepthIR_Fusion(nn.Module):
    def __init__(self, c1, c2, hide_channel=8):
        super().__init__()
        self.conv_depth = Conv(c1, hide_channel, 1, 1)
        self.conv_ir = Conv(c1, hide_channel, 1, 1)
        self.conv_out = Conv(hide_channel * 2, c2, 1, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, depth_feat, ir_feat):
        d = self.conv_depth(depth_feat)
        i = self.conv_ir(ir_feat)

        dot_di = torch.sum(d * i, dim=1, keepdim=True)
        norm_i = torch.norm(i, p=2, dim=1, keepdim=True)
        proj_d_on_i = (dot_di / (norm_i + 1e-6)) * i

        dot_id = torch.sum(i * d, dim=1, keepdim=True)
        norm_d = torch.norm(d, p=2, dim=1, keepdim=True)
        proj_i_on_d = (dot_id / (norm_d + 1e-6)) * d

        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        d_orth = d - alpha * proj_d_on_i
        i_orth = i - beta * proj_i_on_d

        fused = torch.cat([d_orth, i_orth], dim=1)
        return self.conv_out(fused)


class MultiModalFusion(nn.Module):
    """三模态融合：RGB + 深度 + 红外"""
    def __init__(self, c1, c2, hide_channel=8):
        super().__init__()
        self.depthir_fusion = DepthIR_Fusion(c1, c1, hide_channel)
        self.final_conv = Conv(c1 * 2, c2, 1, 1)

    def forward(self, rgb_feat, depth_feat, ir_feat):
        fused_di = self.depthir_fusion(depth_feat, ir_feat)
        fused = torch.cat([rgb_feat, fused_di], dim=1)
        return self.final_conv(fused)