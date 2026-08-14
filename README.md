1920*1080 PNG格式


# 处理数据集
## data文件夹

>├── data/
│   ├── \_\_init__.py          # 对外接口（让train.py只从这里import）
│   ├── dataset.py           # 核心：负责扫路径、读多模态图、返回原始像素
│   ├── transforms.py        # 核心：负责所有数据增强（Mosaic、透视、翻转、HSV等）
│   ├── collate.py           # 负责打包Batch（把多张图叠成张量）
│   ├── utils.py             # 辅助：letterbox（缩放填充）、坐标转换函数
│   └── hyp/            #train.py中使用到的超参数（默认只使用了`hyp.scratch-low.yaml`，其余5个yaml文件为备选



最终通过 `DataLoader` 的 `collate_fn3` 打包成一个 6 元组，会把此6元组交给train.py：

```python
(im_vis, im_ir, im_dep, targets, paths, shapes)
```

**各字段含义：**

| 字段 | 命名（train.py 里） | 形状 | 说明 |
|------|------|------|------|
| 可见光 | `im_vis` | `(B, 3, H, W)` | 三模态之一。B代表batch，3代表通道数，H/B代表图像的高/宽 |
| 红外 | `im_ir` | `(B, 3, H, W)` | 同上 |
| 深度 | `im_dep` | `(B, 3, H, W)` | 同上 |
| 标签 | `targets` | `(N, 6)` | N代表这一批所有图片里目标框的总数。6列含义见下一个表|
| 路径 | `paths` | `tuple[str]` | 每张图的路径（锚点是 visible 的文件路径） |
| 尺寸信息 | `shapes` | `tuple` | 用于验证时把框映射回原图 |

**targets中6 列含义：**
| 列 | 含义 |
|----|------|
| 0 | **图像索引**：该框属于 batch 里第几张图（0 ~ B-1），由 `collate_fn3` 写入 |
| 1 | **class_id**：类别编号 |
| 2 | **cx**：框中心 x，归一化（0~1） |
| 3 | **cy**：框中心 y，归一化（0~1） |
| 4 | **w**：框宽，归一化（0~1） |
| 5 | **h**：框高，归一化（0~1） |






# 构建模型
## models文件夹
>├── models/
│   ├── __init__.py          # 对外接口（让 train.py 只从这里 import DetectionModel）
│   ├── common.py            # 核心：所有基础网络层（Conv、Bottleneck、C3、SPP、Detect 等）
│   ├── yolo.py              # 核心：配置文件解析器（parse_model） + 主模型类（DetectionModel）
│   └── yolov3-spp.yaml          # 核心：网络结构配置文件（描述每层具体参数）

# 构建损失函数&优化器
## losses_and_optimizer文件夹
>├── losses&optimizer/
│   ├── __init__.py          # 对外接口（让 train.py 从这里 import ComputeLoss）
│   ├── loss.py              # 核心：ComputeLoss 类（计算总损失）+ 辅助损失类（FocalLoss、BCEBlurWithLogitsLoss 等）
│   └── optimizer.py         # 优化器配置

# 训练主流程
## train文件夹
>├── train/                        # 训练主文件夹
│   ├── train.py                  # 核心：训练主脚本（包含 train() 函数和 main() 入口）
│   ├── val.py                    # 核心：验证评估脚本（计算 mAP、召回率等指标）
│   └── utils/                    # 工具函数集合（train.py 依赖的辅助函数）




----
---
---

# 各文件内容详细总结

## data文件夹

### 1. `utils.py` — letterbox + 坐标转换

| 函数 | 作用 |
|---|---|
| `letterbox(im, new_shape, color, auto, scaleFill, scaleup, stride)` | 等比缩放并填充图像到目标尺寸,返回 `(img, ratio, pad)`,是推理/训练统一输入尺寸的核心 |
| `xyn2xy(x, w, h, padw, padh)` | 归一化坐标 → 像素坐标(用于分割点/segment) |
| `segment2box(segment, width, height)` | 单个多边形点集 → 外接框 `xyxy` |
| `segments2boxes(segments)` | 一组多边形 → 一组框 `xywh`(用于把分割标签转成检测框标签) |
| `resample_segments(segments, n)` | 把多边形重采样为固定 n 个点(透视变换前上采样) |
| `scale_boxes(img1_shape, boxes, img0_shape, ratio_pad)` | 把预测框从推理尺寸映射回原图尺寸(后处理必需) |
| `xyxy2xywh` / `xywh2xyxy` / `xywhn2xyxy` / `xyxy2xywhn` / `clip_boxes` | 从 `ultralytics` 再导出的框格式互转(与原仓库 `general.py` 相同) |

### 2. `transforms.py` — 数据增强

| 函数/类 | 作用 |
|---|---|
| `Albumentations(size)` | 可选增强管线(依赖 albumentations 库,没装则自动跳过) |
| `augment_hsv(im, hgain, sgain, vgain)` | HSV 色彩空间增强(只对 RGB 有意义,多模态时注意) |
| `hist_equalize(im, clahe, bgr)` | 直方图均衡化 |
| `replicate(im, labels)` | 复制最小的一批框,用于小目标增强 |
| `random_perspective(im, targets, segments, degrees, ...)` | 随机透视/仿射变换(旋转+缩放+平移+剪切),同步变换框 |
| `copy_paste(im, labels, segments, p)` | Copy-Paste 增强(需要分割标签,无 segment 时自动不生效) |
| `cutout(im, labels, p)` | 随机遮挡增强,并剔除被遮 >60% 的框 |
| `mixup(im, labels, im2, labels2)` | MixUp 图像混合 |
| `box_candidates(box1, box2, ...)` | 按宽高/长宽比/面积阈值过滤候选框 |

### 3. `collate.py` — Batch 打包

| 函数 | 作用 |
|---|---|
| `collate_fn(batch)` | 把 `(img, label, path, shapes)` 列表打包成 `(堆叠图像张量, 拼接标签张量, path, shapes)`,并给每个标签写入所属图像索引(供 `build_targets` 用) |
| `collate_fn4(batch)` | quad 模式:每 4 张拼成 1 张大图(2×2)或 2 倍上采样,再打包 |

### 4. `dataset.py` — 数据集核心

| 函数/类 | 作用 |
|---|---|
| `exif_size(img)` | 考虑 EXIF 旋转元数据,返回修正后的 (w, h) |
| `torch_distributed_zero_first(local_rank)` | DDP 下保证每个 rank 只初始化一次缓存 |
| `create_dataloader(...)` | **训练入口**:根据参数构造 `LoadImagesAndLabels` 并返回 DataLoader,处理 worker 数/采样器/随机种子 |
| `InfiniteDataLoader` | 无限重复采样器的 DataLoader(训练用,避免每 epoch 重建 worker) |
| `_RepeatSampler` | 无限重复的采样器 |
| `LoadImages(path, ...)` | 推理/验证用加载器:扫目录/图片/视频,逐张返回 `(path, 预处理后的图, 原图, cap, 信息串)` |
| `LoadImagesAndLabels(path, ...)` | **训练 Dataset**:扫路径、校验并缓存标签(`*.cache`)、矩形批、RAM/磁盘缓存,`__getitem__` 完成 Mosaic/MixUp/letterbox/翻转/HSV 全流程并返回 `(图张量, 标签, 路径, shapes)` |
| `LoadImagesAndLabels.load_image(i)` | 读单张图并缩放到 `img_size` 尺度(**多模态改造点已用 NOTE 注释标出**) |
| `LoadImagesAndLabels.load_mosaic(index)` | 4 图拼 Mosaic 并同步变换标签 |
| `LoadImagesAndLabels.cache_labels(path)` | 多进程扫描所有图+标签,校验合法性,写 `*.cache` 加速后续加载 |
| `LoadImagesAndLabels.check_cache_ram()` | 估算 RAM 是否够整库缓存,不够则关闭缓存 |
| `LoadImagesAndLabels.cache_images_to_disk(i)` | 单张图缓存为 `.npy` |
| `verify_image_label(args)` | 校验单张图+标签对(损坏 JPEG 修复、非法坐标/重复行剔除) |

### 5. `__init__.py` — 对外接口

`train.py` 只需 `from data11111new import LoadImagesAndLabels, create_dataloader`,其余(增强、坐标转换、打包)也都一并导出。

---
## models文件夹


### `__init__.py`
对外接口，从 `.common` 与 `.yolo` 汇聚导出，`train.py` 只需 `from models11111new import DetectionModel`（或 `Model`）。

### `common.py`（基础网络层，全部源自 `common.py`）

| 函数/类 | 作用 |
|---|---|
| `autopad()` | 自动计算“same”卷积的填充量，支持空洞卷积 |
| `Conv` | 卷积 + BN + 激活的基本组合块 |
| `DWConv` | 深度可分离卷积 |
| `DWConvTranspose2d` | 深度可分离转置卷积 |
| `TransformerLayer` | 单层 Transformer 编码块（多头注意力 + 前馈） |
| `TransformerBlock` | 由多层 TransformerLayer 组成的 ViT 块 |
| `Bottleneck` | 残差瓶颈块（Darknet53 核心块） |
| `BottleneckCSP` | CSP 结构瓶颈块 |
| `CrossConv` | 交叉卷积下采样块 |
| `C3` | CSP + 3 卷积 + n 个 Bottleneck |
| `C3x` | C3 变体（内部换 CrossConv） |
| `C3TR` | C3 变体（内部换 TransformerBlock） |
| `C3SPP` | C3 变体（内部换 SPP） |
| `C3Ghost` | C3 变体（内部换 GhostBottleneck） |
| `SPP` | 空间金字塔池化（YOLOv3-SPP 核心模块） |
| `SPPF` | 快速空间金字塔池化 |
| `Focus` | 空间信息折叠进通道维的下采样 |
| `GhostConv` / `GhostBottleneck` | Ghost 轻量卷积/瓶颈块 |
| `Contract` / `Expand` | 空间↔通道维度折叠/展开 |
| `Concat` | 沿指定维度拼接张量（FPN 融合） |
| `MixConv2d` | 混合深度卷积（源自 `experimental.py`） |

### `yolo.py`（解析器 + 主模型，源自 `yolo.py`）

| 函数/类 | 作用 |
|---|---|
| `LOGGER` / `colorstr` / `make_divisible` / `check_version` / `fuse_conv_and_bn` / `initialize_weights` / `model_info` / `scale_img` / `check_anchor_order` | 内联的辅助函数（源自 `general.py`、`torch_utils.py`、`autoanchor.py` 及 ultralytics 依赖的等价实现），保证文件可独立复制 |
| `Detect` | 检测头：特征图 → 预测，推理时生成网格/锚框网格解码 |
| `BaseModel` | 基类：`_forward_once` 前向、`fuse` 层融合、`info` 摘要、`_apply` 设备迁移 |
| `DetectionModel` | 主模型：解析 yaml 构建网络，初始化 stride/锚框/偏置/权重 |
| `Model` | `DetectionModel` 的向后兼容别名 |
| `parse_model` | 核心解析器：将 yaml 配置解析为 `nn.Sequential`，按 depth/width_multiple 缩放 |

### `yolov3-spp.yaml`
Darknet53 骨干 + SPP 颈部 + 3 尺度检测头（P3/8、P4/16、P5/32）的完整配置。

---
## losses_and_optimizer文件夹

### `loss.py`（核心损失）

| 函数/类 | 作用 |
|---|---|
| `bbox_iou()` | 计算两框的 IoU / GIoU / DIoU / CIoU（训练中框回归用 CIoU） |
| `smooth_bce()` | 计算标签平滑后的正/负样本 BCE 目标值 |
| `is_parallel()` | 判断模型是否被 DP/DDP 多卡封装 |
| `de_parallel()` | 剥掉 DP/DDP 封装，取内部单卡模型以访问 `Detect` 属性 |
| `BCEBlurWithLogitsLoss` | 带 alpha 模糊的 BCE 损失，减轻缺失标签副作用 |
| `FocalLoss` | Focal Loss，按预测置信度调制损失以缓解类别不平衡 |
| `QFocalLoss` | Quality Focal Loss，用 `\|true - pred\|` 作调制因子 |
| `ComputeLoss` | **核心总损失类**：汇总分类损失 + 框回归(CIoU)损失 + 目标置信度损失 |
| `ComputeLoss.build_targets()` | 将标签 (img, cls, x, y, w, h) 按层匹配到锚框/网格，生成训练目标 |

### `optimizer.py`（优化器）

| 函数/类 | 作用 |
|---|---|
| `LOGGER` | 日志记录器（等价于 ultralytics 中的 LOGGER） |
| `colorstr()` | 给日志字符串加 ANSI 颜色 |
| `smart_optimizer()` | **核心优化器配置**：按参数类型分组（带衰减权重 / BN 权重 / 偏置），支持 Adam / AdamW / RMSProp / SGD |

### `__init__.py`（对外接口）

导出 `ComputeLoss`、`FocalLoss`、`QFocalLoss`、`BCEBlurWithLogitsLoss`、`smart_optimizer`。

---
## train文件夹
### `train.py`
| 名称 | 作用 |
|------|------|
| `train(hyp, opt, device, callbacks)` | 训练主循环：构建模型/优化器/EMA/数据加载器，逐 epoch 前向、反向、梯度累积、warmup、多尺度训练，每轮验证 mAP，保存 last/best 权重并做早停 |
| `parse_opt(known)` | 解析命令行参数（`--weights/--cfg/--data/--hyp` 等） |
| `main(opt, callbacks)` | 训练入口：校验配置、处理断点续训、初始化 DDP，然后调用 `train()` |
| `run(**kwargs)` | 编程式训练入口：解析参数后用 kwargs 覆盖并调用 `main()` |

### `val.py`
| 名称 | 作用 |
|------|------|
| `save_one_txt(predn, save_conf, shape, file)` | 将单张图检测结果按 YOLO 标签格式写入 txt |
| `save_one_json(predn, jdict, path, class_map)` | 将检测结果保存为 COCO JSON |
| `process_batch(detections, labels, iouv)` | 计算多 IoU 阈值下的正确匹配矩阵 |
| `run(...)` | 验证主流程：推理 → NMS → 统计 P/R/[mAP@0.5](mailto:mAP@0.5)/[mAP@0.5](mailto:mAP@0.5):0.95/损失 |
| `parse_opt()` | 解析验证命令行参数 |
| `main(opt)` | 验证入口，按 `task` 调用 `run()` |

### `/utils`


| 文件 | 作用 |
|------|------|
| `__init__.py` | 从 `ultralytics` 包再导出 `LOGGER`/`TryExcept`/`emojis`/`threaded`，供包内各模块共用 |
| `general.py` | 通用工具：数据集 yaml 解析、文件/路径/尺寸/后缀校验、AMP 检查、YAML 读写、下载解压、类别权重、NMS、坐标缩放、`strip_optimizer` 等 |
| `torch_utils.py` | PyTorch 工具：设备选择、DDP 初始化与同步、模型 `de_parallel`、EMA、断点续训、早停、显存分析 |
| `metrics.py` | 验证指标：`fitness`、逐类 AP（`ap_per_class`/`compute_ap`）、混淆矩阵 `ConfusionMatrix` |
| `callbacks.py` | 训练/验证生命周期回调注册与分发（`Callbacks`） |
| `autoanchor.py` | 锚框工具：检查锚框匹配度，必要时用 k-means + 遗传算法重算锚框 |
| `autobatch.py` | 自动批量：按显存占用估算最优 batch size |
| `downloads.py` | 下载工具：URL 判断、curl/safe 下载、权重缺失时从 GitHub release 自动下载 |
| `plots.py` | 绘图工具：验证结果可视化（`output_to_target`/`plot_images`） |
| `experimental.py` | checkpoint 加载：`attempt_load` + `Ensemble`（多模型集成，单模型透传） |
