import torch
from models.yolo import MultiModalDetectionModel  # 新添加的类

# 1. 创建模型
model = MultiModalDetectionModel(
    cfg='models/yolov3-spp.yaml',
    ch=(3, 3, 3),
    nc=80
)

# 2. 创建随机输入 (模拟三个模态的batch)
batch_size = 2
img_size = 640
im_vis = torch.randn(batch_size, 3, img_size, img_size)
im_ir = torch.randn(batch_size, 3, img_size, img_size)
im_dep = torch.randn(batch_size, 3, img_size, img_size)

# 3. 前向传播
with torch.no_grad():
    output = model(im_vis, im_ir, im_dep)

# 4. 检查输出形状
print(f"Output type: {type(output)}")
if isinstance(output, (list, tuple)):
    for i, o in enumerate(output):
        print(f"Output[{i}] shape: {o.shape}")
else:
    print(f"Output shape: {output.shape}")
print("✅ 前向传播测试通过！")