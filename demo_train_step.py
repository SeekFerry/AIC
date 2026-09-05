"""Minimal tri-stream training-step smoke test.

Run from the repository root:
    python demo_train_step.py --device cuda --imgsz 128

The script uses random data and does not require the real dataset. It checks:
1. three-stream forward in training mode;
2. YOLOv3 loss construction and backward propagation;
3. one optimizer update.
"""

import argparse
from pathlib import Path

import torch
import yaml

from losses_and_optimizer import ComputeLoss, smart_optimizer
from models import MultiModalDetectionModel


REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run one random tri-stream YOLO training step")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--nc", type=int, default=12)
    return parser.parse_args()


def select_device(device_name):
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device(device_name)
    return torch.device("cpu")


def main():
    args = parse_args()
    device = select_device(args.device)

    cfg_path = REPO_ROOT / "models" / "yolov3-spp.yaml"
    hyp_path = REPO_ROOT / "data" / "hyps" / "hyp.scratch-low.yaml"
    with hyp_path.open(encoding="utf-8") as file:
        hyp = yaml.safe_load(file)

    print(f"building model on {device}...", flush=True)
    model = MultiModalDetectionModel(str(cfg_path), ch=(3, 3, 3), nc=args.nc).to(device)
    model.hyp = hyp
    model.nc = args.nc
    model.train()

    images = tuple(
        torch.rand(args.batch_size, 3, args.imgsz, args.imgsz, device=device)
        for _ in range(3)
    )
    # targets: [batch_index, class_index, center_x, center_y, width, height]
    targets = torch.tensor(
        [[0, 0, 0.5, 0.5, 0.25, 0.25]],
        dtype=torch.float32,
        device=device,
    )
    if args.batch_size > 1:
        extra_targets = torch.tensor(
            [[args.batch_size - 1, 1 % args.nc, 0.35, 0.35, 0.2, 0.2]],
            dtype=torch.float32,
            device=device,
        )
        targets = torch.cat((targets, extra_targets), dim=0)

    optimizer = smart_optimizer(model, "SGD", hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    compute_loss = ComputeLoss(model)

    optimizer.zero_grad(set_to_none=True)
    print("running forward...", flush=True)
    predictions = model(*images)
    print("computing loss...", flush=True)
    total_loss, loss_items = compute_loss(predictions, targets)
    print("running backward...", flush=True)
    total_loss.backward()
    print("running optimizer step...", flush=True)
    optimizer.step()

    print(f"device: {device}")
    print(f"inputs: {[tuple(image.shape) for image in images]}")
    print(f"prediction layers: {[tuple(prediction.shape) for prediction in predictions]}")
    print(f"loss: {total_loss.item():.6f}")
    print(f"loss items [box, obj, cls]: {loss_items.tolist()}")
    print("tri-stream forward/loss/backward/optimizer step passed")


if __name__ == "__main__":
    main()
