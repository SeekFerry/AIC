# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
验证 YOLOv3 检测模型(比赛精简版,仅支持 PyTorch .pt 权重)。

数据加载 -> data
辅助工具 -> train/utils

用法:
    $ python train/val.py --weights yolov3-tiny.pt --data data/coco128.yaml --imgsz 640
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # train/ 目录(本脚本所在目录)
REPO_ROOT = ROOT.parent  # 仓库根目录(data11111new/models11111new/losses11111new 所在层级)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 优先把 train/ 加入 PATH,保证 utils 指向本目录
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))  # 保证 data 可导入
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from data import create_dataloader
from utils.callbacks import Callbacks
from utils.experimental import attempt_load
from utils.general import (
    LOGGER,
    TQDM,
    Profile,
    check_dataset,
    check_img_size,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images
from utils.torch_utils import select_device, smart_inference_mode


# 将单张图的检测结果按 YOLO 标签格式(归一化 xywh + cls + 可选 conf)写入 txt
def save_one_txt(predn, save_conf, shape, file):
    """Saves detection results in a text format, including labels and optionally confidence scores.

    Args:
        predn (torch.Tensor): 归一化后的预测结果 (x1, y1, x2, y2, conf, cls)。
        save_conf (bool): 是否保存置信度。
        shape (tuple[int, int]): 原图尺寸 (height, width)。
        file (str | Path): 输出文件路径。
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


# 将检测结果保存为 COCO JSON 格式(含 image_id/category_id/bbox/score)
def save_one_json(predn, jdict, path, class_map):
    """Save detection results in JSON format containing image_id, category_id, bbox, and score per detection.

    Args:
        predn (torch.Tensor): 归一化预测结果 (N, 6),即 (x1, y1, x2, y2, confidence, class)。
        jdict (list): 收集 JSON 结果的列表。
        path (Path): 图像路径。
        class_map (dict[int, int]): 类别 id 到 COCO category id 的映射。
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


# 计算检测框与真实框在多个 IoU 阈值下的正确匹配矩阵
def process_batch(detections, labels, iouv):
    """Computes correct prediction matrix for detections against ground truth labels at various IoU thresholds.

    Args:
        detections (torch.Tensor): 检测结果 (N, 6),即 [x1, y1, x2, y2, confidence, class]。
        labels (torch.Tensor): 真实标签 (M, 5),即 [class, x1, y1, x2, y2]。
        iouv (torch.Tensor): IoU 阈值向量。

    Returns:
        torch.Tensor: 形状 (N, len(iouv)) 的布尔张量,表示每个阈值下的正确预测。
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


# 验证主流程:推理 + NMS + 指标统计,返回 (P, R, mAP@.5, mAP@.5:.95, 损失) 等
@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=REPO_ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=None,
    compute_loss=None,
):
    """Validate a trained YOLOv3 detection model on a dataset and optionally save results in the requested formats.

    Returns:
        (tuple): (results, maps, t),其中 results = (P, R, mAP@0.5, mAP@0.5:0.95, box_loss, obj_loss, cls_loss)。
    """
    if callbacks is None:
        callbacks = Callbacks()
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model(仅支持 PyTorch .pt 权重)
        model = attempt_load(weights, device=device)
        stride, pt, jit, engine = model.stride, True, False, False
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= device.type != "cpu"  # FP16 仅在 CUDA 上可用
        model.half() if half else model.float()

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

        # Warm up after creating the loader so the model receives the same modality layout as validation.
        warmup_size = 1 if pt else batch_size
        if getattr(model, "num_modalities", 1) == 3:
            dummy = torch.zeros((warmup_size, 3, imgsz, imgsz), device=device)
            model(dummy, dummy, dummy)
        else:
            in_ch = next((m.in_channels for m in model.modules() if isinstance(m, torch.nn.Conv2d)), 3)
            model(torch.zeros((warmup_size, in_ch, imgsz, imgsz), device=device))

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    pbar = TQDM(dataloader, desc=s)  # progress bar
    for batch_i, batch in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        multimodal = len(batch) == 6
        if multimodal:
            im_vis, im_ir, im_dep, targets, paths, shapes = batch
            im_plot = im_vis  # 可视化用可见光图
            modal_ims = (im_vis, im_ir, im_dep)
        else:
            im, targets, paths, shapes = batch
            im_plot = im
            modal_ims = (im,)
        with dt[0]:
            if cuda:
                modal_ims = tuple(x.to(device, non_blocking=True) for x in modal_ims)
                targets = targets.to(device)
            modal_ims = tuple((x.half() if half else x.float()) / 255 for x in modal_ims)
            im = torch.cat(modal_ims, dim=1) if multimodal else modal_ims[0]
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            if multimodal:
                output = model(*modal_ims)
            else:
                output = model(im, augment=augment)
            preds, train_out = output if compute_loss else (output, None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im_plot, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
            plot_images(im_plot, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred

        callbacks.run("on_val_batch_end", batch_i, im_plot, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        if not os.path.exists(anno_json):
            anno_json = os.path.join(data["path"], "annotations", "instances_val2017.json")
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


# 解析验证命令行参数
def parse_opt():
    """Parses and returns command-line options for dataset paths, model parameters, and inference settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=REPO_ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=REPO_ROOT / "yolov3-tiny.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=REPO_ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


# 验证入口:根据 task 调用 run()
def main(opt):
    """Executes model validation on the specified dataset."""
    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.warning(f"confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.warning("--save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))
    else:
        raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
