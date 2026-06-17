#!/usr/bin/env python3
"""
Faster R-CNN (MobileNetV2) training on all 5 construction safety datasets.
Lightweight backbone to fit in 6GB GPU VRAM.

  EXP 16 — Roboflow  : ~/FYP/Dataset/Roboflow Dataset/construction safety.yolov8/
  EXP 17 — CHVG      : ~/FYP/Dataset/CHVG-Dataset/
  EXP 18 — SHEL5K    : ~/FYP/Dataset/SHEL5K/Safety Helmet Wearing Dataset/
  EXP 19 — SH17      : ~/FYP/Dataset/SH17dataset/
  EXP 20 — SFCHD     : ~/FYP/Dataset/SFCHD-SCALE/dataset_SFCHD/

Usage:
  python3 train_fasterrcnn_all.py --datasets all
  python3 train_fasterrcnn_all.py --datasets roboflow chvg
  python3 train_fasterrcnn_all.py --datasets sh17 sfchd
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import mobilenet_v2

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("Installing pycocotools...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pycocotools"], check=True)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

# ── Constants ──────────────────────────────────────────────────────────────────
HOME         = Path.home()
RESULTS_BASE = HOME / "FYP/results/FasterRCNN"
SEED         = 42
EPOCHS       = 20   # Aggressive: reduced from 100 for 15-hour timeline
PATIENCE     = 20
BATCH_INIT   = 1    # Aggressive: reduced from 4 for memory efficiency
BATCH_MIN    = 1
IMGSZ        = 352  # Aggressive: reduced from 640 for 15-hour timeline
LR0          = 0.005
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS      = 2
GPU_WARN_C   = 85   # °C

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Model Architecture ─────────────────────────────────────────────────────────

def create_fasterrcnn_mobilenet(num_classes):
    """Create Faster R-CNN with MobileNetV2 backbone (lighter than ResNet50)."""
    from torchvision.models.detection import FasterRCNN
    from torchvision.ops import MultiScaleRoIAlign

    # Load pretrained MobileNetV2
    backbone = mobilenet_v2(weights='DEFAULT').features
    backbone.out_channels = 1280

    # Create RPN anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Create RoI pooler
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Create Faster R-CNN with MobileNetV2 backbone
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model

EXPERIMENTS = {
    "roboflow": {"exp_num": "16", "name": "Roboflow"},
    "chvg":     {"exp_num": "17", "name": "CHVG"},
    "shel5k":   {"exp_num": "18", "name": "SHEL5K"},
    "sh17":     {"exp_num": "19", "name": "SH17"},
    "sfchd":    {"exp_num": "20", "name": "SFCHD"},
}

CLASSES = {
    "roboflow": ["helmet", "no-helmet", "no-vest", "person", "vest"],
    "chvg":     ["person", "vest", "white", "yellow", "head", "blue", "glass", "red"],
    "shel5k":   ["helmet", "head_with_helmet", "person_with_helmet", "face",
                 "head", "person_no_helmet", "person"],
    "sh17":     ["person", "ear", "ear-mufs", "face", "face-guard", "face-mask",
                 "foot", "tool", "glasses", "gloves", "helmet", "hands", "head",
                 "medical-suit", "shoes", "safety-suit", "safety-vest"],
    "sfchd":    ["person", "helmet", "self_clothes", "safety_clothes",
                 "head", "blur_head", "blur_clothes"],
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DATASET_ROOTS = {
    "roboflow": HOME / "FYP/Dataset/Roboflow Dataset/construction safety.yolov8",
    "chvg":     HOME / "FYP/Dataset/CHVG-Dataset",
    "shel5k":   HOME / "FYP/Dataset/SHEL5K/Safety Helmet Wearing Dataset",
    "sh17":     HOME / "FYP/Dataset/SH17dataset/data/archive",
    "sfchd":    HOME / "FYP/Dataset/SFCHD-SCALE/dataset_SFCHD",
}


# ── GPU temperature ────────────────────────────────────────────────────────────

def gpu_temp() -> int | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return int(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


# ── VOC XML parsing ───────────────────────────────────────────────────────────

def parse_voc_xml(xml_path: Path, class_map: dict) -> list:
    """Parse VOC XML and return list of [x1, y1, x2, y2, class_id]."""
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        if w == 0 or h == 0:
            return []

        boxes = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if name not in class_map:
                continue
            bb = obj.find("bndbox")
            x1 = max(0.0,   float(bb.find("xmin").text))
            y1 = max(0.0,   float(bb.find("ymin").text))
            x2 = min(float(w), float(bb.find("xmax").text))
            y2 = min(float(h), float(bb.find("ymax").text))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2, class_map[name]])
        return boxes
    except Exception as exc:
        print(f"    [WARN] {xml_path.name}: {exc}")
        return []


def parse_yolo_txt(txt_path: Path, img_size: tuple, class_map: dict) -> list:
    """Parse YOLO .txt and return list of [x1, y1, x2, y2, class_id]."""
    try:
        w, h = img_size
        boxes = []
        for line in txt_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2, cls_id])
        return boxes
    except Exception as exc:
        print(f"    [WARN] {txt_path.name}: {exc}")
        return []


# ── Custom Dataset ─────────────────────────────────────────────────────────────

class ConstructionSafetyDataset(Dataset):
    """Generic dataset for construction safety with support for multiple formats."""

    def __init__(self, img_paths: list, lbl_sources: list, classes: list, split: str = "train"):
        """
        Args:
            img_paths: list of Path objects for images
            lbl_sources: list of (format, path) tuples where format is 'voc' or 'yolo'
            classes: list of class names
            split: 'train', 'val', or 'test'
        """
        self.img_paths = img_paths
        self.lbl_sources = lbl_sources
        self.classes = classes
        self.class_map = {c: i + 1 for i, c in enumerate(classes)}  # 1-indexed for FasterRCNN
        self.split = split

        # Augmentation for training
        if split == "train":
            self.transform = T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.1),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
            ])
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        lbl_format, lbl_path = self.lbl_sources[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Load boxes and labels
        boxes = []
        if lbl_format == "voc":
            boxes = parse_voc_xml(lbl_path, self.class_map)
        elif lbl_format == "yolo":
            boxes = parse_yolo_txt(lbl_path, (w, h), self.class_map)

        if not boxes:
            boxes = [[0, 0, 1, 1, 0]]

        boxes = np.array(boxes, dtype=np.float32)
        image_tensor = self.transform(image)

        target = {
            "boxes": torch.as_tensor(boxes[:, :4], dtype=torch.float32),
            "labels": torch.as_tensor(boxes[:, 4], dtype=torch.int64),
        }

        return image_tensor, target


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_roboflow_dataset(split: str, classes: list):
    root = DATASET_ROOTS["roboflow"]
    # Roboflow uses 'valid' not 'val'
    split_name = "valid" if split == "val" else split
    img_dir = root / split_name / "images"
    lbl_dir = root / split_name / "labels"

    img_paths = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in IMG_EXTS])
    lbl_sources = [(("yolo" if (lbl_dir / (p.stem + ".txt")).exists() else "voc"),
                    lbl_dir / (p.stem + ".txt")) for p in img_paths]
    return img_paths, lbl_sources


def build_chvg_dataset(split: str, classes: list):
    """CHVG is VOC format."""
    root = DATASET_ROOTS["chvg"]

    # Check if already converted to YOLO format
    yolo_root = HOME / "FYP/Dataset/CHVG-YOLO"
    if (yolo_root / split / "images").exists():
        img_dir = yolo_root / split / "images"
        lbl_dir = yolo_root / split / "labels"
        img_paths = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in IMG_EXTS])
        lbl_sources = [("yolo", lbl_dir / (p.stem + ".txt")) for p in img_paths]
        return img_paths, lbl_sources

    # Fallback: use raw VOC format
    xml_files = sorted(root.glob("*.xml"))
    class_map = {c: i + 1 for i, c in enumerate(classes)}
    img_paths = []
    lbl_sources = []

    for xml_path in xml_files:
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = xml_path.with_suffix(ext)
            if img_path.exists():
                img_paths.append(img_path)
                lbl_sources.append(("voc", xml_path))
                break

    # Split deterministically
    random.Random(SEED).shuffle(list(zip(img_paths, lbl_sources)))
    n = len(img_paths)
    i1 = int(n * 0.70)
    i2 = i1 + int(n * 0.15)

    if split == "train":
        return img_paths[:i1], lbl_sources[:i1]
    elif split == "val":
        return img_paths[i1:i2], lbl_sources[i1:i2]
    else:
        return img_paths[i2:], lbl_sources[i2:]


def build_shel5k_dataset(split: str, classes: list):
    """SHEL5K is VOC format."""
    root = DATASET_ROOTS["shel5k"]
    ann_dir = root / "Annotations"
    img_dir = root / "Images"

    # Check if already converted
    yolo_root = HOME / "FYP/Dataset/SHEL5K-YOLO"
    if (yolo_root / split / "images").exists():
        yolo_img_dir = yolo_root / split / "images"
        yolo_lbl_dir = yolo_root / split / "labels"
        img_paths = sorted([p for p in yolo_img_dir.glob("*.*") if p.suffix.lower() in IMG_EXTS])
        lbl_sources = [("yolo", yolo_lbl_dir / (p.stem + ".txt")) for p in img_paths]
        return img_paths, lbl_sources

    # Fallback: raw VOC
    xml_files = sorted(ann_dir.glob("*.xml"))
    class_map = {c: i + 1 for i, c in enumerate(classes)}
    img_paths = []
    lbl_sources = []

    for xml_path in xml_files:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = img_dir / (xml_path.stem + ext)
            if candidate.exists():
                img_paths.append(candidate)
                lbl_sources.append(("voc", xml_path))
                break

    # Split
    random.Random(SEED).shuffle(list(zip(img_paths, lbl_sources)))
    n = len(img_paths)
    i1 = int(n * 0.70)
    i2 = i1 + int(n * 0.15)

    if split == "train":
        return img_paths[:i1], lbl_sources[:i1]
    elif split == "val":
        return img_paths[i1:i2], lbl_sources[i1:i2]
    else:
        return img_paths[i2:], lbl_sources[i2:]


def build_sh17_dataset(split: str, classes: list):
    """SH17 uses .txt split files."""
    root = DATASET_ROOTS["sh17"]
    img_dir = root / "images"

    # Check converted format first
    yolo_root = HOME / "FYP/Dataset/SH17-YOLO"
    if (yolo_root / f"{split}.txt").exists():
        img_paths = []
        for line in (yolo_root / f"{split}.txt").read_text().strip().split("\n"):
            if line.strip():
                img_paths.append(Path(line.strip()))
        lbl_dir = root / "labels"
        lbl_sources = [("yolo", lbl_dir / (p.stem + ".txt")) for p in img_paths]
        return img_paths, lbl_sources

    # Fallback: use raw splits
    train_names = (root / "train_files.txt").read_text().strip().splitlines()
    val_names = (root / "val_files.txt").read_text().strip().splitlines()

    rng = random.Random(SEED)
    rng.shuffle(val_names)
    n_test = max(50, int(len(val_names) * 0.20))
    test_names = val_names[:n_test]
    val_names = val_names[n_test:]

    if split == "train":
        names = train_names
    elif split == "val":
        names = val_names
    else:
        names = test_names

    img_paths = [img_dir / n for n in names if (img_dir / n).exists()]
    lbl_dir = root / "labels"
    lbl_sources = [("yolo", lbl_dir / (p.stem + ".txt")) for p in img_paths]
    return img_paths, lbl_sources


def build_sfchd_dataset(split: str, classes: list):
    """SFCHD is YOLO format."""
    root = DATASET_ROOTS["sfchd"]
    img_dir = root / "images"
    lbl_dir = root / "labels"

    valid_imgs = sorted([p for p in img_dir.glob("*.jpg") if (lbl_dir / (p.stem + ".txt")).exists()])

    # Split
    random.Random(SEED).shuffle(valid_imgs)
    n = len(valid_imgs)
    i1 = int(n * 0.70)
    i2 = i1 + int(n * 0.15)

    if split == "train":
        img_paths = valid_imgs[:i1]
    elif split == "val":
        img_paths = valid_imgs[i1:i2]
    else:
        img_paths = valid_imgs[i2:]

    lbl_sources = [("yolo", lbl_dir / (p.stem + ".txt")) for p in img_paths]
    return img_paths, lbl_sources


DATASET_BUILDERS = {
    "roboflow": build_roboflow_dataset,
    "chvg":     build_chvg_dataset,
    "shel5k":   build_shel5k_dataset,
    "sh17":     build_sh17_dataset,
    "sfchd":    build_sfchd_dataset,
}


# ── COCO evaluation ────────────────────────────────────────────────────────────

def create_coco_json(dataset: ConstructionSafetyDataset, targets_list: list, predictions: list) -> tuple:
    """Create COCO format JSON for evaluation using test set data.

    Args:
        dataset: The test dataset
        targets_list: List of ground truth target dicts (from DataLoader)
        predictions: List of (boxes, scores, labels) tuples from model

    Returns: (coco_gt_dict, coco_dt_list) for evaluation.
    """
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": cls} for i, cls in enumerate(dataset.classes)],
    }
    coco_dt = []
    ann_id = 1

    # Create GT from test set
    for img_idx, targets in enumerate(targets_list):
        img_id = img_idx + 1
        img_path = dataset.img_paths[img_idx]

        # Get image size
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
        except Exception:
            continue

        coco_gt["images"].append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": str(img_path),
        })

        # Ground truth annotations
        for box, label in zip(targets["boxes"], targets["labels"]):
            x1, y1, x2, y2 = box.tolist()
            w_box = max(1, x2 - x1)
            h_box = max(1, y2 - y1)
            if w_box > 0 and h_box > 0:
                coco_gt["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, w_box, h_box],
                    "area": w_box * h_box,
                    "iscrowd": 0,
                })
                ann_id += 1

    # Create detections from predictions
    for img_idx, (boxes, scores, labels) in enumerate(predictions):
        img_id = img_idx + 1
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w_box = max(1, x2 - x1)
            h_box = max(1, y2 - y1)
            if w_box > 0 and h_box > 0 and score > 0.001:  # Skip very low confidence
                coco_dt.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, w_box, h_box],
                    "score": float(score),
                })

    return coco_gt, coco_dt


def evaluate_coco(coco_gt_dict: dict, coco_dt_list: list, classes: list) -> dict:
    """Evaluate using COCO metrics."""
    try:
        import tempfile

        # Validate inputs
        if not coco_gt_dict.get("images"):
            print("  [WARN] No ground truth images")
            return {"map50": 0.0, "map_all": 0.0, "prec": 0.0, "rec": 0.0,
                   "per_class_ap": {cls: 0.0 for cls in classes}}

        if not coco_dt_list:
            print("  [WARN] No predictions generated")
            return {"map50": 0.0, "map_all": 0.0, "prec": 0.0, "rec": 0.0,
                   "per_class_ap": {cls: 0.0 for cls in classes}}

        print(f"  [DEBUG] GT: {len(coco_gt_dict['images'])} images, {len(coco_gt_dict['annotations'])} annotations")
        print(f"  [DEBUG] DT: {len(coco_dt_list)} predictions")

        # Write temp JSON files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_gt:
            json.dump(coco_gt_dict, f_gt)
            gt_path = f_gt.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_dt:
            json.dump(coco_dt_list, f_dt)
            dt_path = f_dt.name

        # Suppress COCO API output
        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            # Evaluate
            coco_gt = COCO(gt_path)
            coco_dt = coco_gt.loadRes(dt_path)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Extract metrics
            map_all = float(coco_eval.stats[0]) if coco_eval.stats[0] is not None else 0.0
            map50 = float(coco_eval.stats[1]) if coco_eval.stats[1] is not None else 0.0
            prec = float(coco_eval.stats[3]) if coco_eval.stats[3] is not None else 0.0
            rec = float(coco_eval.stats[8]) if coco_eval.stats[8] is not None else 0.0

            # Per-class AP@0.5
            per_class_ap = {}
            for cat_id in coco_gt.getCatIds():
                try:
                    coco_eval_cat = COCOeval(coco_gt, coco_dt, "bbox")
                    coco_eval_cat.params.catIds = [cat_id]
                    coco_eval_cat.evaluate()
                    coco_eval_cat.accumulate()
                    ap50 = float(coco_eval_cat.stats[1]) if coco_eval_cat.stats[1] is not None else 0.0
                    cat_name = coco_gt.loadCats(cat_id)[0]["name"]
                    per_class_ap[cat_name] = ap50
                except Exception as e:
                    cat_name = coco_gt.loadCats(cat_id)[0]["name"]
                    per_class_ap[cat_name] = 0.0

        # Cleanup
        Path(gt_path).unlink()
        Path(dt_path).unlink()

        print(f"  [DEBUG] mAP@0.5={map50:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

        return {
            "map50": map50,
            "map_all": map_all,
            "prec": prec,
            "rec": rec,
            "per_class_ap": per_class_ap,
        }
    except Exception as e:
        print(f"  [ERROR] COCO evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "map50": 0.0,
            "map_all": 0.0,
            "prec": 0.0,
            "rec": 0.0,
            "per_class_ap": {cls: 0.0 for cls in classes},
        }


# ── Training function ──────────────────────────────────────────────────────────

def train_experiment(key: str, classes: list) -> dict:
    exp = EXPERIMENTS[key]
    exp_num = exp["exp_num"]
    exp_name = exp["name"].lower()
    out_dir = RESULTS_BASE / f"exp{exp_num}_fasterrcnn_{exp_name}"
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  EXP {exp_num} — Faster R-CNN ResNet50-FPN | {exp['name']}")
    print("=" * 70)
    print(f"  results   : {out_dir}")
    print(f"  classes   : {len(classes)} → {classes}")
    print(f"  GPU       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    t0 = gpu_temp()
    if t0 is not None:
        print(f"  GPU temp  : {t0}°C")
    print()

    # Build datasets
    print("  Building datasets...")
    builder = DATASET_BUILDERS[key]
    train_imgs, train_lbls = builder("train", classes)
    val_imgs, val_lbls = builder("val", classes)
    test_imgs, test_lbls = builder("test", classes)

    train_ds = ConstructionSafetyDataset(train_imgs, train_lbls, classes, split="train")
    val_ds = ConstructionSafetyDataset(val_imgs, val_lbls, classes, split="val")
    test_ds = ConstructionSafetyDataset(test_imgs, test_lbls, classes, split="test")

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Attempt training with OOM handling
    batch_size = BATCH_INIT
    while batch_size >= BATCH_MIN:
        try:
            print(f"\n  Attempting training with batch_size={batch_size}...")
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                     collate_fn=collate_fn, num_workers=WORKERS, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_fn, num_workers=WORKERS, pin_memory=True)

            # Model setup (using MobileNetV2 for lighter inference)
            model = create_fasterrcnn_mobilenet(num_classes=len(classes) + 1)
            model.to(DEVICE)

            # Optimizer and scheduler
            optimizer = torch.optim.SGD(model.parameters(), lr=LR0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            scaler = GradScaler()

            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            t_start = time.time()

            for epoch in range(EPOCHS):
                model.train()
                total_loss = 0.0
                num_batches = 0

                for images, targets in train_loader:
                    images = [img.to(DEVICE) for img in images]
                    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                    with autocast():
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    scaler.scale(losses).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += losses.item()
                    num_batches += 1

                avg_loss = total_loss / num_batches
                print(f"  Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}", end="")

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), weights_dir / "best.pth")
                    print(" [BEST]")
                else:
                    patience_counter += 1
                    print()

                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

                scheduler.step()

                # GPU temp check
                temp = gpu_temp()
                if temp is not None and temp >= GPU_WARN_C:
                    print(f"\n  *** GPU TEMPERATURE WARNING: {temp}°C ***\n")

            train_time = time.time() - t_start
            print(f"\n  Training complete: {train_time / 60:.1f} min  ({epoch + 1} epochs)")
            break  # Success — exit the OOM retry loop

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"  [OOM] Reducing batch size from {batch_size} to {batch_size // 2}")
                batch_size //= 2
                torch.cuda.empty_cache()
                if batch_size < BATCH_MIN:
                    print(f"  [ERROR] Cannot reduce batch size further. OOM unavoidable.")
                    raise
            else:
                raise

    # Load best model for evaluation
    model.load_state_dict(torch.load(weights_dir / "best.pth", map_location=DEVICE))
    model.eval()

    # Test evaluation
    print(f"\n  Evaluating on test split ({len(test_ds)} images)...")
    predictions = []
    targets_list = []
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            for output in outputs:
                predictions.append((
                    output["boxes"].cpu().numpy(),
                    output["scores"].cpu().numpy(),
                    output["labels"].cpu().numpy(),
                ))
            # Store targets (move to CPU for COCO evaluation)
            for target in targets:
                targets_list.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                })

    # COCO evaluation
    coco_gt, coco_dt = create_coco_json(test_ds, targets_list, predictions)
    metrics = evaluate_coco(coco_gt, coco_dt, classes)
    map50 = metrics["map50"]
    map_all = metrics["map_all"]
    prec = metrics["prec"]
    rec = metrics["rec"]
    per_class_ap = metrics["per_class_ap"]

    # F1 score
    f1 = 2 * prec * rec / (prec + rec + 1e-9) if (prec + rec) > 0 else 0.0

    # FPS benchmark
    print(f"\n  Measuring inference speed ...")
    latencies = []
    for img, _ in test_ds:
        t = time.perf_counter()
        with torch.no_grad():
            model([img.to(DEVICE)])
        latencies.append((time.perf_counter() - t) * 1000)
    avg_ms = np.mean(latencies) if latencies else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    # Write metrics.txt
    lines = [
        "=" * 70,
        f"  EXPERIMENT {exp_num} — Faster R-CNN ResNet50-FPN on {exp['name']}",
        "=" * 70,
        "",
        "  SETUP",
        f"    Model            : Faster R-CNN ResNet50-FPN (PyTorch)",
        f"    Dataset          : {exp['name']}",
        f"    Classes ({len(classes):2d})      : {classes}",
        f"    Epochs trained   : {epoch + 1}  (max={EPOCHS}, patience={PATIENCE})",
        f"    Training time    : {train_time / 60:.1f} min",
        f"    Batch size       : {batch_size}",
        f"    GPU              : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        "",
        "  TEST SET METRICS",
        f"    mAP@0.5          : {map50:.4f}  ({map50 * 100:.2f}%)",
        f"    mAP@0.5:0.95     : {map_all:.4f}  ({map_all * 100:.2f}%)",
        f"    Precision        : {prec:.4f}  ({prec * 100:.2f}%)",
        f"    Recall           : {rec:.4f}  ({rec * 100:.2f}%)",
        f"    F1               : {f1:.4f}",
        "",
        "  INFERENCE (single image, GPU, 640×640)",
        f"    Avg latency      : {avg_ms:.2f} ms",
        f"    FPS              : {fps:.1f}",
        "",
        "  PER-CLASS AP@0.5",
    ]
    for cls, ap in per_class_ap.items():
        lines.append(f"    {cls:<24}: {ap:.4f}  ({ap * 100:.2f}%)")

    lines += [
        "",
        "  TRAINING CONFIG",
        f"    Batch / Img size : {batch_size} / {IMGSZ}×{IMGSZ}",
        f"    Optimizer        : SGD  lr0={LR0}  momentum={MOMENTUM}",
        f"    Weight decay     : {WEIGHT_DECAY}  |  Seed: {SEED}",
        f"    AMP              : True",
        "",
        f"  Weights  : {weights_dir / 'best.pth'}",
        f"  Run dir  : {out_dir}",
        "=" * 70,
    ]
    report = "\n".join(lines)
    print("\n" + report)

    metrics_path = out_dir / "metrics.txt"
    metrics_path.write_text(report + "\n")
    print(f"\n  Saved → {metrics_path}")

    return {
        "key":      key,
        "name":     exp["name"],
        "exp_num":  exp_num,
        "map50":    map50,
        "map_all":  map_all,
        "prec":     prec,
        "rec":      rec,
        "f1":       f1,
        "fps":      fps,
        "lat_ms":   avg_ms,
        "epochs":   epoch + 1,
        "time_min": train_time / 60,
    }


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    sep = "=" * 100
    hdr = (f"  {'Dataset':<12} {'Epochs':>7} {'mAP@0.5':>10} {'Precision':>11}"
           f" {'Recall':>8} {'F1':>7} {'FPS':>7} {'Latency(ms)':>13}")
    print(f"\n{sep}\n  Faster R-CNN EXPERIMENT SUMMARY\n{sep}")
    print(hdr)
    print(f"  {'-' * 96}")
    for r in results:
        print(
            f"  {r['name']:<12} {r['epochs']:>7d} {r['map50']*100:>9.2f}%"
            f" {r['prec']*100:>10.2f}% {r['rec']*100:>7.2f}%"
            f" {r['f1']:>7.4f} {r['fps']:>7.1f} {r['lat_ms']:>12.2f}"
        )
    print(sep)

    summary_path = HOME / "FYP/results/summary/fasterrcnn_experiments_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"{sep}\n  Faster R-CNN EXPERIMENT SUMMARY\n{sep}\n{hdr}\n  {'-'*96}\n")
        for r in results:
            f.write(
                f"  {r['name']:<12} {r['epochs']:>7d} {r['map50']*100:>9.2f}%"
                f" {r['prec']*100:>10.2f}% {r['rec']*100:>7.2f}%"
                f" {r['f1']:>7.4f} {r['fps']:>7.1f} {r['lat_ms']:>12.2f}\n"
            )
        f.write(f"{sep}\n")
    print(f"\n  Summary saved → {summary_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Faster R-CNN on all 5 construction safety datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 train_fasterrcnn_all.py --datasets all
  python3 train_fasterrcnn_all.py --datasets roboflow chvg
  python3 train_fasterrcnn_all.py --datasets sh17 sfchd
        """,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=["roboflow", "chvg", "shel5k", "sh17", "sfchd", "all"],
        metavar="DATASET",
        help="Datasets to train (choices: roboflow chvg shel5k sh17 sfchd all)",
    )
    args = parser.parse_args()

    keys = list(EXPERIMENTS) if "all" in args.datasets else [
        k for k in args.datasets if k in EXPERIMENTS
    ]
    if not keys:
        print("No valid datasets specified.")
        return 1

    print("=" * 70)
    print("  Faster R-CNN — All Experiments")
    print("=" * 70)
    print(f"  Running  : {[EXPERIMENTS[k]['name'] for k in keys]}")
    print(f"  GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    t = gpu_temp()
    if t:
        print(f"  GPU temp : {t}°C")
    print()

    results: list[dict] = []
    errors:  list[tuple] = []

    for key in keys:
        try:
            print(f"\n{'─' * 70}")
            print(f"  Resolving dataset: {EXPERIMENTS[key]['name']} ...")
            classes = CLASSES[key]
            metrics = train_experiment(key, classes)
            results.append(metrics)
        except Exception:
            tb = traceback.format_exc()
            print(f"\n  FAILED — EXP {EXPERIMENTS[key]['exp_num']} ({EXPERIMENTS[key]['name']}):\n{tb}")
            errors.append((key, tb))
            continue

    if results:
        print_summary(results)

    if errors:
        print(f"\n  Experiments that failed: {[e[0] for e in errors]}")
        return 1 if len(errors) == len(keys) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
