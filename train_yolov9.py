#!/usr/bin/env python3
"""
YOLOv9s training on all 5 construction safety datasets.

  EXP 06 — Roboflow  : ~/FYP/Dataset/Roboflow Dataset/construction safety.yolov8/
  EXP 07 — CHVG      : reuses ~/FYP/Dataset/CHVG-YOLO/   (converted by YOLOv8 run)
  EXP 08 — SHEL5K    : reuses ~/FYP/Dataset/SHEL5K-YOLO/ (converted by YOLOv8 run)
  EXP 09 — SH17      : reuses ~/FYP/Dataset/SH17-YOLO/   (prepared by YOLOv8 run)
  EXP 10 — SFCHD     : reuses ~/FYP/Dataset/SFCHD-YOLO/  (prepared by YOLOv8 run)
                        Falls back to building the split if SFCHD-YOLO is absent.

Usage:
  python3 train_yolov9_all.py --datasets all
  python3 train_yolov9_all.py --datasets roboflow chvg shel5k
  python3 train_yolov9_all.py --datasets sh17 sfchd
"""

import argparse
import random
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ── Constants ──────────────────────────────────────────────────────────────────
HOME         = Path.home()
WEIGHTS      = HOME / "FYP/results/weights/yolov9s.pt"
RESULTS_BASE = HOME / "FYP/results/YOLOv9"
SEED         = 42
EPOCHS       = 100
PATIENCE     = 20
BATCH        = 8
IMGSZ        = 640
LR0          = 0.01
OPTIMIZER    = "SGD"
DEVICE       = 0
WORKERS      = 4
GPU_WARN_C   = 85   # °C

torch.manual_seed(SEED)

EXPERIMENTS = {
    "roboflow": {"exp_num": "06", "name": "Roboflow"},
    "chvg":     {"exp_num": "07", "name": "CHVG"},
    "shel5k":   {"exp_num": "08", "name": "SHEL5K"},
    "sh17":     {"exp_num": "09", "name": "SH17"},
    "sfchd":    {"exp_num": "10", "name": "SFCHD"},
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


# ── VOC XML → YOLO .txt (fallback converter) ──────────────────────────────────

def voc_to_yolo(xml_path: Path, class_map: dict, out_txt: Path) -> bool:
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        if w == 0 or h == 0:
            return False
        lines = []
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
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{class_map[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))
        return True
    except Exception as exc:
        print(f"    [WARN] {xml_path.name}: {exc}")
        return False


def split_list(items, train_r: float = 0.70, val_r: float = 0.15):
    lst = list(items)
    random.Random(SEED).shuffle(lst)
    n = len(lst)
    i1 = int(n * train_r)
    i2 = i1 + int(n * val_r)
    return lst[:i1], lst[i1:i2], lst[i2:]


def write_yaml(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)


# ── Dataset resolution ─────────────────────────────────────────────────────────
# Each function returns the absolute path to a ready data.yaml.
# Primary strategy: reuse the YOLO-format datasets already created by the
# YOLOv8 training run.  Fallback: recreate them if missing.

def resolve_roboflow() -> str:
    yaml_path = HOME / "FYP/results/configs/exp01_roboflow_abs.yaml"
    if yaml_path.exists():
        print("  [Roboflow] Using existing data.yaml.")
        return str(yaml_path)

    # Fallback: write the yaml from scratch (original dataset paths are fixed)
    print("  [Roboflow] Writing data.yaml ...")
    base = HOME / "FYP/Dataset/Roboflow Dataset/construction safety.yolov8"
    write_yaml(yaml_path, {
        "train": str(base / "train/images"),
        "val":   str(base / "valid/images"),
        "test":  str(base / "test/images"),
        "nc":    5,
        "names": CLASSES["roboflow"],
    })
    return str(yaml_path)


def resolve_chvg() -> str:
    yaml_path = HOME / "FYP/Dataset/CHVG-YOLO/data.yaml"
    if yaml_path.exists():
        print("  [CHVG] Reusing converted dataset from YOLOv8 run.")
        return str(yaml_path)

    # Fallback: convert from scratch
    print("  [CHVG] CHVG-YOLO missing — converting VOC → YOLO ...")
    src       = HOME / "FYP/Dataset/CHVG-Dataset"
    out       = HOME / "FYP/Dataset/CHVG-YOLO"
    classes   = CLASSES["chvg"]
    class_map = {c: i for i, c in enumerate(classes)}

    pairs = []
    for xml in sorted(src.glob("*.xml")):
        for ext in (".jpg", ".jpeg", ".png"):
            img = xml.with_suffix(ext)
            if img.exists():
                pairs.append((img, xml))
                break

    print(f"  [CHVG] {len(pairs)} image–XML pairs")
    train_p, val_p, test_p = split_list(pairs)

    for split, split_pairs in [("train", train_p), ("val", val_p), ("test", test_p)]:
        img_dir = out / split / "images"
        lbl_dir = out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        ok = sum(
            1 for img_path, xml_path in split_pairs
            if shutil.copy2(img_path, img_dir / img_path.name) is not None
            or voc_to_yolo(xml_path, class_map, lbl_dir / (img_path.stem + ".txt"))
        )
        print(f"    {split}: {len(split_pairs)} images")

    write_yaml(yaml_path, {
        "path":  str(out),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(classes),
        "names": classes,
    })
    return str(yaml_path)


def resolve_shel5k() -> str:
    yaml_path = HOME / "FYP/Dataset/SHEL5K-YOLO/data.yaml"
    if yaml_path.exists():
        print("  [SHEL5K] Reusing converted dataset from YOLOv8 run.")
        return str(yaml_path)

    # Fallback: convert from scratch
    print("  [SHEL5K] SHEL5K-YOLO missing — converting VOC → YOLO ...")
    src       = HOME / "FYP/Dataset/SHEL5K/Safety Helmet Wearing Dataset"
    out       = HOME / "FYP/Dataset/SHEL5K-YOLO"
    classes   = CLASSES["shel5k"]
    class_map = {c: i for i, c in enumerate(classes)}
    ann_dir   = src / "Annotations"
    img_dir   = src / "Images"

    pairs = []
    for xml in sorted(ann_dir.glob("*.xml")):
        for ext in (".png", ".jpg", ".jpeg"):
            img = img_dir / (xml.stem + ext)
            if img.exists():
                pairs.append((img, xml))
                break

    print(f"  [SHEL5K] {len(pairs)} image–XML pairs")
    train_p, val_p, test_p = split_list(pairs)

    for split, split_pairs in [("train", train_p), ("val", val_p), ("test", test_p)]:
        sp_img = out / split / "images"
        sp_lbl = out / split / "labels"
        sp_img.mkdir(parents=True, exist_ok=True)
        sp_lbl.mkdir(parents=True, exist_ok=True)
        for img_path, xml_path in split_pairs:
            shutil.copy2(img_path, sp_img / img_path.name)
            voc_to_yolo(xml_path, class_map, sp_lbl / (img_path.stem + ".txt"))
        print(f"    {split}: {len(split_pairs)} images")

    write_yaml(yaml_path, {
        "path":  str(out),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(classes),
        "names": classes,
    })
    return str(yaml_path)


def resolve_sh17() -> str:
    yaml_path = HOME / "FYP/Dataset/SH17-YOLO/data.yaml"
    if yaml_path.exists():
        print("  [SH17] Reusing prepared dataset from YOLOv8 run.")
        return str(yaml_path)

    # Fallback: build absolute-path split files
    print("  [SH17] SH17-YOLO missing — building split files ...")
    src     = HOME / "FYP/Dataset/SH17dataset/data/archive"
    out     = HOME / "FYP/Dataset/SH17-YOLO"
    img_dir = src / "images"

    train_names = (src / "train_files.txt").read_text().strip().splitlines()
    val_names   = (src / "val_files.txt").read_text().strip().splitlines()

    rng = random.Random(SEED)
    rng.shuffle(val_names)
    n_test    = max(50, int(len(val_names) * 0.20))
    test_names = val_names[:n_test]
    val_names  = val_names[n_test:]

    out.mkdir(parents=True, exist_ok=True)

    def write_abs(names, dest):
        valid = [str(img_dir / n) for n in names if (img_dir / n).exists()]
        dest.write_text("\n".join(valid) + "\n")
        return len(valid)

    n_tr = write_abs(train_names, out / "train.txt")
    n_va = write_abs(val_names,   out / "val.txt")
    n_te = write_abs(test_names,  out / "test.txt")
    print(f"  [SH17] Train: {n_tr} | Val: {n_va} | Test: {n_te}")

    classes = CLASSES["sh17"]
    write_yaml(yaml_path, {
        "path":  str(src),
        "train": str(out / "train.txt"),
        "val":   str(out / "val.txt"),
        "test":  str(out / "test.txt"),
        "nc":    len(classes),
        "names": classes,
    })
    return str(yaml_path)


def resolve_sfchd() -> str:
    yaml_path = HOME / "FYP/Dataset/SFCHD-YOLO/data.yaml"
    if yaml_path.exists():
        print("  [SFCHD] Reusing prepared dataset from YOLOv8 run.")
        return str(yaml_path)

    # Fallback: rebuild 70/15/15 split from local images
    print("  [SFCHD] SFCHD-YOLO missing — building split from local images ...")
    src     = HOME / "FYP/Dataset/SFCHD-SCALE/dataset_SFCHD"
    out     = HOME / "FYP/Dataset/SFCHD-YOLO"
    img_dir = src / "images"
    lbl_dir = src / "labels"

    valid = sorted(
        p for p in img_dir.glob("*.jpg")
        if (lbl_dir / (p.stem + ".txt")).exists()
    )
    print(f"  [SFCHD] {len(valid)} images with matching labels")

    train_imgs, val_imgs, test_imgs = split_list(valid)
    out.mkdir(parents=True, exist_ok=True)
    for name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        (out / f"{name}.txt").write_text("\n".join(str(p) for p in imgs) + "\n")
    print(f"  [SFCHD] Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    classes = CLASSES["sfchd"]
    write_yaml(yaml_path, {
        "path":  str(src),
        "train": str(out / "train.txt"),
        "val":   str(out / "val.txt"),
        "test":  str(out / "test.txt"),
        "nc":    len(classes),
        "names": classes,
    })
    return str(yaml_path)


RESOLVE_FNS = {
    "roboflow": resolve_roboflow,
    "chvg":     resolve_chvg,
    "shel5k":   resolve_shel5k,
    "sh17":     resolve_sh17,
    "sfchd":    resolve_sfchd,
}

# Where to find test images for the FPS benchmark
TEST_IMG_SOURCES = {
    "roboflow": ("dir", HOME / "FYP/Dataset/Roboflow Dataset/construction safety.yolov8/test/images"),
    "chvg":     ("dir", HOME / "FYP/Dataset/CHVG-YOLO/test/images"),
    "shel5k":   ("dir", HOME / "FYP/Dataset/SHEL5K-YOLO/test/images"),
    "sh17":     ("txt", HOME / "FYP/Dataset/SH17-YOLO/test.txt"),
    "sfchd":    ("txt", HOME / "FYP/Dataset/SFCHD-YOLO/test.txt"),
}


def get_test_images(key: str, n: int = 100) -> list[Path]:
    kind, source = TEST_IMG_SOURCES[key]
    if kind == "dir":
        imgs = sorted(p for p in Path(source).glob("*.*") if p.suffix.lower() in IMG_EXTS)
    else:
        txt = Path(source)
        if not txt.exists():
            return []
        imgs = [Path(l.strip()) for l in txt.read_text().splitlines() if l.strip()]
    return [p for p in imgs if p.exists()][:n]


# ── Core training function ─────────────────────────────────────────────────────

def train_experiment(key: str, yaml_path: str) -> dict:
    exp      = EXPERIMENTS[key]
    exp_num  = exp["exp_num"]
    exp_name = exp["name"].lower()
    out_name = f"exp{exp_num}_yolov9_{exp_name}"
    out_dir  = RESULTS_BASE / out_name
    classes  = CLASSES[key]

    print("\n" + "=" * 70)
    print(f"  EXP {exp_num} — YOLOv9s | {exp['name']}")
    print("=" * 70)
    print(f"  data.yaml : {yaml_path}")
    print(f"  results   : {out_dir}")
    print(f"  classes   : {len(classes)} → {classes}")
    print(f"  GPU       : {torch.cuda.get_device_name(DEVICE)}")
    t0 = gpu_temp()
    if t0 is not None:
        print(f"  GPU temp  : {t0}°C")
    print()

    # ── temperature callback ───────────────────────────────────────────────────
    def on_train_epoch_end(trainer):
        temp = gpu_temp()
        if temp is not None and temp >= GPU_WARN_C:
            print(f"\n  *** GPU TEMPERATURE WARNING: {temp}°C (epoch {trainer.epoch + 1}) ***")

    # ── train ─────────────────────────────────────────────────────────────────
    model = YOLO(str(WEIGHTS))
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    t_start = time.time()
    model.train(
        data         = yaml_path,
        epochs       = EPOCHS,
        patience     = PATIENCE,
        batch        = BATCH,
        imgsz        = IMGSZ,
        optimizer    = OPTIMIZER,
        lr0          = LR0,
        momentum     = 0.937,
        weight_decay = 0.0005,
        amp          = True,
        seed         = SEED,
        device       = DEVICE,
        workers      = WORKERS,
        project      = str(RESULTS_BASE),
        name         = out_name,
        exist_ok     = True,
        verbose      = True,
        plots        = True,
        save         = True,
        cache        = False,
    )
    train_time = time.time() - t_start

    try:
        epochs_run = model.trainer.epoch + 1   # 0-indexed internally
    except Exception:
        epochs_run = EPOCHS
    print(f"\n  Training complete: {train_time / 60:.1f} min  ({epochs_run} epochs)")

    # ── test evaluation ───────────────────────────────────────────────────────
    print(f"\n  Evaluating on test split ...")
    best_pt    = out_dir / "weights" / "best.pt"
    model_eval = YOLO(str(best_pt))

    test_m = model_eval.val(
        data    = yaml_path,
        split   = "test",
        batch   = BATCH,
        imgsz   = IMGSZ,
        device  = DEVICE,
        verbose = True,
    )

    map50   = float(test_m.box.map50)
    map5095 = float(test_m.box.map)
    prec    = float(test_m.box.mp)
    rec     = float(test_m.box.mr)
    f1      = 2 * prec * rec / (prec + rec + 1e-9)

    # ── per-class AP@0.5 ──────────────────────────────────────────────────────
    per_class: dict[str, float] = {}
    if hasattr(test_m.box, "ap50") and test_m.box.ap50 is not None:
        for i, cls in enumerate(classes):
            if i < len(test_m.box.ap50):
                per_class[cls] = float(test_m.box.ap50[i])

    # ── FPS benchmark ─────────────────────────────────────────────────────────
    print(f"\n  Measuring inference speed ...")
    test_imgs = get_test_images(key, n=100)
    if not test_imgs:
        print("  [WARN] No test images found for FPS benchmark")
        avg_ms, fps = 0.0, 0.0
    else:
        for img in test_imgs[:5]:           # warmup
            model_eval.predict(str(img), imgsz=IMGSZ, device=DEVICE, verbose=False)
        latencies = []
        for img in test_imgs[5:]:           # benchmark
            t = time.perf_counter()
            model_eval.predict(str(img), imgsz=IMGSZ, device=DEVICE, verbose=False)
            latencies.append((time.perf_counter() - t) * 1000)
        avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
        fps    = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    # ── write metrics.txt ─────────────────────────────────────────────────────
    gpu_name = torch.cuda.get_device_name(DEVICE)
    lines = [
        "=" * 70,
        f"  EXPERIMENT {exp_num} — YOLOv9s on {exp['name']}",
        "=" * 70,
        "",
        "  SETUP",
        f"    Model            : YOLOv9s ({WEIGHTS.name})",
        f"    Dataset          : {exp['name']}",
        f"    Classes ({len(classes):2d})      : {classes}",
        f"    Epochs trained   : {epochs_run}  (max={EPOCHS}, patience={PATIENCE})",
        f"    Training time    : {train_time / 60:.1f} min",
        f"    GPU              : {gpu_name}",
        "",
        "  TEST SET METRICS",
        f"    mAP@0.5          : {map50:.4f}  ({map50 * 100:.2f}%)",
        f"    mAP@0.5:0.95     : {map5095:.4f}  ({map5095 * 100:.2f}%)",
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
    for cls, ap in per_class.items():
        lines.append(f"    {cls:<24}: {ap:.4f}  ({ap * 100:.2f}%)")
    lines += [
        "",
        "  TRAINING CONFIG",
        f"    Batch / Img size : {BATCH} / {IMGSZ}×{IMGSZ}",
        f"    Optimizer        : {OPTIMIZER}  lr0={LR0}  momentum=0.937",
        f"    AMP              : True  |  Seed: {SEED}",
        "",
        f"  Weights  : {best_pt}",
        f"  Run dir  : {out_dir}",
        "=" * 70,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.txt"
    metrics_path.write_text(report + "\n")
    print(f"\n  Saved → {metrics_path}")

    return {
        "key":      key,
        "name":     exp["name"],
        "exp_num":  exp_num,
        "map50":    map50,
        "map5095":  map5095,
        "prec":     prec,
        "rec":      rec,
        "f1":       f1,
        "fps":      fps,
        "lat_ms":   avg_ms,
        "epochs":   epochs_run,
        "time_min": train_time / 60,
    }


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    sep = "=" * 95
    hdr = (f"  {'Dataset':<12} {'Epochs':>7} {'mAP@0.5':>9} {'Precision':>11}"
           f" {'Recall':>8} {'F1':>7} {'FPS':>7} {'Latency(ms)':>13}")
    print(f"\n{sep}\n  YOLOv9s EXPERIMENT SUMMARY\n{sep}")
    print(hdr)
    print(f"  {'-' * 91}")
    for r in results:
        print(
            f"  {r['name']:<12} {r['epochs']:>7d} {r['map50'] * 100:>8.2f}%"
            f" {r['prec'] * 100:>10.2f}% {r['rec'] * 100:>7.2f}%"
            f" {r['f1']:>7.4f} {r['fps']:>7.1f} {r['lat_ms']:>12.2f}"
        )
    print(sep)

    summary_path = HOME / "FYP/results/summary/yolov9_experiments_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"{sep}\n  YOLOv9s EXPERIMENT SUMMARY\n{sep}\n{hdr}\n  {'-'*91}\n")
        for r in results:
            f.write(
                f"  {r['name']:<12} {r['epochs']:>7d} {r['map50']*100:>8.2f}%"
                f" {r['prec']*100:>10.2f}% {r['rec']*100:>7.2f}%"
                f" {r['f1']:>7.4f} {r['fps']:>7.1f} {r['lat_ms']:>12.2f}\n"
            )
        f.write(f"{sep}\n")
    print(f"\n  Summary saved → {summary_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLOv9s on all 5 construction safety datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 train_yolov9_all.py --datasets all
  python3 train_yolov9_all.py --datasets roboflow chvg shel5k
  python3 train_yolov9_all.py --datasets sh17 sfchd
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
    print("  YOLOv9s — All Experiments")
    print("=" * 70)
    print(f"  Running  : {[EXPERIMENTS[k]['name'] for k in keys]}")
    print(f"  Weights  : {WEIGHTS}")
    print(f"  GPU      : {torch.cuda.get_device_name(DEVICE)}")
    print(f"  VRAM     : {torch.cuda.get_device_properties(DEVICE).total_memory / 1e9:.1f} GB")
    t = gpu_temp()
    if t:
        print(f"  GPU temp : {t}°C")
    print()

    if not WEIGHTS.exists():
        print(f"  ERROR: weights not found at {WEIGHTS}")
        print("  Run: python3 -c \"from ultralytics import YOLO; YOLO('yolov9s.pt')\"")
        print("  then move yolov9s.pt to ~/FYP/results/")
        return 1

    results: list[dict] = []
    errors:  list[tuple] = []

    for key in keys:
        try:
            print(f"\n{'─' * 70}")
            print(f"  Resolving dataset: {EXPERIMENTS[key]['name']} ...")
            yaml_path = RESOLVE_FNS[key]()
            metrics   = train_experiment(key, yaml_path)
            results.append(metrics)
        except Exception:
            tb = traceback.format_exc()
            print(f"\n  FAILED — EXP {EXPERIMENTS[key]['exp_num']} ({EXPERIMENTS[key]['name']}):\n{tb}")
            errors.append((key, tb))

    if results:
        print_summary(results)

    if errors:
        print(f"\n  Experiments that failed: {[e[0] for e in errors]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
