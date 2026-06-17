#!/usr/bin/env python3
"""
Dataset integrity checker for PPE/Safety datasets.
Checks structure, file counts, pairing, class IDs, and overall readiness.
"""

import os
import glob
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from collections import defaultdict

HOME = Path.home()

DATASETS = {
    "SH17": {
        "path": HOME / "FYP/Dataset/SH17dataset",
        "format": "yolo",
        "expected_images": 8099,
        "expected_classes": 17,
    },
    "SFCHD": {
        "path": HOME / "FYP/Dataset/SFCHD-SCALE/dataset_SFCHD",
        "format": "yolo_coco",  # labels in zip, annotations in COCO JSON
        "expected_images": 12373,
        "expected_classes": 7,
    },
    "CHVG": {
        "path": HOME / "FYP/Dataset/CHVG-Dataset",
        "format": "voc",
        "expected_images": 1699,
        "expected_classes": 8,
    },
    "SHEL5K": {
        "path": HOME / "FYP/Dataset/SHEL5K/Safety Helmet Wearing Dataset",
        "format": "voc",
        "expected_images": 5000,
        "expected_classes": 6,
    },
    "RoboflowConstruction": {
        "path": HOME / "FYP/Dataset/Roboflow Dataset/construction safety.yolov8",
        "format": "yolo",
        "expected_images": None,
        "expected_classes": 5,
    },
}

SEP = "=" * 70


def header(name):
    print(f"\n{SEP}")
    print(f"  DATASET: {name}")
    print(SEP)


def check(label, ok, detail=""):
    status = "  [PASS]" if ok else "  [FAIL]"
    suffix = f"  →  {detail}" if detail else ""
    print(f"{status}  {label}{suffix}")
    return ok


def find_images(root):
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        imgs.extend(glob.glob(str(root / "**" / ext), recursive=True))
    return imgs


def find_labels(root, ext):
    return glob.glob(str(root / "**" / f"*{ext}"), recursive=True)


def folder_size_mb(path):
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def paired_stems(img_paths, label_paths):
    img_stems = {Path(p).stem for p in img_paths}
    lbl_stems = {Path(p).stem for p in label_paths}
    missing_labels = img_stems - lbl_stems
    extra_labels = lbl_stems - img_stems
    return missing_labels, extra_labels


def empty_files(paths):
    return [p for p in paths if os.path.getsize(p) == 0]


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset checkers
# ─────────────────────────────────────────────────────────────────────────────

def check_yolo(name, cfg):
    """Standard YOLO dataset: images/ + labels/ possibly under train/val/test splits."""
    root = cfg["path"]
    passes = []

    # 1. Folder exists
    exists = root.exists()
    passes.append(check("Folder exists", exists, str(root)))
    if not exists:
        return False

    # Look for images recursively
    all_images = find_images(root)
    all_labels = find_labels(root, ".txt")
    # Exclude non-annotation txt files
    all_labels = [p for p in all_labels
                  if Path(p).name not in ("classes.txt",) and
                  not any(kw in p for kw in ("/new_split_yolo/", "/Vision/", "files.txt", "README"))]

    # 2. Image count
    n_img = len(all_images)
    exp = cfg.get("expected_images")
    if exp:
        img_ok = abs(n_img - exp) / max(exp, 1) < 0.05  # within 5%
        passes.append(check("Image count", img_ok,
                            f"{n_img} found  (expected ~{exp})"))
    else:
        passes.append(check("Image count", n_img > 0, f"{n_img} found"))

    # 3. Label count
    n_lbl = len(all_labels)
    passes.append(check("Label (.txt) files found", n_lbl > 0, f"{n_lbl} files"))

    # 4. Pairing
    missing, extra = paired_stems(all_images, all_labels)
    passes.append(check("Images and labels paired",
                        len(missing) == 0,
                        f"{len(missing)} images missing label" +
                        (f", {len(extra)} orphan labels" if extra else "")))

    # 5. Missing labels
    if missing:
        sample = list(missing)[:5]
        print(f"         Sample missing: {sample}")

    # 6. Empty label files
    empties = empty_files(all_labels)
    # Empty txts are valid in YOLO (background images), just report
    check("Empty label files (background images OK)",
          True, f"{len(empties)} empty labels found")
    passes.append(True)

    # 7. Class IDs
    class_ids = set()
    sample_count = min(500, n_lbl)
    for lp in all_labels[:sample_count]:
        try:
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_ids.add(int(parts[0]))
        except Exception:
            pass
    exp_cls = cfg.get("expected_classes")
    cls_ok = (len(class_ids) == exp_cls) if exp_cls else len(class_ids) > 0
    passes.append(check("Unique class IDs",
                        cls_ok,
                        f"Found IDs: {sorted(class_ids)}  (expected {exp_cls} classes)"))

    # 8. Train/val/test splits
    splits_found = []
    for split in ("train", "val", "valid", "test"):
        if (root / split).exists():
            splits_found.append(split)
    # Also check for split txt files
    for txt in ("train_files.txt", "val_files.txt"):
        if any(txt in str(p) for p in root.rglob("*.txt")):
            splits_found.append(txt)
    has_splits = len(splits_found) >= 2
    passes.append(check("Train/val splits present", has_splits,
                        f"Found: {splits_found if splits_found else 'none'}"))

    # 9. Size
    size_mb = folder_size_mb(root)
    passes.append(check("Dataset size", size_mb > 1, f"{size_mb:.1f} MB"))

    return all(passes)


def check_voc(name, cfg):
    """Pascal VOC XML dataset."""
    root = cfg["path"]
    passes = []

    # 1. Folder exists
    exists = root.exists()
    passes.append(check("Folder exists", exists, str(root)))
    if not exists:
        return False

    all_images = find_images(root)
    all_labels = find_labels(root, ".xml")

    # 2. Image count
    n_img = len(all_images)
    exp = cfg.get("expected_images")
    if exp:
        img_ok = abs(n_img - exp) / max(exp, 1) < 0.05
        passes.append(check("Image count", img_ok,
                            f"{n_img} found  (expected ~{exp})"))
    else:
        passes.append(check("Image count", n_img > 0, f"{n_img} found"))

    # 3. Annotation count
    n_lbl = len(all_labels)
    passes.append(check("Annotation (.xml) files found", n_lbl > 0, f"{n_lbl} files"))

    # 4. Pairing
    missing, extra = paired_stems(all_images, all_labels)
    passes.append(check("Images and labels paired",
                        len(missing) == 0,
                        f"{len(missing)} images missing annotation" +
                        (f", {len(extra)} orphan XMLs" if extra else "")))

    # 5. Missing
    if missing:
        sample = list(missing)[:5]
        print(f"         Sample missing: {sample}")

    # 6. Empty files
    empties = empty_files(all_labels)
    passes.append(check("No empty annotation files", len(empties) == 0,
                        f"{len(empties)} empty XMLs"))

    # 7. Class names from XML
    class_names = set()
    sample_count = min(200, n_lbl)
    parse_errors = 0
    for xp in all_labels[:sample_count]:
        try:
            tree = ET.parse(xp)
            for obj in tree.findall(".//object"):
                nm = obj.find("name")
                if nm is not None:
                    class_names.add(nm.text.strip())
        except ET.ParseError:
            parse_errors += 1
    exp_cls = cfg.get("expected_classes")
    cls_ok = len(class_names) > 0
    passes.append(check("Class names parsed from XML",
                        cls_ok,
                        f"Found: {sorted(class_names)}" +
                        (f"  (expected {exp_cls} classes)" if exp_cls else "") +
                        (f"  [{parse_errors} parse errors]" if parse_errors else "")))

    # 8. Folder structure
    subdirs = [d.name for d in root.iterdir() if d.is_dir()]
    has_structure = any(d in subdirs for d in ("Annotations", "Images", "JPEGImages",
                                                "train", "val", "test"))
    passes.append(check("Folder structure",
                        has_structure or n_img > 0,
                        f"Subdirs: {subdirs if subdirs else 'flat'}" ))

    # 9. Size
    size_mb = folder_size_mb(root)
    passes.append(check("Dataset size", size_mb > 1, f"{size_mb:.1f} MB"))

    return all(passes)


def check_sfchd(name, cfg):
    """SFCHD has COCO JSON annotations + labels.zip (not extracted)."""
    root = cfg["path"]
    passes = []

    # 1. Folder exists
    exists = root.exists()
    passes.append(check("Folder exists", exists, str(root)))
    if not exists:
        return False

    # 2. Check COCO JSON annotations
    ann_dir = root / "annotations"
    json_files = list(ann_dir.glob("*.json")) if ann_dir.exists() else []
    passes.append(check("COCO JSON annotation files", len(json_files) > 0,
                        f"Found: {[f.name for f in json_files]}"))

    # Count unique image filenames across all JSON files (all_data.json overlaps train/val/test)
    unique_image_names = set()
    total_ann_json = 0
    categories = set()
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            for img in data.get("images", []):
                unique_image_names.add(img.get("file_name", img.get("id")))
            total_ann_json += len(data.get("annotations", []))
            for cat in data.get("categories", []):
                categories.add(cat["name"])
        except Exception as e:
            print(f"         JSON parse error in {jf.name}: {e}")

    total_img_json = len(unique_image_names)
    exp = cfg.get("expected_images")
    img_ok = abs(total_img_json - exp) / max(exp, 1) < 0.05 if exp else total_img_json > 0
    passes.append(check("Unique image entries in JSON annotations",
                        img_ok,
                        f"{total_img_json} unique images  (expected ~{exp})"))

    # 3. Physical images — count annotated filenames that exist on disk
    images_dir = root / "images"
    found_annotated = sum(
        1 for n in unique_image_names if (images_dir / n).exists()
    )
    missing_annotated = len(unique_image_names) - found_annotated
    img_count_ok = missing_annotated == 0 or missing_annotated / max(len(unique_image_names), 1) < 0.01
    passes.append(check("Annotated images present on disk",
                        img_count_ok,
                        f"{found_annotated}/{len(unique_image_names)} annotated images found in images/"))

    # 4. Labels
    labels_zip = root / "labels.zip"
    labels_dir = root / "labels"
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        passes.append(check("YOLO label files (extracted)",
                            len(label_files) > 0,
                            f"{len(label_files)} .txt files in labels/"))
    elif labels_zip.exists():
        import zipfile
        with zipfile.ZipFile(labels_zip) as zf:
            txt_count = sum(1 for n in zf.namelist() if n.endswith(".txt"))
        passes.append(check("YOLO label files (in labels.zip — NOT extracted)",
                            False,
                            f"labels.zip contains {txt_count} .txt files — must be extracted before training"))
    else:
        passes.append(check("YOLO label files", False, "labels.zip missing and labels/ dir missing"))

    # 5. Empty labels — only if extracted
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        empties = empty_files([str(p) for p in label_files])
        passes.append(check("Empty label check", True,
                            f"{len(empties)} empty labels"))

    # 6. Classes
    exp_cls = cfg.get("expected_classes")
    cls_ok = len(categories) == exp_cls if exp_cls else len(categories) > 0
    passes.append(check("Class names from JSON",
                        cls_ok,
                        f"{sorted(categories)}  (expected {exp_cls})"))

    # 7. Splits
    new_split = root / "new_split_yolo"
    split_txts = list(new_split.glob("*.txt")) if new_split.exists() else []
    has_splits = any("train" in f.name for f in split_txts) and any("val" in f.name for f in split_txts)
    passes.append(check("Train/val split files",
                        has_splits,
                        f"Found in new_split_yolo/: {[f.name for f in split_txts]}"))

    # 8. Size
    size_mb = folder_size_mb(root)
    passes.append(check("Dataset size", size_mb > 1, f"{size_mb:.1f} MB"))

    return all(passes)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

results = {}

for ds_name, cfg in DATASETS.items():
    header(ds_name)
    fmt = cfg["format"]
    try:
        if ds_name == "SFCHD":
            ok = check_sfchd(ds_name, cfg)
        elif fmt == "yolo":
            ok = check_yolo(ds_name, cfg)
        elif fmt == "voc":
            ok = check_voc(ds_name, cfg)
        else:
            ok = False
    except Exception as e:
        print(f"  [ERROR]  Unexpected error: {e}")
        ok = False
    results[ds_name] = ok

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  SUMMARY")
print(SEP)
for ds_name, ok in results.items():
    status = "PASS" if ok else "FAIL"
    marker = "✓" if ok else "✗"
    print(f"  [{status}]  {marker}  {ds_name}")
print()
all_ok = all(results.values())
if all_ok:
    print("  All datasets PASSED.")
else:
    failed = [n for n, ok in results.items() if not ok]
    print(f"  {len(failed)} dataset(s) need attention: {', '.join(failed)}")
print(SEP)
