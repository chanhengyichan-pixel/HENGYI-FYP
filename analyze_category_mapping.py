#!/usr/bin/env python3
"""Analyze class-to-category-ID mapping for both Roboflow and CHVG."""

import sys
sys.path.insert(0, '/home/heng/FYP')

import torch
import json
import tempfile
import contextlib
import io
from pathlib import Path
from train_ssd_all import (
    ConstructionSafetyDataset, DATASET_BUILDERS, CLASSES, DEVICE, IMGSZ,
    create_coco_json, collate_fn
)
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HOME = Path.home()

def analyze_mapping(dataset_key: str, model_path: Path):
    """Analyze category mapping for a dataset."""

    print(f"\n{'='*80}")
    print(f"  CATEGORY MAPPING ANALYSIS: {dataset_key.upper()}")
    print(f"{'='*80}\n")

    classes = CLASSES[dataset_key]
    print(f"  Dataset classes (from CLASSES['{dataset_key}']):")
    print(f"  Classes list = {classes}")
    print(f"  Order = {list(enumerate(classes, 1))}\n")

    # Build dataset
    builder = DATASET_BUILDERS[dataset_key]
    test_imgs, test_lbls = builder("test", classes)
    test_ds = ConstructionSafetyDataset(test_imgs, test_lbls, classes, split="test")

    print(f"  ConstructionSafetyDataset.class_map (from __init__):")
    print(f"  {test_ds.class_map}\n")

    # Load model and run inference
    model = ssd300_vgg16(weights='DEFAULT')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    predictions = []
    targets_list = []
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            for output in outputs:
                keep = output["scores"] > 0.01
                predictions.append((
                    output["boxes"][keep].cpu().numpy(),
                    output["scores"][keep].cpu().numpy(),
                    output["labels"][keep].cpu().numpy(),
                ))
            for target in targets:
                targets_list.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                })

    # Create COCO JSON
    coco_gt, coco_dt = create_coco_json(test_ds, targets_list, predictions)

    print(f"  COCO JSON categories (from create_coco_json):")
    print(f"  {coco_gt['categories']}\n")

    # Write to temp files and load with COCO
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_gt:
        json.dump(coco_gt, f_gt)
        gt_path = f_gt.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_dt:
        json.dump(coco_dt, f_dt)
        dt_path = f_dt.name

    # Load with pycocotools
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt_obj = COCO(gt_path)

    print(f"  COCOeval.getCatIds() — Category IDs in COCO object:")
    cat_ids = coco_gt_obj.getCatIds()
    print(f"  {cat_ids}\n")

    print(f"  Category ID → Name mapping (from COCO.loadCats):")
    for cat_id in cat_ids:
        cat_name = coco_gt_obj.loadCats(cat_id)[0]["name"]
        print(f"    {cat_id} → {cat_name}")
    print()

    print(f"  Per-class AP loop iteration order (what will be evaluated):")
    for i, cat_id in enumerate(cat_ids, 1):
        cat_name = coco_gt_obj.loadCats(cat_id)[0]["name"]
        print(f"    {i}. cat_id={cat_id}, cat_name={cat_name}")
    print()

    # Check for mismatches
    print(f"  {'─'*80}")
    print(f"  CONSISTENCY CHECK")
    print(f"  {'─'*80}\n")

    expected_mapping = {i+1: classes[i] for i in range(len(classes))}
    actual_mapping = {cat_id: coco_gt_obj.loadCats(cat_id)[0]["name"] for cat_id in cat_ids}

    print(f"  Expected (from dataset.classes): {expected_mapping}")
    print(f"  Actual (from COCO object):       {actual_mapping}\n")

    if expected_mapping == actual_mapping:
        print(f"  ✓ MATCH: Category mapping is consistent!")
    else:
        print(f"  ✗ MISMATCH: Category mapping differs!")
        for cat_id in expected_mapping:
            if cat_id not in actual_mapping:
                print(f"    Missing cat_id {cat_id}")
            elif expected_mapping[cat_id] != actual_mapping[cat_id]:
                print(f"    cat_id {cat_id}: expected '{expected_mapping[cat_id]}', got '{actual_mapping[cat_id]}'")

    # Cleanup
    Path(gt_path).unlink()
    Path(dt_path).unlink()

# Analyze both datasets
print("\n" + "="*80)
print("  CATEGORY MAPPING ANALYSIS FOR ROBOFLOW AND CHVG")
print("="*80)

roboflow_model = HOME / "FYP/results/SSD/exp21_ssd_roboflow/weights/best.pth"
chvg_model = HOME / "FYP/results/SSD/exp22_ssd_chvg/weights/best.pth"

analyze_mapping("roboflow", roboflow_model)
analyze_mapping("chvg", chvg_model)

print(f"\n{'='*80}")
print(f"  CONCLUSION")
print(f"{'='*80}\n")
print(f"  If mappings are identical for both datasets, the issue is NOT in class ordering.")
print(f"  The missing summarize() was the root cause.\n")
