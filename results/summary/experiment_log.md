# FYP Experiment Log — YOLOv8 Construction Safety Detection

**Project:** Comparative evaluation of object detection models for PPE/safety compliance monitoring  
**Dataset sources:** Roboflow Construction Safety (EXP 01), SFCHD-SCALE (EXP 02, planned)  
**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Framework:** Ultralytics 8.4.51, PyTorch 2.11.0+cu128, Python 3.12.3  
**venv:** `/home/heng/edge-ai-safety-monitor-second/.venv`

---

## EXP 01 — YOLOv8s | Roboflow Construction Safety

**Date:** 2026-06-06  
**Status:** COMPLETE

### Dataset

| Split | Images | Source |
|-------|--------|--------|
| Train | 997 | Roboflow workspace: `chan-heng-yi-s-workspace` |
| Val   | 119 | project: `construction-safety-gsnvb-func8` |
| Test  | 90  | license: CC BY 4.0 |

**Classes (5):** `helmet`, `no-helmet`, `no-vest`, `person`, `vest`  
**Dataset path:** `~/FYP/Dataset/Roboflow Dataset/construction safety.yolov8/`  
**Data config:** `~/FYP/results/configs/exp01_roboflow_abs.yaml` (absolute paths)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8s (`yolov8s.pt`) — 11.1M params, 28.7 GFLOPs |
| Pre-trained weights | ImageNet (Ultralytics official) |
| Epochs | 100 (early stopping patience=20) |
| Batch size | 8 |
| Image size | 640×640 |
| Optimizer | SGD (lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005) |
| Warmup | 3 epochs (bias lr=0.1, momentum=0.8) |
| AMP | Enabled (mixed precision) |
| Seed | 42 |
| Workers | 4 |
| Cache | Disabled |
| Augmentation | Mosaic, HSV, RandomFlip-LR, RandAugment, Erasing (p=0.4) |

### Training Outcome

- **Epochs run:** 38 of 100 (early stopping triggered)
- **Best checkpoint epoch:** 18 (by fitness = 0.1×mAP50 + 0.9×mAP50-95)
- **Training time:** 11.1 minutes
- **Stopping reason:** No improvement in fitness over 20 consecutive epochs (epochs 19–38)

### Epoch-by-Epoch Progress (val split)

| Epoch | Train Loss (box/cls/dfl) | Val mAP@0.5 | Val mAP@0.5:95 | Precision | Recall |
|-------|--------------------------|-------------|----------------|-----------|--------|
| 1  | 1.428 / 1.627 / 1.444 | 67.0% | 37.8% | 57.6% | 62.5% |
| 2  | 1.298 / 0.957 / 1.266 | 75.3% | 42.6% | 68.3% | 72.0% |
| 3  | 1.291 / 0.951 / 1.253 | 77.3% | 42.7% | 70.0% | 81.0% |
| 4  | 1.305 / 0.914 / 1.261 | 83.8% | 44.4% | 74.2% | 82.4% |
| 5  | 1.298 / 0.885 / 1.262 | 80.7% | 43.2% | 73.6% | 77.5% |
| 8  | 1.299 / 0.867 / 1.252 | 82.9% | 46.4% | 73.6% | 78.3% |
| 10 | 1.273 / 0.810 / 1.249 | 80.7% | 44.8% | 69.8% | 81.2% |
| 13 | 1.255 / 0.783 / 1.245 | 82.6% | 45.2% | 81.8% | 78.5% |
| 17 | 1.219 / 0.741 / 1.235 | 85.8% | 48.0% | 81.0% | 80.3% |
| **18** | **1.210 / 0.756 / 1.224** | **84.2%** | **48.2%** | **76.8%** | **81.8%** |
| 19 | 1.202 / 0.725 / 1.220 | 84.5% | 46.9% | 80.4% | 77.9% |
| 24 | 1.157 / 0.681 / 1.189 | 84.0% | 45.8% | 88.7% | 75.5% |
| 29 | 1.138 / 0.655 / 1.181 | 85.8% | 45.9% | 82.6% | 80.3% |
| 35 | 1.084 / 0.621 / 1.154 | **87.4%** | 48.1% | 82.3% | 83.1% |
| 38 | 1.047 / 0.586 / 1.137 | 83.2% | 46.3% | 89.7% | 78.2% |

> Note: Early stopping saved `best.pt` at epoch 18 (highest fitness). Epoch 35 had the highest raw val mAP50 (87.4%) but marginally lower mAP50-95 (48.1% vs 48.2%), so fitness was fractionally lower.

### Test Set Metrics (best.pt evaluated on 90-image test split)

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **80.01%** |
| mAP@0.5:0.95 | 40.90% |
| **Precision** | **70.41%** |
| **Recall** | **81.52%** |
| F1 (P×R harmonic mean) | 0.756 |

### Per-Class AP@0.5 (test split)

| Class | AP@0.5 | Notes |
|-------|--------|-------|
| helmet | 90.7% | PPE present — well represented |
| person | 88.8% | Dominant class |
| vest | 85.1% | PPE present — well represented |
| no-vest | 68.6% | Violation class — harder |
| no-helmet | 66.8% | Violation class — harder |
| **Mean** | **80.0%** | |

> Pattern: Violation classes (no-helmet, no-vest) score ~18–22pp lower than their PPE-present counterparts. Expected — violation instances are fewer in the training set.

### Inference Performance (GPU, single image, 640×640)

| Metric | Value |
|--------|-------|
| Avg latency | 18.67 ms |
| FPS | **53.6** |
| Device | NVIDIA RTX 4050 Laptop GPU |

> 5-image warmup, then benchmarked on up to 45 test images. Single-image latency reflects realistic deployment conditions (not batched).

### Saved Artifacts

| Artifact | Path |
|----------|------|
| Best weights | `~/FYP/results/exp01_yolov8_roboflow/weights/best.pt` |
| Last weights | `~/FYP/results/exp01_yolov8_roboflow/weights/last.pt` |
| Metrics report | `~/FYP/results/exp01_yolov8_roboflow/metrics.txt` |
| Training CSV | `~/FYP/results/exp01_yolov8_roboflow/results.csv` |
| Training log | `~/FYP/results/exp01_yolov8_roboflow/train_log.txt` |
| Plots | `results.png`, `confusion_matrix.png`, `BoxPR_curve.png`, `labels.jpg`, … |
| Args | `~/FYP/results/exp01_yolov8_roboflow/args.yaml` |

### Observations & Notes

1. **Rapid early convergence:** mAP50 jumped from 67% → 84% in just 4 epochs, suggesting the ImageNet pre-trained backbone transfers very effectively to this domain.
2. **Early stopping at epoch 38:** Only 38 of 100 epochs were needed. The fitness metric plateaued around epoch 18–19; later epochs showed minor mAP50 oscillation but no sustained improvement in mAP50-95.
3. **Violation class gap:** The model is significantly weaker on `no-helmet` and `no-vest` (66–69%) compared to presence classes (85–91%). This is the primary area for improvement in future experiments (data augmentation, oversampling, or a larger dataset like SFCHD-SCALE).
4. **Inference speed:** 53.6 FPS at single-image latency is suitable for real-time deployment at 640px resolution.
5. **Val→Test gap:** Val mAP50 peaked at 87.4% but test mAP50 is 80.0% — a 7.4pp gap indicating some overfitting to the val distribution or test set difficulty.

---

## EXP 02 — Planned (next session)

**Dataset:** SFCHD-SCALE  
**Dataset path:** `~/FYP/Dataset/SFCHD-SCALE/`  
**Download script:** `~/FYP/Dataset/SFCHD-SCALE/download_sfchd_images.py`  
**Model:** TBD (YOLOv8s same config for fair comparison, or upgrade to YOLOv8m)  
**Goal:** Compare performance on a larger/different construction safety dataset vs EXP 01 Roboflow results  
**Results target:** `~/FYP/results/exp02_*/`

---

## Comparison Table (to be filled)

| Experiment | Model | Dataset | Epochs | mAP@0.5 | Precision | Recall | FPS |
|------------|-------|---------|--------|---------|-----------|--------|-----|
| EXP 01 | YOLOv8s | Roboflow (997 train) | 38/100 | 80.0% | 70.4% | 81.5% | 53.6 |
| EXP 02 | — | SFCHD-SCALE | — | — | — | — | — |
