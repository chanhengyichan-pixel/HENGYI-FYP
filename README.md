# Edge AI-Based PPE Compliance Monitoring System
### Final Year Project — Multimedia University (MMU)
**Faculty of Information Science and Technology (FIST)**

---

## Project Overview

This project presents a systematic comparative evaluation of object 
detection algorithms for real-time Personal Protective Equipment (PPE) 
compliance monitoring in industrial workplace safety applications.

Five object detection algorithms were evaluated across five public PPE 
datasets, combining detection accuracy metrics with computational 
efficiency metrics to identify the most suitable algorithm for 
Edge AI deployment.

---

## Final Result

| Rank | Algorithm | Avg mAP@0.5 | Avg Recall | Avg FPS | Score |
|------|-----------|-------------|------------|---------|-------|
| 🥇 1st | YOLOv8 | 78.63% | 74.93% | 78.7 | 0.9024 |
| 🥈 2nd | YOLOv11 | 77.42% | 73.43% | 81.2 | 0.8947 |
| 🥉 3rd | YOLOv9 | 78.01% | 73.85% | 62.7 | 0.7967 |
| 4th | SSD MobileNet V2 | 35.13% | 42.91% | 50.5 | 0.1000 |

**✅ Recommended Model: YOLOv8**

---

## Algorithms Evaluated

| Algorithm | Architecture | Role |
|-----------|-------------|------|
| YOLOv8 | One-stage (YOLO generation) | Generational baseline |
| YOLOv9 | One-stage (YOLO generation) | Intermediate generation |
| YOLOv11 | One-stage (YOLO generation) | Most recent generation |
| SSD MobileNet V2 | Lightweight one-stage | Edge-oriented comparison |
| Faster R-CNN | Two-stage | ⚠️ Excluded — hardware limitation (see below) |

---

## Datasets Used

| Dataset | Domain | Images | Classes |
|---------|--------|--------|---------|
| SH17 | Manufacturing | 8,099 | 17 |
| SFCHD | Chemical/Hazardous | 12,373 | 7 |
| CHVG | Construction | 1,699 | 8 |
| SHEL5K | Construction (helmet-focused) | ~5,000 | 6 |
| Roboflow Construction Safety | Construction | 1,206 | 5 |

> ⚠️ Datasets are NOT included in this repository due to size (~5GB).
> See [DATASETS.md](DATASETS.md) for download links and setup instructions.

---

## Experimental Design

- **Design:** 5 × 5 factorial (5 algorithms × 5 datasets)
- **Planned experiments:** 25
- **Completed experiments:** 20
- **Excluded:** Faster R-CNN (EXP16-20) — GPU memory limitation
  (RTX 4050, 6GB VRAM insufficient for two-stage architecture)

---

## Evaluation Metrics

| Category | Metrics |
|----------|---------|
| Detection Accuracy | mAP@0.5, Precision, Recall, F1-Score |
| Computational Efficiency | FPS, Latency (ms) |

**Priority metric: Recall** (safety-critical — missed detection = undetected violation)

---

## Complete Results

| Algorithm | Dataset | mAP@0.5 | Precision | Recall | F1 | FPS | Latency |
|-----------|---------|---------|-----------|--------|----|-----|---------|
| YOLOv8 | Roboflow | 80.01% | 70.41% | 81.52% | 0.7556 | 53.6 | 18.67ms |
| YOLOv8 | CHVG | 92.26% | 88.61% | 88.95% | 0.8878 | 126.0 | 7.93ms |
| YOLOv8 | SHEL5K | 77.04% | 91.31% | 71.93% | 0.8047 | 93.0 | 10.76ms |
| YOLOv8 | SH17 | 64.71% | 79.09% | 57.22% | 0.6640 | 7.7 | 130.70ms |
| YOLOv8 | SFCHD | 79.14% | 82.27% | 75.05% | 0.7850 | 113.4 | 8.82ms |
| YOLOv9 | Roboflow | 78.43% | 84.67% | 72.99% | 0.7840 | 61.3 | 16.32ms |
| YOLOv9 | CHVG | 93.32% | 88.91% | 90.31% | 0.8961 | 88.1 | 11.36ms |
| YOLOv9 | SHEL5K | 77.18% | 90.76% | 72.81% | 0.8080 | 75.8 | 13.20ms |
| YOLOv9 | SH17 | 62.59% | 71.66% | 58.81% | 0.6460 | 6.5 | 152.94ms |
| YOLOv9 | SFCHD | 78.52% | 83.00% | 74.34% | 0.7843 | 81.6 | 12.25ms |
| YOLOv11 | Roboflow | 75.59% | 72.04% | 74.95% | 0.7347 | 68.3 | 14.65ms |
| YOLOv11 | CHVG | 92.14% | 88.26% | 88.10% | 0.8818 | 129.4 | 7.73ms |
| YOLOv11 | SHEL5K | 77.24% | 91.27% | 72.15% | 0.8059 | 92.1 | 10.86ms |
| YOLOv11 | SH17 | 63.21% | 80.18% | 56.29% | 0.6615 | 7.3 | 137.34ms |
| YOLOv11 | SFCHD | 78.91% | 83.28% | 75.67% | 0.7930 | 108.7 | 9.20ms |
| SSD MobileNet V2 | Roboflow | 50.66% | 65.18% | 58.59% | 0.6171 | 51.2 | 19.53ms |
| SSD MobileNet V2 | CHVG | 54.91% | 69.94% | 55.04% | 0.6160 | 53.5 | 18.68ms |
| SSD MobileNet V2 | SHEL5K | 33.99% | 66.70% | 33.77% | 0.4484 | 52.8 | 18.93ms |
| SSD MobileNet V2 | SH17 | 10.88% | 62.90% | 26.96% | 0.3774 | 43.3 | 23.07ms |
| SSD MobileNet V2 | SFCHD | 25.20% | 83.75% | 40.18% | 0.5431 | 51.6 | 19.38ms |

---

## Repository Structure

```
HENGYI-FYP/

├── train_yolov8.py              # YOLOv8 training script (all 5 datasets)
├── train_yolov9.py              # YOLOv9 training script (all 5 datasets)
├── train_yolov11.py             # YOLOv11 training script (all 5 datasets)
├── train_ssd.py                 # SSD MobileNet V2 training script
├── train_fasterrcnn.py          # Faster R-CNN (excluded — see hardware note)
├── check_datasets.py            # Dataset integrity verification utility
├── analyze_category_mapping.py  # Class mapping analysis utility
├── DATASETS.md                  # Dataset download links and setup guide
├── THESIS_EXPERIMENTAL_RESULTS_CHECKLIST.md
└── results/
    ├── YOLOv8/                  # EXP01-05 results
    ├── YOLOv9/                  # EXP06-10 results
    ├── YOLOv11/                 # EXP11-15 results
    ├── FasterRCNN/              # EXP16-20 (excluded, hardware limitation)
    ├── SSD/                     # EXP21-25 results
    ├── configs/                 # 20 YAML training configuration files
    ├── summary/                 # Per-algorithm experiment summaries
    ├── logs/                    # 20 training log files
    └── weights/                 # Pre-trained model weights
```

---

## Hardware & Software

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) |
| CPU | AMD Ryzen 7 7445HS |
| RAM | 16GB |
| OS | Linux Mint 22.2 |
| Python | 3.12.3 |
| PyTorch | 2.11.0 (CUDA 12.8) |
| Ultralytics | 8.4.51 |
| CUDA | 12.8 |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Momentum | 0.937 |
| Weight Decay | 0.0005 |
| Batch Size | 8 (YOLO family) |
| Image Size | 640×640 (YOLO family) |
| Max Epochs | ~100 with early stopping |
| Mixed Precision | Enabled (AMP) |
| Random Seed | 42 |

---

## Key Findings

1. **YOLOv8 ranked 1st** under a recall-prioritised multi-criteria 
   evaluation framework (weighted score: 0.9024)
2. **YOLO-family algorithms** achieved similar accuracy (77-79% avg mAP) 
   while SSD MobileNet V2 was substantially lower (35% avg mAP)
3. **CHVG was the easiest dataset** — highest mAP for all algorithms
4. **SH17 was the hardest dataset** — lowest mAP AND lowest FPS for 
   all YOLO-family algorithms (17-class complexity)
5. **SSD MobileNet V2** maintained stable FPS across all datasets 
   including SH17, but at the cost of significantly lower accuracy

---

## Academic Context

- **Institution:** Multimedia University (MMU), Malaysia
- **Faculty:** Faculty of Information Science and Technology (FIST)
- **Project Type:** Final Year Project (FYP)
- **Year:** 2026

---
