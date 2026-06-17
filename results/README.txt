FYP EXPERIMENT RESULTS — OVERVIEW
===================================
Project: Edge AI-Based PPE Compliance Monitoring System
         for Industrial Workplace Safety

EXPERIMENT NUMBERING
--------------------
EXP01-05  : YOLOv8        — 5 datasets (Roboflow, CHVG, SHEL5K, SH17, SFCHD)
EXP06-10  : YOLOv9        — 5 datasets (Roboflow, CHVG, SHEL5K, SH17, SFCHD)
EXP11-15  : YOLOv11       — 5 datasets (Roboflow, CHVG, SHEL5K, SH17, SFCHD)
EXP16-20  : Faster R-CNN  — EXCLUDED (hardware limitation, see FasterRCNN/README.txt)
EXP21-25  : SSD300        — 5 datasets (Roboflow, CHVG, SHEL5K, SH17, SFCHD)

COMPLETION STATUS
-----------------
Total planned   : 25 experiments
Total completed : 20 experiments
Excluded        : 5 experiments (Faster R-CNN, EXP16-20)

FOLDER STRUCTURE
----------------
results/
├── YOLOv8/        EXP01-05  — all 5 completed
├── YOLOv9/        EXP06-10  — all 5 completed
├── YOLOv11/       EXP11-15  — all 5 completed
├── FasterRCNN/    EXP16-20  — excluded (see README inside)
├── SSD/           EXP21-25  — all 5 completed
├── configs/       Training configuration files
├── logs/          Training logs (19 files)
├── summary/       Experiment summary files (6 files)
└── weights/       Pre-trained model weights
                   (yolov8s.pt, yolov9s.pt, yolo11s.pt)

EVALUATION METRICS REPORTED
----------------------------
mAP@0.5 | Precision | Recall | F1-Score | FPS | Latency (ms)

HARDWARE USED
-------------
GPU : NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
CPU : AMD Ryzen 7 7445HS
RAM : 16GB
OS  : Linux Mint 22.2
