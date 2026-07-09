"""
Read-only inspection of trained YOLOv8 model class metadata.
Does not run inference or modify any files.

Run from the fyp2/ directory:
    python scripts/inspect_classes.py
"""

from ultralytics import YOLO

MODELS = [
    "/home/heng/FYP/results/YOLOv8/exp01_roboflow/weights/best.pt",
    "/home/heng/FYP/results/YOLOv8/exp02_chvg/weights/best.pt",
]

for path in MODELS:
    print("=" * 60)
    print(f"Weights : {path}")
    model = YOLO(path)
    print(f"Classes : {model.names}")
    print(f"Count   : {len(model.names)}")
    print()
