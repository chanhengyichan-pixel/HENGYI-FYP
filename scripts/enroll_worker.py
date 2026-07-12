"""
Offline worker enrollment script.

Captures face samples from the webcam, trains/updates the LBPH face
recognizer, saves the model, and registers the worker in WorkerStore.

Usage:
    cd /path/to/fyp2
    source venv/bin/activate
    python scripts/enroll_worker.py "Worker Name"

This script is STANDALONE — it does NOT use CameraCapture, SharedState,
or the main pipeline.  It opens the webcam directly for a short enrollment
session and releases it when done, following the same pattern as
scripts/inspect_classes.py.

Do NOT run this while main.py is running (both would compete for the
same webcam device).

Manual verification after enrolling multiple workers
-----------------------------------------------------
After enrolling Worker A then Worker B, run main.py and stand in front
of the camera as each worker in turn.  The dashboard's worker name field
should show the correct name for each person — if it shows Worker B's
name while Worker A is present (or vice versa), the model is corrupted
and re-enrollment is needed.  This is the exact scenario the train()/
update() bug would have broken silently.
"""

import sys
from pathlib import Path

# Allow imports from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.config_loader import load_config
from storage.worker_store import WorkerStore

_NUM_SAMPLES = 20       # face samples to capture per worker
_SAMPLE_SIZE = (100, 100)  # resize all ROIs to this before training (consistent input)
_CASCADE_PATH = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/enroll_worker.py \"Worker Name\"")
        sys.exit(1)

    worker_name = sys.argv[1].strip()
    if not worker_name:
        print("Error: worker name cannot be empty.")
        sys.exit(1)

    config = load_config()
    model_path = Path(config["face"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    worker_store = WorkerStore(config)
    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # If a model already exists, load it so we append to it rather than
    # overwriting other enrolled workers.
    if model_path.exists():
        recognizer.read(str(model_path))
        print(f"Existing model loaded from {model_path} — will update with new worker.")
    else:
        print(f"No existing model found — creating new model at {model_path}.")

    label_id = worker_store.add_worker(worker_name)
    print(f"\nRegistered '{worker_name}' with label_id={label_id}.")
    print(f"Opening webcam — look directly at the camera.\n")

    cap = cv2.VideoCapture(config["camera"]["source"])
    if not cap.isOpened():
        print("Error: could not open webcam.")
        worker_store.close()
        sys.exit(1)

    samples: list[np.ndarray] = []
    labels: list[int] = []

    try:
        while len(samples) < _NUM_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            if len(faces) == 0:
                print(f"  No face detected — adjust position... ({len(samples)}/{_NUM_SAMPLES})")
                continue

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            roi = gray[y : y + h, x : x + w]
            roi_resized = cv2.resize(roi, _SAMPLE_SIZE)
            samples.append(roi_resized)
            labels.append(label_id)
            print(f"  Captured {len(samples)}/{_NUM_SAMPLES}")

    finally:
        cap.release()

    if len(samples) < _NUM_SAMPLES:
        print(f"\nOnly captured {len(samples)} samples — enrollment incomplete.")
        worker_store.close()
        sys.exit(1)

    print(f"\nTraining model on {len(samples)} samples...")
    # train() replaces the model's entire training set — use it only for the
    # first-ever enrollment.  For subsequent workers, read() above already put
    # the recognizer into a trained state; update() appends without discarding
    # existing data.  Calling train() here when a model was already loaded would
    # silently erase every previously enrolled worker.
    if model_path.exists():
        recognizer.update(samples, np.array(labels))
    else:
        recognizer.train(samples, np.array(labels))
    recognizer.write(str(model_path))

    print(f"Model saved to {model_path}.")
    print(f"\nEnrollment complete: '{worker_name}' (label_id={label_id}) is now registered.")

    existing = worker_store.get_all()
    print(f"\nAll registered workers ({len(existing)}):")
    for w in existing:
        marker = " ← new" if w["label_id"] == label_id else ""
        print(f"  [{w['label_id']}] {w['name']}  (enrolled {w['enrolled_at']}){marker}")

    worker_store.close()


if __name__ == "__main__":
    main()
