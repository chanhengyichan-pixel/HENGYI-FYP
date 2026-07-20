"""
Offline worker enrollment script.

Captures face samples from the webcam, trains/updates the LBPH face
recognizer, saves the model, and registers the worker in WorkerStore.

Usage (standalone):
    cd /path/to/fyp2
    source venv/bin/activate
    python scripts/enroll_worker.py "Worker Name"

This script is STANDALONE — it does NOT use CameraCapture, SharedState,
or the main pipeline.  It opens the webcam directly for a short enrollment
session and releases it when done, following the same pattern as
scripts/inspect_classes.py.

Do NOT run this while main.py is running (both would compete for the
same webcam device).  The main.py menu (Choice 2) calls enroll_worker()
from this module directly and guards against this by only presenting
enrollment when the dashboard is not running.

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

# Allow imports from the project root whether run standalone or imported.
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.config_loader import load_config
from storage.worker_store import WorkerStore

_NUM_SAMPLES = 20
_SAMPLE_SIZE = (100, 100)
_CASCADE_PATH = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")


def enroll_worker(worker_name: str, config: dict) -> bool:
    """Capture face samples, train/update the LBPH model, and register the worker.

    Parameters
    ----------
    worker_name : str
        Display name for the worker.  Must be non-empty (caller's responsibility).
    config : dict
        Loaded application config (same object passed to all other components).

    Returns
    -------
    bool
        True if enrollment completed successfully, False if it failed — e.g.
        webcam could not be opened, or fewer than _NUM_SAMPLES faces were
        captured before the webcam loop exited.

    Side effects
    ------------
    - Writes/updates models/face_recognizer.yml (path from config["face"]["model_path"]).
    - Inserts a new row in the workers database (config["workers"]["db_path"]).
    Both writes are permanent; a False return means neither was completed
    (WorkerStore entry IS written before capture so that label_id is reserved,
    but the model is only written on success).
    """
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
        return False

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
        return False

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
        marker = " <- new" if w["label_id"] == label_id else ""
        print(f"  [{w['label_id']}] {w['name']}  (enrolled {w['enrolled_at']}){marker}")

    worker_store.close()
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts/enroll_worker.py "Worker Name"')
        sys.exit(1)

    name = sys.argv[1].strip()
    if not name:
        print("Error: worker name cannot be empty.")
        sys.exit(1)

    cfg = load_config()
    success = enroll_worker(name, cfg)
    sys.exit(0 if success else 1)
