"""
Face detection and recognition using OpenCV's LBPH recognizer.

Face detection: Haar cascade (haarcascade_frontalface_default.xml),
bundled with opencv-python — zero extra dependency.

Face recognition: cv2.face.LBPHFaceRecognizer (opencv-contrib-python).

LBPH distance convention — IMPORTANT:
  LBPH's predict() returns (label_id, distance) where a LOWER distance
  means a BETTER (more confident) match.  This is the OPPOSITE of the
  detector confidence scores used elsewhere in this project (where higher
  confidence = better match).  match_threshold in config is the MAXIMUM
  acceptable distance — predictions with distance > match_threshold are
  treated as "no confident match."  This inversion is documented here
  because it is an easy source of confusion and an easy threshold-direction
  bug to introduce.

Graceful degradation:
  If the model file does not exist yet (no workers enrolled), the
  recognizer operates in a "no known workers" state — identify() returns
  None for every frame without crashing.  This allows the system to start
  and run normally before any enrollment has taken place.

Dependency injection:
  The recognizer and cascade objects can be injected via constructor
  parameters (_recognizer, _cascade).  This is used in unit tests to
  supply stubs without touching real model files or cv2 internals,
  following the injectable-clock pattern established in SessionController.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum face size for Haar cascade detection (pixels).
# Smaller faces are likely noise or very distant — skip them.
_MIN_FACE_SIZE = (60, 60)


class FaceRecognizer:
    def __init__(self, config: dict, worker_store,
                 _recognizer=None, _cascade=None):
        """
        Parameters
        ----------
        config       : full project config dict; reads config["face"]
        worker_store : WorkerStore instance for label_id → name lookup
        _recognizer  : injectable LBPH recognizer (for testing)
        _cascade     : injectable CascadeClassifier (for testing)
        """
        face_cfg = config["face"]
        self._model_path = Path(face_cfg["model_path"])
        self._match_threshold = float(face_cfg["match_threshold"])
        self._worker_store = worker_store

        self._cascade = _cascade or cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )

        self._recognizer = _recognizer or cv2.face.LBPHFaceRecognizer_create()
        self._trained = False

        if self._model_path.exists():
            self._recognizer.read(str(self._model_path))
            self._trained = True
            logger.info("FaceRecognizer: model loaded from %s", self._model_path)
        else:
            logger.warning(
                "FaceRecognizer: no model file at %s — operating in "
                "'no known workers' state until first enrollment.",
                self._model_path,
            )

    def detect_face(self, frame: np.ndarray) -> np.ndarray | None:
        """Detect faces in frame; return the largest face ROI (grey) or None.

        Returns None when no face is found — a person facing away, low
        light, or an unenrolled worker are all valid "no face" cases and
        must not cause an error.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=_MIN_FACE_SIZE
        )
        if len(faces) == 0:
            return None
        # Use the largest detected face (by area) to reduce noise from
        # background false positives.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return gray[y : y + h, x : x + w]

    def identify(self, frame: np.ndarray) -> tuple[str, float] | None:
        """Identify the person in frame.

        Returns (worker_name, distance) if a confident match is found,
        or None if:
          - no face is detected in the frame, OR
          - the model has not been trained yet (no enrolled workers), OR
          - the best match distance exceeds match_threshold (no confident ID).

        Remember: LBPH distance is LOWER for better matches.  A return of
        None does not mean "no worker present" — it means "cannot
        confidently identify who is present."  Callers must NOT treat None
        as evidence of a worker swap; only a positive mismatch between two
        confident IDs is evidence of a swap.
        """
        if not self._trained:
            return None

        roi = self.detect_face(frame)
        if roi is None:
            return None

        label_id, distance = self._recognizer.predict(roi)

        # LBPH distance: lower = better.  Reject if too far from any training sample.
        if distance > self._match_threshold:
            return None

        name = self._worker_store.get_name(label_id)
        if name is None:
            # label_id returned by model but not in WorkerStore —
            # model and DB are out of sync (e.g. manual DB edit).  Treat as no match.
            logger.warning(
                "FaceRecognizer: label_id %d not found in WorkerStore — "
                "model and database may be out of sync.", label_id
            )
            return None

        return (name, distance)
