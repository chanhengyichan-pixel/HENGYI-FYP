import logging

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Detection:
    """
    Plain data container for a single detection result.

    Attributes
    ----------
    class_id   : int   — model class index
    class_name : str   — human-readable label
    confidence : float — detection confidence in [0, 1]
    x1, y1    : int   — top-left corner in pixel coords
    x2, y2    : int   — bottom-right corner in pixel coords
    track_id   : int | None — stable per-object ID from Detector.track();
                 None from Detector.detect() or when the tracker has not
                 confirmed the object yet
    """

    __slots__ = ("class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
                 "track_id")

    def __init__(self, class_id, class_name, confidence, x1, y1, x2, y2,
                 track_id=None):
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.confidence = float(confidence)
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.track_id = int(track_id) if track_id is not None else None

    def to_dict(self):
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2},
            "track_id": self.track_id,
        }

    def __repr__(self):
        return (
            f"Detection(class_id={self.class_id}, class_name='{self.class_name}', "
            f"confidence={self.confidence:.4f}, "
            f"bbox=({self.x1},{self.y1},{self.x2},{self.y2}), track_id={self.track_id})"
        )


class Detector:
    def __init__(self, config: dict):
        det = config["detector"]
        self._weights = det["weights"]
        self._conf_threshold = float(det["confidence_threshold"])
        self._imgsz = int(det.get("imgsz", 640))
        self._tracker = det.get("tracker", "bytetrack.yaml")

        logger.info("Loading model from %s", self._weights)
        self._model = YOLO(self._weights)
        logger.info(
            "Model loaded — %d classes: %s",
            len(self._model.names),
            self._model.names,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame (as returned by OpenCV).

        Returns a list of Detection objects whose confidence is at or above
        the configured threshold. Returns an empty list if no detections
        meet the threshold. Never raises on an empty frame — caller is
        responsible for not passing None.
        """
        results = self._model(frame, conf=self._conf_threshold,
                              imgsz=self._imgsz, verbose=False)
        return self._to_detections(results)

    def track(self, frame: np.ndarray) -> list[Detection]:
        """
        Like detect(), but runs the multi-object tracker (ByteTrack) so each
        object keeps the same track_id across consecutive frames.

        persist=True keeps tracker state inside this Detector's model
        between calls — so each camera needs its OWN Detector instance,
        otherwise IDs from two different cameras would be mixed together.
        """
        results = self._model.track(frame, conf=self._conf_threshold,
                                    imgsz=self._imgsz, tracker=self._tracker,
                                    persist=True, verbose=False)
        return self._to_detections(results)

    def _to_detections(self, results) -> list[Detection]:
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = None
                if getattr(box, "id", None) is not None:
                    track_id = box.id[0].item()
                detections.append(
                    Detection(
                        class_id=box.cls[0].item(),
                        class_name=self._model.names[int(box.cls[0].item())],
                        confidence=box.conf[0].item(),
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        track_id=track_id,
                    )
                )

        return detections
