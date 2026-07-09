"""
Smoke-test for Milestone 2: Detector.

Loads the model, runs detect() on output/camera_test.jpg, prints raw
detections, and saves an annotated copy to output/detector_test.jpg.
No cv2.imshow anywhere.

Run from the fyp2/ directory:
    python verify_detector.py
"""

import logging
from pathlib import Path

import cv2

from core.config_loader import load_config
from core.detector import Detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEST_IMAGE = Path("output/camera_test.jpg")
OUT_IMAGE = Path("output/detector_test.jpg")


def main():
    config = load_config()
    detector = Detector(config)

    frame = cv2.imread(str(TEST_IMAGE))
    if frame is None:
        logger.error("Could not read test image at %s", TEST_IMAGE)
        return

    logger.info("Running detect() on %s (shape=%s)", TEST_IMAGE, frame.shape)
    detections = detector.detect(frame)

    print(f"\n--- Raw detections ({len(detections)}) ---")
    for d in detections:
        print(d)
    print(f"--- End ---\n")

    # Draw boxes on a copy for visual inspection — drawing code is
    # intentionally in this script only, not in detector.py.
    annotated = frame.copy()
    for d in detections:
        cv2.rectangle(annotated, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(
            annotated, label, (d.x1, max(d.y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )

    OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_IMAGE), annotated)
    logger.info("Annotated image saved to %s", OUT_IMAGE)


if __name__ == "__main__":
    main()
