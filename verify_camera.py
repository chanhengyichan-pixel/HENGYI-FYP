"""
Smoke-test for Step 3: CameraCapture + SharedState.

Starts the camera, waits for a frame, saves it to output/camera_test.jpg,
then stops cleanly. No display window is opened.

Run from the fyp2/ directory:
    python verify_camera.py
"""

import logging
import time
from pathlib import Path

import cv2

from core.config_loader import load_config
from core.shared_state import SharedState
from core.camera import CameraCapture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    state = SharedState()
    camera = CameraCapture(config, state)

    try:
        camera.start()
        logger.info("Camera status: %s", state.get_camera_status())

        # Wait up to 3 seconds for the first frame.
        deadline = time.time() + 3.0
        frame = None
        while time.time() < deadline:
            frame = state.get_frame()
            if frame is not None:
                break
            time.sleep(0.05)

        if frame is None:
            logger.error("No frame received within 3 seconds.")
            return

        out_path = Path("output/camera_test.jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
        logger.info("Frame saved to %s (shape=%s)", out_path, frame.shape)

    finally:
        camera.stop()
        logger.info("Camera status after stop: %s", state.get_camera_status())


if __name__ == "__main__":
    main()
