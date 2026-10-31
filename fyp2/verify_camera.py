"""
Smoke-test: CameraCapture + SharedState.

Starts one configured camera, waits for a frame, saves it to
output/camera_test.jpg (or output/camera_test_room.jpg), then stops
cleanly. No display window is opened — works over SSH on the Pi.

Run from the fyp2/ directory:
    python verify_camera.py            # entrance camera
    python verify_camera.py room       # room camera
"""

import logging
import sys
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
    name = sys.argv[1] if len(sys.argv) > 1 else "entrance"
    config = load_config()
    state = SharedState()
    camera = CameraCapture(config["cameras"][name], state, name=name)

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

        suffix = "" if name == "entrance" else f"_{name}"
        out_path = Path(f"output/camera_test{suffix}.jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
        logger.info("Frame saved to %s (shape=%s)", out_path, frame.shape)

    finally:
        camera.stop()
        logger.info("Camera status after stop: %s", state.get_camera_status())


if __name__ == "__main__":
    main()
