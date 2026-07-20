import logging
import threading

import cv2

logger = logging.getLogger(__name__)


class CameraCapture:
    def __init__(self, config: dict, shared_state):
        cam = config["camera"]
        self._source = cam["source"]
        self._width = cam["width"]
        self._height = cam["height"]
        self._max_failed_reads = cam["max_failed_reads"]

        self._shared_state = shared_state
        self._cap = None
        self._thread = None
        self._running = False

    def start(self):
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source '{self._source}'. "
                "Check that the device is connected and not in use."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        self._running = True
        self._shared_state.set_camera_status("running")

        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()
        logger.info("Camera started (source=%s, %dx%d)", self._source, self._width, self._height)

    def _capture_loop(self):
        failed_reads = 0

        while self._running:
            ret, frame = self._cap.read()

            if ret:
                failed_reads = 0
                self._shared_state.set_frame(frame)
            else:
                failed_reads += 1
                logger.warning("Failed to read frame (%d/%d)", failed_reads, self._max_failed_reads)

                if failed_reads >= self._max_failed_reads:
                    logger.error(
                        "Camera read failed %d times in a row — stopping capture.",
                        self._max_failed_reads,
                    )
                    self._running = False
                    self._shared_state.set_camera_status("disconnected")
                    self._cap.release()
                    return

        self._cap.release()
        logger.info("Camera capture loop exited cleanly.")

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        # Release only if _capture_loop didn't already release on failure.
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        if self._shared_state.get_camera_status() != "disconnected":
            self._shared_state.set_camera_status("stopped")
        logger.info("Camera stopped.")
