"""
CameraCapture — background capture thread for one logical camera.

Backends
--------
  opencv    — cv2.VideoCapture. source is a webcam index (int) or a video
              file path (str). Used on the laptop, and on the Pi for USB
              webcams.
  picamera2 — Raspberry Pi CSI camera (Camera Module 3). source is the CSI
              camera number (0 or 1). Picamera2 is imported lazily so the
              laptop never needs it installed.

Both backends hand BGR numpy frames to SharedState, so everything
downstream (detector, dashboard) is backend-agnostic.

Video files
-----------
When source is a file and loop_video is true, the file restarts when it
ends instead of counting as a failed read. This makes recorded test
footage behave like a live camera. Video files are also paced to their
native FPS so a 30 FPS recording plays back in real time — otherwise the
whole file would be read in a second and the time-based zone rules would
never fire.
"""

import logging
import threading
import time

import cv2

logger = logging.getLogger(__name__)


class _OpenCVSource:
    def __init__(self, source, width, height, loop_video):
        self._source = source
        self._is_file = isinstance(source, str) and not source.isdigit()
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source '{self._source}'. "
                "Check that the device is connected and not in use."
            )
        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._loop_video = loop_video
        fps = self._cap.get(cv2.CAP_PROP_FPS) if self._is_file else 0
        self._frame_interval = 1.0 / fps if fps and fps > 0 else 0.0
        self._next_frame_time = time.monotonic()

    def read(self):
        if self._frame_interval:
            delay = self._next_frame_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_frame_time = max(self._next_frame_time + self._frame_interval,
                                        time.monotonic() - self._frame_interval)

        ret, frame = self._cap.read()
        if not ret and self._is_file and self._loop_video:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return ret, frame

    def release(self):
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()


class _Picamera2Source:
    def __init__(self, source, width, height):
        from picamera2 import Picamera2  # Pi only — lazy import

        self._cam = Picamera2(int(source))
        # RGB888 in libcamera terms is BGR byte order in numpy — exactly
        # what OpenCV and the rest of the pipeline expect.
        cfg = self._cam.create_video_configuration(
            main={"size": (int(width), int(height)), "format": "RGB888"}
        )
        self._cam.configure(cfg)
        self._cam.start()

    def read(self):
        try:
            return True, self._cam.capture_array("main")
        except Exception:
            logger.exception("Picamera2 capture failed")
            return False, None

    def release(self):
        try:
            self._cam.stop()
            self._cam.close()
        except Exception:
            pass


class CameraCapture:
    def __init__(self, cam_config: dict, shared_state, name: str = "camera"):
        self._name = name
        self._backend = cam_config.get("backend", "opencv")
        self._source = cam_config["source"]
        self._width = cam_config["width"]
        self._height = cam_config["height"]
        self._max_failed_reads = cam_config["max_failed_reads"]
        self._loop_video = bool(cam_config.get("loop_video", False))

        self._shared_state = shared_state
        self._src = None
        self._thread = None
        self._running = False

    def _open(self):
        if self._backend == "opencv":
            return _OpenCVSource(self._source, self._width, self._height, self._loop_video)
        if self._backend == "picamera2":
            return _Picamera2Source(self._source, self._width, self._height)
        raise ValueError(f"Unknown camera backend '{self._backend}' for {self._name}")

    def start(self):
        self._src = self._open()

        self._running = True
        self._shared_state.set_camera_status("running")

        self._thread = threading.Thread(target=self._capture_loop, daemon=True,
                                        name=f"CameraThread-{self._name}")
        self._thread.start()
        logger.info("Camera %s started (backend=%s, source=%s, %dx%d)",
                    self._name, self._backend, self._source, self._width, self._height)

    def _capture_loop(self):
        failed_reads = 0

        while self._running:
            ret, frame = self._src.read()

            if ret:
                failed_reads = 0
                self._shared_state.set_frame(frame)
            else:
                failed_reads += 1
                logger.warning("Camera %s: failed to read frame (%d/%d)",
                               self._name, failed_reads, self._max_failed_reads)

                if failed_reads >= self._max_failed_reads:
                    logger.error(
                        "Camera %s read failed %d times in a row — stopping capture.",
                        self._name, self._max_failed_reads,
                    )
                    self._running = False
                    self._shared_state.set_camera_status("disconnected")
                    self._src.release()
                    return

        self._src.release()
        logger.info("Camera %s capture loop exited cleanly.", self._name)

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        # Release only if _capture_loop didn't already release on failure.
        if self._src is not None:
            self._src.release()
        if self._shared_state.get_camera_status() != "disconnected":
            self._shared_state.set_camera_status("stopped")
        logger.info("Camera %s stopped.", self._name)
