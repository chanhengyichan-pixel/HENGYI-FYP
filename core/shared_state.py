import threading


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._camera_status = "stopped"

    def set_frame(self, frame):
        with self._lock:
            self._frame = frame

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def set_camera_status(self, status: str):
        with self._lock:
            self._camera_status = status

    def get_camera_status(self) -> str:
        with self._lock:
            return self._camera_status
