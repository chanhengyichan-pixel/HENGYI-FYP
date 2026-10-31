import threading


class SharedState:
    """Thread-safe hand-off between one camera, its pipeline and the dashboard.

    One instance per camera (entrance, room). Each pipeline only ever reads
    and writes its own camera's instance.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        # Incremented on every set_frame(). Pipelines remember the last id
        # they processed so the same frame is never run through YOLO twice
        # (on the laptop inference is faster than the camera, so without
        # this the pipeline would re-process duplicate frames).
        self._frame_id = 0
        self._camera_status = "stopped"
        # list of Detection objects as returned by Detector.detect().
        # Stored untyped to avoid importing detector.py here.
        # Callers that need JSON call .to_dict() on each element themselves.
        self._latest_detections = []
        self._session_state = None  # SessionState enum value, or None before first update
        self._fps = 0.0             # pipeline inference throughput (not camera capture FPS)
        # FrameVerdict.debug_info from the most recent evaluate() call, or
        # None. Only ever non-None when decision_engine.debug is enabled —
        # see core/decision_engine.py. Used exclusively by the dashboard's
        # diagnostic overlay (dashboard/app.py _draw_debug_overlay).
        self._latest_debug_info = None
        # Room camera only: list of per-person zone dicts from ZoneMonitor
        # (track_id, zone name/type, seconds inside, ppe_ok). Empty for the
        # entrance camera.
        self._zone_status = []

    # --- frame ---

    def set_frame(self, frame):
        with self._lock:
            self._frame = frame
            self._frame_id += 1

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_frame_with_id(self):
        """Return (frame_copy, frame_id); frame_copy is None before the first frame."""
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            return frame, self._frame_id

    # --- camera_status ---

    def set_camera_status(self, status: str):
        with self._lock:
            self._camera_status = status

    def get_camera_status(self) -> str:
        with self._lock:
            return self._camera_status

    # --- latest_detections ---

    def set_latest_detections(self, detections: list):
        with self._lock:
            self._latest_detections = detections

    def get_latest_detections(self) -> list:
        with self._lock:
            return list(self._latest_detections)

    # --- session_state ---

    def set_session_state(self, state):
        with self._lock:
            self._session_state = state

    def get_session_state(self):
        with self._lock:
            return self._session_state

    # --- fps ---

    def set_fps(self, fps: float):
        with self._lock:
            self._fps = fps

    def get_fps(self) -> float:
        with self._lock:
            return self._fps

    # --- latest_debug_info (decision_engine diagnostic overlay) ---

    def set_latest_debug_info(self, debug_info) -> None:
        with self._lock:
            self._latest_debug_info = debug_info

    def get_latest_debug_info(self):
        with self._lock:
            return self._latest_debug_info

    # --- zone_status (room camera) ---

    def set_zone_status(self, zone_status: list) -> None:
        with self._lock:
            self._zone_status = zone_status

    def get_zone_status(self) -> list:
        with self._lock:
            return list(self._zone_status)
