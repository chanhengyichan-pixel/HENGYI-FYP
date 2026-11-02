"""
CameraManager — starts and stops each camera's threads while the program runs.

Every enabled camera has two threads: CameraCapture (reads frames) and its
pipeline (entrance Pipeline or RoomPipeline: YOLO + rules + alerts). The
dashboard's Cameras page calls this class to add, restart, disable or remove
ONE camera without touching the others, so the PPE checks of the cameras
that keep running are never interrupted.

Safe stop order: pipeline first (it reads the camera's frames), then the
capture thread, which releases the device (USB / CSI / video file) so it can
be used again straight away.

How many cameras can run depends on the machine, not on a fixed number.
Before starting another camera the manager checks:
  - camera_capacity.max_active   (0 = no fixed limit)
  - free memory (/proc/meminfo)  >= camera_capacity.min_free_memory_mb
and refuses with a clear reason (CapacityError) if either fails. While
running, overload() reports cameras whose AI frame rate fell below
camera_capacity.min_fps — the Cameras page shows it as a warning.

Statuses (status()):
  running       capture thread delivering frames
  disconnected  the device stopped sending frames (unplugged, video ended)
  error         the camera could not be started (reason in "error")
  stopped       not started (disabled cameras are reported as "disabled"
                by the dashboard, which knows the saved setting)
"""

from __future__ import annotations

import logging
import threading
import time

from core.camera import CameraCapture
from core.shared_state import SharedState

logger = logging.getLogger(__name__)

# Seconds a new camera gets before its frame rate counts for overload().
_WARMUP_SECONDS = 15


class CapacityError(RuntimeError):
    """This machine cannot run another camera right now."""


def free_memory_mb() -> float | None:
    """MemAvailable in MB (Linux), or None when it cannot be read."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def friendly_error(exc: Exception, cam: dict) -> str:
    """Short reason for the dashboard, e.g. 'Device not found'."""
    text = str(exc)
    if isinstance(exc, ImportError) or "picamera2" in text.lower():
        return "Raspberry Pi camera software not available (Picamera2 runs only on the Pi)"
    if isinstance(exc, CapacityError):
        return text
    if "Cannot open camera source" in text:
        src = cam.get("source")
        if isinstance(src, str) and not src.strip().lstrip("-").isdigit():
            return f"Video file not found or unreadable: {src}"
        return "Device not found or busy (is it plugged in, or used by another program?)"
    return text.split("\n")[0][:160] or type(exc).__name__


class _Runtime:
    def __init__(self, cfg, capture, pipeline, zone_monitor):
        self.cfg = cfg
        self.capture = capture
        self.pipeline = pipeline
        self.zone_monitor = zone_monitor
        self.started_at = time.monotonic()


class CameraManager:
    """make_pipeline(cfg, state, capture) -> (pipeline, zone_monitor or None).

    states / zone_monitors are plain dicts that the dashboard holds on to;
    they are updated in place. A camera keeps the same SharedState across
    restarts so open video streams continue."""

    def __init__(self, config: dict, make_pipeline, capture_cls=CameraCapture):
        self._config = config
        self._make_pipeline = make_pipeline
        self._capture_cls = capture_cls
        cap = config.get("camera_capacity") or {}
        self.max_active = int(cap.get("max_active", 0) or 0)
        self.min_free_mb = float(cap.get("min_free_memory_mb", 0) or 0)
        self.min_fps = float(cap.get("min_fps", 0) or 0)
        self.states: dict = {"entrance": SharedState()}   # dashboard always reads "entrance"
        self.zone_monitors: dict = {}
        self._rt: dict[str, _Runtime] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.RLock()

    # ── queries ──
    def running_ids(self) -> list[str]:
        with self._lock:
            return list(self._rt)

    def is_running(self, cid: str) -> bool:
        with self._lock:
            return cid in self._rt

    def running_cfg(self, cid: str) -> dict | None:
        with self._lock:
            rt = self._rt.get(cid)
            return dict(rt.cfg) if rt else None

    def status(self, cid: str) -> dict:
        with self._lock:
            if cid in self._errors:
                return {"status": "error", "error": self._errors[cid]}
            rt = self._rt.get(cid)
            if rt is None:
                return {"status": "stopped", "error": None}
            st = self.states[cid]
            if st.get_camera_status() == "disconnected":
                return {"status": "disconnected",
                        "error": "No video signal: the camera was unplugged or stopped sending frames"}
            return {"status": "running", "error": None, "fps": round(st.get_fps(), 1)}

    def capacity_problem(self, extra: int = 1) -> str | None:
        """Why another camera cannot start now, or None."""
        with self._lock:
            active = len(self._rt)
        if self.max_active and active + extra > self.max_active:
            return (f"This system is set to run at most {self.max_active} camera(s) at once "
                    f"({active} running). Disable another camera first, or raise "
                    f"camera_capacity.max_active in the config if the hardware can handle it.")
        free = free_memory_mb()
        if self.min_free_mb and free is not None and free < self.min_free_mb:
            return (f"Not enough free memory to start another camera: {free:.0f} MB left, "
                    f"{self.min_free_mb:.0f} MB needed. Disable another camera first.")
        return None

    def overload(self) -> list[dict]:
        """Running cameras whose AI frame rate is below camera_capacity.min_fps."""
        if not self.min_fps:
            return []
        now, slow = time.monotonic(), []
        with self._lock:
            for cid, rt in self._rt.items():
                if now - rt.started_at < _WARMUP_SECONDS:
                    continue
                st = self.states[cid]
                fps = st.get_fps()
                if st.get_camera_status() == "running" and fps < self.min_fps:
                    slow.append({"id": cid, "fps": round(fps, 1)})
        return slow

    # ── actions ──
    def start(self, cfg: dict, check_capacity: bool = True) -> dict:
        """Start one camera. Returns status(); a camera that fails to open
        is kept in "error" with the reason. Raises CapacityError."""
        cid = cfg["id"]
        with self._lock:
            if cid in self._rt:
                self.stop(cid)
            if check_capacity:
                problem = self.capacity_problem()
                if problem:
                    raise CapacityError(problem)
            self._errors.pop(cid, None)
            state = self.states.setdefault(cid, SharedState())
            capture = self._capture_cls(cfg, state, name=cid)
            try:
                capture.start()
            except Exception as e:          # unplugged, busy, missing file, not a Pi...
                logger.error("Camera %s could not start: %s", cid, e)
                try:
                    capture.stop()
                except Exception:
                    pass
                state.set_camera_status("stopped")
                self._errors[cid] = friendly_error(e, cfg)
                return self.status(cid)
            try:
                pipeline, zone_monitor = self._make_pipeline(cfg, state, capture)
                pipeline.start()
            except Exception as e:
                logger.exception("Pipeline for %s could not start", cid)
                capture.stop()
                self._errors[cid] = f"AI processing could not start: {e}"
                return self.status(cid)
            self._rt[cid] = _Runtime(dict(cfg), capture, pipeline, zone_monitor)
            if zone_monitor is not None:
                self.zone_monitors[cid] = zone_monitor
            logger.info("Camera %s (%s) started.", cid, cfg.get("name", cid))
            return self.status(cid)

    def stop(self, cid: str) -> None:
        """Stop one camera's threads and release the device."""
        with self._lock:
            self._errors.pop(cid, None)
            rt = self._rt.pop(cid, None)
            self.zone_monitors.pop(cid, None)
            if rt is None:
                return
            # Pipeline first: it reads this camera's frames.
            try:
                rt.pipeline.stop()
            finally:
                rt.capture.stop()
            self.states[cid].set_camera_status("stopped")
            logger.info("Camera %s stopped and released.", cid)

    def restart(self, cfg: dict) -> dict:
        """Stop (if running) and start again — also used for Retry. The
        camera being restarted does not count against the capacity."""
        with self._lock:
            was_running = cfg["id"] in self._rt
            self.stop(cfg["id"])
            return self.start(cfg, check_capacity=not was_running)

    def remove(self, cid: str) -> None:
        with self._lock:
            self.stop(cid)
            self._errors.pop(cid, None)
            if cid != "entrance":
                self.states.pop(cid, None)

    def note_error(self, cid: str, message: str) -> None:
        """Record why a camera is not running (e.g. refused at startup)."""
        with self._lock:
            self._errors[cid] = message

    def stop_all(self) -> None:
        with self._lock:
            for cid in list(self._rt):
                self.stop(cid)
