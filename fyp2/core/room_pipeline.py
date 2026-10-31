"""
Room AI processing pipeline (hazard / restricted zone monitoring).

Reads frames from the room camera's SharedState and runs, per new frame:

  Detector.track()  -> detections with person track IDs
  DecisionEngine    -> which tracked person is missing which PPE
  ZoneMonitor       -> time-based zone rules, fired ZoneEvents

Results go back to SharedState (detections, zone status, fps) for the
dashboard, and each fired ZoneEvent is dispatched to the alert handlers
through AlertManager.notify_zone().

Structure mirrors core/pipeline.py (entrance): same new-frame-only check,
same disconnected-camera handling, same rolling-window FPS. The room
camera needs its OWN Detector instance — tracker state lives inside the
model and must not mix with the entrance camera's frames.
"""

import logging
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

_FPS_WINDOW = 30


class RoomPipeline:
    def __init__(self, config, shared_state, detector, decision_engine,
                 zone_monitor, alert_manager=None):
        self._shared_state = shared_state
        self._detector = detector
        self._decision_engine = decision_engine
        self._zone_monitor = zone_monitor
        self._alert_manager = alert_manager

        self._running = False
        self._thread = None
        self._ts_window: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._was_disconnected = False
        self._last_fps_log_time = 0.0
        self._last_frame_id = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="RoomPipelineThread")
        self._thread.start()
        logger.info("Room pipeline started.")

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Room pipeline stopped.")

    def _loop(self):
        while self._running:
            if self._shared_state.get_camera_status() == "disconnected":
                if not self._was_disconnected:
                    logger.warning("Room camera disconnected — zone monitoring paused.")
                    self._was_disconnected = True
                time.sleep(0.1)
                continue
            if self._was_disconnected:
                logger.info("Room camera reconnected — zone monitoring resumed.")
                self._was_disconnected = False

            frame, frame_id = self._shared_state.get_frame_with_id()
            if frame is None or frame_id == self._last_frame_id:
                time.sleep(0.005)
                continue
            self._last_frame_id = frame_id

            t_start = time.monotonic()
            self.process(frame)

            self._ts_window.append(t_start)
            if len(self._ts_window) >= 2:
                elapsed = self._ts_window[-1] - self._ts_window[0]
                fps = (len(self._ts_window) - 1) / elapsed if elapsed > 0 else 0.0
                self._shared_state.set_fps(fps)
                if t_start - self._last_fps_log_time >= 5.0:
                    logger.info("Room pipeline FPS: %.1f", fps)
                    self._last_fps_log_time = t_start

        logger.info("Room pipeline loop exited.")

    def process(self, frame):
        """One full detect -> evaluate -> zone-rule cycle. Public for tests."""
        detections = self._detector.track(frame)
        verdict = self._decision_engine.evaluate(detections)
        events, zone_status = self._zone_monitor.update(detections, verdict)

        self._shared_state.set_latest_detections(detections)
        self._shared_state.set_zone_status(zone_status)

        for event in events:
            logger.warning("Zone event: %s in '%s' (track %d) %s",
                           event.event_type, event.zone, event.track_id,
                           event.violations or "")
            if self._alert_manager is not None:
                self._alert_manager.notify_zone(event, frame)
        return events
