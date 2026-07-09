"""
Thread 1 — AI processing pipeline.

Reads frames from SharedState, runs Detector -> DecisionEngine ->
SessionController in sequence, and writes results back to SharedState.

This loop runs on the CPU-bound inference cycle.  The fps value written
to shared_state.set_fps() is INFERENCE THROUGHPUT — how many
detect+evaluate+session_controller cycles the pipeline completes per
second.  This is NOT the camera capture FPS (camera FPS is constrained
by the hardware and OpenCV's read loop in CameraCapture).  On a laptop
these numbers may be close; on a Raspberry Pi 5 inference throughput
will be significantly lower than capture FPS.  Never conflate them in
performance reporting.

FPS measurement: rolling window over the last _FPS_WINDOW frame-start
timestamps.  FPS = (window_size - 1) / (newest_ts - oldest_ts).
Rolling window is chosen over exponential moving average because it is
deterministic and easy to explain — the reported number always reflects
exactly the last N cycles.

Dependencies are injected at construction, not created internally, so
the pipeline can be unit-tested with stubs (no real camera or model
required).

Camera-disconnected behaviour
------------------------------
Each loop iteration checks camera_status before processing. If the
status is "disconnected", the loop skips the detect/evaluate/
session_controller.update cycle entirely and does NOT pass the last
stale frame through the AI stack.  This matters because after a
disconnect, get_frame() continues to return the final captured frame
(CameraCapture does not clear SharedState on failure), so a frame-None
check alone is insufficient to detect the failure.

What happens to in-progress session state on disconnect:
  - The session_controller state is left frozen at whatever it was.
    The pipeline simply stops feeding it verdicts.
  - Rationale: a camera disconnect may be transient (cable bump,
    USB re-enumerate). Forcing a session reset on every disconnect
    would abort a legitimate evaluation that could resume in seconds.
    If the camera reconnects (status returns to "running"), the
    pipeline resumes normally from the frozen state.
  - For time-based states (GRANTED, DENIED, COOLDOWN): the hold is NOT
    actively monitored during a disconnect.  session_controller.update()
    is not called while disconnected, so nothing is checking the clock.
    What actually happens is a retroactive catch-up: on the first frame
    processed after reconnect, update() is called and its elapsed-time
    check (self._clock() - self._hold_start) reflects true wall-clock
    time regardless of the gap.  If the hold duration has already passed
    by that point, the transition to COOLDOWN (or IDLE from COOLDOWN)
    fires immediately on that first post-reconnect call — not gradually
    during the outage.  A sufficiently long disconnect therefore causes
    the hold to appear already-expired the moment inference resumes,
    which is the correct and expected behaviour.
  - This decision should be revisited if the hardware shows frequent
    spurious disconnects that leave sessions permanently frozen.

The disconnected warning is logged only on the first cycle after
entering the disconnected state, not every cycle (rate-limiting).
"""

import logging
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

_FPS_WINDOW = 30  # number of frame timestamps kept for rolling FPS


class Pipeline:
    def __init__(self, config, shared_state, camera, detector,
                 decision_engine, session_controller, alert_manager=None):
        self._shared_state = shared_state
        self._detector = detector
        self._decision_engine = decision_engine
        self._session_controller = session_controller
        self._alert_manager = alert_manager  # optional; None disables alerts

        self._running = False
        self._thread = None
        self._ts_window: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._was_disconnected = False  # tracks transition into disconnected
        self._prev_session_state = None  # tracks last seen state for transition logging
        self._last_fps_log_time: float = 0.0  # monotonic time of last FPS log

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="PipelineThread"
        )
        self._thread.start()
        logger.info("Pipeline started.")

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Pipeline stopped.")

    def _loop(self):
        while self._running:
            camera_status = self._shared_state.get_camera_status()
            if camera_status == "disconnected":
                if not self._was_disconnected:
                    logger.warning(
                        "Camera disconnected — pipeline halting inference. "
                        "Session state frozen until camera recovers."
                    )
                    self._was_disconnected = True
                time.sleep(0.1)
                continue

            # Reset the disconnected flag if the camera recovered.
            if self._was_disconnected:
                logger.info("Camera reconnected — pipeline resuming inference.")
                self._was_disconnected = False

            frame = self._shared_state.get_frame()
            if frame is None:
                # Camera not yet producing frames — yield and retry.
                time.sleep(0.005)
                continue

            t_start = time.monotonic()

            detections = self._detector.detect(frame)
            verdict = self._decision_engine.evaluate(detections)
            session_state = self._session_controller.update(verdict)

            self._shared_state.set_latest_detections(detections)
            self._shared_state.set_session_state(session_state)

            if self._alert_manager is not None:
                self._alert_manager.notify(session_state, verdict, frame)

            # Log session state transitions (only on change, not every frame).
            if session_state != self._prev_session_state:
                logger.info(
                    "Session state: %s -> %s",
                    self._prev_session_state.value if self._prev_session_state else "None",
                    session_state.value,
                )
                self._prev_session_state = session_state

            self._ts_window.append(t_start)
            if len(self._ts_window) >= 2:
                elapsed = self._ts_window[-1] - self._ts_window[0]
                fps = (len(self._ts_window) - 1) / elapsed if elapsed > 0 else 0.0
                self._shared_state.set_fps(fps)
                # Log FPS roughly every 5 seconds (time-based, not frame-count-based).
                if t_start - self._last_fps_log_time >= 5.0:
                    logger.info("Pipeline FPS: %.1f", fps)
                    self._last_fps_log_time = t_start

        logger.info("Pipeline loop exited.")
