"""
Entrance AI processing pipeline (PPE ALLOW/DENY check point).

Reads frames from the entrance camera's SharedState, runs Detector ->
DecisionEngine -> SessionController in sequence, and writes results back
to SharedState. The room camera has its own pipeline
(core/room_pipeline.py).

New frames only
---------------
Each loop iteration processes a frame only if its frame_id differs from
the last one processed. On the laptop, inference (~110 cycles/s) is
faster than the camera (~30 FPS); without this check the same frame was
run through YOLO several times and counted as several votes.

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

import dataclasses
import logging
import threading
import time
from collections import deque

from core.session_controller import SessionState

logger = logging.getLogger(__name__)

_FPS_WINDOW = 30  # number of frame timestamps kept for rolling FPS


class Pipeline:
    def __init__(self, config, shared_state, camera, detector,
                 decision_engine, session_controller, alert_manager=None,
                 pose_checker=None):
        self._shared_state = shared_state
        self._detector = detector
        self._decision_engine = decision_engine
        self._session_controller = session_controller
        # Calibration/diagnostic flag — see core/decision_engine.py's module
        # docstring ("Debug diagnostic mode"). Read once at construction,
        # not re-read every frame, since config does not change at runtime.
        self._decision_debug = bool(config["decision_engine"]["debug"])
        self._alert_manager = alert_manager      # optional; None disables alerts
        # Optional PoseWearChecker (core/pose_wear_check.py): turns PPE that
        # is held rather than worn into a violation. None = off.
        self._pose_checker = pose_checker

        self._running = False
        self._thread = None
        self._ts_window: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._was_disconnected = False      # tracks transition into disconnected
        self._prev_session_state = None     # tracks last seen state for transition logging
        self._last_fps_log_time: float = 0.0  # monotonic time of last FPS log
        self._last_frame_id = 0               # last frame_id run through the model

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

    def _alert_verdict(self, session_state, verdict):
        """During a GRANTED/DENIED hold, give handlers the violations the
        whole evaluation window agreed on (SessionController.
        decision_violations) instead of this single frame's — so the voice
        prompt and the event log name what was really missing."""
        if session_state not in (SessionState.GRANTED, SessionState.DENIED):
            return verdict
        violations = getattr(self._session_controller, "decision_violations", None)
        if not isinstance(violations, list):
            return verdict
        if session_state is SessionState.DENIED and not violations:
            violations = list(verdict.violations) or ["no-helmet", "no-vest"]
        return dataclasses.replace(verdict, violations=violations)

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

            frame, frame_id = self._shared_state.get_frame_with_id()
            if frame is None or frame_id == self._last_frame_id:
                # No frame yet, or no NEW frame since the last cycle —
                # yield and retry.
                time.sleep(0.005)
                continue
            self._last_frame_id = frame_id

            t_start = time.monotonic()

            detections = self._detector.detect(frame)
            verdict = self._decision_engine.evaluate(detections, debug=self._decision_debug)
            if self._pose_checker is not None:
                verdict = self._pose_checker.check(frame, verdict)

            # Diagnostic overlay data — only ever non-None when debug is on
            # (see core/decision_engine.py). Written unconditionally (cheap
            # None-write when off) so the dashboard always has a defined
            # value to read rather than stale data from a previous run.
            self._shared_state.set_latest_debug_info(verdict.debug_info)

            session_state = self._session_controller.update(verdict)

            self._shared_state.set_latest_detections(detections)
            self._shared_state.set_session_state(session_state)

            if self._alert_manager is not None:
                self._alert_manager.notify(session_state,
                                           self._alert_verdict(session_state, verdict),
                                           frame)

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
