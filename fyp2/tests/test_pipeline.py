"""
Unit tests for core/pipeline.py.

No real camera, no real YOLO model.  All dependencies are stubs or
MagicMocks.  SharedState is real (it's a thin threading wrapper — faking
it would just re-test threading.Lock, not pipeline logic).

An threading.Event is used wherever the test needs to wait for the
pipeline loop to process at least one frame, making the tests
deterministic without sleeping for a fixed duration.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from core.shared_state import SharedState
from core.pipeline import Pipeline
from core.decision_engine import ComplianceStatus, FrameVerdict
from core.session_controller import SessionController, SessionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_CONFIG = {"session": {"evaluating_seconds": 1.5, "min_votes": 3,
                            "decision_hold_seconds": 5.0,
                            "cooldown_seconds": 3.0},
               "decision_engine": {"debug": False}}

_FAKE_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
_FAKE_DETECTIONS = []
_FAKE_VERDICT = FrameVerdict(status=ComplianceStatus.NO_PERSON)
_FAKE_SESSION_STATE = SessionState.IDLE


def make_stubs(processed_event: threading.Event | None = None):
    """
    Return (shared_state, detector_stub, de_stub, sc_stub).

    If processed_event is given, detector.detect() sets it on first call
    so tests can block until the pipeline has processed at least one frame.
    """
    shared_state = SharedState()
    shared_state.set_frame(_FAKE_FRAME.copy())

    def _detect(frame):
        if processed_event is not None:
            processed_event.set()
        return _FAKE_DETECTIONS

    detector = MagicMock()
    detector.detect.side_effect = _detect

    de = MagicMock()
    de.evaluate.return_value = _FAKE_VERDICT

    sc = MagicMock()
    sc.update.return_value = _FAKE_SESSION_STATE

    return shared_state, detector, de, sc


class FrameFeeder:
    """Keep publishing NEW frames into a SharedState from a background thread.

    The pipeline only processes a frame whose frame_id it has not seen, so
    tests that need several pipeline cycles must keep supplying frames —
    like a real camera does.
    """

    def __init__(self, shared_state, interval=0.005):
        self._state = shared_state
        self._interval = interval
        self._running = False
        self._thread = None

    def __enter__(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._running = False
        self._thread.join(timeout=1)

    def _run(self):
        while self._running:
            self._state.set_frame(_FAKE_FRAME.copy())
            time.sleep(self._interval)


def make_pipeline(shared_state, detector, de, sc):
    return Pipeline(BASE_CONFIG, shared_state,
                    camera=None,  # camera is not used by Pipeline directly
                    detector=detector,
                    decision_engine=de,
                    session_controller=sc)


# ---------------------------------------------------------------------------
# Call order
# ---------------------------------------------------------------------------

class TestCallOrder:
    def test_pipeline_calls_detect_then_evaluate_then_update(self):
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        pipeline = make_pipeline(shared_state, detector, de, sc)

        call_order = []
        detector.detect.side_effect = lambda f: (call_order.append("detect") or event.set() or _FAKE_DETECTIONS)
        de.evaluate.side_effect = lambda d, **kw: (call_order.append("evaluate") or _FAKE_VERDICT)
        sc.update.side_effect = lambda v: (call_order.append("update") or _FAKE_SESSION_STATE)

        pipeline.start()
        event.wait(timeout=2)
        pipeline.stop()

        assert call_order[:3] == ["detect", "evaluate", "update"]

    def test_detect_receives_frame_from_shared_state(self):
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        pipeline = make_pipeline(shared_state, detector, de, sc)

        received_frames = []
        def _detect(frame):
            received_frames.append(frame)
            event.set()
            return _FAKE_DETECTIONS
        detector.detect.side_effect = _detect

        pipeline.start()
        event.wait(timeout=2)
        pipeline.stop()

        assert len(received_frames) >= 1
        assert received_frames[0].shape == _FAKE_FRAME.shape

    def test_evaluate_receives_detections_from_detect(self):
        fake_dets = [SimpleNamespace(class_name="helmet")]
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        detector.detect.side_effect = lambda f: (event.set() or fake_dets)
        pipeline = make_pipeline(shared_state, detector, de, sc)

        pipeline.start()
        event.wait(timeout=2)
        pipeline.stop()

        de.evaluate.assert_called_with(fake_dets, debug=False)

    def test_update_receives_verdict_from_evaluate(self):
        fake_verdict = FrameVerdict(status=ComplianceStatus.COMPLIANT, person_count=1)
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        de.evaluate.side_effect = lambda d, **kw: (event.set() or fake_verdict)
        pipeline = make_pipeline(shared_state, detector, de, sc)

        pipeline.start()
        event.wait(timeout=2)
        pipeline.stop()

        sc.update.assert_called_with(fake_verdict)


# ---------------------------------------------------------------------------
# SharedState writes
# ---------------------------------------------------------------------------

class TestSharedStateWrites:
    def test_latest_detections_written_to_shared_state(self):
        fake_dets = [SimpleNamespace(class_name="vest")]
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        detector.detect.side_effect = lambda f: (event.set() or fake_dets)
        pipeline = make_pipeline(shared_state, detector, de, sc)

        pipeline.start()
        event.wait(timeout=2)
        time.sleep(0.02)   # let the write happen after event is set
        pipeline.stop()

        assert shared_state.get_latest_detections() == fake_dets

    def test_session_state_written_to_shared_state(self):
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        sc.update.side_effect = lambda v: (event.set() or SessionState.EVALUATING)
        pipeline = make_pipeline(shared_state, detector, de, sc)

        pipeline.start()
        event.wait(timeout=2)
        time.sleep(0.02)
        pipeline.stop()

        assert shared_state.get_session_state() is SessionState.EVALUATING

    def test_fps_written_after_multiple_frames(self):
        shared_state, detector, de, sc = make_stubs()
        count = {"n": 0}
        event = threading.Event()

        def _detect(f):
            count["n"] += 1
            if count["n"] >= 3:
                event.set()
            return _FAKE_DETECTIONS
        detector.detect.side_effect = _detect

        pipeline = make_pipeline(shared_state, detector, de, sc)
        with FrameFeeder(shared_state):
            pipeline.start()
            event.wait(timeout=3)
            time.sleep(0.02)
            pipeline.stop()

        assert shared_state.get_fps() > 0.0

    def test_same_frame_is_processed_only_once(self):
        # One frame published, pipeline left running: detect() runs once,
        # not once per loop iteration.
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        pipeline = make_pipeline(shared_state, detector, de, sc)
        pipeline.start()
        event.wait(timeout=2)
        time.sleep(0.1)
        pipeline.stop()

        assert detector.detect.call_count == 1


# ---------------------------------------------------------------------------
# None-frame tolerance (camera not ready yet)
# ---------------------------------------------------------------------------

class TestNoneFrameTolerance:
    def test_none_frame_does_not_crash(self):
        shared_state = SharedState()
        # Frame is NOT set — get_frame() returns None.
        detector = MagicMock()
        de = MagicMock()
        sc = MagicMock()
        pipeline = Pipeline(BASE_CONFIG, shared_state, None, detector, de, sc)

        pipeline.start()
        time.sleep(0.05)   # let the loop spin a few times on None
        pipeline.stop()

        # detect() must never have been called
        detector.detect.assert_not_called()

    def test_none_frame_then_real_frame_processes_correctly(self):
        shared_state = SharedState()
        # Start with no frame, then inject one after a short delay.
        event = threading.Event()
        detector = MagicMock()
        detector.detect.side_effect = lambda f: (event.set() or _FAKE_DETECTIONS)
        de = MagicMock()
        de.evaluate.return_value = _FAKE_VERDICT
        sc = MagicMock()
        sc.update.return_value = _FAKE_SESSION_STATE
        pipeline = Pipeline(BASE_CONFIG, shared_state, None, detector, de, sc)

        pipeline.start()
        time.sleep(0.03)                              # loop spins on None
        shared_state.set_frame(_FAKE_FRAME.copy())    # now inject a frame
        event.wait(timeout=2)
        pipeline.stop()

        detector.detect.assert_called()


# ---------------------------------------------------------------------------
# Session state transition logging
# ---------------------------------------------------------------------------

class TestSessionStateLogging:
    def test_state_change_produces_log_line(self, caplog):
        """A transition between two different states emits exactly one INFO line."""
        import logging
        shared_state = SharedState()
        shared_state.set_frame(_FAKE_FRAME.copy())
        shared_state.set_camera_status("running")

        # Return IDLE on call 1, PERSON_DETECTED on call 2+.
        # Using a stable sentinel after the transition avoids StopIteration
        # if the loop runs extra iterations before stop() is called.
        call_idx = {"n": 0}
        def _update(v):
            call_idx["n"] += 1
            return SessionState.IDLE if call_idx["n"] < 2 else SessionState.PERSON_DETECTED
        sc = MagicMock()
        sc.update.side_effect = _update

        detector = MagicMock()
        event = threading.Event()
        det_count = {"n": 0}
        def _detect(f):
            det_count["n"] += 1
            if det_count["n"] >= 2:
                event.set()
            return _FAKE_DETECTIONS
        detector.detect.side_effect = _detect

        de = MagicMock()
        de.evaluate.return_value = _FAKE_VERDICT

        pipeline = Pipeline(BASE_CONFIG, shared_state, None, detector, de, sc)

        with caplog.at_level(logging.INFO, logger="core.pipeline"), FrameFeeder(shared_state):
            pipeline.start()
            event.wait(timeout=2)
            time.sleep(0.02)
            pipeline.stop()

        transition_lines = [r.message for r in caplog.records
                            if "Session state:" in r.message]
        # First log: None -> idle; second log: idle -> person_detected
        assert any("idle" in l and "person_detected" in l for l in transition_lines)

    def test_same_state_produces_no_transition_log(self, caplog):
        """Consecutive frames with the same session state produce no transition log."""
        import logging
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        # SC always returns IDLE — state never changes.
        sc.update.return_value = SessionState.IDLE

        pipeline = make_pipeline(shared_state, detector, de, sc)

        with caplog.at_level(logging.INFO, logger="core.pipeline"):
            pipeline.start()
            event.wait(timeout=2)
            # Let it process several more frames to confirm no repeated logs.
            time.sleep(0.05)
            pipeline.stop()

        transition_lines = [r.message for r in caplog.records
                            if "Session state:" in r.message]
        # First call transitions from None -> idle (logged once), then no more.
        assert len(transition_lines) == 1


# ---------------------------------------------------------------------------
# Camera-disconnected handling
# ---------------------------------------------------------------------------

class TestCameraDisconnected:
    def test_disconnected_status_prevents_detect_calls(self):
        # Shared state has a frame AND camera_status == "disconnected".
        # Pipeline must not call detect() in this case.
        shared_state = SharedState()
        shared_state.set_frame(_FAKE_FRAME.copy())
        shared_state.set_camera_status("disconnected")

        detector = MagicMock()
        de = MagicMock()
        sc = MagicMock()
        pipeline = Pipeline(BASE_CONFIG, shared_state, None, detector, de, sc)

        pipeline.start()
        time.sleep(0.1)   # let the loop spin several times
        pipeline.stop()

        detector.detect.assert_not_called()

    def test_hold_expiry_caught_up_retroactively_on_reconnect(self):
        """
        Confirm that a GRANTED hold which expires during a camera disconnect
        is detected retroactively on the first post-reconnect update() call,
        not lost or silently ignored.

        Mechanism: the SessionController's clock is wall-clock-based.
        While disconnected, the pipeline stops calling update() so nothing
        checks the clock.  On the first frame after reconnect, update() runs
        and its elapsed-time check fires immediately because enough time has
        already passed — retroactive catch-up, not continuous expiry.
        """
        fake_clock = [0.0]
        sc_config = {"session": {"evaluating_seconds": 0.0, "min_votes": 5,
                                  "decision_hold_seconds": 10.0,
                                  "cooldown_seconds": 3.0}}
        sc = SessionController(sc_config, clock=lambda: fake_clock[0])

        # Drive SessionController to GRANTED via direct update() calls.
        # IDLE -> PERSON_DETECTED -> EVALUATING (5 frames) -> GRANTED
        compliant = FrameVerdict(status=ComplianceStatus.COMPLIANT, person_count=1)
        sc.update(compliant)                 # → PERSON_DETECTED
        for _ in range(5):
            sc.update(compliant)             # → EVALUATING, then → GRANTED
        assert sc.state is SessionState.GRANTED

        # Advance fake clock past the hold duration — hold is now "expired"
        # but update() has not been called yet, so the SC doesn't know yet.
        fake_clock[0] = 15.0  # > decision_hold_seconds (10.0)

        # Wire SC into a pipeline with a disconnected camera.
        shared_state = SharedState()
        shared_state.set_frame(_FAKE_FRAME.copy())
        shared_state.set_camera_status("disconnected")

        # Detector/DE return a COMPLIANT verdict so the SC gets a real input.
        detector = MagicMock()
        detector.detect.return_value = _FAKE_DETECTIONS
        de = MagicMock()
        de.evaluate.return_value = compliant

        pipeline = Pipeline(BASE_CONFIG, shared_state, None, detector, de, sc)
        pipeline.start()
        time.sleep(0.05)                              # pipeline spins on disconnected

        # Confirm SC was not updated while disconnected.
        assert sc.state is SessionState.GRANTED       # still frozen

        # Reconnect — pipeline should call update() on next frame.
        event = threading.Event()
        original_detect = detector.detect.side_effect
        detector.detect.side_effect = lambda f: (event.set() or _FAKE_DETECTIONS)

        shared_state.set_camera_status("running")
        event.wait(timeout=2)
        time.sleep(0.02)   # let the SharedState write complete
        pipeline.stop()

        # The first post-reconnect update() should have caught up:
        # hold already expired (clock=15 > hold=10) → COOLDOWN.
        assert shared_state.get_session_state() is SessionState.COOLDOWN

    def test_pipeline_resumes_after_reconnect(self):
        # Start disconnected, then flip to "running" — pipeline should detect.
        shared_state = SharedState()
        shared_state.set_frame(_FAKE_FRAME.copy())
        shared_state.set_camera_status("disconnected")

        event = threading.Event()
        detector = MagicMock()
        detector.detect.side_effect = lambda f: (event.set() or _FAKE_DETECTIONS)
        de = MagicMock()
        de.evaluate.return_value = _FAKE_VERDICT
        sc = MagicMock()
        sc.update.return_value = _FAKE_SESSION_STATE
        pipeline = Pipeline(BASE_CONFIG, shared_state, None, detector, de, sc)

        pipeline.start()
        time.sleep(0.05)                              # spinning on disconnected
        shared_state.set_camera_status("running")     # simulate reconnect
        event.wait(timeout=2)
        pipeline.stop()

        detector.detect.assert_called()


# ---------------------------------------------------------------------------
# Stop halts the loop
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_halts_shared_state_writes(self):
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        pipeline = make_pipeline(shared_state, detector, de, sc)

        pipeline.start()
        event.wait(timeout=2)   # at least one frame processed
        pipeline.stop()

        call_count_after_stop = detector.detect.call_count
        time.sleep(0.1)         # let any in-flight iteration finish

        # Allow at most 1 extra call that was in-flight when stop() was called.
        assert detector.detect.call_count <= call_count_after_stop + 1

    def test_stop_joins_thread(self):
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        pipeline = make_pipeline(shared_state, detector, de, sc)

        pipeline.start()
        event.wait(timeout=2)
        pipeline.stop()

        assert not pipeline._thread.is_alive()


# ---------------------------------------------------------------------------
# Alerts receive the window's decision violations
# ---------------------------------------------------------------------------

class TestAlertVerdict:
    def test_denied_alert_uses_session_decision_violations(self):
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        de.evaluate.return_value = FrameVerdict(
            status=ComplianceStatus.NON_COMPLIANT,
            violations=["no-helmet", "no-vest"], person_count=1)   # flickery frame
        sc.update.return_value = SessionState.DENIED
        sc.decision_violations = ["no-helmet"]                      # window majority
        alerts = MagicMock()
        pipeline = Pipeline(BASE_CONFIG, shared_state, None, detector, de, sc, alerts)
        pipeline.start()
        event.wait(timeout=2)
        time.sleep(0.05)
        pipeline.stop()

        state, verdict, _ = alerts.notify.call_args[0]
        assert state is SessionState.DENIED
        assert verdict.violations == ["no-helmet"]

    def test_non_decision_states_pass_frame_verdict_through(self):
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs(event)
        sc.update.return_value = SessionState.EVALUATING
        alerts = MagicMock()
        pipeline = Pipeline(BASE_CONFIG, shared_state, None, detector, de, sc, alerts)
        pipeline.start()
        event.wait(timeout=2)
        time.sleep(0.05)
        pipeline.stop()

        assert alerts.notify.call_args[0][1] is _FAKE_VERDICT


class TestPoseChecker:
    def test_pose_checker_verdict_goes_to_session_controller(self):
        event = threading.Event()
        shared_state, detector, de, sc = make_stubs()
        checked = FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                               violations=["no-vest"], person_count=1)
        pose = MagicMock()
        pose.check.return_value = checked
        sc.update.side_effect = lambda v: (event.set(), _FAKE_SESSION_STATE)[1]
        pipeline = Pipeline(BASE_CONFIG, shared_state, None, detector, de, sc,
                            pose_checker=pose)
        pipeline.start()
        event.wait(timeout=2)
        pipeline.stop()

        frame, verdict = pose.check.call_args[0]
        assert frame.shape == _FAKE_FRAME.shape
        assert verdict is _FAKE_VERDICT
        assert sc.update.call_args[0][0] is checked
