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

BASE_CONFIG = {"session": {"evaluating_frame_count": 15,
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
        pipeline.start()
        event.wait(timeout=3)
        time.sleep(0.02)
        pipeline.stop()

        assert shared_state.get_fps() > 0.0


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

        with caplog.at_level(logging.INFO, logger="core.pipeline"):
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
        sc_config = {"session": {"evaluating_frame_count": 5,
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

# ---------------------------------------------------------------------------
# Identity invalidation (face recognition worker-swap detection)
# ---------------------------------------------------------------------------

class TestIdentityInvalidation:
    """
    Tests for the two-point identity check added in Milestone 12.

    Uses a real SessionController (small window) and a stub FaceRecognizer
    injected via the face_recognizer= parameter so no LBPH model is needed.

    The face_recognizer stub has an .identify attribute that tests can
    control to return different identities at the opening vs. closing check.
    """

    _SC_CONFIG = {"session": {"evaluating_frame_count": 3,
                               "decision_hold_seconds": 10.0,
                               "cooldown_seconds": 3.0},
                  "decision_engine": {"debug": False}}

    @staticmethod
    def _make_fr_stub(two_results: list):
        """
        Return a stub FaceRecognizer whose identify() returns:
          - two_results[0] on the first call (opening identity check)
          - two_results[1] on the second call (closing identity check)
          - two_results[1] on all subsequent calls (pipeline may keep running)
        """
        stub = MagicMock()
        # Pad with the second value repeated to avoid StopIteration.
        padded = list(two_results) + [two_results[-1]] * 100
        stub.identify.side_effect = padded
        return stub

    def _run_session(self, fr_stub, compliant=True):
        """
        Drive one full session through the pipeline and return (final_state, shared_state).

        Frame sequence for evaluating_frame_count=3:
          Frame 1 (NO_PERSON): IDLE
          Frame 2 (person):    IDLE → PERSON_DETECTED  [identity check 1]
          Frame 3 (person):    PERSON_DETECTED → EVALUATING (vote 1)
          Frame 4 (person):    EVALUATING (vote 2)
          Frame 5 (person):    EVALUATING (vote 3 → decision)  [identity check 2, before update]
        """
        fake_clock = [0.0]
        sc = SessionController(self._SC_CONFIG, clock=lambda: fake_clock[0])
        shared_state = SharedState()

        verdict_seq = [
            FrameVerdict(status=ComplianceStatus.NO_PERSON),
            FrameVerdict(status=ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT, person_count=1),
            FrameVerdict(status=ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT, person_count=1),
            FrameVerdict(status=ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT, person_count=1),
            FrameVerdict(status=ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT, person_count=1),
        ]
        de = MagicMock()
        de.evaluate.side_effect = list(verdict_seq) + [verdict_seq[-1]] * 100

        done_event = threading.Event()
        frame_count = [0]

        def _detect(frame):
            frame_count[0] += 1
            if frame_count[0] >= 5:
                done_event.set()
            return []

        detector = MagicMock()
        detector.detect.side_effect = _detect

        shared_state.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))

        pipeline = Pipeline(self._SC_CONFIG, shared_state, None, detector, de,
                            sc, face_recognizer=fr_stub)
        pipeline.start()
        done_event.wait(timeout=3)
        time.sleep(0.05)
        pipeline.stop()

        return shared_state.get_session_state(), shared_state

    def test_matching_identities_complete_normally(self):
        """Same name at open and close → session proceeds to GRANTED."""
        fr = self._make_fr_stub([
            ("Alice", 30.0),   # opening check
            ("Alice", 32.0),   # closing check (same person)
        ])
        final_state, ss = self._run_session(fr, compliant=True)
        assert final_state is SessionState.GRANTED

    def test_differing_identities_trigger_reset(self):
        """Different names at open and close → session_controller.reset() is called."""
        # We can't check final state — after reset(), the pipeline immediately
        # continues and may complete a new session.  Instead verify reset() fired.
        swap_event = threading.Event()
        fake_clock = [0.0]
        sc = SessionController(self._SC_CONFIG, clock=lambda: fake_clock[0])
        original_reset = sc.reset

        def _spy_reset():
            original_reset()
            swap_event.set()

        sc.reset = _spy_reset

        fr = self._make_fr_stub([
            ("Alice", 30.0),   # opening check
            ("Bob", 28.0),     # closing check (different person → swap)
        ])

        shared_state = SharedState()
        verdict_seq = [
            FrameVerdict(status=ComplianceStatus.NO_PERSON),
            *(FrameVerdict(status=ComplianceStatus.COMPLIANT, person_count=1),) * 100,
        ]
        de = MagicMock()
        de.evaluate.side_effect = verdict_seq

        detector = MagicMock()
        detector.detect.return_value = []
        shared_state.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))

        pipeline = Pipeline(self._SC_CONFIG, shared_state, None, detector, de,
                            sc, face_recognizer=fr)
        pipeline.start()
        triggered = swap_event.wait(timeout=3)
        pipeline.stop()

        assert triggered, "session_controller.reset() was never called — swap not detected"

    def test_none_opening_identity_does_not_invalidate(self):
        """If opening check returns None, closing check cannot trigger a swap."""
        fr = self._make_fr_stub([
            None,              # opening check: face not visible
            ("Bob", 28.0),    # closing check: somebody identified
        ])
        final_state, ss = self._run_session(fr, compliant=True)
        # No invalidation — None on opening side
        assert final_state is SessionState.GRANTED

    def test_none_closing_identity_does_not_invalidate(self):
        """If closing check returns None, no invalidation even if opening was confident."""
        fr = self._make_fr_stub([
            ("Alice", 30.0),   # opening check: confident
            None,              # closing check: face not visible
        ])
        final_state, ss = self._run_session(fr, compliant=True)
        assert final_state is SessionState.GRANTED

    def test_both_none_does_not_invalidate(self):
        """Two None results do not trigger a swap."""
        fr = self._make_fr_stub([None, None])
        final_state, ss = self._run_session(fr, compliant=True)
        assert final_state is SessionState.GRANTED

    def test_worker_name_cleared_on_swap(self):
        """After swap invalidation, set_worker_name('unknown') is called."""
        # Spy on shared_state.set_worker_name; collect every value it's called with.
        # After a swap the pipeline must call it with "unknown" before anything else.
        unknown_event = threading.Event()
        shared_state = SharedState()
        original_set = shared_state.set_worker_name
        set_calls: list[str] = []

        def _spy_set(name: str) -> None:
            original_set(name)
            set_calls.append(name)
            if name == "unknown":
                unknown_event.set()

        shared_state.set_worker_name = _spy_set

        fake_clock = [0.0]
        sc = SessionController(self._SC_CONFIG, clock=lambda: fake_clock[0])
        fr = self._make_fr_stub([("Alice", 30.0), ("Bob", 28.0)])

        verdict_seq = [
            FrameVerdict(status=ComplianceStatus.NO_PERSON),
            *(FrameVerdict(status=ComplianceStatus.COMPLIANT, person_count=1),) * 100,
        ]
        de = MagicMock()
        de.evaluate.side_effect = verdict_seq
        detector = MagicMock()
        detector.detect.return_value = []
        shared_state.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))

        pipeline = Pipeline(self._SC_CONFIG, shared_state, None, detector, de,
                            sc, face_recognizer=fr)
        pipeline.start()
        triggered = unknown_event.wait(timeout=3)
        pipeline.stop()

        assert triggered, "set_worker_name('unknown') was never called after swap"
        # The call sequence must have been: "Alice" (opening), then "unknown" (swap clear).
        assert "Alice" in set_calls
        unknown_idx = next(i for i, v in enumerate(set_calls) if v == "unknown")
        alice_idx = set_calls.index("Alice")
        assert alice_idx < unknown_idx, "worker_name should be set to Alice before being cleared"

    def test_no_face_recognizer_still_completes_session(self):
        """face_recognizer=None disables identity checks; session completes normally."""
        fake_clock = [0.0]
        sc = SessionController(self._SC_CONFIG, clock=lambda: fake_clock[0])
        shared_state = SharedState()
        verdict_seq = [
            FrameVerdict(status=ComplianceStatus.NO_PERSON),
            *(FrameVerdict(status=ComplianceStatus.COMPLIANT, person_count=1),) * 10,
        ]
        de = MagicMock()
        de.evaluate.side_effect = verdict_seq

        done_event = threading.Event()
        frame_count = [0]

        def _detect(frame):
            frame_count[0] += 1
            if frame_count[0] >= 5:
                done_event.set()
            return []

        detector = MagicMock()
        detector.detect.side_effect = _detect
        shared_state.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))

        pipeline = Pipeline(self._SC_CONFIG, shared_state, None, detector, de,
                            sc, face_recognizer=None)
        pipeline.start()
        done_event.wait(timeout=3)
        time.sleep(0.05)
        pipeline.stop()

        assert shared_state.get_session_state() is SessionState.GRANTED


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
