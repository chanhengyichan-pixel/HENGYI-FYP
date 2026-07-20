"""
Unit tests for alerts/screenshot_handler.py.

No camera, no real model.  Frames are minimal numpy arrays.
cv2.imwrite is patched so tests don't touch the filesystem.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, call

from alerts.screenshot_handler import ScreenshotHandler
from core.decision_engine import ComplianceStatus, FrameVerdict
from core.session_controller import SessionState

GRANTED = SessionState.GRANTED
DENIED = SessionState.DENIED
EVALUATING = SessionState.EVALUATING
IDLE = SessionState.IDLE

COMPLIANT_VERDICT = FrameVerdict(status=ComplianceStatus.COMPLIANT, person_count=1)
NON_COMPLIANT_VERDICT = FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                                      violations=["no-helmet"], person_count=1)
FAKE_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

BASE_CONFIG = {"alerts": {"screenshot_dir": "/tmp/test_screenshots"}}


@pytest.fixture
def handler(tmp_path):
    config = {"alerts": {"screenshot_dir": str(tmp_path)}}
    return ScreenshotHandler(config)


class TestFiresOnce:
    def test_fires_on_granted_entry(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(GRANTED, COMPLIANT_VERDICT, FAKE_FRAME, None)
        mock_write.assert_called_once()

    def test_fires_on_denied_entry(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(DENIED, NON_COMPLIANT_VERDICT, FAKE_FRAME, None)
        mock_write.assert_called_once()

    def test_does_not_fire_again_on_second_granted_frame(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(GRANTED, COMPLIANT_VERDICT, FAKE_FRAME, None)  # entry → fires
            handler.handle(GRANTED, COMPLIANT_VERDICT, FAKE_FRAME, None)  # same hold → debounce
        mock_write.assert_called_once()

    def test_does_not_fire_again_on_subsequent_denied_frames(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            for _ in range(5):
                handler.handle(DENIED, NON_COMPLIANT_VERDICT, FAKE_FRAME, None)
        mock_write.assert_called_once()


class TestDebounceReset:
    def test_fires_again_after_state_leaves_and_re_enters_granted(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(GRANTED, COMPLIANT_VERDICT, FAKE_FRAME, None)   # fires (1)
            handler.handle(COOLDOWN := SessionState.COOLDOWN, COMPLIANT_VERDICT, FAKE_FRAME, None)  # reset
            handler.handle(IDLE, COMPLIANT_VERDICT, FAKE_FRAME, None)       # reset
            handler.handle(GRANTED, COMPLIANT_VERDICT, FAKE_FRAME, None)   # fires again (2)
        assert mock_write.call_count == 2

    def test_denied_then_granted_fires_twice(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(DENIED, NON_COMPLIANT_VERDICT, FAKE_FRAME, None)   # fires (1)
            handler.handle(COOLDOWN := SessionState.COOLDOWN, COMPLIANT_VERDICT, FAKE_FRAME, None)
            handler.handle(IDLE, COMPLIANT_VERDICT, FAKE_FRAME, None)
            handler.handle(GRANTED, COMPLIANT_VERDICT, FAKE_FRAME, None)      # fires (2)
        assert mock_write.call_count == 2

    def test_non_trigger_states_never_fire(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            for state in (IDLE, EVALUATING, SessionState.PERSON_DETECTED,
                          SessionState.COOLDOWN):
                handler.handle(state, COMPLIANT_VERDICT, FAKE_FRAME, None)
        mock_write.assert_not_called()


class TestFilename:
    def test_filename_contains_verdict_label(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(DENIED, NON_COMPLIANT_VERDICT, FAKE_FRAME, None)
        saved_path = mock_write.call_args[0][0]
        assert "denied" in saved_path

    def test_filename_contains_verdict_label_granted(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(GRANTED, COMPLIANT_VERDICT, FAKE_FRAME, None)
        saved_path = mock_write.call_args[0][0]
        assert "granted" in saved_path

    def test_filename_ends_with_jpg(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(GRANTED, COMPLIANT_VERDICT, FAKE_FRAME, None)
        saved_path = mock_write.call_args[0][0]
        assert saved_path.endswith(".jpg")

    def test_correct_frame_passed_to_imwrite(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(GRANTED, COMPLIANT_VERDICT, FAKE_FRAME, None)
        frame_arg = mock_write.call_args[0][1]
        assert (frame_arg == FAKE_FRAME).all()


class TestNoneFrame:
    def test_none_frame_does_not_crash(self, handler):
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(GRANTED, COMPLIANT_VERDICT, None, None)  # must not raise
        mock_write.assert_not_called()

    def test_none_frame_still_debounces(self, handler):
        """Even with no frame, the state is consumed so no retry spam on next frames."""
        with patch("alerts.screenshot_handler.cv2.imwrite") as mock_write:
            handler.handle(GRANTED, COMPLIANT_VERDICT, None, None)   # skipped (no frame)
            handler.handle(GRANTED, COMPLIANT_VERDICT, FAKE_FRAME, None)  # debounced — NOT saved
        mock_write.assert_not_called()
