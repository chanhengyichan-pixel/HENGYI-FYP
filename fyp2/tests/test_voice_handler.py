"""
Unit tests for alerts/voice_handler.py.

No real audio device required — subprocess.Popen is patched throughout.
"""

import logging
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from alerts.voice_handler import VoiceHandler
from core.decision_engine import ComplianceStatus, FrameVerdict
from core.session_controller import SessionState

GRANTED = SessionState.GRANTED
DENIED  = SessionState.DENIED
IDLE    = SessionState.IDLE
EVALUATING = SessionState.EVALUATING

COMPLIANT     = FrameVerdict(status=ComplianceStatus.COMPLIANT, person_count=1)
NON_COMPLIANT = FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                              violations=["no-helmet"], person_count=1)


@pytest.fixture
def handler(tmp_path):
    """VoiceHandler with real placeholder WAV files that exist on disk."""
    granted = tmp_path / "granted.wav"
    denied  = tmp_path / "denied.wav"
    granted.write_bytes(b"RIFF")  # minimal stub — content not checked by handler
    denied.write_bytes(b"RIFF")
    config = {"voice": {"granted_sound": str(granted), "denied_sound": str(denied)}}
    return VoiceHandler(config)


# ---------------------------------------------------------------------------
# Debounce — fires once per state entry
# ---------------------------------------------------------------------------

class TestDebounce:
    def test_fires_once_on_granted_entry(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen") as mock_popen:
            handler.handle(GRANTED, COMPLIANT, None, None)
        mock_popen.assert_called_once()

    def test_fires_once_on_denied_entry(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen") as mock_popen:
            handler.handle(DENIED, NON_COMPLIANT, None, None)
        mock_popen.assert_called_once()

    def test_does_not_fire_again_on_subsequent_granted_frames(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen") as mock_popen:
            handler.handle(GRANTED, COMPLIANT, None, None)  # entry — fires
            handler.handle(GRANTED, COMPLIANT, None, None)  # same hold — debounce
            handler.handle(GRANTED, COMPLIANT, None, None)
        mock_popen.assert_called_once()

    def test_does_not_fire_again_on_subsequent_denied_frames(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen") as mock_popen:
            for _ in range(5):
                handler.handle(DENIED, NON_COMPLIANT, None, None)
        mock_popen.assert_called_once()

    def test_fires_again_after_state_resets(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen") as mock_popen:
            handler.handle(GRANTED, COMPLIANT, None, None)            # fires (1)
            handler.handle(SessionState.COOLDOWN, COMPLIANT, None, None)  # reset
            handler.handle(IDLE, COMPLIANT, None, None)               # reset
            handler.handle(GRANTED, COMPLIANT, None, None)            # fires (2)
        assert mock_popen.call_count == 2

    def test_denied_then_granted_fires_twice(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen") as mock_popen:
            handler.handle(DENIED, NON_COMPLIANT, None, None)             # fires (1)
            handler.handle(SessionState.COOLDOWN, COMPLIANT, None, None)
            handler.handle(IDLE, COMPLIANT, None, None)
            handler.handle(GRANTED, COMPLIANT, None, None)                # fires (2)
        assert mock_popen.call_count == 2


# ---------------------------------------------------------------------------
# Non-trigger states never fire
# ---------------------------------------------------------------------------

class TestNonTriggerStates:
    def test_non_trigger_states_do_not_play(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen") as mock_popen:
            for state in (IDLE, EVALUATING, SessionState.PERSON_DETECTED,
                          SessionState.COOLDOWN):
                handler.handle(state, COMPLIANT, None, None)
        mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# Correct sound file per state
# ---------------------------------------------------------------------------

class TestCorrectSoundPlayed:
    def test_granted_plays_granted_sound(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen") as mock_popen:
            handler.handle(GRANTED, COMPLIANT, None, None)
        args = mock_popen.call_args[0][0]  # first positional arg: the command list
        assert "granted" in args[-1]

    def test_denied_plays_denied_sound(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen") as mock_popen:
            handler.handle(DENIED, NON_COMPLIANT, None, None)
        args = mock_popen.call_args[0][0]
        assert "denied" in args[-1]


# ---------------------------------------------------------------------------
# Missing audio file — graceful failure
# ---------------------------------------------------------------------------

class TestMissingAudioFile:
    def test_missing_file_does_not_crash(self, tmp_path):
        config = {"voice": {
            "granted_sound": str(tmp_path / "nonexistent_granted.wav"),
            "denied_sound":  str(tmp_path / "nonexistent_denied.wav"),
        }}
        handler = VoiceHandler(config)  # must not raise

        with patch("alerts.voice_handler.subprocess.Popen") as mock_popen:
            handler.handle(GRANTED, COMPLIANT, None, None)  # must not raise
        mock_popen.assert_not_called()

    def test_missing_file_logs_warning_at_construction(self, tmp_path, caplog):
        config = {"voice": {
            "granted_sound": str(tmp_path / "missing.wav"),
            "denied_sound":  str(tmp_path / "also_missing.wav"),
        }}
        with caplog.at_level(logging.WARNING, logger="alerts.voice_handler"):
            VoiceHandler(config)
        assert any("not found" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Non-blocking — handle() must return quickly
# ---------------------------------------------------------------------------

class TestNonBlocking:
    def test_handle_does_not_wait_on_subprocess(self, handler):
        """
        Confirm that handle() never calls .wait() or .communicate() on the
        Popen result — the only mechanism that would cause the pipeline thread
        to block for the duration of audio playback.

        The non-blocking guarantee is that Popen is called and its return
        value is discarded; no blocking join is performed.
        """
        mock_process = MagicMock()

        with patch("alerts.voice_handler.subprocess.Popen", return_value=mock_process):
            handler.handle(GRANTED, COMPLIANT, None, None)

        mock_process.wait.assert_not_called()
        mock_process.communicate.assert_not_called()
