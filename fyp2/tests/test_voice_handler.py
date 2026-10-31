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
from core.zone_monitor import HAZARD_PPE, RESTRICTED_ENTRY, ZoneEvent
from core.decision_engine import ComplianceStatus, FrameVerdict
from core.session_controller import SessionState

GRANTED = SessionState.GRANTED
DENIED  = SessionState.DENIED
IDLE    = SessionState.IDLE
EVALUATING = SessionState.EVALUATING

COMPLIANT     = FrameVerdict(status=ComplianceStatus.COMPLIANT, person_count=1)
NON_COMPLIANT = FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                              violations=["no-helmet"], person_count=1)

# Popen mock whose process has already finished (poll() returns an exit
# code), so the "another prompt is still playing" check never blocks.
_FINISHED = {"return_value.poll.return_value": 0}


def _voice_config(tmp_path, create=True):
    names = {"granted_sound": "granted.wav", "denied_sound": "denied.wav",
             "denied_no_helmet_sound": "denied_no_helmet.wav",
             "denied_no_vest_sound": "denied_no_vest.wav",
             "hazard_sound": "hazard_ppe.wav",
             "hazard_no_helmet_sound": "hazard_no_helmet.wav",
             "hazard_no_vest_sound": "hazard_no_vest.wav",
             "restricted_sound": "restricted.wav"}
    cfg = {}
    for key, name in names.items():
        path = tmp_path / name
        if create:
            path.write_bytes(b"RIFF")  # minimal stub — content not checked by handler
        cfg[key] = str(path)
    return {"voice": cfg}


@pytest.fixture
def handler(tmp_path):
    """VoiceHandler with real placeholder WAV files that exist on disk."""
    return VoiceHandler(_voice_config(tmp_path))


# ---------------------------------------------------------------------------
# Debounce — fires once per state entry
# ---------------------------------------------------------------------------

class TestDebounce:
    def test_fires_once_on_granted_entry(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(GRANTED, COMPLIANT, None)
        mock_popen.assert_called_once()

    def test_fires_once_on_denied_entry(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(DENIED, NON_COMPLIANT, None)
        mock_popen.assert_called_once()

    def test_does_not_fire_again_on_subsequent_granted_frames(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(GRANTED, COMPLIANT, None)  # entry — fires
            handler.handle(GRANTED, COMPLIANT, None)  # same hold — debounce
            handler.handle(GRANTED, COMPLIANT, None)
        mock_popen.assert_called_once()

    def test_does_not_fire_again_on_subsequent_denied_frames(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            for _ in range(5):
                handler.handle(DENIED, NON_COMPLIANT, None)
        mock_popen.assert_called_once()

    def test_fires_again_after_state_resets(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(GRANTED, COMPLIANT, None)            # fires (1)
            handler.handle(SessionState.COOLDOWN, COMPLIANT, None)  # reset
            handler.handle(IDLE, COMPLIANT, None)               # reset
            handler.handle(GRANTED, COMPLIANT, None)            # fires (2)
        assert mock_popen.call_count == 2

    def test_denied_then_granted_fires_twice(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(DENIED, NON_COMPLIANT, None)             # fires (1)
            handler.handle(SessionState.COOLDOWN, COMPLIANT, None)
            handler.handle(IDLE, COMPLIANT, None)
            handler.handle(GRANTED, COMPLIANT, None)                # fires (2)
        assert mock_popen.call_count == 2


# ---------------------------------------------------------------------------
# Non-trigger states never fire
# ---------------------------------------------------------------------------

class TestNonTriggerStates:
    def test_non_trigger_states_do_not_play(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            for state in (IDLE, EVALUATING, SessionState.PERSON_DETECTED,
                          SessionState.COOLDOWN):
                handler.handle(state, COMPLIANT, None)
        mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# Correct sound file per state
# ---------------------------------------------------------------------------

class TestCorrectSoundPlayed:
    def test_granted_plays_granted_sound(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(GRANTED, COMPLIANT, None)
        args = mock_popen.call_args[0][0]  # first positional arg: the command list
        assert "granted" in args[-1]

    def test_denied_plays_denied_sound(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(DENIED, NON_COMPLIANT, None)
        args = mock_popen.call_args[0][0]
        assert "denied" in args[-1]


# ---------------------------------------------------------------------------
# Missing audio file — graceful failure
# ---------------------------------------------------------------------------

class TestMissingAudioFile:
    def test_missing_file_does_not_crash(self, tmp_path):
        config = _voice_config(tmp_path, create=False)
        handler = VoiceHandler(config)  # must not raise

        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(GRANTED, COMPLIANT, None)  # must not raise
        mock_popen.assert_not_called()

    def test_missing_file_logs_warning_at_construction(self, tmp_path, caplog):
        config = _voice_config(tmp_path, create=False)
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
            handler.handle(GRANTED, COMPLIANT, None)

        mock_process.wait.assert_not_called()
        mock_process.communicate.assert_not_called()


# ---------------------------------------------------------------------------
# Room zone events
# ---------------------------------------------------------------------------

def _event(event_type):
    return ZoneEvent(event_type=event_type, zone="Z", track_id=1)


class TestZoneEvents:
    def test_hazard_event_plays_hazard_sound(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle_zone_event(_event(HAZARD_PPE), None)
        assert "hazard_ppe" in mock_popen.call_args[0][0][-1]

    def test_restricted_event_plays_restricted_sound(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle_zone_event(_event(RESTRICTED_ENTRY), None)
        assert "restricted" in mock_popen.call_args[0][0][-1]


class TestOverlap:
    def test_new_prompt_skipped_while_one_is_playing(self, handler):
        playing = {"return_value.poll.return_value": None}  # still running
        with patch("alerts.voice_handler.subprocess.Popen", **playing) as mock_popen:
            handler.handle(DENIED, NON_COMPLIANT, None)
            handler.handle_zone_event(_event(HAZARD_PPE), None)
        mock_popen.assert_called_once()

    def test_restricted_interrupts_current_prompt(self, handler):
        playing = {"return_value.poll.return_value": None}
        with patch("alerts.voice_handler.subprocess.Popen", **playing) as mock_popen:
            handler.handle(DENIED, NON_COMPLIANT, None)
            handler.handle_zone_event(_event(RESTRICTED_ENTRY), None)
        assert mock_popen.call_count == 2
        mock_popen.return_value.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# Prompt names only the missing item
# ---------------------------------------------------------------------------

def _verdict(*violations):
    return FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                        violations=list(violations), person_count=1)


def _played(mock_popen):
    return mock_popen.call_args[0][0][-1]


class TestMissingItemPrompt:
    def test_denied_only_helmet(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(DENIED, _verdict("no-helmet"), None)
        assert _played(mock_popen).endswith("denied_no_helmet.wav")

    def test_denied_only_vest(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(DENIED, _verdict("no-vest"), None)
        assert _played(mock_popen).endswith("denied_no_vest.wav")

    def test_denied_both_uses_general_prompt(self, handler):
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle(DENIED, _verdict("no-helmet", "no-vest"), None)
        assert _played(mock_popen).endswith("/denied.wav")

    def test_hazard_only_vest(self, handler):
        ev = ZoneEvent(event_type=HAZARD_PPE, zone="Z", track_id=1, violations=["no-vest"])
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            handler.handle_zone_event(ev, None)
        assert _played(mock_popen).endswith("hazard_no_vest.wav")

    def test_per_item_prompt_optional(self, tmp_path):
        cfg = _voice_config(tmp_path)
        del cfg["voice"]["denied_no_helmet_sound"]
        h = VoiceHandler(cfg)
        with patch("alerts.voice_handler.subprocess.Popen", **_FINISHED) as mock_popen:
            h.handle(DENIED, _verdict("no-helmet"), None)
        assert _played(mock_popen).endswith("/denied.wav")
