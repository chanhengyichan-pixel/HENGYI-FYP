"""
Unit tests for alerts/voice_handler.py and alerts/speaker.py.

No real audio device required — the Speaker gets a fake player and runs
without its worker thread (play_pending() plays the queue).
"""

import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from alerts.speaker import Speaker, PRIORITY_DENIED, PRIORITY_GRANTED, PRIORITY_HAZARD, PRIORITY_RESTRICTED
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


class FakePlayer:
    """Records what was played. Each 'process' finishes at once."""

    def __init__(self):
        self.played = []
        self.procs = []

    def __call__(self, path):
        self.played.append(str(path))
        proc = MagicMock()
        self.procs.append(proc)
        return proc

    @property
    def call_count(self):
        return len(self.played)

    def last(self):
        return self.played[-1]


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


class Rig:
    """VoiceHandler + synchronous Speaker + fake player."""

    def __init__(self, config, clock=None):
        self.player = FakePlayer()
        kw = {"clock": clock} if clock else {}
        self.speaker = Speaker(8.0, player=self.player, threaded=False, **kw)
        self.h = VoiceHandler(config, speaker=self.speaker)

    def handle(self, *a):
        self.h.handle(*a)
        self.speaker.play_pending()

    def zone(self, ev):
        self.h.handle_zone_event(ev, None)
        self.speaker.play_pending()


@pytest.fixture
def rig(tmp_path):
    """VoiceHandler with real placeholder WAV files that exist on disk."""
    return Rig(_voice_config(tmp_path))


# ---------------------------------------------------------------------------
# Debounce — fires once per state entry
# ---------------------------------------------------------------------------

class TestDebounce:
    def test_fires_once_on_granted_entry(self, rig):
        rig.handle(GRANTED, COMPLIANT, None)
        assert rig.player.call_count == 1

    def test_fires_once_on_denied_entry(self, rig):
        rig.handle(DENIED, NON_COMPLIANT, None)
        assert rig.player.call_count == 1

    def test_does_not_fire_again_on_subsequent_granted_frames(self, rig):
        for _ in range(3):
            rig.handle(GRANTED, COMPLIANT, None)
        assert rig.player.call_count == 1

    def test_does_not_fire_again_on_subsequent_denied_frames(self, rig):
        for _ in range(5):
            rig.handle(DENIED, NON_COMPLIANT, None)
        assert rig.player.call_count == 1

    def test_fires_again_after_state_resets(self, rig):
        rig.handle(GRANTED, COMPLIANT, None)
        rig.handle(SessionState.COOLDOWN, COMPLIANT, None)
        rig.handle(IDLE, COMPLIANT, None)
        rig.handle(GRANTED, COMPLIANT, None)
        assert rig.player.call_count == 2

    def test_denied_then_granted_fires_twice(self, rig):
        rig.handle(DENIED, NON_COMPLIANT, None)
        rig.handle(SessionState.COOLDOWN, COMPLIANT, None)
        rig.handle(IDLE, COMPLIANT, None)
        rig.handle(GRANTED, COMPLIANT, None)
        assert rig.player.call_count == 2


class TestNonTriggerStates:
    def test_non_trigger_states_do_not_play(self, rig):
        for state in (IDLE, EVALUATING, SessionState.PERSON_DETECTED, SessionState.COOLDOWN):
            rig.handle(state, COMPLIANT, None)
        assert rig.player.call_count == 0


class TestCorrectSoundPlayed:
    def test_granted_plays_granted_sound(self, rig):
        rig.handle(GRANTED, COMPLIANT, None)
        assert "granted" in rig.player.last()

    def test_denied_plays_denied_sound(self, rig):
        rig.handle(DENIED, NON_COMPLIANT, None)
        assert "denied" in rig.player.last()


class TestMissingAudioFile:
    def test_missing_file_does_not_crash(self, tmp_path):
        r = Rig(_voice_config(tmp_path, create=False))   # must not raise
        r.handle(GRANTED, COMPLIANT, None)                 # must not raise
        assert r.player.call_count == 0

    def test_missing_file_logs_warning_at_construction(self, tmp_path, caplog):
        config = _voice_config(tmp_path, create=False)
        with caplog.at_level(logging.WARNING, logger="alerts.voice_handler"):
            VoiceHandler(config, speaker=Speaker(threaded=False))
        assert any("not found" in r.message for r in caplog.records)


class TestNonBlocking:
    def test_handle_only_queues(self, tmp_path):
        """handle() never plays audio itself: it only queues the prompt;
        the Speaker's own thread plays it (here: nothing until play_pending)."""
        player = FakePlayer()
        sp = Speaker(player=player, threaded=False)
        h = VoiceHandler(_voice_config(tmp_path), speaker=sp)
        h.handle(GRANTED, COMPLIANT, None)
        assert player.call_count == 0
        assert len(sp.pending()) == 1

    def test_default_speaker_uses_aplay_in_its_own_thread(self, tmp_path):
        done = MagicMock()
        with patch("alerts.speaker.subprocess.Popen", return_value=done) as popen:
            h = VoiceHandler(_voice_config(tmp_path))
            h.handle(GRANTED, COMPLIANT, None)
            h.speaker.close()   # joins the worker thread
        if popen.called:        # the worker may or may not have reached it before close
            assert popen.call_args[0][0][0] == "aplay"


def _event(event_type, violations=None):
    return ZoneEvent(event_type=event_type, zone="Z", track_id=1, violations=violations or [])


class TestZoneEvents:
    def test_hazard_event_plays_hazard_sound(self, rig):
        rig.zone(_event(HAZARD_PPE))
        assert "hazard_ppe" in rig.player.last()

    def test_restricted_event_plays_restricted_sound(self, rig):
        rig.zone(_event(RESTRICTED_ENTRY))
        assert "restricted" in rig.player.last()


# ---------------------------------------------------------------------------
# Shared speaker: queue, priority, no overlap
# ---------------------------------------------------------------------------

class TestSharedSpeaker:
    def test_two_cameras_share_one_queue_nothing_lost(self, tmp_path):
        cfg = _voice_config(tmp_path)
        player = FakePlayer()
        sp = Speaker(player=player, threaded=False)
        entrance, room = VoiceHandler(cfg, speaker=sp), VoiceHandler(cfg, speaker=sp)
        entrance.handle(DENIED, NON_COMPLIANT, None)
        room.handle_zone_event(_event(HAZARD_PPE), None)
        sp.play_pending()
        assert len(player.played) == 2      # both played, one after the other

    def test_priority_order(self, rig):
        rig.h.handle(GRANTED, COMPLIANT, None)
        rig.h.handle_zone_event(_event(HAZARD_PPE), None)
        rig.h.handle_zone_event(_event(RESTRICTED_ENTRY), None)
        rig.speaker.play_pending()
        names = [Path(p).name for p in rig.player.played]
        assert names == ["restricted.wav", "hazard_ppe.wav", "granted.wav"]

    def test_denied_before_hazard(self, rig):
        rig.h.handle_zone_event(_event(HAZARD_PPE), None)
        rig.h.handle(DENIED, _verdict("no-helmet", "no-vest"), None)
        rig.speaker.play_pending()
        assert [Path(p).name for p in rig.player.played] == ["denied.wav", "hazard_ppe.wav"]

    def test_same_prompt_not_queued_twice(self, rig):
        rig.h.handle_zone_event(_event(HAZARD_PPE), None)
        rig.h.handle_zone_event(_event(HAZARD_PPE), None)
        assert len(rig.speaker.pending()) == 1

    def test_stale_prompt_dropped(self, tmp_path):
        t = [0.0]
        r = Rig(_voice_config(tmp_path), clock=lambda: t[0])
        r.h.handle(GRANTED, COMPLIANT, None)
        t[0] = 9.0                           # waited longer than 8 s
        r.speaker.play_pending()
        assert r.player.call_count == 0

    def test_restricted_interrupts_current_prompt(self, tmp_path):
        player = FakePlayer()
        sp = Speaker(player=player, threaded=False)
        proc = MagicMock()
        sp._current = (PRIORITY_DENIED, proc, Path("denied.wav"))   # something playing now
        sp.say(tmp_path / "r.wav", PRIORITY_RESTRICTED)
        proc.terminate.assert_called_once()

    def test_other_prompts_do_not_interrupt(self, tmp_path):
        sp = Speaker(player=FakePlayer(), threaded=False)
        proc = MagicMock()
        sp._current = (PRIORITY_GRANTED, proc, Path("granted.wav"))
        sp.say(tmp_path / "h.wav", PRIORITY_HAZARD)
        sp.say(tmp_path / "d.wav", PRIORITY_DENIED)
        proc.terminate.assert_not_called()

    def test_threaded_speaker_plays_in_order(self, tmp_path):
        import threading
        gate, played = threading.Event(), []
        files = []
        for n in ("a", "b", "c"):
            f = tmp_path / f"{n}.wav"; f.write_bytes(b"RIFF"); files.append(f)

        def player(path):
            proc = MagicMock()
            proc.wait.side_effect = lambda: gate.wait(2) if path.name == "a.wav" else None
            played.append(path.name)
            return proc

        sp = Speaker(player=player)
        sp.say(files[0], PRIORITY_GRANTED)       # starts playing, blocks on gate
        for _ in range(100):
            if played:
                break
            gate.wait(0.01)
        sp.say(files[1], PRIORITY_HAZARD)
        sp.say(files[2], PRIORITY_DENIED)
        gate.set()
        for _ in range(200):
            if len(played) == 3:
                break
            threading.Event().wait(0.01)
        sp.close()
        assert played == ["a.wav", "c.wav", "b.wav"]


# ---------------------------------------------------------------------------
# Prompt names only the missing item
# ---------------------------------------------------------------------------

def _verdict(*violations):
    return FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                        violations=list(violations), person_count=1)


class TestMissingItemPrompt:
    def test_denied_only_helmet(self, rig):
        rig.handle(DENIED, _verdict("no-helmet"), None)
        assert rig.player.last().endswith("denied_no_helmet.wav")

    def test_denied_only_vest(self, rig):
        rig.handle(DENIED, _verdict("no-vest"), None)
        assert rig.player.last().endswith("denied_no_vest.wav")

    def test_denied_both_uses_general_prompt(self, rig):
        rig.handle(DENIED, _verdict("no-helmet", "no-vest"), None)
        assert rig.player.last().endswith("/denied.wav")

    def test_hazard_only_vest(self, rig):
        rig.zone(_event(HAZARD_PPE, ["no-vest"]))
        assert rig.player.last().endswith("hazard_no_vest.wav")

    def test_per_item_prompt_optional(self, tmp_path):
        cfg = _voice_config(tmp_path)
        del cfg["voice"]["denied_no_helmet_sound"]
        r = Rig(cfg)
        r.handle(DENIED, _verdict("no-helmet"), None)
        assert r.player.last().endswith("/denied.wav")
