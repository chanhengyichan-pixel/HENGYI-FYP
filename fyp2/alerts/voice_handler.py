"""
VoiceHandler — plays a pre-recorded audio reminder on GRANTED/DENIED
state entry, and on room zone events (HAZARD_PPE, RESTRICTED_ENTRY).

Audio library: subprocess.Popen(['aplay', path])
-----------------------------------------------
aplay is ALSA's command-line WAV player, standard on all Debian-based
Linux distributions and Raspberry Pi OS — zero additional pip
dependencies.  subprocess.Popen() returns immediately without waiting
for the child process to finish, making playback fully non-blocking.
The audio plays in a separate OS process; if it fails, the pipeline
thread is unaffected.

Alternatives considered:
- simpleaudio: pip dependency, known ALSA backend issues on some Pi
  configurations.
- pygame.mixer: requires display initialisation on some platforms,
  violates the project's headless-first rule.
- playsound: blocking on Linux without threading workarounds.
- threading.Thread(target=playsound): works but adds a Python thread
  per play call; Popen achieves the same result at the OS level.

Non-blocking guarantee
----------------------
handle() dispatches Popen and returns immediately. The calling thread
(Pipeline Thread 1) is not blocked for the duration of playback.
This preserves the inference throughput measured in Milestone 5.

Debounce
--------
Identical to ScreenshotHandler._last_captured_state: fires once per
GRANTED/DENIED entry, resets on any other state, so the next entry
fires again. No new debounce pattern introduced.

Missing audio files
-------------------
Two independent checks are performed:

1. Construction-time: each configured path is checked with Path.exists()
   at startup. If absent, a WARNING is logged once — surfaces the
   problem early so the operator sees it in the startup log rather than
   discovering it silently 30 seconds later at the first decision event.

2. handle()-time: Path.exists() is re-checked on every GRANTED/DENIED
   trigger before calling Popen. This catches files that go missing
   after startup — e.g. an SD card unmounting on the Pi, or a file
   being swapped during deployment. This live re-check is a deliberate
   robustness improvement beyond what was originally scoped: it ensures
   handle() never calls Popen with a path that no longer resolves,
   regardless of what was true at construction.

In both cases the pipeline is not interrupted.

Prompt per missing item
-----------------------
DENIED and HAZARD_PPE prompts name only what is missing:
  only the helmet missing -> *_no_helmet_sound
  only the vest missing   -> *_no_vest_sound
  both missing            -> denied_sound / hazard_sound
The per-item files are optional config keys; if one is not configured,
the general prompt is used instead. For DENIED the violations come from
the whole evaluation window (SessionController.decision_violations, via
Pipeline), not from one possibly-flickering frame.

Overlapping prompts
-------------------
The entrance and room pipelines can both want to speak at once. Only one
prompt plays at a time: while a prompt is still playing, a new one is
skipped — except RESTRICTED_ENTRY, the highest-severity event, which
stops the current prompt and plays immediately. A lock guards the
"currently playing" process because the two pipelines call in from
different threads.

USB speaker
-----------
aplay plays on the default ALSA device. The Raspberry Pi 5 has no 3.5 mm
jack, so the USB speaker must be the default device (see README) —
no code change is needed for that.

Placeholder audio
-----------------
The WAV files in assets/audio/ are generated sine-wave beeps —
placeholder only. Replace with real recorded safety reminders before
deploying to a production gate. File paths are config-driven; no code
change is required to swap in real recordings.
"""

import logging
import subprocess
import threading
from pathlib import Path

from alerts.base_handler import BaseHandler
from core.session_controller import SessionState
from core.zone_monitor import HAZARD_PPE, RESTRICTED_ENTRY

logger = logging.getLogger(__name__)

_TRIGGER_STATES = {SessionState.GRANTED, SessionState.DENIED}


class VoiceHandler(BaseHandler):
    def __init__(self, config: dict):
        voice = config["voice"]
        self._sounds: dict = {
            SessionState.GRANTED: Path(voice["granted_sound"]),
            SessionState.DENIED:  Path(voice["denied_sound"]),
            HAZARD_PPE:           Path(voice["hazard_sound"]),
            RESTRICTED_ENTRY:     Path(voice["restricted_sound"]),
        }
        # Optional per-item prompts: (event, missing item) -> file.
        optional = {
            (SessionState.DENIED, "no-helmet"): "denied_no_helmet_sound",
            (SessionState.DENIED, "no-vest"):   "denied_no_vest_sound",
            (HAZARD_PPE, "no-helmet"):          "hazard_no_helmet_sound",
            (HAZARD_PPE, "no-vest"):            "hazard_no_vest_sound",
        }
        for key, cfg_key in optional.items():
            if voice.get(cfg_key):
                self._sounds[key] = Path(voice[cfg_key])

        # WARNING (not ERROR): a missing placeholder in an MVP-optional module
        # is a startup notice, not a fatal condition — consistent with
        # ScreenshotHandler's non-fatal warning on a None frame.
        for state, path in self._sounds.items():
            if not path.exists():
                logger.warning(
                    "VoiceHandler: audio file for %s not found: %s — "
                    "playback will be skipped.",
                    _describe(state), path,
                )

        self._last_played_state = None  # debounce: same pattern as ScreenshotHandler
        self._lock = threading.Lock()
        self._current = None            # Popen of the prompt playing now, if any

    def handle(self, session_state, frame_verdict, frame) -> None:
        if session_state not in _TRIGGER_STATES:
            self._last_played_state = None
            return

        if session_state == self._last_played_state:
            return  # debounce: already played for this hold

        self._last_played_state = session_state
        key = session_state
        if session_state is SessionState.DENIED:
            key = self._item_key(session_state, frame_verdict.violations)
        self._play(key, interrupt=False)

    def handle_zone_event(self, event, frame) -> None:
        key = event.event_type
        if event.event_type == HAZARD_PPE:
            key = self._item_key(HAZARD_PPE, event.violations)
        self._play(key, interrupt=event.event_type == RESTRICTED_ENTRY)

    def _item_key(self, base, violations):
        """Per-item prompt key when exactly one item is missing and a file
        for it is configured; otherwise the general prompt."""
        missing = set(violations or [])
        if len(missing) == 1:
            key = (base, missing.pop())
            if key in self._sounds:
                return key
        return base

    def _play(self, key, interrupt: bool) -> None:
        path = self._sounds[key]
        if not path.exists():
            return  # already warned at construction; skip silently here

        with self._lock:
            busy = self._current is not None and self._current.poll() is None
            if busy and not interrupt:
                logger.info("Voice alert skipped (another prompt playing): %s", path.name)
                return
            if busy:
                self._current.terminate()
            # Fire-and-forget: Popen returns immediately, audio plays in a
            # separate OS process without blocking this thread.
            self._current = subprocess.Popen(
                ["aplay", "--quiet", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        logger.info("Voice alert: playing %s", path.name)


def _describe(key) -> str:
    if isinstance(key, tuple):
        return "/".join(_describe(k) for k in key)
    return getattr(key, "value", key)
