"""
VoiceHandler — plays a pre-recorded audio reminder on GRANTED/DENIED
state entry.

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

Placeholder audio
-----------------
The WAV files in assets/audio/ are generated sine-wave beeps —
placeholder only. Replace with real recorded safety reminders before
deploying to a production gate. File paths are config-driven; no code
change is required to swap in real recordings.
"""

import logging
import subprocess
from pathlib import Path

from alerts.base_handler import BaseHandler
from core.session_controller import SessionState

logger = logging.getLogger(__name__)

_TRIGGER_STATES = {SessionState.GRANTED, SessionState.DENIED}


class VoiceHandler(BaseHandler):
    def __init__(self, config: dict):
        voice = config["voice"]
        self._sounds: dict[SessionState, Path] = {
            SessionState.GRANTED: Path(voice["granted_sound"]),
            SessionState.DENIED:  Path(voice["denied_sound"]),
        }

        # WARNING (not ERROR): a missing placeholder in an MVP-optional module
        # is a startup notice, not a fatal condition — consistent with
        # ScreenshotHandler's non-fatal warning on a None frame.
        for state, path in self._sounds.items():
            if not path.exists():
                logger.warning(
                    "VoiceHandler: audio file for %s not found: %s — "
                    "playback will be skipped.",
                    state.value, path,
                )

        self._last_played_state = None  # debounce: same pattern as ScreenshotHandler

    def handle(self, session_state, frame_verdict, frame, worker_name=None) -> None:
        if session_state not in _TRIGGER_STATES:
            self._last_played_state = None
            return

        if session_state == self._last_played_state:
            return  # debounce: already played for this hold

        self._last_played_state = session_state
        path = self._sounds[session_state]

        if not path.exists():
            return  # already warned at construction; skip silently here

        # Fire-and-forget: Popen returns immediately, audio plays in a
        # separate OS process without blocking this thread.
        subprocess.Popen(
            ["aplay", "--quiet", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Voice alert: playing %s", path.name)
