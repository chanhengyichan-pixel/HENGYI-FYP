"""
Speaker — the ONE shared USB speaker, used by every camera's VoiceHandler.

Why one speaker for both cameras
--------------------------------
The Raspberry Pi 5 has no 3.5 mm jack, so audio goes to a USB speaker.
Two USB speakers would each need their own ALSA device set up by hand;
in a small demo area one speaker is heard from both the entrance and the
room. The prompts already say where they come from ("Access denied ..."
is the entrance, "... hazard area ..." / "... restricted area ..." is the
room), so no extra location announcement is needed.

Queue instead of skipping
-------------------------
Both pipelines can raise an alert at the same moment. Prompts are put in
a queue and played one after another by a single worker thread, so they
never talk over each other and none is lost:

  priority 0  RESTRICTED_ENTRY  stops the prompt playing now, plays first
  priority 1  entrance DENIED
  priority 2  HAZARD_PPE
  priority 3  entrance GRANTED

Same priority -> first come, first served. A prompt that has waited longer
than voice.max_wait_seconds is dropped (it would describe something that
is already over), and a prompt already waiting in the queue is not added
twice.

Threading
---------
say() only puts an item in the queue and returns at once, so the pipeline
threads (YOLO) are never blocked by audio. The worker thread is the only
place that starts aplay and waits for it.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PRIORITY_RESTRICTED = 0
PRIORITY_DENIED = 1
PRIORITY_HAZARD = 2
PRIORITY_GRANTED = 3


def _aplay(path: Path):
    return subprocess.Popen(["aplay", "--quiet", str(path)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Speaker:
    """player(path) -> process with wait() / terminate() / poll() (default: aplay).
    threaded=False: no worker thread; call play_pending() yourself (tests)."""

    def __init__(self, max_wait_seconds: float = 8.0, player=_aplay,
                 clock=time.monotonic, threaded: bool = True):
        self._max_wait = float(max_wait_seconds)
        self._player = player
        self._clock = clock
        self._queue: list = []              # heap of (priority, seq, queued_at, path)
        self._seq = itertools.count()
        self._cond = threading.Condition()
        self._current = None                # (priority, process, path) playing now
        self._closed = False
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
            self._thread.start()

    def say(self, path: Path, priority: int) -> None:
        """Queue a prompt. Never blocks the caller."""
        with self._cond:
            if self._closed:
                return
            if any(p == path for _, _, _, p in self._queue):
                return                       # already waiting
            if self._current is not None and self._current[2] == path:
                return                       # this exact prompt is playing now
            heapq.heappush(self._queue, (priority, next(self._seq), self._clock(), path))
            cur = self._current
            if priority == PRIORITY_RESTRICTED and cur is not None and cur[0] != PRIORITY_RESTRICTED:
                try:
                    cur[1].terminate()       # highest severity: cut the current prompt
                except Exception:
                    pass
            self._cond.notify()

    def pending(self) -> list[Path]:
        with self._cond:
            return [p for _, _, _, p in sorted(self._queue)]

    def _next(self, block: bool):
        """Pop the next prompt that is not too old, or None."""
        with self._cond:
            while True:
                while not self._queue and block and not self._closed:
                    self._cond.wait()
                if self._closed or not self._queue:
                    return None
                priority, _, queued_at, path = heapq.heappop(self._queue)
                if self._clock() - queued_at > self._max_wait:
                    logger.info("Voice alert dropped (waited too long): %s", path.name)
                    continue
                return priority, path

    def _play_one(self, priority: int, path: Path) -> None:
        if not path.exists():
            return
        try:
            proc = self._player(path)
        except Exception as e:
            logger.error("Speaker: could not play %s: %s", path.name, e)
            return
        logger.info("Voice alert: playing %s", path.name)
        with self._cond:
            self._current = (priority, proc, path)
        try:
            proc.wait()
        except Exception:
            pass
        finally:
            with self._cond:
                self._current = None

    def play_pending(self) -> None:
        """Play everything queued now, in priority order (no worker thread)."""
        while (item := self._next(block=False)) is not None:
            self._play_one(*item)

    def _run(self) -> None:
        while (item := self._next(block=True)) is not None:
            self._play_one(*item)

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._queue.clear()
            if self._current is not None:
                try:
                    self._current[1].terminate()
                except Exception:
                    pass
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
