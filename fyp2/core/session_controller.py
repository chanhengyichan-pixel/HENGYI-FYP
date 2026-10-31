"""
Access Session Controller — state machine for PPE gate decisions.

State machine
-------------
  IDLE
    → PERSON_DETECTED  (first frame where verdict is not NO_PERSON)

  PERSON_DETECTED
    → IDLE             (next frame is NO_PERSON — person walked away)
    → EVALUATING       (next frame has a person — start collecting votes)

  EVALUATING  (collect votes for evaluating_seconds, and at least min_votes)
    → IDLE             (any frame is NO_PERSON — person left mid-evaluation,
                        abort and reset; counting absence as a vote would
                        bias the result based on how quickly someone arrives)
    → GRANTED          (majority COMPLIANT after full window)
    → DENIED           (majority NON_COMPLIANT, or tie, after full window)

  GRANTED / DENIED  (held for decision_hold_seconds, time-based)
    → COOLDOWN

  COOLDOWN  (held for cooldown_seconds, time-based)
    → IDLE

Time checks use an injectable clock callable (default: time.monotonic).
time.monotonic is used instead of time.time to avoid jumps from NTP
corrections or manual system-clock changes.

Time-based evaluation window
----------------------------
The window is measured in seconds (evaluating_seconds), not frames. An
earlier version collected a fixed 15 frames: on the laptop (~110
pipeline cycles/s) that took 0.15 s, on a Raspberry Pi (~2-5 FPS) it
would take 3-7 s — the same config behaved completely differently per
device. A time window makes the gate feel the same everywhere.
min_votes guards the other extreme: on a very slow device the window
simply stays open until at least min_votes frames have been seen, so a
decision is never made from one or two frames.

Tie-breaking
------------
The vote count is not fixed under a time window, so ties are possible.
Ties are resolved as DENIED (fail-safe: ambiguity never grants access).

Note on DECISION
----------------
The architecture brief's state machine lists DECISION (majority
vote) as a distinct phase between EVALUATING and GRANTED/DENIED.
This implementation does not expose DECISION as a separate
SessionState. The majority vote is computed in _decide(), called
synchronously within the same update() call that completes the
EVALUATING window (self._votes.count(...) over a fixed-size
in-memory list). This computation has negligible, effectively
zero duration relative to the frame processing pipeline, so there
is no meaningful interval during which the system observably
"is" in a decision phase — it transitions directly from EVALUATING
to GRANTED or DENIED within one update() call. Adding a DECISION
enum value would increase state machine complexity (an extra
branch, an extra SharedState read for the dashboard to handle)
without changing any externally observable behavior. If vote
computation ever becomes non-trivial (e.g. per-person spatial
attribution under a future Strategy B), this decision should be
revisited.

Decision violations
-------------------
Alongside each vote, the violations of that frame are counted. When the
window closes, `decision_violations` holds the violations present in at
least half of the NON_COMPLIANT votes — e.g. ["no-helmet"] if the person
wore a vest throughout but no helmet. Alert handlers use this (not the
last single frame, which can flicker) to say exactly what is missing.

Known limitation — session identity
------------------------------------
The session controller assumes person-presence continuity, not
verified per-person identity.  If Worker A leaves the camera's field
of view and Worker B enters before any single frame reports NO_PERSON
(a fast handoff), the in-progress session continues as if it were
still Worker A.

This is accepted by design: the entrance check point is a marked
one-person square in front of the camera, so a gap-free handoff is
unlikely in practice, and the project deliberately stores no personal
identity (no face recognition). If needed later, the room camera's
tracker IDs could be reused here to detect a change of person.

Coupling note
-------------
This module imports FrameVerdict and ComplianceStatus from
core.decision_engine — that is the intended coupling.  These types are
the contract between the decision engine and the session controller.
"""

from __future__ import annotations

import enum
import time as _time_module
from dataclasses import dataclass, field
from typing import Callable

from core.decision_engine import ComplianceStatus, FrameVerdict


class SessionState(enum.Enum):
    IDLE = "idle"
    PERSON_DETECTED = "person_detected"
    EVALUATING = "evaluating"
    GRANTED = "granted"
    DENIED = "denied"
    COOLDOWN = "cooldown"


class SessionController:
    def __init__(self, config: dict, clock: Callable[[], float] = _time_module.monotonic):
        sess = config["session"]
        self._evaluating_seconds = float(sess["evaluating_seconds"])
        self._min_votes = int(sess.get("min_votes", 1))
        self._decision_hold_seconds = float(sess["decision_hold_seconds"])
        self._cooldown_seconds = float(sess["cooldown_seconds"])
        self._clock = clock

        self._state = SessionState.IDLE
        self._votes: list[ComplianceStatus] = []
        self._violation_counts: dict[str, int] = {}
        self._decision_violations: list[str] = []
        self._eval_start: float = 0.0
        self._hold_start: float = 0.0
        self._cooldown_start: float = 0.0

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def decision_violations(self) -> list[str]:
        """Violations behind the current GRANTED/DENIED decision (see module
        docstring). Empty for GRANTED. Valid until the next decision."""
        return list(self._decision_violations)

    def reset(self) -> None:
        """Force the session back to IDLE, discarding any in-progress evaluation.

        Safe to call from any state.
        """
        self._state = SessionState.IDLE
        self._votes = []
        self._eval_start = 0.0
        self._hold_start = 0.0
        self._cooldown_start = 0.0

    def update(self, verdict: FrameVerdict) -> SessionState:
        """
        Advance the state machine by one frame verdict.

        Should be called once per processed frame.  Returns the current
        SessionState after any transition.
        """
        if self._state is SessionState.IDLE:
            if verdict.status is not ComplianceStatus.NO_PERSON:
                self._state = SessionState.PERSON_DETECTED
            return self._state

        if self._state is SessionState.PERSON_DETECTED:
            if verdict.status is ComplianceStatus.NO_PERSON:
                self._state = SessionState.IDLE
            else:
                self._votes = []
                self._violation_counts = {}
                self._state = SessionState.EVALUATING
                self._eval_start = self._clock()
                self._record_vote(verdict)
            return self._state

        if self._state is SessionState.EVALUATING:
            if verdict.status is ComplianceStatus.NO_PERSON:
                # Person left — abort evaluation, return to IDLE.
                self._votes = []
                self._state = SessionState.IDLE
                return self._state
            self._record_vote(verdict)
            window_elapsed = self._clock() - self._eval_start >= self._evaluating_seconds
            if window_elapsed and len(self._votes) >= self._min_votes:
                self._state = self._decide()
                self._hold_start = self._clock()
            return self._state

        if self._state in (SessionState.GRANTED, SessionState.DENIED):
            if self._clock() - self._hold_start >= self._decision_hold_seconds:
                self._state = SessionState.COOLDOWN
                self._cooldown_start = self._clock()
            return self._state

        if self._state is SessionState.COOLDOWN:
            if self._clock() - self._cooldown_start >= self._cooldown_seconds:
                self._state = SessionState.IDLE
            return self._state

        return self._state  # unreachable, satisfies type checkers

    def _record_vote(self, verdict: FrameVerdict) -> None:
        # Only COMPLIANT and NON_COMPLIANT reach here (NO_PERSON is handled above).
        if verdict.status in (ComplianceStatus.COMPLIANT, ComplianceStatus.NON_COMPLIANT):
            self._votes.append(verdict.status)
            if verdict.status is ComplianceStatus.NON_COMPLIANT:
                for v in verdict.violations:
                    self._violation_counts[v] = self._violation_counts.get(v, 0) + 1

    def _decide(self) -> SessionState:
        compliant_count = self._votes.count(ComplianceStatus.COMPLIANT)
        non_compliant_count = self._votes.count(ComplianceStatus.NON_COMPLIANT)
        # Ties resolve as DENIED (fail-safe).
        if compliant_count > non_compliant_count:
            self._decision_violations = []
            return SessionState.GRANTED
        half = non_compliant_count / 2.0
        self._decision_violations = [
            v for v, n in sorted(self._violation_counts.items()) if n >= half
        ]
        return SessionState.DENIED
