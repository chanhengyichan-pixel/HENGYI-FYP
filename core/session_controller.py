"""
Access Session Controller — state machine for PPE gate decisions.

State machine
-------------
  IDLE
    → PERSON_DETECTED  (first frame where verdict is not NO_PERSON)

  PERSON_DETECTED
    → IDLE             (next frame is NO_PERSON — person walked away)
    → EVALUATING       (next frame has a person — start collecting votes)

  EVALUATING  (collect evaluating_frame_count verdicts)
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

Tie-breaking
------------
evaluating_frame_count defaults to 15 (odd) — ties cannot occur with an
odd window.  If someone configures an even count, ties are resolved as
DENIED (fail-safe: ambiguity never grants access).

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
        self._evaluating_frame_count = int(sess["evaluating_frame_count"])
        self._decision_hold_seconds = float(sess["decision_hold_seconds"])
        self._cooldown_seconds = float(sess["cooldown_seconds"])
        self._clock = clock

        self._state = SessionState.IDLE
        self._votes: list[ComplianceStatus] = []
        self._hold_start: float = 0.0
        self._cooldown_start: float = 0.0

    @property
    def state(self) -> SessionState:
        return self._state

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
                self._state = SessionState.EVALUATING
                self._record_vote(verdict)
            return self._state

        if self._state is SessionState.EVALUATING:
            if verdict.status is ComplianceStatus.NO_PERSON:
                # Person left — abort evaluation, return to IDLE.
                self._votes = []
                self._state = SessionState.IDLE
                return self._state
            self._record_vote(verdict)
            if len(self._votes) >= self._evaluating_frame_count:
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

    def _decide(self) -> SessionState:
        compliant_count = self._votes.count(ComplianceStatus.COMPLIANT)
        non_compliant_count = self._votes.count(ComplianceStatus.NON_COMPLIANT)
        # Ties resolve as DENIED (fail-safe).
        if compliant_count > non_compliant_count:
            return SessionState.GRANTED
        return SessionState.DENIED
