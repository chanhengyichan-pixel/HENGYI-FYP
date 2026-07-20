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

Known limitation — session identity
------------------------------------
The session controller assumes person-presence continuity, not
verified per-person identity.  If Worker A leaves the camera's field
of view and Worker B enters before any single frame reports NO_PERSON
(e.g. a fast handoff, or A and B briefly overlapping), the controller
continues the in-progress EVALUATING session as if it were still
Worker A — votes accumulated for A are counted toward the session that
will grant or deny B.

Root cause: FrameVerdict carries no person-identity information.  This
is a direct consequence of the Strategy A detector schema chosen in
Milestone 2: the YOLOv8 output is a flat list of class detections with
no spatial tracking, no re-identification embedding, and no per-track
ID.  The controller has no signal to distinguish "same person, still
present" from "different person, no gap between them."

A bounding-box jump heuristic was considered as a stopgap (flag large
between-frame centroid displacement as a probable new person) but was
explicitly deferred: it would be unreliable under occlusion, lateral
movement, or multiple simultaneous detections and would add complexity
without actually solving the problem.

Resolution: deferred by design to Milestone 12 (face recognition +
worker registration), where real per-person identity enters the system
via a dedicated recognition module.  That is the correct place to fix
this — not here.

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

    @property
    def is_last_evaluating_frame(self) -> bool:
        """True when the NEXT update() call will complete the evaluation window.

        Used by Pipeline to time the second identity check — identify() is
        called on this frame (before update()), so that if a worker swap is
        detected, reset() can be called instead of allowing the decision to
        proceed.  Checked before update(), not after, because after the final
        update() the state is already GRANTED or DENIED.
        """
        return (
            self._state is SessionState.EVALUATING
            and len(self._votes) == self._evaluating_frame_count - 1
        )

    def reset(self) -> None:
        """Force the session back to IDLE, discarding any in-progress evaluation.

        Called by Pipeline when a worker swap is detected: two confident face
        identity checks within the same session returned different names.
        This is the deferred fix for the session-identity gap documented in
        the module docstring — previously, a worker swap with no NO_PERSON
        frame between them was indistinguishable from one continuous person.

        Safe to call from any state (e.g. if somehow called during COOLDOWN,
        it will still reset cleanly to IDLE).
        """
        self._state = SessionState.IDLE
        self._votes = []
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
