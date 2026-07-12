"""
Unit tests for core/session_controller.py.

No camera, no model, no YOLO import.  FrameVerdict and ComplianceStatus
are imported directly — the intended coupling between session_controller
and decision_engine.

Clock injection: tests pass a list-backed fake clock whose current value
is faked_clock[0].  Tests advance time by mutating faked_clock[0].
"""

import pytest
from core.decision_engine import ComplianceStatus, FrameVerdict
from core.session_controller import SessionController, SessionState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "session": {
        "evaluating_frame_count": 5,   # small for test speed
        "decision_hold_seconds": 10.0,
        "cooldown_seconds": 3.0,
    }
}

COMPLIANT = FrameVerdict(status=ComplianceStatus.COMPLIANT, person_count=1)
NON_COMPLIANT = FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                              violations=["no-helmet"], person_count=1)
NO_PERSON = FrameVerdict(status=ComplianceStatus.NO_PERSON)


def make_controller(config=None, clock_val=0.0):
    """Return (controller, fake_clock_list).  Mutate fake_clock[0] to advance time."""
    fake_clock = [clock_val]
    ctrl = SessionController(config or BASE_CONFIG, clock=lambda: fake_clock[0])
    return ctrl, fake_clock


def feed(ctrl, verdict, n):
    """Call update() n times with the same verdict; return the last state."""
    state = None
    for _ in range(n):
        state = ctrl.update(verdict)
    return state


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_idle(self):
        ctrl, _ = make_controller()
        assert ctrl.state is SessionState.IDLE

    def test_no_person_stays_idle(self):
        ctrl, _ = make_controller()
        state = ctrl.update(NO_PERSON)
        assert state is SessionState.IDLE


# ---------------------------------------------------------------------------
# IDLE → PERSON_DETECTED
# ---------------------------------------------------------------------------

class TestPersonDetected:
    def test_compliant_frame_leaves_idle(self):
        ctrl, _ = make_controller()
        state = ctrl.update(COMPLIANT)
        assert state is SessionState.PERSON_DETECTED

    def test_non_compliant_frame_leaves_idle(self):
        ctrl, _ = make_controller()
        state = ctrl.update(NON_COMPLIANT)
        assert state is SessionState.PERSON_DETECTED

    def test_person_disappears_returns_to_idle(self):
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)           # → PERSON_DETECTED
        state = ctrl.update(NO_PERSON)   # person left
        assert state is SessionState.IDLE


# ---------------------------------------------------------------------------
# PERSON_DETECTED → EVALUATING
# ---------------------------------------------------------------------------

class TestEvaluating:
    def test_second_person_frame_enters_evaluating(self):
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)          # → PERSON_DETECTED
        state = ctrl.update(COMPLIANT)  # → EVALUATING
        assert state is SessionState.EVALUATING

    def test_stays_evaluating_before_window_full(self):
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)   # PERSON_DETECTED
        # evaluating_frame_count=5; send 4 frames — still evaluating
        state = feed(ctrl, COMPLIANT, 4)
        assert state is SessionState.EVALUATING

    def test_full_window_exits_evaluating(self):
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)   # PERSON_DETECTED
        # 5 frames completes the window (evaluating_frame_count=5)
        state = feed(ctrl, COMPLIANT, 5)
        assert state is not SessionState.EVALUATING


# ---------------------------------------------------------------------------
# Full cycle — all-compliant
# ---------------------------------------------------------------------------

class TestAllCompliantCycle:
    def test_all_compliant_grants(self):
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)        # PERSON_DETECTED
        feed(ctrl, COMPLIANT, 5)      # EVALUATING → GRANTED
        assert ctrl.state is SessionState.GRANTED

    def test_granted_transitions_to_cooldown_after_hold(self):
        ctrl, clock = make_controller()
        ctrl.update(COMPLIANT)
        feed(ctrl, COMPLIANT, 5)      # → GRANTED
        clock[0] = 10.0               # advance past decision_hold_seconds (10)
        ctrl.update(COMPLIANT)        # trigger time check
        assert ctrl.state is SessionState.COOLDOWN

    def test_cooldown_returns_to_idle(self):
        ctrl, clock = make_controller()
        ctrl.update(COMPLIANT)
        feed(ctrl, COMPLIANT, 5)      # → GRANTED
        clock[0] = 10.0
        ctrl.update(COMPLIANT)        # → COOLDOWN
        clock[0] = 13.1               # advance past cooldown_seconds (3)
        ctrl.update(COMPLIANT)        # → IDLE
        assert ctrl.state is SessionState.IDLE


# ---------------------------------------------------------------------------
# Full cycle — all non-compliant
# ---------------------------------------------------------------------------

class TestAllNonCompliantCycle:
    def test_all_non_compliant_denied(self):
        ctrl, _ = make_controller()
        ctrl.update(NON_COMPLIANT)
        feed(ctrl, NON_COMPLIANT, 5)
        assert ctrl.state is SessionState.DENIED

    def test_denied_transitions_to_cooldown_after_hold(self):
        ctrl, clock = make_controller()
        ctrl.update(NON_COMPLIANT)
        feed(ctrl, NON_COMPLIANT, 5)  # → DENIED
        clock[0] = 10.0
        ctrl.update(NON_COMPLIANT)
        assert ctrl.state is SessionState.COOLDOWN


# ---------------------------------------------------------------------------
# Majority vote — mixed frames
# ---------------------------------------------------------------------------

class TestMajorityVote:
    def test_majority_compliant_grants(self):
        # 3 compliant, 2 non-compliant → GRANTED
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)  # PERSON_DETECTED
        for v in [COMPLIANT, COMPLIANT, COMPLIANT, NON_COMPLIANT, NON_COMPLIANT]:
            ctrl.update(v)
        assert ctrl.state is SessionState.GRANTED

    def test_majority_non_compliant_denied(self):
        # 2 compliant, 3 non-compliant → DENIED
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)
        for v in [COMPLIANT, COMPLIANT, NON_COMPLIANT, NON_COMPLIANT, NON_COMPLIANT]:
            ctrl.update(v)
        assert ctrl.state is SessionState.DENIED

    def test_tie_is_denied(self):
        # evaluating_frame_count=4 (even) forces a tie scenario
        config = {"session": {"evaluating_frame_count": 4,
                               "decision_hold_seconds": 5.0,
                               "cooldown_seconds": 3.0}}
        ctrl, _ = make_controller(config)
        ctrl.update(COMPLIANT)   # PERSON_DETECTED
        for v in [COMPLIANT, COMPLIANT, NON_COMPLIANT, NON_COMPLIANT]:
            ctrl.update(v)
        # 2 vs 2 tie → DENIED (fail-safe)
        assert ctrl.state is SessionState.DENIED


# ---------------------------------------------------------------------------
# Person leaves mid-EVALUATING
# ---------------------------------------------------------------------------

class TestPersonLeavesEvaluating:
    def test_no_person_mid_evaluation_resets_to_idle(self):
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)          # PERSON_DETECTED
        ctrl.update(COMPLIANT)          # EVALUATING, vote 1
        ctrl.update(COMPLIANT)          # vote 2
        state = ctrl.update(NO_PERSON)  # person left → IDLE
        assert state is SessionState.IDLE

    def test_votes_cleared_after_abort(self):
        # After reset, a fresh person should trigger a fresh evaluation.
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)          # PERSON_DETECTED
        ctrl.update(COMPLIANT)          # EVALUATING
        ctrl.update(NO_PERSON)          # → IDLE (votes cleared)
        ctrl.update(COMPLIANT)          # → PERSON_DETECTED again
        state = ctrl.update(COMPLIANT)  # → EVALUATING (fresh, not still counting old votes)
        assert state is SessionState.EVALUATING
        # Only 1 vote collected in this new session (not 3 from before the abort).
        # Feed 3 more — total should be 4, still evaluating (window=5).
        feed(ctrl, COMPLIANT, 3)
        assert ctrl.state is SessionState.EVALUATING

    def test_person_returns_can_complete_evaluation(self):
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)    # PERSON_DETECTED
        ctrl.update(COMPLIANT)    # EVALUATING
        ctrl.update(NO_PERSON)    # → IDLE
        ctrl.update(COMPLIANT)    # PERSON_DETECTED
        feed(ctrl, COMPLIANT, 5)  # fresh evaluation → GRANTED
        assert ctrl.state is SessionState.GRANTED


# ---------------------------------------------------------------------------
# Hold and cooldown timing edge cases
# ---------------------------------------------------------------------------

class TestTimingEdgeCases:
    def test_hold_not_expired_stays_granted(self):
        ctrl, clock = make_controller()
        ctrl.update(COMPLIANT)
        feed(ctrl, COMPLIANT, 5)   # → GRANTED, hold_start = 0.0
        clock[0] = 9.9             # just under decision_hold_seconds (10)
        ctrl.update(COMPLIANT)
        assert ctrl.state is SessionState.GRANTED

    def test_cooldown_not_expired_stays_cooldown(self):
        ctrl, clock = make_controller()
        ctrl.update(COMPLIANT)
        feed(ctrl, COMPLIANT, 5)   # → GRANTED
        clock[0] = 10.0
        ctrl.update(COMPLIANT)     # → COOLDOWN, cooldown_start = 10.0
        clock[0] = 12.9            # just under cooldown_seconds (3), so 10+3=13
        ctrl.update(COMPLIANT)
        assert ctrl.state is SessionState.COOLDOWN

    def test_new_person_during_hold_is_ignored(self):
        # GRANTED hold should not be interrupted by further frame verdicts.
        ctrl, clock = make_controller()
        ctrl.update(COMPLIANT)
        feed(ctrl, COMPLIANT, 5)     # → GRANTED
        # Feed many non-compliant frames — hold must not be affected.
        feed(ctrl, NON_COMPLIANT, 20)
        assert ctrl.state is SessionState.GRANTED

    def test_new_person_during_cooldown_is_ignored(self):
        ctrl, clock = make_controller()
        ctrl.update(COMPLIANT)
        feed(ctrl, COMPLIANT, 5)   # → GRANTED
        clock[0] = 10.0
        ctrl.update(COMPLIANT)     # → COOLDOWN
        feed(ctrl, NON_COMPLIANT, 20)
        assert ctrl.state is SessionState.COOLDOWN


# ---------------------------------------------------------------------------
# Session identity limitation (documented behavior, not a bug)
# ---------------------------------------------------------------------------

class TestSessionIdentityLimitation:
    def test_fast_worker_handoff_without_empty_frame_continues_session(self):
        # DOCUMENTED LIMITATION — see module docstring "Known limitation —
        # session identity" for full explanation.
        #
        # Scenario: Worker A partially evaluates (3 non-compliant votes),
        # then immediately leaves and Worker B (compliant) enters — but there
        # is no NO_PERSON frame between them.  The controller has no identity
        # signal, so it cannot detect the handoff and continues the in-progress
        # session, mixing A's votes with B's.
        #
        # This test locks in the CURRENT behavior.  It is NOT asserting the
        # desired long-term behavior (which would be a fresh session for B).
        # Do not "fix" this test without implementing per-person identity in
        # Milestone 12.
        ctrl, _ = make_controller()

        ctrl.update(NON_COMPLIANT)              # IDLE → PERSON_DETECTED (Worker A, no vote yet)
        ctrl.update(NON_COMPLIANT)              # → EVALUATING, vote 1 (A: non-compliant)
        ctrl.update(NON_COMPLIANT)              # vote 2 (A: non-compliant)
        ctrl.update(NON_COMPLIANT)              # vote 3 (A: non-compliant)
        # Worker A leaves / Worker B enters — no NO_PERSON frame.
        # From here all frames are B (compliant), but session is NOT reset.
        ctrl.update(COMPLIANT)                  # vote 4 (B: compliant, but mixed in)
        ctrl.update(COMPLIANT)                  # vote 5 — window complete

        # Current (limited) behavior: session was never reset, so the vote
        # tally is 3 NON_COMPLIANT (A) vs 2 COMPLIANT (B) → DENIED.
        # If identity were tracked properly, B's session should start fresh
        # and would eventually GRANT after 5 compliant frames.
        assert ctrl.state is SessionState.DENIED  # mixed A+B votes → wrong outcome for B
