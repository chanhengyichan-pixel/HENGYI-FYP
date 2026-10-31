"""
Unit tests for core/session_controller.py.

No camera, no model, no YOLO import.  FrameVerdict and ComplianceStatus
are imported directly — the intended coupling between session_controller
and decision_engine.

Clock injection: tests pass a list-backed fake clock whose current value
is faked_clock[0].  Tests advance time by mutating faked_clock[0].

Most tests use evaluating_seconds=0 with min_votes=5, so the window closes
after exactly 5 votes without moving the clock — that keeps the hold and
cooldown timing tests simple. TestTimeWindow covers the time-based window.
"""

import pytest
from core.decision_engine import ComplianceStatus, FrameVerdict
from core.session_controller import SessionController, SessionState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "session": {
        "evaluating_seconds": 0.0,     # window closes on min_votes alone
        "min_votes": 5,
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
        # min_votes=5; send 4 frames — still evaluating
        state = feed(ctrl, COMPLIANT, 4)
        assert state is SessionState.EVALUATING

    def test_full_window_exits_evaluating(self):
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)   # PERSON_DETECTED
        # 5 frames completes the window (min_votes=5)
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
        # min_votes=4 (even) forces a tie scenario
        config = {"session": {"evaluating_seconds": 0.0, "min_votes": 4,
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
        # This test locks in the CURRENT behavior, accepted by design for a
        # one-person check point with no identity storage.
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


# ---------------------------------------------------------------------------
# Time-based evaluation window
# ---------------------------------------------------------------------------

TIME_CONFIG = {"session": {"evaluating_seconds": 1.5, "min_votes": 3,
                           "decision_hold_seconds": 5.0, "cooldown_seconds": 3.0}}


class TestTimeWindow:
    def test_many_fast_frames_do_not_close_window_early(self):
        # Laptop case: lots of frames inside 1.5 s must not decide early.
        ctrl, clock = make_controller(TIME_CONFIG)
        ctrl.update(COMPLIANT)                 # PERSON_DETECTED
        for i in range(40):                    # 40 frames in 1.0 s
            clock[0] = i * 0.025
            ctrl.update(COMPLIANT)
        assert ctrl.state is SessionState.EVALUATING

    def test_window_closes_after_evaluating_seconds(self):
        ctrl, clock = make_controller(TIME_CONFIG)
        ctrl.update(COMPLIANT)                 # PERSON_DETECTED
        for t in (0.0, 0.5, 1.0, 1.5):         # ~2 FPS, Pi-like
            clock[0] = t
            ctrl.update(COMPLIANT)
        assert ctrl.state is SessionState.GRANTED

    def test_slow_device_waits_for_min_votes(self):
        # Very slow device: 1.5 s passed but only 2 votes — keep collecting.
        ctrl, clock = make_controller(TIME_CONFIG)
        ctrl.update(NON_COMPLIANT)             # PERSON_DETECTED
        clock[0] = 0.0
        ctrl.update(NON_COMPLIANT)             # vote 1, window starts
        clock[0] = 2.0
        ctrl.update(NON_COMPLIANT)             # vote 2 — time up, votes short
        assert ctrl.state is SessionState.EVALUATING
        clock[0] = 2.5
        ctrl.update(NON_COMPLIANT)             # vote 3 — decide
        assert ctrl.state is SessionState.DENIED

    def test_window_restarts_after_person_leaves(self):
        ctrl, clock = make_controller(TIME_CONFIG)
        ctrl.update(COMPLIANT)
        clock[0] = 0.0
        ctrl.update(COMPLIANT)                 # window starts at 0.0
        clock[0] = 1.0
        ctrl.update(NO_PERSON)                 # abort → IDLE
        clock[0] = 1.2
        ctrl.update(COMPLIANT)                 # PERSON_DETECTED
        ctrl.update(COMPLIANT)                 # new window starts at 1.2
        clock[0] = 2.0
        ctrl.update(COMPLIANT)
        clock[0] = 2.2
        ctrl.update(COMPLIANT)                 # 1.0 s into new window
        assert ctrl.state is SessionState.EVALUATING


# ---------------------------------------------------------------------------
# decision_violations — what the voice prompt should name
# ---------------------------------------------------------------------------

def _nc(*violations):
    return FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                        violations=list(violations), person_count=1)


class TestDecisionViolations:
    def test_only_helmet_missing_throughout(self):
        ctrl, _ = make_controller()
        ctrl.update(_nc("no-helmet"))
        feed(ctrl, _nc("no-helmet"), 5)
        assert ctrl.state is SessionState.DENIED
        assert ctrl.decision_violations == ["no-helmet"]

    def test_one_flickering_frame_does_not_add_a_violation(self):
        # Vest worn; one frame misses it. Majority says only the helmet.
        ctrl, _ = make_controller()
        ctrl.update(_nc("no-helmet"))
        for v in [_nc("no-helmet"), _nc("no-helmet", "no-vest"), _nc("no-helmet"),
                  _nc("no-helmet"), _nc("no-helmet")]:
            ctrl.update(v)
        assert ctrl.decision_violations == ["no-helmet"]

    def test_both_missing(self):
        ctrl, _ = make_controller()
        ctrl.update(_nc("no-helmet", "no-vest"))
        feed(ctrl, _nc("no-helmet", "no-vest"), 5)
        assert ctrl.decision_violations == ["no-helmet", "no-vest"]

    def test_granted_has_no_violations(self):
        ctrl, _ = make_controller()
        ctrl.update(COMPLIANT)
        feed(ctrl, COMPLIANT, 5)
        assert ctrl.decision_violations == []

    def test_counts_reset_between_sessions(self):
        ctrl, clock = make_controller()
        ctrl.update(_nc("no-vest"))
        feed(ctrl, _nc("no-vest"), 5)                 # DENIED, no-vest
        clock[0] = 10.0
        ctrl.update(COMPLIANT)                        # → COOLDOWN
        clock[0] = 13.1
        ctrl.update(COMPLIANT)                        # → IDLE
        ctrl.update(_nc("no-helmet"))                 # PERSON_DETECTED
        feed(ctrl, _nc("no-helmet"), 5)
        assert ctrl.decision_violations == ["no-helmet"]
