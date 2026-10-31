"""
Unit tests for scripts/enrollment_server.py's pure logic.

No real camera, no real Flask server, no cv2.imshow. Tests exercise
evaluate_capture_guidance() (synthetic face-box/brightness inputs) and
CaptureCooldown (injectable fake clock) in isolation.
"""

import pytest

from scripts.enrollment_server import (
    CaptureCooldown, EnrollmentState, evaluate_capture_guidance, validate_pose_stages,
)

CFG = {
    "min_face_area_ratio": 0.05,
    "max_face_area_ratio": 0.35,
    "center_tolerance_ratio": 0.35,
    "brightness_threshold": 60,
}

POSE_STAGES = [
    {"name": "center", "label": "Center", "instruction": "Look straight ahead and hold still", "samples": 5},
    {"name": "slight_left", "label": "Slight Left", "instruction": "Turn slightly LEFT and hold still", "samples": 5},
    {"name": "slight_right", "label": "Slight Right", "instruction": "Turn slightly RIGHT and hold still", "samples": 5},
    {"name": "slight_down", "label": "Slight Down", "instruction": "Tilt your head slightly DOWN and hold still", "samples": 5},
]

ENROLLMENT_CFG = {**CFG, "num_samples": 20, "pose_stages": POSE_STAGES}

FRAME_W, FRAME_H = 640, 480


def _centered_face_box(area_ratio=0.15):
    """A face box centered in the frame with the given area ratio."""
    area = area_ratio * FRAME_W * FRAME_H
    side = int(area ** 0.5)
    x = (FRAME_W - side) // 2
    y = (FRAME_H - side) // 2
    return (x, y, side, side)


class TestNoFace:
    def test_zero_faces_returns_no_face_message(self):
        r = evaluate_capture_guidance(0, None, FRAME_W, FRAME_H, 100.0, CFG)
        assert r.valid is False
        assert r.message == "No face detected — please face the camera"


class TestMultipleFaces:
    def test_two_faces_returns_multiple_faces_message(self):
        r = evaluate_capture_guidance(2, _centered_face_box(), FRAME_W, FRAME_H, 100.0, CFG)
        assert r.valid is False
        assert r.message == "Multiple faces detected — only one person should be in frame"

    def test_multiple_faces_wins_over_simultaneous_brightness_problem(self):
        # Brightness is also failing (too dark), but face_count > 1 must
        # win — priority order, not "first problem found."
        r = evaluate_capture_guidance(2, _centered_face_box(), FRAME_W, FRAME_H, 5.0, CFG)
        assert r.message == "Multiple faces detected — only one person should be in frame"


class TestLighting:
    def test_dark_frame_returns_lighting_message(self):
        r = evaluate_capture_guidance(1, _centered_face_box(), FRAME_W, FRAME_H, 30.0, CFG)
        assert r.valid is False
        assert r.message == "Lighting is too dark — move to a brighter area"

    def test_brightness_exactly_at_threshold_passes_lighting_check(self):
        # >= threshold passes (< threshold fails) — boundary is inclusive.
        r = evaluate_capture_guidance(1, _centered_face_box(), FRAME_W, FRAME_H, 60.0, CFG)
        assert r.message != "Lighting is too dark — move to a brighter area"

    def test_lighting_wins_over_simultaneous_distance_problem(self):
        too_small = (300, 220, 20, 20)  # tiny box, area ratio far below min
        r = evaluate_capture_guidance(1, too_small, FRAME_W, FRAME_H, 5.0, CFG)
        assert r.message == "Lighting is too dark — move to a brighter area"


class TestDistance:
    def test_face_too_small_returns_move_closer(self):
        tiny = (310, 230, 15, 15)  # area ratio ~= 0.0007, well below 0.05
        r = evaluate_capture_guidance(1, tiny, FRAME_W, FRAME_H, 100.0, CFG)
        assert r.valid is False
        assert r.message == "Move closer to the camera"

    def test_face_too_large_returns_move_back(self):
        big = _centered_face_box(area_ratio=0.5)  # above max 0.35
        r = evaluate_capture_guidance(1, big, FRAME_W, FRAME_H, 100.0, CFG)
        assert r.valid is False
        assert r.message == "Move back from the camera"

    def test_distance_wins_over_simultaneous_centering_problem(self):
        # Small AND off-center — distance check must fire first.
        tiny_offcenter = (20, 20, 15, 15)
        r = evaluate_capture_guidance(1, tiny_offcenter, FRAME_W, FRAME_H, 100.0, CFG)
        assert r.message == "Move closer to the camera"


class TestCentering:
    def test_face_far_from_center_returns_move_to_center(self):
        # Correct size (area ratio ~0.15, well within the acceptable
        # distance range), but pushed to the far edge of the frame so
        # only the centering check fails.
        edge = (0, 0, 214, 214)
        r = evaluate_capture_guidance(1, edge, FRAME_W, FRAME_H, 100.0, CFG)
        assert r.valid is False
        assert r.message == "Move to the center of the frame"

    def test_face_within_tolerance_of_center_passes(self):
        r = evaluate_capture_guidance(1, _centered_face_box(), FRAME_W, FRAME_H, 100.0, CFG)
        assert r.valid is True


class TestAllChecksPass:
    def test_valid_frame_returns_hold_still(self):
        r = evaluate_capture_guidance(1, _centered_face_box(), FRAME_W, FRAME_H, 150.0, CFG)
        assert r.valid is True
        assert r.message == "Hold still"


# ---------------------------------------------------------------------------
# Pose-stage-aware instruction text
# ---------------------------------------------------------------------------

class TestPoseStageInstructions:
    def test_valid_frame_returns_active_stage_instruction(self):
        r = evaluate_capture_guidance(
            1, _centered_face_box(), FRAME_W, FRAME_H, 150.0, CFG,
            stage_instruction="Turn slightly LEFT and hold still",
        )
        assert r.valid is True
        assert r.message == "Turn slightly LEFT and hold still"

    def test_different_stages_produce_different_instruction_text(self):
        r_left = evaluate_capture_guidance(
            1, _centered_face_box(), FRAME_W, FRAME_H, 150.0, CFG,
            stage_instruction=POSE_STAGES[1]["instruction"],
        )
        r_down = evaluate_capture_guidance(
            1, _centered_face_box(), FRAME_W, FRAME_H, 150.0, CFG,
            stage_instruction=POSE_STAGES[3]["instruction"],
        )
        assert r_left.message != r_down.message
        assert r_left.message == "Turn slightly LEFT and hold still"
        assert r_down.message == "Tilt your head slightly DOWN and hold still"

    def test_failing_check_ignores_stage_instruction(self):
        # A failing frame must still return the corrective message, not
        # the stage directive — stage_instruction only applies once every
        # guidance check has already passed.
        r = evaluate_capture_guidance(
            0, None, FRAME_W, FRAME_H, 150.0, CFG,
            stage_instruction="Turn slightly LEFT and hold still",
        )
        assert r.message == "No face detected — please face the camera"


# ---------------------------------------------------------------------------
# validate_pose_stages — fail-fast config validation
# ---------------------------------------------------------------------------

class TestValidatePoseStages:
    def test_valid_config_does_not_raise(self):
        validate_pose_stages(ENROLLMENT_CFG)  # must not raise

    def test_mismatched_totals_raises_value_error(self):
        bad_cfg = {**ENROLLMENT_CFG, "num_samples": 25}  # stages still sum to 20
        with pytest.raises(ValueError, match="pose_stages samples sum to 20"):
            validate_pose_stages(bad_cfg)

    def test_empty_pose_stages_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_pose_stages({**ENROLLMENT_CFG, "pose_stages": []})


# ---------------------------------------------------------------------------
# EnrollmentState — overall + per-stage progress reporting
# ---------------------------------------------------------------------------

class TestEnrollmentStateStageProgress:
    def test_snapshot_reports_overall_and_stage_progress_independently(self):
        state = EnrollmentState(target_samples=20, stage_count=4)
        state.update_progress(
            captured=7, stage_index=2, stage_label="Slight Left",
            stage_captured=2, stage_target=5,
        )
        snap = state.snapshot()
        assert snap["captured"] == 7
        assert snap["target"] == 20
        assert snap["stage_index"] == 2
        assert snap["stage_count"] == 4
        assert snap["stage_label"] == "Slight Left"
        assert snap["stage_captured"] == 2
        assert snap["stage_target"] == 5

    def test_finish_sets_status_and_message_without_touching_progress(self):
        state = EnrollmentState(target_samples=20, stage_count=4)
        state.update_progress(20, 4, "Slight Down", 5, 5)
        state.finish(True, "Enrollment complete.")
        snap = state.snapshot()
        assert snap["status"] == "success"
        assert snap["message"] == "Enrollment complete."
        assert snap["captured"] == 20  # untouched by finish()


class TestCaptureCooldown:
    def test_ready_before_any_capture(self):
        cd = CaptureCooldown(0.2, clock=lambda: 100.0)
        assert cd.ready() is True

    def test_not_ready_immediately_after_capture(self):
        t = [100.0]
        cd = CaptureCooldown(0.2, clock=lambda: t[0])
        cd.mark_captured()
        assert cd.ready() is False

    def test_not_ready_before_cooldown_elapses(self):
        t = [100.0]
        cd = CaptureCooldown(0.2, clock=lambda: t[0])
        cd.mark_captured()
        t[0] = 100.15  # 0.15s elapsed, < 0.2s cooldown
        assert cd.ready() is False

    def test_ready_once_cooldown_elapses(self):
        t = [100.0]
        cd = CaptureCooldown(0.2, clock=lambda: t[0])
        cd.mark_captured()
        t[0] = 100.2  # exactly the cooldown boundary — inclusive
        assert cd.ready() is True

    def test_ready_well_after_cooldown_elapses(self):
        t = [100.0]
        cd = CaptureCooldown(0.2, clock=lambda: t[0])
        cd.mark_captured()
        t[0] = 105.0
        assert cd.ready() is True

    def test_repeated_captures_each_reset_the_cooldown(self):
        t = [100.0]
        cd = CaptureCooldown(0.2, clock=lambda: t[0])
        cd.mark_captured()
        t[0] = 100.3
        assert cd.ready() is True
        cd.mark_captured()
        t[0] = 100.35  # only 0.05s since the second capture
        assert cd.ready() is False
