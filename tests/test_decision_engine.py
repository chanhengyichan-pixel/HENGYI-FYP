"""
Unit tests for core/decision_engine.py.

No camera, no model, no YOLO import. Inputs are SimpleNamespace stubs
that satisfy the duck-type contract: .class_name, .confidence,
.x1/.y1/.x2/.y2. Geometry is now load-bearing (containment + region
validity), so every detection stub needs real, deliberately chosen box
coordinates — unlike the old class-name-only stubs.

Test person box convention: 200 wide x 400 tall, at (100, 100) ->
(300, 500), unless a test needs a second person. With the default
config (containment_threshold=0.60, head top_ratio=0.25 width_ratio=0.55,
torso top_ratio=0.25 bottom_ratio=0.75 width_ratio=0.80):
  - head region: y in [100, 200], x in [145, 255]  (band half-width 55)
  - torso region: y in [200, 400], x in [120, 280]  (band half-width 80)
"""

import pytest
from types import SimpleNamespace

from core.decision_engine import DecisionEngine, ComplianceStatus, FrameVerdict


CONFIG = {
    "decision_engine": {
        "containment_threshold": 0.60,
        "head": {"top_ratio": 0.25, "width_ratio": 0.55},
        "torso": {"top_ratio": 0.25, "bottom_ratio": 0.75, "width_ratio": 0.80},
    }
}


def _det(class_name, x1, y1, x2, y2, confidence=0.9):
    return SimpleNamespace(
        class_name=class_name, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2
    )


def _person(x1=100, y1=100, x2=300, y2=500, confidence=0.9):
    return _det("person", x1, y1, x2, y2, confidence)


@pytest.fixture
def engine():
    return DecisionEngine(CONFIG)


# ---------------------------------------------------------------------------
# No person
# ---------------------------------------------------------------------------

class TestNoPerson:
    def test_empty_list_returns_no_person(self, engine):
        v = engine.evaluate([])
        assert v.status is ComplianceStatus.NO_PERSON

    def test_no_person_has_zero_person_count(self, engine):
        v = engine.evaluate([])
        assert v.person_count == 0

    def test_only_ppe_no_person_returns_no_person(self, engine):
        # Helmet/vest boxes visible but no person box — odd but possible.
        v = engine.evaluate([
            _det("helmet", 145, 100, 255, 180),
            _det("vest", 120, 220, 280, 380),
        ])
        assert v.status is ComplianceStatus.NO_PERSON


# ---------------------------------------------------------------------------
# Compliant — single person, correctly positioned PPE
# ---------------------------------------------------------------------------

class TestCompliant:
    def _detections(self):
        person = _person()
        helmet = _det("helmet", 150, 100, 250, 190)   # centroid (200, 145) -> head-valid
        vest = _det("vest", 130, 230, 270, 370)        # centroid (200, 300) -> torso-valid
        return [person, helmet, vest]

    def test_person_helmet_vest_is_compliant(self, engine):
        v = engine.evaluate(self._detections())
        assert v.status is ComplianceStatus.COMPLIANT

    def test_compliant_has_no_violations(self, engine):
        v = engine.evaluate(self._detections())
        assert v.violations == []

    def test_compliant_person_count(self, engine):
        v = engine.evaluate(self._detections())
        assert v.person_count == 1

    def test_is_compliant_helper(self, engine):
        v = engine.evaluate(self._detections())
        assert v.is_compliant() is True


# ---------------------------------------------------------------------------
# Helmet exists in frame but is not actually worn (Problem 1/2/3 regression)
# ---------------------------------------------------------------------------

class TestHelmetOutsideHeadRegion:
    def test_helmet_at_torso_height_is_non_compliant(self, engine):
        # Helmet-class box positioned down at torso height, not the head.
        person = _person()
        helmet_at_torso = _det("helmet", 150, 260, 250, 350)
        vest = _det("vest", 130, 230, 270, 370)
        v = engine.evaluate([person, helmet_at_torso, vest])
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations

    def test_helmet_far_to_side_beyond_width_ratio_is_non_compliant(self, engine):
        # High containment vertically, but centroid well outside the
        # horizontal head band (person width 200, half-band 55 -> valid
        # x in [145, 255]). Box centered at x=350 is far outside.
        person = _person()
        helmet_far_side = _det("helmet", 320, 100, 380, 190)
        vest = _det("vest", 130, 230, 270, 370)
        v = engine.evaluate([person, helmet_far_side, vest])
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations


# ---------------------------------------------------------------------------
# Helmet held beside the head — high containment, centroid outside band
# ---------------------------------------------------------------------------

class TestHelmetHeldBesideHead:
    def test_high_containment_but_outside_horizontal_band(self, engine):
        # Box overlaps the person heavily (so containment passes) but is
        # shifted sideways so its centroid falls outside width_ratio band.
        person = _person()
        held_helmet = _det("helmet", 255, 105, 300, 195)  # centroid x=277.5 > 255
        vest = _det("vest", 130, 230, 270, 370)
        v = engine.evaluate([person, held_helmet, vest])
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations


# ---------------------------------------------------------------------------
# Two people — claim-and-remove double-counting regression tests
# ---------------------------------------------------------------------------

class TestTwoPeopleClaimAndRemove:
    def test_only_one_helmet_in_frame_for_two_people(self, engine):
        # Person A (higher confidence) gets the helmet; person B does not.
        person_a = _person(x1=0, y1=0, x2=200, y2=400, confidence=0.95)
        person_b = _person(x1=300, y1=0, x2=500, y2=400, confidence=0.90)
        # Positioned in A's head region: person A center_x=100, band [45,155].
        helmet_for_a = _det("helmet", 70, 0, 130, 90)
        vest_a = _det("vest", 40, 100, 160, 300)
        vest_b = _det("vest", 340, 100, 460, 300)

        v = engine.evaluate([person_a, person_b, helmet_for_a, vest_a, vest_b])

        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations

        results_by_conf = sorted(v.person_results, key=lambda r: r["person_index"])
        # person_index 0 == person_a (sorted by confidence desc, A first)
        assert results_by_conf[0]["helmet_ok"] is True
        assert results_by_conf[1]["helmet_ok"] is False

    def test_two_people_two_helmets_both_compliant(self, engine):
        person_a = _person(x1=0, y1=0, x2=200, y2=400, confidence=0.95)
        person_b = _person(x1=300, y1=0, x2=500, y2=400, confidence=0.90)
        helmet_a = _det("helmet", 70, 0, 130, 90)     # A's head region
        helmet_b = _det("helmet", 370, 0, 430, 90)    # B's head region
        vest_a = _det("vest", 40, 100, 160, 300)
        vest_b = _det("vest", 340, 100, 460, 300)

        v = engine.evaluate([person_a, person_b, helmet_a, helmet_b, vest_a, vest_b])

        assert v.status is ComplianceStatus.COMPLIANT
        assert v.person_count == 2
        assert all(r["helmet_ok"] and r["vest_ok"] for r in v.person_results)


# ---------------------------------------------------------------------------
# Explicit no-helmet / no-vest signals, correctly in-region
# ---------------------------------------------------------------------------

class TestExplicitViolationSignals:
    def test_no_helmet_box_in_region_is_explicit_violation(self, engine):
        person = _person()
        no_helmet = _det("no-helmet", 150, 100, 250, 190)
        vest = _det("vest", 130, 230, 270, 370)
        v = engine.evaluate([person, no_helmet, vest])
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations

    def test_no_vest_box_in_region_is_explicit_violation(self, engine):
        person = _person()
        helmet = _det("helmet", 150, 100, 250, 190)
        no_vest = _det("no-vest", 130, 230, 270, 370)
        v = engine.evaluate([person, helmet, no_vest])
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-vest" in v.violations

    def test_both_no_helmet_and_no_vest(self, engine):
        person = _person()
        no_helmet = _det("no-helmet", 150, 100, 250, 190)
        no_vest = _det("no-vest", 130, 230, 270, 370)
        v = engine.evaluate([person, no_helmet, no_vest])
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations
        assert "no-vest" in v.violations


# ---------------------------------------------------------------------------
# Ambiguous / no-signal case (fail-safe rule)
# ---------------------------------------------------------------------------

class TestAmbiguousNoSignal:
    def test_person_only_no_ppe_is_non_compliant(self, engine):
        v = engine.evaluate([_person()])
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations
        assert "no-vest" in v.violations

    def test_person_with_helmet_but_no_vest_signal(self, engine):
        person = _person()
        helmet = _det("helmet", 150, 100, 250, 190)
        v = engine.evaluate([person, helmet])
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-vest" in v.violations
        assert "no-helmet" not in v.violations

    def test_person_with_vest_but_no_helmet_signal(self, engine):
        person = _person()
        vest = _det("vest", 130, 230, 270, 370)
        v = engine.evaluate([person, vest])
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations
        assert "no-vest" not in v.violations


# ---------------------------------------------------------------------------
# person_results field
# ---------------------------------------------------------------------------

class TestPersonResults:
    def test_person_results_populated_independent_of_aggregate(self, engine):
        person_a = _person(x1=0, y1=0, x2=200, y2=400, confidence=0.95)
        person_b = _person(x1=300, y1=0, x2=500, y2=400, confidence=0.90)
        helmet_a = _det("helmet", 70, 0, 130, 90)
        vest_a = _det("vest", 40, 100, 160, 300)
        # person_b gets nothing.

        v = engine.evaluate([person_a, person_b, helmet_a, vest_a])

        assert len(v.person_results) == 2
        a_result = v.person_results[0]
        b_result = v.person_results[1]

        assert a_result["person_index"] == 0
        assert a_result["helmet_ok"] is True
        assert a_result["vest_ok"] is True
        assert a_result["violations"] == []

        assert b_result["person_index"] == 1
        assert b_result["helmet_ok"] is False
        assert b_result["vest_ok"] is False
        assert set(b_result["violations"]) == {"no-helmet", "no-vest"}

        # Aggregate view still reflects the union, unaffected by per-person detail.
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert set(v.violations) == {"no-helmet", "no-vest"}

    def test_person_results_empty_list_shape_for_no_person(self, engine):
        v = engine.evaluate([])
        assert v.person_results == []


# ---------------------------------------------------------------------------
# Containment threshold behavior
# ---------------------------------------------------------------------------

class TestContainmentThreshold:
    def test_low_containment_in_region_still_rejected(self, engine):
        # Box centroid lands in the head region (200, 100), but the box
        # extends far above the person's top edge (y1=20 vs person y1=100),
        # so containment = 8000/16000 = 0.5, below the 0.60 threshold.
        person = _person()
        mostly_outside_helmet = _det("helmet", 150, 20, 250, 180)
        vest = _det("vest", 130, 230, 270, 370)
        v = engine.evaluate([person, mostly_outside_helmet, vest])
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations


# ---------------------------------------------------------------------------
# Debug diagnostic mode — calibration tool, gated by the debug parameter
# ---------------------------------------------------------------------------

class TestDebugDiagnostics:
    def _detections(self):
        person = _person()
        helmet = _det("helmet", 150, 100, 250, 190)   # head-valid
        vest = _det("vest", 130, 230, 270, 370)        # torso-valid
        return [person, helmet, vest]

    def test_debug_true_produces_log_output(self, engine, caplog):
        with caplog.at_level("DEBUG", logger="core.decision_engine"):
            engine.evaluate(self._detections(), debug=True)
        assert len(caplog.records) > 0
        assert any("claim-check" in r.message for r in caplog.records)
        assert any("claim-outcome" in r.message for r in caplog.records)
        assert any("Person 0:" in r.message for r in caplog.records)
        assert any("Frame summary" in r.message for r in caplog.records)

    def test_debug_false_by_default_produces_no_log_output(self, engine, caplog):
        with caplog.at_level("DEBUG", logger="core.decision_engine"):
            engine.evaluate(self._detections())
        assert len(caplog.records) == 0

    def test_debug_false_explicit_produces_no_log_output(self, engine, caplog):
        with caplog.at_level("DEBUG", logger="core.decision_engine"):
            engine.evaluate(self._detections(), debug=False)
        assert len(caplog.records) == 0

    def test_debug_false_leaves_debug_info_none(self, engine):
        v = engine.evaluate(self._detections())
        assert v.debug_info is None

    def test_debug_false_explicit_leaves_debug_info_none(self, engine):
        v = engine.evaluate(self._detections(), debug=False)
        assert v.debug_info is None

    def test_debug_true_populates_debug_info_structure(self, engine):
        v = engine.evaluate(self._detections(), debug=True)
        assert v.debug_info is not None

        assert v.debug_info["frame"] == {
            "person_count": 1,
            "helmets_assigned": 1,
            "vests_assigned": 1,
            "status": "compliant",
            "violations": [],
        }

        assert len(v.debug_info["persons"]) == 1
        p = v.debug_info["persons"][0]
        assert p["person_index"] == 0
        assert p["person_status"] == "COMPLIANT"
        assert p["helmet"]["assigned"] == "helmet"
        assert p["vest"]["assigned"] == "vest"

        helmet_candidate = p["helmet"]["candidates"][0]
        assert helmet_candidate["class_name"] == "helmet"
        assert helmet_candidate["status"] == "assigned"
        assert helmet_candidate["reasons"] == []
        assert helmet_candidate["containment_ok"] is True
        assert helmet_candidate["region_ok"] is True

    def test_x_axis_and_y_axis_rejections_produce_different_reason_text(self, engine, caplog):
        # Same person both times; one helmet fails only the X band, the
        # other fails only the Y band. The specific reason text must differ
        # so a calibration engineer can tell which axis is the problem.
        person = _person()
        # High containment, but centroid.x=277.5 is outside x in [145,255].
        helmet_x_fail = _det("helmet", 255, 105, 300, 195)
        v_x = engine.evaluate([person, helmet_x_fail], debug=True)
        x_candidate = v_x.debug_info["persons"][0]["helmet"]["candidates"][0]
        assert x_candidate["x_ok"] is False
        assert x_candidate["y_ok"] is True
        assert any("centroid X outside" in r for r in x_candidate["reasons"])
        assert not any("centroid Y outside" in r for r in x_candidate["reasons"])

        person2 = _person()
        # centroid.y=260 is outside head y in [100,200]; x=200 is in-band.
        helmet_y_fail = _det("helmet", 150, 230, 250, 290)
        v_y = engine.evaluate([person2, helmet_y_fail], debug=True)
        y_candidate = v_y.debug_info["persons"][0]["helmet"]["candidates"][0]
        assert y_candidate["y_ok"] is False
        assert y_candidate["x_ok"] is True
        assert any("centroid Y outside" in r for r in y_candidate["reasons"])
        assert not any("centroid X outside" in r for r in y_candidate["reasons"])

        # The distinguishing text must actually differ between the two cases.
        assert x_candidate["reasons"] != y_candidate["reasons"]

    def test_claimed_by_other_reported_for_later_person(self, engine):
        # Person A (higher confidence, processed first) and person C both
        # have head regions overlapping the same physical helmet box —
        # A claims it; C's candidate list must report it as
        # claimed_by_other, not omit it silently.
        person_a = _person(x1=0, y1=0, x2=200, y2=400, confidence=0.95)
        person_c = _person(x1=50, y1=0, x2=250, y2=400, confidence=0.80)
        helmet_for_a = _det("helmet", 70, 0, 130, 90)

        v = engine.evaluate([person_a, person_c, helmet_for_a], debug=True)

        c_helmet_candidates = v.debug_info["persons"][1]["helmet"]["candidates"]
        claimed_entries = [c for c in c_helmet_candidates if c["status"] == "claimed_by_other"]
        assert len(claimed_entries) == 1
        assert claimed_entries[0]["claimed_by"] == 0
        assert "already claimed by person 0" in claimed_entries[0]["reasons"][0]
