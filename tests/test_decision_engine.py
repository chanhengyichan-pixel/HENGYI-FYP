"""
Unit tests for core/decision_engine.py.

No camera, no model, no YOLO import.  Inputs are _Det stubs that
satisfy the duck-type contract (any object with .class_name: str).
"""

import pytest
from types import SimpleNamespace

from core.decision_engine import DecisionEngine, ComplianceStatus, FrameVerdict


def _det(*class_names):
    """Return a list of detection stubs with the given class names."""
    return [SimpleNamespace(class_name=name) for name in class_names]


engine = DecisionEngine()


# ---------------------------------------------------------------------------
# No person
# ---------------------------------------------------------------------------

class TestNoPerson:
    def test_empty_list_returns_no_person(self):
        v = engine.evaluate(_det())
        assert v.status is ComplianceStatus.NO_PERSON

    def test_no_person_has_zero_person_count(self):
        v = engine.evaluate(_det())
        assert v.person_count == 0

    def test_only_ppe_no_person_returns_no_person(self):
        # Helmet and vest visible but no person bounding box — odd but possible.
        v = engine.evaluate(_det("helmet", "vest"))
        assert v.status is ComplianceStatus.NO_PERSON


# ---------------------------------------------------------------------------
# Compliant
# ---------------------------------------------------------------------------

class TestCompliant:
    def test_person_helmet_vest_is_compliant(self):
        v = engine.evaluate(_det("person", "helmet", "vest"))
        assert v.status is ComplianceStatus.COMPLIANT

    def test_compliant_has_no_violations(self):
        v = engine.evaluate(_det("person", "helmet", "vest"))
        assert v.violations == []

    def test_compliant_person_count(self):
        v = engine.evaluate(_det("person", "helmet", "vest"))
        assert v.person_count == 1

    def test_is_compliant_helper(self):
        v = engine.evaluate(_det("person", "helmet", "vest"))
        assert v.is_compliant() is True


# ---------------------------------------------------------------------------
# Non-compliant — explicit violation signals
# ---------------------------------------------------------------------------

class TestExplicitViolations:
    def test_no_helmet_signal_is_non_compliant(self):
        v = engine.evaluate(_det("person", "no-helmet", "vest"))
        assert v.status is ComplianceStatus.NON_COMPLIANT

    def test_no_vest_signal_is_non_compliant(self):
        v = engine.evaluate(_det("person", "helmet", "no-vest"))
        assert v.status is ComplianceStatus.NON_COMPLIANT

    def test_both_violation_signals(self):
        v = engine.evaluate(_det("person", "no-helmet", "no-vest"))
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations
        assert "no-vest" in v.violations

    def test_violations_listed_correctly(self):
        v = engine.evaluate(_det("person", "no-helmet", "vest"))
        assert "no-helmet" in v.violations

    def test_is_compliant_false_on_violation(self):
        v = engine.evaluate(_det("person", "no-helmet", "vest"))
        assert v.is_compliant() is False


# ---------------------------------------------------------------------------
# Ambiguous / no-signal case (fail-safe rule)
# ---------------------------------------------------------------------------

class TestAmbiguousNoSignal:
    def test_person_only_no_ppe_is_non_compliant(self):
        # Person detected, no PPE box of any polarity — ambiguous, fail-safe.
        v = engine.evaluate(_det("person"))
        assert v.status is ComplianceStatus.NON_COMPLIANT

    def test_person_only_violations_include_missing_ppe(self):
        v = engine.evaluate(_det("person"))
        assert "no-helmet" in v.violations
        assert "no-vest" in v.violations

    def test_person_with_helmet_but_no_vest_signal(self):
        # Helmet confirmed, vest signal absent entirely — fail-safe on vest.
        v = engine.evaluate(_det("person", "helmet"))
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-vest" in v.violations
        assert "no-helmet" not in v.violations

    def test_person_with_vest_but_no_helmet_signal(self):
        v = engine.evaluate(_det("person", "vest"))
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-helmet" in v.violations
        assert "no-vest" not in v.violations


# ---------------------------------------------------------------------------
# Multiple people
# ---------------------------------------------------------------------------

class TestMultiplePeople:
    def test_two_compliant_people(self):
        # Two full sets of PPE — both compliant.
        v = engine.evaluate(_det("person", "helmet", "vest", "person", "helmet", "vest"))
        assert v.status is ComplianceStatus.COMPLIANT
        assert v.person_count == 2

    def test_one_compliant_one_non_compliant_person(self):
        # One person has full PPE; other has a no-helmet signal.
        # Frame must be NON_COMPLIANT — any violation blocks the gate.
        v = engine.evaluate(_det("person", "helmet", "vest", "person", "no-helmet", "vest"))
        assert v.status is ComplianceStatus.NON_COMPLIANT

    def test_non_compliant_person_count_is_total(self):
        v = engine.evaluate(_det("person", "no-helmet", "person", "helmet", "vest"))
        assert v.person_count == 2

    def test_two_people_no_ppe_signals_non_compliant(self):
        v = engine.evaluate(_det("person", "person"))
        assert v.status is ComplianceStatus.NON_COMPLIANT


# ---------------------------------------------------------------------------
# Multi-person PPE masking (count comparison fix)
# ---------------------------------------------------------------------------

class TestMultiPersonMasking:
    def test_person_a_full_ppe_person_b_no_signal(self):
        # 2 people, only 1 set of PPE boxes — second person is uncovered.
        # helmet_count(1) < person_count(2) → NON_COMPLIANT.
        v = engine.evaluate(_det("person", "helmet", "vest", "person"))
        assert v.status is ComplianceStatus.NON_COMPLIANT

    def test_person_a_full_ppe_person_b_helmet_only(self):
        # 2 people, 2 helmets but only 1 vest — vest count insufficient.
        v = engine.evaluate(_det("person", "helmet", "vest", "person", "helmet"))
        assert v.status is ComplianceStatus.NON_COMPLIANT
        assert "no-vest" in v.violations

    def test_three_people_two_sets_of_ppe(self):
        # 3 people, only 2 helmets and 2 vests — count falls short for 3rd person.
        v = engine.evaluate(_det("person", "helmet", "vest",
                                 "person", "helmet", "vest",
                                 "person"))
        assert v.status is ComplianceStatus.NON_COMPLIANT

    def test_two_people_two_full_sets_compliant(self):
        # 2 people, 2 helmets, 2 vests — counts match, no violations.
        v = engine.evaluate(_det("person", "helmet", "vest",
                                 "person", "helmet", "vest"))
        assert v.status is ComplianceStatus.COMPLIANT
        assert v.person_count == 2
