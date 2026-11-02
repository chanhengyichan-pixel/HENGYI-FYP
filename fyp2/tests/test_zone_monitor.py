"""
Unit tests for core/zone_monitor.py.

Detections are SimpleNamespace stubs; the FrameVerdict is built by hand so
each test controls exactly which person is missing which PPE. Time comes
from a list-backed fake clock, as in test_session_controller.py.
"""

from types import SimpleNamespace

import pytest

from core.decision_engine import ComplianceStatus, FrameVerdict
from core.zone_monitor import HAZARD_PPE, RESTRICTED_ENTRY, ZoneMonitor

# Hazard zone on the left half of the floor, restricted on the right.
CONFIG = {"zones": {
    "hazard_seconds": 2.0,
    "restricted_seconds": 1.0,
    "grace_seconds": 1.0,
    "cooldown_seconds": 30.0,
    "areas": [
        {"name": "Machine", "type": "hazard",
         "polygon": [[0, 200], [300, 200], [300, 480], [0, 480]]},
        {"name": "Server", "type": "restricted",
         "polygon": [[340, 200], [640, 200], [640, 480], [340, 480]]},
    ],
}}

IN_HAZARD = (100, 50, 200, 400)       # feet at (150, 400)
IN_RESTRICTED = (450, 50, 550, 400)   # feet at (500, 400)
OUTSIDE = (100, 0, 200, 150)          # feet at (150, 150) — above both zones


def person(box, track_id=1):
    x1, y1, x2, y2 = box
    return SimpleNamespace(class_name="person", confidence=0.9, x1=x1, y1=y1,
                           x2=x2, y2=y2, track_id=track_id)


def verdict(*people_violations):
    """people_violations: (detection, [violations]) pairs."""
    results = [{"person_index": i, "box": (d.x1, d.y1, d.x2, d.y2),
                "helmet_ok": "no-helmet" not in v, "vest_ok": "no-vest" not in v,
                "violations": v}
               for i, (d, v) in enumerate(people_violations)]
    return FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                        person_results=results, person_count=len(results))


def make_monitor():
    clock = [0.0]
    return ZoneMonitor(CONFIG, clock=lambda: clock[0]), clock


def run(monitor, clock, times, dets_and_violations):
    """Feed the same people at each time in `times`; return all fired events."""
    fired = []
    for t in times:
        clock[0] = t
        dets = [d for d, _ in dets_and_violations]
        events, _ = monitor.update(dets, verdict(*dets_and_violations))
        fired.extend(events)
    return fired


class TestHazard:
    def test_missing_ppe_fires_after_hazard_seconds(self):
        m, clock = make_monitor()
        p = person(IN_HAZARD)
        events = run(m, clock, [0.0, 0.5, 1.0, 1.5], [(p, ["no-helmet"])])
        assert events == []
        events = run(m, clock, [2.0], [(p, ["no-helmet"])])
        assert len(events) == 1
        assert events[0].event_type == HAZARD_PPE
        assert events[0].zone == "Machine"
        assert events[0].violations == ["no-helmet"]
        assert events[0].severity == "high"

    def test_full_ppe_in_hazard_zone_never_fires(self):
        m, clock = make_monitor()
        p = person(IN_HAZARD)
        events = run(m, clock, [i * 0.5 for i in range(10)], [(p, [])])
        assert events == []

    def test_short_dropout_within_grace_keeps_timer(self):
        m, clock = make_monitor()
        p = person(IN_HAZARD)
        run(m, clock, [0.0, 1.0], [(p, ["no-vest"])])
        run(m, clock, [1.3], [])                  # person missed for one frame
        events = run(m, clock, [1.4, 2.0], [(p, ["no-vest"])])
        assert [e.event_type for e in events] == [HAZARD_PPE]

    def test_gap_longer_than_grace_restarts_timer(self):
        m, clock = make_monitor()
        p = person(IN_HAZARD)
        run(m, clock, [0.0, 1.5], [(p, ["no-vest"])])
        events = run(m, clock, [3.0, 3.5, 4.5], [(p, ["no-vest"])])  # 1.5 s gap
        assert events == []                        # restarted at 3.0, needs 5.0


class TestRestricted:
    def test_anyone_fires_after_restricted_seconds(self):
        m, clock = make_monitor()
        p = person(IN_RESTRICTED)
        events = run(m, clock, [0.0, 0.5, 1.0], [(p, [])])  # full PPE doesn't matter
        assert [e.event_type for e in events] == [RESTRICTED_ENTRY]
        assert events[0].severity == "critical"
        assert events[0].violations == []

    def test_outside_zone_never_fires(self):
        m, clock = make_monitor()
        p = person(OUTSIDE)
        assert run(m, clock, [i * 0.5 for i in range(10)], [(p, ["no-helmet"])]) == []


class TestCooldown:
    def test_same_person_fires_once_per_cooldown(self):
        m, clock = make_monitor()
        p = person(IN_RESTRICTED)
        events = run(m, clock, [i * 0.5 for i in range(40)], [(p, [])])   # 20 s inside
        assert len(events) == 1

    def test_fires_again_after_cooldown(self):
        m, clock = make_monitor()
        p = person(IN_RESTRICTED)
        events = run(m, clock, [i * 0.5 for i in range(70)], [(p, [])])   # 35 s inside
        assert len(events) == 2

    def test_cooldown_is_per_person(self):
        m, clock = make_monitor()
        a = person(IN_RESTRICTED, track_id=1)
        b = person((560, 50, 620, 400), track_id=2)
        events = run(m, clock, [0.0, 0.5, 1.0], [(a, []), (b, [])])
        assert sorted(e.track_id for e in events) == [1, 2]


class TestUntracked:
    def test_person_without_track_id_ignored(self):
        m, clock = make_monitor()
        p = person(IN_RESTRICTED, track_id=None)
        assert run(m, clock, [0.0, 1.0, 2.0], [(p, [])]) == []

    def test_non_person_detections_ignored(self):
        m, clock = make_monitor()
        helmet = SimpleNamespace(class_name="helmet", confidence=0.9, x1=450, y1=50,
                                 x2=500, y2=400, track_id=7)
        clock[0] = 5.0
        events, status = m.update([helmet], verdict())
        assert events == [] and status == []


class TestStatus:
    def test_status_reports_zone_and_seconds(self):
        m, clock = make_monitor()
        p = person(IN_HAZARD)
        run(m, clock, [0.0], [(p, ["no-helmet"])])
        clock[0] = 0.8
        _, status = m.update([p], verdict((p, ["no-helmet"])))
        assert status == [{"track_id": 1, "zone": "Machine", "zone_type": "hazard",
                           "seconds": 0.8, "ppe_ok": False, "violations": ["no-helmet"]}]

    def test_person_missing_from_verdict_is_treated_as_no_ppe(self):
        # Fail-safe: if PPE attribution is unavailable, assume not equipped.
        m, clock = make_monitor()
        p = person(IN_HAZARD)
        fired = []
        for t in (0.0, 1.0, 2.0):
            clock[0] = t
            events, _ = m.update([p], FrameVerdict(status=ComplianceStatus.NO_PERSON))
            fired.extend(events)
        assert [e.event_type for e in fired] == [HAZARD_PPE]


class TestFlickerFilter:
    def _monitor(self, ratio):
        cfg = {"zones": {**CONFIG["zones"], "hazard_min_missing_ratio": ratio}}
        clock = [0.0]
        return ZoneMonitor(cfg, clock=lambda: clock[0]), clock

    def _feed(self, m, clock, pattern):
        """pattern: list of (time, violations) for one person in the hazard zone."""
        p = person(IN_HAZARD)
        fired = []
        for t, v in pattern:
            clock[0] = t
            events, _ = m.update([p], verdict((p, v)))
            fired.extend(events)
        return fired

    def _flicker(self):
        # Worn helmet that drops out every other frame for 3 s (50% missing).
        return [(i * 0.25, ["no-helmet"] if i % 2 == 0 else []) for i in range(13)]

    def test_flicker_fires_without_filter(self):
        m, clock = self._monitor(0.0)
        assert len(self._feed(m, clock, self._flicker())) == 1

    def test_flicker_ignored_with_filter(self):
        m, clock = self._monitor(0.7)
        assert self._feed(m, clock, self._flicker()) == []

    def test_really_missing_still_fires_with_filter(self):
        m, clock = self._monitor(0.7)
        # Missing in 8 of 9 frames over 2 s.
        pattern = [(i * 0.25, [] if i == 4 else ["no-helmet"]) for i in range(9)]
        events = self._feed(m, clock, pattern)
        assert len(events) == 1 and events[0].event_type == HAZARD_PPE
