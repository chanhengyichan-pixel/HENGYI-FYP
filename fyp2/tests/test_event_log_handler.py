"""Unit tests for alerts/event_log_handler.py with a real EventStore."""

from core.decision_engine import ComplianceStatus, FrameVerdict
from core.session_controller import SessionState
from core.zone_monitor import HAZARD_PPE, RESTRICTED_ENTRY, ZoneEvent
from alerts.event_log_handler import EventLogHandler
from storage.event_store import EventStore

DENIED_VERDICT = FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                              violations=["no-helmet"], person_count=1)


def make(tmp_path):
    store = EventStore({"storage": {"db_path": str(tmp_path / "e.db")}})
    return EventLogHandler(store), store


class TestEntrance:
    def test_denied_logged_once_per_hold(self, tmp_path):
        h, store = make(tmp_path)
        for _ in range(5):
            h.handle(SessionState.DENIED, DENIED_VERDICT, None)
        events = store.get_recent(10)
        assert len(events) == 1
        e = events[0]
        assert e["event_type"] == "ENTRY_DENIED"
        assert e["camera"] == "entrance"
        assert e["severity"] == "warning"
        assert e["violations"] == ["no-helmet"]

    def test_logs_again_after_state_resets(self, tmp_path):
        h, store = make(tmp_path)
        h.handle(SessionState.GRANTED, FrameVerdict(status=ComplianceStatus.COMPLIANT), None)
        h.handle(SessionState.COOLDOWN, FrameVerdict(status=ComplianceStatus.NO_PERSON), None)
        h.handle(SessionState.GRANTED, FrameVerdict(status=ComplianceStatus.COMPLIANT), None)
        assert len(store.get_recent(10)) == 2


class TestZoneEvents:
    def test_hazard_event_stored_with_zone_and_snapshot(self, tmp_path):
        h, store = make(tmp_path)
        ev = ZoneEvent(event_type=HAZARD_PPE, zone="Machine", track_id=4,
                       violations=["no-vest"], snapshot="output/screenshots/a.jpg")
        h.handle_zone_event(ev, None)
        e = store.get_recent(1)[0]
        assert e["event_type"] == HAZARD_PPE
        assert e["camera"] == "room"
        assert e["zone"] == "Machine"
        assert e["severity"] == "high"
        assert e["violations"] == ["no-vest"]
        assert e["snapshot"] == "output/screenshots/a.jpg"

    def test_restricted_event_is_critical(self, tmp_path):
        h, store = make(tmp_path)
        h.handle_zone_event(ZoneEvent(event_type=RESTRICTED_ENTRY, zone="Server",
                                      track_id=1), None)
        assert store.get_recent(1)[0]["severity"] == "critical"
