"""
Unit tests for alerts/alert_manager.py.

No camera, no model.  Handlers are MagicMocks or minimal fakes.
"""

import logging
import pytest
from unittest.mock import MagicMock, call

from alerts.alert_manager import AlertManager
from alerts.base_handler import BaseHandler
from core.decision_engine import ComplianceStatus, FrameVerdict
from core.session_controller import SessionState
from core.zone_monitor import RESTRICTED_ENTRY, ZoneEvent

IDLE = SessionState.IDLE
VERDICT = FrameVerdict(status=ComplianceStatus.NO_PERSON)
FRAME = None


def make_handler():
    h = MagicMock(spec=BaseHandler)
    return h


class TestAllHandlersCalled:
    def test_single_handler_called(self):
        h = make_handler()
        am = AlertManager([h])
        am.notify(IDLE, VERDICT, FRAME)
        h.handle.assert_called_once_with(IDLE, VERDICT, FRAME)

    def test_multiple_handlers_all_called(self):
        h1, h2, h3 = make_handler(), make_handler(), make_handler()
        am = AlertManager([h1, h2, h3])
        am.notify(IDLE, VERDICT, FRAME)
        h1.handle.assert_called_once_with(IDLE, VERDICT, FRAME)
        h2.handle.assert_called_once_with(IDLE, VERDICT, FRAME)
        h3.handle.assert_called_once_with(IDLE, VERDICT, FRAME)

    def test_correct_arguments_forwarded(self):
        h = make_handler()
        am = AlertManager([h])
        verdict = FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                               violations=["no-helmet"], person_count=1)
        am.notify(SessionState.DENIED, verdict, b"fake_frame")
        h.handle.assert_called_once_with(SessionState.DENIED, verdict, b"fake_frame")

    def test_empty_handler_list_does_not_raise(self):
        am = AlertManager([])
        am.notify(IDLE, VERDICT, FRAME)  # should not raise


class TestExceptionIsolation:
    def test_one_handler_raising_does_not_stop_others(self):
        bad = make_handler()
        bad.handle.side_effect = RuntimeError("handler blew up")
        good = make_handler()

        am = AlertManager([bad, good])
        am.notify(IDLE, VERDICT, FRAME)  # must not raise

        good.handle.assert_called_once_with(IDLE, VERDICT, FRAME)

    def test_exception_is_logged_not_re_raised(self, caplog):
        bad = make_handler()
        bad.handle.side_effect = ValueError("oops")

        am = AlertManager([bad])
        with caplog.at_level(logging.ERROR, logger="alerts.alert_manager"):
            am.notify(IDLE, VERDICT, FRAME)

        assert any("MagicMock" in r.message or "raised" in r.message
                   for r in caplog.records)

    def test_second_handler_still_called_after_first_raises(self):
        h1 = make_handler()
        h1.handle.side_effect = Exception("first fails")
        h2 = make_handler()
        h3 = make_handler()
        h3.handle.side_effect = Exception("third also fails")

        am = AlertManager([h1, h2, h3])
        am.notify(IDLE, VERDICT, FRAME)

        h2.handle.assert_called_once()


class TestZoneEvents:
    def test_zone_event_forwarded_to_every_handler_in_order(self):
        calls = []
        h1, h2 = make_handler(), make_handler()
        h1.handle_zone_event.side_effect = lambda e, f: calls.append("h1")
        h2.handle_zone_event.side_effect = lambda e, f: calls.append("h2")
        event = ZoneEvent(event_type=RESTRICTED_ENTRY, zone="Z", track_id=3)
        AlertManager([h1, h2]).notify_zone(event, b"frame")
        h1.handle_zone_event.assert_called_once_with(event, b"frame")
        assert calls == ["h1", "h2"]

    def test_zone_handler_raising_does_not_stop_others(self):
        bad, good = make_handler(), make_handler()
        bad.handle_zone_event.side_effect = RuntimeError("boom")
        event = ZoneEvent(event_type=RESTRICTED_ENTRY, zone="Z", track_id=3)
        AlertManager([bad, good]).notify_zone(event, None)
        good.handle_zone_event.assert_called_once()

    def test_base_handler_ignores_zone_events_by_default(self):
        class EntranceOnly(BaseHandler):
            def handle(self, session_state, frame_verdict, frame):
                pass
        EntranceOnly().handle_zone_event(
            ZoneEvent(event_type=RESTRICTED_ENTRY, zone="Z", track_id=1), None)
