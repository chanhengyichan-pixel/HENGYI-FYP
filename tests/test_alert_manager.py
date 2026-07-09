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
