"""
AlertManager — event dispatcher for alert handlers.

Constructed with a list of BaseHandler instances (dependency-injected).
Calls handle() on every registered handler for each pipeline cycle.

Exception isolation
-------------------
Each handler is called inside an individual try/except.  If a handler
raises, the exception is logged and execution continues to the next
handler.  One broken handler must not silently prevent others from
firing — e.g. a failing voice handler must not block the screenshot
handler from saving evidence of a denial.
"""

import logging

from alerts.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, handlers: list[BaseHandler]):
        self._handlers = handlers

    def notify(self, session_state, frame_verdict, frame) -> None:
        """Dispatch the current pipeline state to all registered handlers."""
        for handler in self._handlers:
            try:
                handler.handle(session_state, frame_verdict, frame)
            except Exception:
                logger.exception(
                    "Handler %s raised an unhandled exception — skipping.",
                    type(handler).__name__,
                )
