"""
EventLogHandler — records GRANTED/DENIED decision events to the
SQLite-backed EventStore.

Implemented as a BaseHandler so it slots into the AlertManager
dispatcher alongside ScreenshotHandler and VoiceHandler.  "Record this
decision" is semantically identical to "screenshot this decision" —
keeping it as a handler means zero pipeline.py changes and the same
fan-out guarantee (exception in one handler doesn't prevent others).

Debounce: identical to ScreenshotHandler._last_captured_state — records
once per state entry, resets on any non-trigger state.

Event format stored:
    {
        "timestamp":   "2026-07-09 07:15:12",  # local time, human-readable
        "verdict":     "granted" | "denied",
        "violations":  [],                      # from frame_verdict.violations
        "worker_name": "Alice" | "unknown",     # None coerced to "unknown"
    }

worker_name coercion: None (unidentified / face recognition not configured)
is stored as the string "unknown" for display consistency.  The distinction
between "face recognition off" and "face visible but no match" is not
preserved at the storage layer — both appear as "unknown" in the event log.
"""

import logging
from datetime import datetime

from alerts.base_handler import BaseHandler
from core.session_controller import SessionState

logger = logging.getLogger(__name__)

_TRIGGER_STATES = {SessionState.GRANTED, SessionState.DENIED}


class EventLogHandler(BaseHandler):
    def __init__(self, event_store):
        self._event_store = event_store
        self._last_recorded_state = None

    def handle(self, session_state, frame_verdict, frame,
               worker_name: str | None = None) -> None:
        if session_state not in _TRIGGER_STATES:
            self._last_recorded_state = None
            return

        if session_state == self._last_recorded_state:
            return  # debounce

        self._last_recorded_state = session_state

        event = {
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "verdict":     session_state.value,
            "violations":  list(frame_verdict.violations),
            "worker_name": worker_name if worker_name is not None else "unknown",
        }
        self._event_store.append(event)
        logger.info("Event logged: %s", event)
