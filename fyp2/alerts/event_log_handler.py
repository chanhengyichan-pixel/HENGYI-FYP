"""
EventLogHandler — records events to the SQLite-backed EventStore.

Two sources:
  handle()            — entrance GRANTED/DENIED decisions (debounced: once
                        per state entry, same pattern as ScreenshotHandler)
  handle_zone_event() — room HAZARD_PPE / RESTRICTED_ENTRY events (already
                        rate-limited per person by ZoneMonitor's cooldown)

Implemented as a BaseHandler so it slots into the AlertManager
dispatcher alongside ScreenshotHandler, VoiceHandler and LedHandler —
same fan-out guarantee (an exception in one handler doesn't prevent
others).

Event format stored (see storage/event_store.py):
    {
        "timestamp":  "2026-07-09 07:15:12",   # local time, human-readable
        "event_type": "ENTRY_GRANTED" | "ENTRY_DENIED"
                      | "HAZARD_PPE" | "RESTRICTED_ENTRY",
        "verdict":    "granted" | "denied" | "violation",
        "camera":     "entrance" | "room",
        "zone":       None | "Machine area" | ...,
        "severity":   "info" | "warning" | "high" | "critical",
        "violations": ["no-helmet", ...],
        "snapshot":   None | "output/screenshots/....jpg",
    }

No personal data is stored: there is no worker name or face data, only
what happened, where, and when.
"""

import logging
from datetime import datetime

from alerts.base_handler import BaseHandler
from core.session_controller import SessionState

logger = logging.getLogger(__name__)

_TRIGGER_STATES = {SessionState.GRANTED, SessionState.DENIED}

_ENTRANCE_EVENT = {
    SessionState.GRANTED: ("ENTRY_GRANTED", "info"),
    SessionState.DENIED:  ("ENTRY_DENIED", "warning"),
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class EventLogHandler(BaseHandler):
    def __init__(self, event_store, camera: str = "entrance"):
        self._event_store = event_store
        # Which entrance camera's decisions this instance logs (one
        # instance per entrance camera, each with its own debounce).
        self._camera = camera
        self._last_recorded_state = None

    def handle(self, session_state, frame_verdict, frame) -> None:
        if session_state not in _TRIGGER_STATES:
            self._last_recorded_state = None
            return

        if session_state == self._last_recorded_state:
            return  # debounce

        self._last_recorded_state = session_state
        event_type, severity = _ENTRANCE_EVENT[session_state]

        event = {
            "timestamp":  _now(),
            "event_type": event_type,
            "verdict":    session_state.value,
            "camera":     self._camera,
            "zone":       None,
            "severity":   severity,
            "violations": list(frame_verdict.violations),
            "snapshot":   None,
        }
        self._event_store.append(event)
        logger.info("Event logged: %s", event)

    def handle_zone_event(self, event, frame) -> None:
        record = {
            "timestamp":  _now(),
            "event_type": event.event_type,
            "verdict":    "violation",
            "camera":     event.camera,
            "zone":       event.zone,
            "severity":   event.severity,
            "violations": list(event.violations),
            "snapshot":   event.snapshot,
        }
        self._event_store.append(record)
        logger.info("Event logged: %s", record)
