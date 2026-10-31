"""
Abstract base class for all alert handlers.

Contract
--------
Every handler implements:

    handle(session_state, frame_verdict, frame) -> None

and may override:

    handle_zone_event(event, frame) -> None      (default: do nothing)

handle() — entrance check point
-------------------------------
session_state : SessionState
    The current session controller state at the time of the alert call.
    Handlers use this to decide whether to act (e.g. only on GRANTED/DENIED).

frame_verdict : FrameVerdict
    The compliance verdict for the current frame.  Handlers that need
    violation details (e.g. which PPE was missing) read it from here.

frame : numpy.ndarray | None
    The current BGR camera frame.  May be None if the camera has not yet
    produced a frame.  Handlers that save or display the frame must check
    for None before use.

handle_zone_event() — room camera
---------------------------------
event : core.zone_monitor.ZoneEvent
    A HAZARD_PPE or RESTRICTED_ENTRY event that fired on this frame.
    Called once per event (ZoneMonitor already applies the per-person
    cooldown), so handlers do not need their own debounce here.
    The event is mutable: ScreenshotHandler sets event.snapshot, which
    EventLogHandler (registered after it) stores.

frame : numpy.ndarray
    The room camera frame the event fired on.

Design notes
------------
- handle() is called on EVERY entrance pipeline iteration (not just on
  session_state transitions).  Each handler is responsible for its own
  debounce / rate-limiting logic if it should not fire every frame.
- The entrance and room pipelines run in different threads, so a handler
  that keeps state shared by both methods must guard it with a lock.
- One handler raising an exception must NOT prevent others from running.
  AlertManager catches and logs per-handler exceptions.
"""

from abc import ABC, abstractmethod


class BaseHandler(ABC):
    @abstractmethod
    def handle(self, session_state, frame_verdict, frame) -> None:
        """Process one entrance pipeline cycle's worth of state."""

    def handle_zone_event(self, event, frame) -> None:
        """Process one fired room zone event. Default: ignore."""
