"""
Abstract base class for all alert handlers.

Contract
--------
Every handler implements a single method:

    handle(session_state, frame_verdict, frame) -> None

Parameters
----------
session_state : SessionState
    The current session controller state at the time of the alert call.
    Handlers use this to decide whether to act (e.g. only on GRANTED/DENIED).

frame_verdict : FrameVerdict
    The compliance verdict for the current frame.  Handlers that need
    violation details (e.g. which PPE was missing) read it from here.

frame : numpy.ndarray | None
    The current BGR camera frame, as returned by shared_state.get_frame().
    May be None if the camera has not yet produced a frame.  Handlers that
    save or display the frame must check for None before use.

Design notes
------------
- Handlers are called on EVERY pipeline loop iteration (not just on
  session_state transitions).  Each handler is responsible for its own
  debounce / rate-limiting logic if it should not fire every frame.
- One handler raising an exception must NOT prevent others from running.
  AlertManager catches and logs per-handler exceptions.
"""

from abc import ABC, abstractmethod


class BaseHandler(ABC):
    @abstractmethod
    def handle(self, session_state, frame_verdict, frame) -> None:
        """Process one pipeline cycle's worth of state."""
