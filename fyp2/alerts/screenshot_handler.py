"""
ScreenshotHandler — saves a frame to disk when a GRANTED or DENIED
decision is reached, and for every room zone event (HAZARD_PPE,
RESTRICTED_ENTRY).

Fires exactly once per state entry, not once per frame during the hold.

Debounce pattern
----------------
_last_captured_state tracks the SessionState for which a screenshot was
most recently saved.  A screenshot is taken only when:
  - session_state is GRANTED or DENIED, AND
  - session_state differs from _last_captured_state (new entry, not a
    continuation of the same hold).

_last_captured_state is reset to None whenever the state is anything
other than GRANTED or DENIED, so the next time either of those states
is entered, the handler fires again.  This is the same single-flag
pattern used by Pipeline._was_disconnected — one boolean (or value)
tracks a transition, not a counter or timer.

Frame handling
--------------
If frame is None (camera not ready), the screenshot is skipped and a
warning is logged.  The _last_captured_state is still updated so the
handler does not retry on every subsequent frame of the same hold.
"""

import logging
from datetime import datetime
from pathlib import Path

import cv2

from alerts.base_handler import BaseHandler
from core.session_controller import SessionState

logger = logging.getLogger(__name__)

_TRIGGER_STATES = {SessionState.GRANTED, SessionState.DENIED}


class ScreenshotHandler(BaseHandler):
    def __init__(self, config: dict, camera: str = "entrance"):
        # camera: which entrance camera this instance serves (one instance
        # per entrance camera, so each has its own debounce state).
        self._camera = camera
        # Fail fast on missing key — consistent with camera/detector/session config
        # handling. A misconfigured path is caught at startup, not silently at
        # the first GRANTED/DENIED event mid-run.
        screenshot_dir = config["alerts"]["screenshot_dir"]
        self._screenshot_dir = Path(screenshot_dir)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_captured_state = None

    def handle(self, session_state, frame_verdict, frame) -> None:
        if session_state not in _TRIGGER_STATES:
            # Outside a decision hold — reset so next entry fires.
            self._last_captured_state = None
            return

        if session_state == self._last_captured_state:
            # Already fired for this hold — debounce.
            return

        self._last_captured_state = session_state
        verdict_label = session_state.value  # "granted" or "denied"

        if frame is None:
            logger.warning(
                "ScreenshotHandler: no frame available at %s — skipping save.",
                verdict_label,
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "" if self._camera == "entrance" else f"_{self._camera}"
        filename = self._screenshot_dir / f"{timestamp}_{verdict_label}{suffix}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info("Screenshot saved: %s", filename)

    def handle_zone_event(self, event, frame) -> None:
        """Save the room frame for a zone event and record its path on the event."""
        if frame is None:
            logger.warning("ScreenshotHandler: no frame for %s — skipping save.",
                           event.event_type)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (self._screenshot_dir /
                    f"{timestamp}_{event.event_type.lower()}_{event.camera}.jpg")
        cv2.imwrite(str(filename), frame)
        event.snapshot = str(filename)
        logger.info("Screenshot saved: %s", filename)
