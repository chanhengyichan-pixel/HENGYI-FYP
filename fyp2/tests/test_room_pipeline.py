"""
Unit tests for core/room_pipeline.py.

Real DecisionEngine and ZoneMonitor (they are pure logic); the detector is
a stub returning fixed tracked detections, so no model is needed.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from core.decision_engine import DecisionEngine
from core.room_pipeline import RoomPipeline
from core.shared_state import SharedState
from core.zone_monitor import RESTRICTED_ENTRY, ZoneMonitor

CONFIG = {
    "decision_engine": {"containment_threshold": 0.6, "debug": False,
                        "head": {"top_ratio": 0.25, "width_ratio": 0.55},
                        "torso": {"top_ratio": 0.25, "bottom_ratio": 0.75,
                                  "width_ratio": 0.8}},
    "zones": {"hazard_seconds": 2.0, "restricted_seconds": 1.0,
              "grace_seconds": 1.0, "cooldown_seconds": 30.0,
              "areas": [{"name": "Server", "type": "restricted",
                         "polygon": [[340, 200], [640, 200], [640, 480], [340, 480]]}]},
}

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
PERSON_IN_RESTRICTED = SimpleNamespace(class_name="person", confidence=0.9,
                                       x1=450, y1=50, x2=550, y2=400, track_id=5)


def make(clock):
    state = SharedState()
    detector = MagicMock()
    detector.track.return_value = [PERSON_IN_RESTRICTED]
    alerts = MagicMock()
    monitor = ZoneMonitor(CONFIG, clock=lambda: clock[0])
    rp = RoomPipeline(CONFIG, state, detector, DecisionEngine(CONFIG), monitor, alerts)
    return rp, state, detector, alerts


class TestProcess:
    def test_restricted_event_dispatched_after_threshold(self):
        clock = [0.0]
        rp, state, _, alerts = make(clock)
        rp.process(FRAME)
        alerts.notify_zone.assert_not_called()
        clock[0] = 1.0
        rp.process(FRAME)
        alerts.notify_zone.assert_called_once()
        event, frame = alerts.notify_zone.call_args[0]
        assert event.event_type == RESTRICTED_ENTRY and event.track_id == 5

    def test_shared_state_gets_detections_and_zone_status(self):
        clock = [0.0]
        rp, state, _, _ = make(clock)
        rp.process(FRAME)
        assert state.get_latest_detections() == [PERSON_IN_RESTRICTED]
        assert state.get_zone_status()[0]["zone"] == "Server"

    def test_uses_tracking_not_plain_detect(self):
        clock = [0.0]
        rp, _, detector, _ = make(clock)
        rp.process(FRAME)
        detector.track.assert_called_once()
        detector.detect.assert_not_called()


class TestLoop:
    def test_loop_processes_each_new_frame_once(self):
        clock = [0.0]
        rp, state, detector, _ = make(clock)
        done = threading.Event()
        detector.track.side_effect = lambda f: (done.set() or [])
        state.set_camera_status("running")
        state.set_frame(FRAME.copy())
        rp.start()
        done.wait(timeout=2)
        time.sleep(0.1)
        rp.stop()
        assert detector.track.call_count == 1
