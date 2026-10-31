"""
Unit tests for dashboard/app.py's _draw_debug_overlay().

No Flask app construction, no camera — a plain numpy frame array and a
debug_info dict/None shaped like FrameVerdict.debug_info
(core/decision_engine.py). Confirms the overlay is a genuine no-op (no
drawing) when debug_info is None, and that a well-formed debug_info does
not raise.
"""

from unittest.mock import patch

import numpy as np

from dashboard.app import _draw_debug_overlay


def _fake_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _sample_debug_info():
    return {
        "persons": [
            {
                "person_index": 0,
                "box": (100, 100, 300, 500),
                "confidence": 0.9,
                "head_region": (100.0, 200.0, 145.0, 255.0),
                "torso_region": (200.0, 400.0, 120.0, 280.0),
                "helmet": {
                    "assigned": "helmet",
                    "candidates": [
                        {
                            "class_name": "helmet",
                            "box": (150, 100, 250, 190),
                            "centroid": (200.0, 145.0),
                            "containment": 1.0,
                            "containment_ok": True,
                            "x_ok": True,
                            "y_ok": True,
                            "region_ok": True,
                            "claimed_by": None,
                            "status": "assigned",
                            "reasons": [],
                        },
                    ],
                },
                "vest": {
                    "assigned": None,
                    "candidates": [
                        {
                            "class_name": "vest",
                            "box": (130, 230, 270, 370),
                            "centroid": (200.0, 300.0),
                            "containment": 0.4,
                            "containment_ok": False,
                            "x_ok": True,
                            "y_ok": True,
                            "region_ok": True,
                            "claimed_by": None,
                            "status": "rejected",
                            "reasons": ["containment below threshold (0.40 < 0.60)"],
                        },
                    ],
                },
                "person_status": "NON_COMPLIANT",
            }
        ],
        "frame": {
            "person_count": 1,
            "helmets_assigned": 1,
            "vests_assigned": 0,
            "status": "non_compliant",
            "violations": ["no-vest"],
        },
    }


class TestDebugOverlayNoOp:
    def test_none_debug_info_draws_nothing(self):
        frame = _fake_frame()
        with patch("dashboard.app.cv2") as mock_cv2:
            _draw_debug_overlay(frame, None)
        mock_cv2.rectangle.assert_not_called()
        mock_cv2.putText.assert_not_called()
        mock_cv2.drawMarker.assert_not_called()


class TestDebugOverlayDrawing:
    def test_valid_debug_info_does_not_raise(self):
        frame = _fake_frame()
        _draw_debug_overlay(frame, _sample_debug_info())  # must not raise

    def test_valid_debug_info_draws_person_box_and_regions(self):
        frame = _fake_frame()
        with patch("dashboard.app.cv2") as mock_cv2:
            _draw_debug_overlay(frame, _sample_debug_info())
        # person box + head region + torso region = 3 rectangle calls
        assert mock_cv2.rectangle.call_count == 3
        # one marker per PPE candidate (1 helmet + 1 vest)
        assert mock_cv2.drawMarker.call_count == 2


# ---------------------------------------------------------------------------
# Flask routes (two cameras, zone overlay, events API)
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from core.shared_state import SharedState
from core.zones import Zone
from dashboard.app import _draw_room_detections, _draw_zones, create_app
from storage.event_store import EventStore


def _app(tmp_path, with_room=True):
    shots = tmp_path / "shots"
    shots.mkdir(exist_ok=True)
    config = {"decision_engine": {"debug_overlay": False},
              "alerts": {"screenshot_dir": str(shots)}}
    store = EventStore({"storage": {"db_path": str(tmp_path / "e.db")}})
    store.append({"timestamp": "2026-10-30 10:00:00", "verdict": "violation",
                  "violations": [], "event_type": "RESTRICTED_ENTRY", "camera": "room",
                  "zone": "Server", "severity": "critical", "snapshot": None})
    states = {"entrance": SharedState(), "room": SharedState() if with_room else None}
    zones = [Zone("Server", "restricted", ((340, 200), (640, 200), (640, 480), (340, 480)))]
    return create_app(config, states, store, zones=zones).test_client(), states


class TestRoutes:
    def test_status_includes_room_block(self, tmp_path):
        client, states = _app(tmp_path)
        states["room"].set_zone_status([{"track_id": 1, "zone": "Server",
                                         "zone_type": "restricted", "seconds": 0.5,
                                         "ppe_ok": True, "violations": []}])
        data = client.get("/api/status").get_json()
        assert data["room"]["enabled"] is True
        assert data["room"]["zone_status"][0]["zone"] == "Server"
        assert data["recent_events"][0]["event_type"] == "RESTRICTED_ENTRY"

    def test_status_room_disabled(self, tmp_path):
        client, _ = _app(tmp_path, with_room=False)
        assert client.get("/api/status").get_json()["room"] == {"enabled": False}

    def test_room_feed_404_when_disabled(self, tmp_path):
        client, _ = _app(tmp_path, with_room=False)
        assert client.get("/video_feed/room").status_code == 404

    def test_events_filter_by_type(self, tmp_path):
        client, _ = _app(tmp_path)
        data = client.get("/api/events?type=ENTRY_DENIED").get_json()
        assert data["events"] == []
        data = client.get("/api/events?type=RESTRICTED_ENTRY").get_json()
        assert len(data["events"]) == 1

    def test_index_shows_room_panel_only_when_enabled(self, tmp_path):
        client, _ = _app(tmp_path)
        assert b"/video_feed/room" in client.get("/").data
        client, _ = _app(tmp_path, with_room=False)
        assert b"/video_feed/room" not in client.get("/").data

    def test_snapshot_route_serves_screenshot(self, tmp_path):
        client, _ = _app(tmp_path)
        (tmp_path / "shots" / "a.jpg").write_bytes(b"jpg")
        assert client.get("/snapshots/a.jpg").data == b"jpg"

    def test_snapshot_route_cannot_escape_directory(self, tmp_path):
        # e.db lives one level above the screenshot directory.
        client, _ = _app(tmp_path)
        assert client.get("/snapshots/../e.db").status_code == 404


class TestRoomOverlay:
    def test_draw_zones_and_people_does_not_raise(self):
        frame = _fake_frame()
        zones = [Zone("Server", "restricted", ((340, 200), (640, 200), (640, 480), (340, 480)))]
        status = [{"track_id": 3, "zone": "Server", "zone_type": "restricted",
                   "seconds": 1.2, "ppe_ok": True, "violations": []}]
        dets = [SimpleNamespace(class_name="person", confidence=0.9, x1=450, y1=50,
                                x2=550, y2=400, track_id=3)]
        _draw_zones(frame, zones, status)
        _draw_room_detections(frame, dets, status)
        assert frame.any()
