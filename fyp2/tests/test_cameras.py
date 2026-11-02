"""Camera setup: camera list, per-camera zones, several room cameras, Cameras page."""

import json
import re

import numpy as np
import pytest

from core.camera_registry import load_cameras, next_room_id, save_cameras, validate_cameras
from core.shared_state import SharedState
from core.zone_monitor import ZoneMonitor
from core.zones import load_active_zones, save_zones, zones_from_areas
from dashboard.app import create_app
from storage.event_store import EventStore
from storage.user_store import UserStore

AREA = {"name": "Bay", "type": "hazard", "polygon": [[0, 0], [100, 0], [100, 100]]}


def _config(tmp_path):
    shots = tmp_path / "shots"
    shots.mkdir(exist_ok=True)
    return {
        "decision_engine": {"debug_overlay": False},
        "alerts": {"screenshot_dir": str(shots)},
        "storage": {"db_path": str(tmp_path / "e.db")},
        "dashboard": {},
        "camera_setup_file": str(tmp_path / "cameras.json"),
        "cameras": {
            "entrance": {"enabled": True, "backend": "opencv", "source": 0, "width": 640,
                         "height": 480, "max_failed_reads": 10},
            "room": {"enabled": True, "backend": "opencv", "source": 1, "width": 1280,
                     "height": 720, "max_failed_reads": 10},
        },
        "zones": {"file": str(tmp_path / "zones.json"), "hazard_seconds": 2,
                  "restricted_seconds": 1, "grace_seconds": 1, "cooldown_seconds": 30,
                  "areas": [{"name": "Old", "type": "restricted", "polygon": [[0, 0], [9, 0], [9, 9]]}]},
    }


class TestRegistry:
    def test_from_config_then_file(self, tmp_path):
        cfg = _config(tmp_path)
        cams, src = load_cameras(cfg)
        assert [c["id"] for c in cams] == ["entrance", "room"] and src == "config.yaml"
        assert cams[1]["role"] == "room" and cams[1]["width"] == 1280
        save_cameras(cfg, [
            {"id": "entrance", "name": "Main gate", "backend": "opencv", "source": 2},
            {"id": "room", "name": "Workshop", "backend": "opencv", "source": 0},
            {"id": "room2", "name": "Store", "backend": "picamera2", "source": "1", "enabled": False}])
        cams, src = load_cameras(cfg)
        assert src.endswith("cameras.json")
        assert [c["name"] for c in cams] == ["Main gate", "Workshop", "Store"]
        assert cams[1]["width"] == 1280          # size kept from config.yaml
        assert cams[2]["source"] == 1 and cams[2]["enabled"] is False

    def test_validation(self):
        ok = {"id": "room", "name": "A", "backend": "opencv", "source": 0}
        with pytest.raises(ValueError, match="already used"):
            validate_cameras([{**ok, "id": "entrance"}, ok])
        with pytest.raises(ValueError, match="twice"):
            validate_cameras([ok, ok])
        with pytest.raises(ValueError, match="bad id"):
            validate_cameras([{**ok, "id": "kitchen"}])
        # no fixed 2-camera / 3-room limit: several entrances and rooms are fine
        many = [{**ok, "id": f"room{i}" if i else "room", "source": i} for i in range(5)]
        many += [{**ok, "id": "entrance", "source": 10}, {**ok, "id": "entrance2", "source": 11}]
        assert [c["id"] for c in validate_cameras(many)][:3] == ["entrance", "entrance2", "room"]
        with pytest.raises(ValueError, match="At most"):
            validate_cameras([{**ok, "id": f"room{i}" if i else "room", "source": i} for i in range(13)])
        validate_cameras([{**ok, "enabled": False}])     # all off is allowed
        # two cameras on the same source is fine when one is off
        validate_cameras([{**ok, "id": "entrance"}, {**ok, "enabled": False}])

    def test_source_labels_and_ids(self):
        from core.camera_registry import describe_source, next_id
        assert describe_source("opencv", 0)["label"] == "USB Camera – /dev/video0"
        assert describe_source("picamera2", 1) == {"kind": "csi", "label": "Raspberry Pi Camera – CSI 1"}
        assert describe_source("opencv", "videos/a.mp4") == {"kind": "file", "label": "Video File – videos/a.mp4"}
        assert next_id([{"id": "entrance"}, {"id": "room"}], "entrance") == "entrance2"

    def test_next_room_id(self):
        assert next_room_id([{"id": "entrance"}]) == "room"
        assert next_room_id([{"id": "room"}, {"id": "room2"}]) == "room3"


class TestPerCameraZones:
    def test_each_camera_keeps_its_own_zones(self, tmp_path):
        cfg = _config(tmp_path)
        assert [z.name for z in load_active_zones(cfg, "room")[0]] == ["Old"]
        assert load_active_zones(cfg, "room2")[0] == []
        save_zones(cfg, zones_from_areas([AREA]), "room2")
        assert [z.name for z in load_active_zones(cfg, "room")[0]] == ["Old"]
        assert [z.name for z in load_active_zones(cfg, "room2")[0]] == ["Bay"]

    def test_old_single_camera_file_belongs_to_room(self, tmp_path):
        cfg = _config(tmp_path)
        (tmp_path / "zones.json").write_text(json.dumps({"zones": [AREA]}))
        assert [z.name for z in load_active_zones(cfg, "room")[0]] == ["Bay"]

    def test_zone_events_carry_camera_id(self, tmp_path):
        zm = ZoneMonitor(_config(tmp_path), camera="room2")
        assert zm.camera == "room2"


@pytest.fixture
def two_rooms(tmp_path):
    cfg = _config(tmp_path)
    users = UserStore(cfg)
    users.add("boss", "secret1", "admin")
    store = EventStore(cfg)
    store.append({"timestamp": "2026-11-02 10:00:00", "verdict": "violation", "violations": [],
                  "event_type": "RESTRICTED_ENTRY", "camera": "room2", "zone": "Bay"})
    states = {"entrance": SharedState(), "room": SharedState(), "room2": SharedState()}
    for st in states.values():
        st.set_frame(np.zeros((120, 160, 3), np.uint8))
    zms = {"room": ZoneMonitor(cfg, camera="room"),
           "room2": ZoneMonitor(cfg, zones=zones_from_areas([AREA]), camera="room2")}
    cams = [{"id": "entrance", "role": "entrance", "name": "Gate", "enabled": True, "backend": "opencv", "source": 0},
            {"id": "room", "role": "room", "name": "Workshop", "enabled": True, "backend": "opencv", "source": 1},
            {"id": "room2", "role": "room", "name": "Store", "enabled": True, "backend": "opencv", "source": 2}]
    restarted = []
    client = create_app(cfg, states, store, user_store=users, zone_monitors=zms, cameras=cams,
                        on_restart=lambda: restarted.append(1)).test_client()
    html = client.get("/login").get_data(as_text=True)
    tok = re.search(r'name="csrf" value="([0-9a-f]+)"', html).group(1)
    client.post("/login", data={"username": "boss", "password": "secret1", "csrf": tok})
    tok = re.search(r'name="csrf" content="([0-9a-f]+)"',
                    client.get("/settings/cameras").get_data(as_text=True)).group(1)
    return client, zms, cfg, tok, restarted


class TestSeveralRoomCameras:
    def test_live_page_and_status(self, two_rooms):
        client, *_ = two_rooms
        html = client.get("/").get_data(as_text=True)
        assert "/video_feed/room2" in html and "Store — zone monitoring" in html
        data = client.get("/api/status").get_json()
        assert [r["name"] for r in data["rooms"]] == ["Workshop", "Store"]
        assert data["rooms"][1]["events"][0]["zone"] == "Bay"
        assert data["rooms"][0]["events"] == []
        assert data["room_events"][0]["camera"] == "room2"

    def test_zone_editor_per_camera(self, two_rooms):
        client, zms, cfg, tok, _ = two_rooms
        assert client.get("/settings/zones/data?camera=room2").get_json()["zones"][0]["name"] == "Bay"
        assert client.get("/settings/zones/data?camera=entrance").status_code == 404
        r = client.post("/settings/zones/data?camera=room", json={"zones": [{**AREA, "name": "Press"}]},
                        headers={"X-CSRF-Token": tok})
        assert r.status_code == 200
        assert [z.name for z in zms["room"].zones] == ["Press"]
        assert [z.name for z in zms["room2"].zones] == ["Bay"]     # untouched
        r = client.get("/settings/zones?camera=room2")          # old address
        assert r.status_code == 302 and r.headers["Location"].endswith("/settings/cameras/room2?tab=zones")

    def test_cameras_page_save_and_restart(self, two_rooms):
        client, _, cfg, tok, restarted = two_rooms
        d = client.get("/settings/cameras/data").get_json()
        assert d["capacity"]["live"] is False       # no manager: changes apply on restart
        new = [{"id": "entrance", "name": "Gate", "backend": "opencv", "source": 1},
               {"id": "room", "name": "Workshop", "backend": "opencv", "source": 0}]
        r = client.post("/settings/cameras/data", json={"cameras": new}, headers={"X-CSRF-Token": tok})
        assert r.get_json()["ok"] and json.load(open(cfg["camera_setup_file"]))["cameras"][0]["source"] == 1
        bad = client.post("/settings/cameras/data", json={"cameras": [new[0], {**new[1], "source": 1}]},
                          headers={"X-CSRF-Token": tok})
        assert bad.status_code == 400
        assert client.post("/settings/cameras/restart", headers={"X-CSRF-Token": tok}).get_json()["ok"]
        import time
        time.sleep(0.8)
        assert restarted == [1]

    def test_preview_of_running_camera(self, two_rooms):
        client, *_ = two_rooms
        r = client.get("/settings/cameras/preview.jpg?backend=opencv&source=2")
        assert r.status_code == 200 and r.mimetype == "image/jpeg"
        assert client.get("/settings/cameras/preview.jpg?backend=opencv&source=http://x").status_code == 404

    def test_statistics_zone_rows_name_the_camera(self, two_rooms):
        client, *_ = two_rooms
        rows = client.get("/api/summary?days=0").get_json()["room"]["by_zone"]
        assert rows[0]["camera"] == "room2"


class TestCameraPages:
    def test_status_source_and_zone_counts(self, two_rooms):
        client, zms, cfg, tok, _ = two_rooms
        save_cameras(cfg, [{"id": "entrance", "name": "Gate", "backend": "opencv", "source": 0},
                           {"id": "room", "name": "Workshop", "backend": "opencv", "source": 1},
                           {"id": "room2", "name": "Store", "backend": "opencv", "source": 2},
                           {"id": "room3", "name": "Yard", "backend": "opencv", "source": 3, "enabled": False}])
        d = client.get("/settings/cameras/data").get_json()
        by_id = {c["id"]: c for c in d["cameras"]}
        assert [by_id[i]["status"] for i in ("entrance", "room", "room2", "room3")] == \
            ["running", "running", "running", "disabled"]
        assert by_id["room2"]["zones"] == {"hazard": 1, "restricted": 0}
        assert "zones" not in by_id["entrance"]
        assert by_id["room"]["source_info"]["label"] == "USB Camera – /dev/video1"
        assert d["next_ids"] == {"entrance": "entrance2", "room": "room4"}

    def test_zones_for_camera_not_running_yet(self, two_rooms, monkeypatch):
        client, zms, cfg, tok, _ = two_rooms
        save_cameras(cfg, [{"id": "entrance", "name": "Gate", "backend": "opencv", "source": 0},
                           {"id": "room", "name": "Workshop", "backend": "opencv", "source": 1},
                           {"id": "room3", "name": "Yard", "backend": "opencv", "source": 5}])
        assert client.get("/settings/cameras/room3").status_code == 200
        assert client.get("/settings/zones/data?camera=room3").get_json() == \
            {"camera": "room3", "running": False, "zones": []}
        r = client.post("/settings/zones/data?camera=room3", json={"zones": [AREA]},
                        headers={"X-CSRF-Token": tok})
        assert r.get_json()["ok"] and r.get_json()["running"] is False
        assert [z.name for z in load_active_zones(cfg, "room3")[0]] == ["Bay"]
        assert {c["id"]: c for c in client.get("/settings/cameras/data").get_json()["cameras"]}["room3"]["zones"]["hazard"] == 1
        import dashboard.zone_editor as ze
        monkeypatch.setattr(ze, "grab_frame", lambda *a: np.zeros((48, 64, 3), np.uint8))
        assert client.get("/settings/zones/frame.jpg?camera=room3").mimetype == "image/jpeg"
        assert client.get("/settings/zones/data?camera=room9").status_code == 404
