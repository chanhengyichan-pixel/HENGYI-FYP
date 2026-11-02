"""Settings > Room zones: validation, saving, live swap into ZoneMonitor."""

import json
import re

import numpy as np
import pytest

from core.shared_state import SharedState
from core.zone_monitor import ZoneMonitor
from core.zones import Zone, load_active_zones, save_zones, zones_from_areas
from dashboard.app import create_app
from storage.event_store import EventStore
from storage.user_store import UserStore

AREA = {"name": "Machine area", "type": "hazard", "polygon": [[0, 0], [100, 0], [100, 100]]}


def _config(tmp_path):
    shots = tmp_path / "shots"
    shots.mkdir(exist_ok=True)
    return {"decision_engine": {"debug_overlay": False},
            "alerts": {"screenshot_dir": str(shots)},
            "storage": {"db_path": str(tmp_path / "e.db")},
            "dashboard": {},
            "zones": {"file": str(tmp_path / "zones.json"),
                      "hazard_seconds": 2, "restricted_seconds": 1,
                      "grace_seconds": 1, "cooldown_seconds": 30,
                      "areas": [{"name": "Old", "type": "restricted",
                                 "polygon": [[0, 0], [10, 0], [10, 10]]}]}}


class TestZonesFile:
    def test_validation_messages(self):
        with pytest.raises(ValueError, match="type"):
            zones_from_areas([{**AREA, "type": "danger"}])
        with pytest.raises(ValueError, match="3-"):
            zones_from_areas([{**AREA, "polygon": [[0, 0], [1, 1]]}])
        with pytest.raises(ValueError, match="twice"):
            zones_from_areas([AREA, {**AREA, "name": "machine AREA"}])
        with pytest.raises(ValueError, match="name"):
            zones_from_areas([{**AREA, "name": "  "}])
        with pytest.raises(ValueError):
            zones_from_areas("not a list")

    def test_file_overrides_config(self, tmp_path):
        cfg = _config(tmp_path)
        zones, src = load_active_zones(cfg)
        assert [z.name for z in zones] == ["Old"] and src == "config.yaml"
        save_zones(cfg, zones_from_areas([AREA]))
        zones, src = load_active_zones(cfg)
        assert [z.name for z in zones] == ["Machine area"] and src.endswith("zones.json")

    def test_monitor_set_zones(self, tmp_path):
        zm = ZoneMonitor(_config(tmp_path))
        zm.set_zones(zones_from_areas([AREA]))
        assert [z.name for z in zm.zones] == ["Machine area"]


@pytest.fixture
def admin(tmp_path):
    cfg = _config(tmp_path)
    users = UserStore(cfg)
    users.add("boss", "secret1", "admin")
    users.add("guard", "secret1", "viewer")
    room = SharedState()
    room.set_frame(np.zeros((360, 640, 3), np.uint8))
    zm = ZoneMonitor(cfg)
    client = create_app(cfg, {"entrance": SharedState(), "room": room},
                        EventStore(cfg), user_store=users, zone_monitor=zm).test_client()

    def login(name):
        html = client.get("/login").get_data(as_text=True)
        tok = re.search(r'name="csrf" value="([0-9a-f]+)"', html).group(1)
        client.post("/login", data={"username": name, "password": "secret1", "csrf": tok})
        html = client.get("/settings/cameras" if name == "boss" else "/").get_data(as_text=True)
        m = re.search(r'name="csrf"(?: content|.*?value)="([0-9a-f]+)"', html)
        return m.group(1) if m else None
    return client, zm, cfg, login


class TestEditorRoutes:
    def test_save_applies_live_and_writes_file(self, admin):
        client, zm, cfg, login = admin
        tok = login("boss")
        r = client.post("/settings/zones/data", json={"zones": [AREA]},
                        headers={"X-CSRF-Token": tok})
        assert r.status_code == 200 and r.get_json()["ok"]
        assert [z.name for z in zm.zones] == ["Machine area"]
        saved = json.load(open(cfg["zones"]["file"]))
        assert saved["cameras"]["room"][0]["polygon"][1] == [100.0, 0.0]
        assert client.get("/api/status").get_json()["room"]["zones"][0]["name"] == "Machine area"

    def test_bad_zone_rejected_and_nothing_changes(self, admin):
        client, zm, _, login = admin
        tok = login("boss")
        r = client.post("/settings/zones/data", json={"zones": [{**AREA, "polygon": [[0, 0]]}]},
                        headers={"X-CSRF-Token": tok})
        assert r.status_code == 400 and "points" in r.get_json()["error"]
        assert [z.name for z in zm.zones] == ["Old"]

    def test_save_needs_csrf(self, admin):
        client, zm, _, login = admin
        login("boss")
        assert client.post("/settings/zones/data", json={"zones": [AREA]}).status_code == 400

    def test_frame_and_page(self, admin):
        client, _, _, login = admin
        login("boss")
        assert client.get("/settings/zones").status_code == 302      # moved to the camera page
        page = client.get("/settings/cameras/room?tab=zones").get_data(as_text=True)
        assert 'id="p-zones"' in page and "b-add" in page
        assert 'id="p-zones"' not in client.get("/settings/cameras/entrance").get_data(as_text=True)
        assert client.get("/settings/cameras/nope").status_code == 404
        r = client.get("/settings/zones/frame.jpg")
        assert r.status_code == 200 and r.mimetype == "image/jpeg"

    def test_viewer_blocked(self, admin):
        client, _, _, login = admin
        login("guard")
        assert client.get("/settings/zones").status_code == 403
        assert client.get("/settings/cameras/room").status_code == 403
        assert client.get("/settings/zones/frame.jpg").status_code == 403
        assert client.get("/settings/zones/data").status_code == 403
