"""CameraManager: start / stop single cameras live, safe stop order, capacity, statuses;
and the Cameras page actions (add, update, disable, enable, remove) on top of it."""

import json
import re

import numpy as np
import pytest

from core.camera_manager import CameraManager, CapacityError, friendly_error
from core.camera_registry import save_cameras
from dashboard.app import create_app
from storage.event_store import EventStore
from storage.user_store import UserStore

LOG = []


class FakeCapture:
    def __init__(self, cfg, state, name="cam"):
        self.cfg, self.state, self.name = cfg, state, name

    def start(self):
        if self.cfg["source"] == "missing.mp4":
            raise RuntimeError("Cannot open camera source 'missing.mp4'. Check ...")
        self.state.set_camera_status("running")
        self.state.set_frame(np.zeros((48, 64, 3), np.uint8))
        LOG.append(("capture.start", self.name))

    def stop(self):
        LOG.append(("capture.stop", self.name))


class FakePipeline:
    def __init__(self, name):
        self.name = name

    def start(self):
        LOG.append(("pipeline.start", self.name))

    def stop(self):
        LOG.append(("pipeline.stop", self.name))


class FakeZoneMonitor:
    zones = []

    def set_zones(self, z):
        self.zones = z


def make_pipeline(cfg, state, capture):
    return FakePipeline(cfg["id"]), (FakeZoneMonitor() if cfg["role"] == "room" else None)


def cam(cid, source, **kw):
    return {"id": cid, "role": "entrance" if cid.startswith("entrance") else "room",
            "name": cid.title(), "enabled": True, "backend": "opencv", "source": source, **kw}


@pytest.fixture(autouse=True)
def _clear():
    LOG.clear()


def _mgr(**cap):
    return CameraManager({"camera_capacity": cap}, make_pipeline, capture_cls=FakeCapture)


class TestManager:
    def test_start_stop_order_and_release(self):
        m = _mgr()
        assert m.start(cam("room", 1))["status"] == "running"
        assert "room" in m.zone_monitors and m.is_running("room")
        m.stop("room")
        # pipeline stops before the capture (which releases the device)
        assert LOG[-2:] == [("pipeline.stop", "room"), ("capture.stop", "room")]
        assert not m.is_running("room") and "room" not in m.zone_monitors
        assert m.states["room"].get_camera_status() == "stopped"

    def test_error_keeps_others_running(self):
        m = _mgr()
        m.start(cam("entrance", 0))
        st = m.start(cam("room2", "missing.mp4"))
        assert st == {"status": "error", "error": "Video file not found or unreadable: missing.mp4"}
        assert m.running_ids() == ["entrance"]
        assert ("pipeline.start", "room2") not in LOG     # no AI thread for a dead camera

    def test_disconnected(self):
        m = _mgr()
        m.start(cam("room", 1))
        m.states["room"].set_camera_status("disconnected")
        assert m.status("room")["status"] == "disconnected"
        assert m.restart(cam("room", 1))["status"] == "running"     # Retry

    def test_capacity_limit_but_restart_allowed(self):
        m = _mgr(max_active=2)
        m.start(cam("entrance", 0))
        m.start(cam("room", 1))
        with pytest.raises(CapacityError, match="at most 2"):
            m.start(cam("room2", 2))
        assert m.restart(cam("room", 1))["status"] == "running"     # restart is not an extra camera
        assert "at most 2" in m.capacity_problem()

    def test_memory_check(self, monkeypatch):
        import core.camera_manager as cm
        monkeypatch.setattr(cm, "free_memory_mb", lambda: 150.0)
        m = _mgr(min_free_memory_mb=300)
        with pytest.raises(CapacityError, match="150 MB left"):
            m.start(cam("room", 1))

    def test_overload(self, monkeypatch):
        import core.camera_manager as cm
        m = _mgr(min_fps=2)
        m.start(cam("room", 1))
        monkeypatch.setattr(cm, "_WARMUP_SECONDS", -1)
        m.states["room"].set_fps(1.2)
        assert m.overload() == [{"id": "room", "fps": 1.2}]

    def test_remove_drops_state(self):
        m = _mgr()
        m.start(cam("room3", 3))
        m.remove("room3")
        assert "room3" not in m.states and "entrance" in m.states

    def test_friendly_errors(self):
        assert "Device not found" in friendly_error(RuntimeError("Cannot open camera source '0'"), cam("room", 0))
        assert "Pi" in friendly_error(ImportError("No module named picamera2"), cam("room", 0, backend="picamera2"))


# ── Cameras page actions with a live manager ──

@pytest.fixture
def live(tmp_path):
    shots = tmp_path / "shots"
    shots.mkdir()
    cfg = {"decision_engine": {"debug_overlay": False}, "alerts": {"screenshot_dir": str(shots)},
           "storage": {"db_path": str(tmp_path / "e.db")}, "dashboard": {},
           "camera_setup_file": str(tmp_path / "cameras.json"),
           "camera_capacity": {"max_active": 3},
           "cameras": {"entrance": {"enabled": True, "backend": "opencv", "source": 0, "width": 640,
                                    "height": 480, "max_failed_reads": 10}},
           "zones": {"file": str(tmp_path / "zones.json"), "hazard_seconds": 2, "restricted_seconds": 1,
                     "grace_seconds": 1, "cooldown_seconds": 30, "areas": []}}
    users = UserStore(cfg)
    users.add("boss", "secret1", "admin")
    m = CameraManager(cfg, make_pipeline, capture_cls=FakeCapture)
    from core.camera_registry import load_cameras
    for c in load_cameras(cfg)[0]:
        m.start(c)
    client = create_app(cfg, m.states, EventStore(cfg), user_store=users,
                        zone_monitors=m.zone_monitors, manager=m).test_client()
    html = client.get("/login").get_data(as_text=True)
    tok = re.search(r'name="csrf" value="([0-9a-f]+)"', html).group(1)
    client.post("/login", data={"username": "boss", "password": "secret1", "csrf": tok})
    tok = re.search(r'name="csrf" content="([0-9a-f]+)"',
                    client.get("/settings/cameras").get_data(as_text=True)).group(1)

    def post(url, body):
        return client.post(url, json=body, headers={"X-CSRF-Token": tok})
    return client, m, cfg, post


class TestLiveActions:
    def test_add_entrance_and_room_live(self, live):
        client, m, cfg, post = live
        d = post("/settings/cameras/add", {"role": "entrance", "name": "Back door", "source": 5}).get_json()
        assert d["ok"] and d["id"] == "entrance2" and m.is_running("entrance2")
        d = post("/settings/cameras/add", {"role": "room", "name": "Bay", "source": 6}).get_json()
        assert d["id"] == "room" and "room" in m.zone_monitors
        by_id = {c["id"]: c for c in d["cameras"]}
        assert by_id["entrance2"]["role"] == "entrance" and by_id["room"]["status"] == "running"
        # live page and status list both entrances
        status = client.get("/api/status").get_json()
        assert [e["id"] for e in status["entrances"]] == ["entrance", "entrance2"]
        assert [r["name"] for r in status["rooms"]] == ["Bay"]
        assert "/video_feed/entrance2" in client.get("/").get_data(as_text=True)
        # max_active = 3 reached: the next camera is saved but disabled, with the reason
        d = post("/settings/cameras/add", {"role": "room", "name": "Yard", "source": 7}).get_json()
        assert "at most 3" in d["warning"] and not m.is_running("room2")
        assert {c["id"]: c for c in d["cameras"]}["room2"]["status"] == "disabled"
        assert d["capacity"]["add_problem"]

    def test_bad_input(self, live):
        _, _, _, post = live
        assert post("/settings/cameras/add", {"role": "kitchen", "name": "x", "source": 1}).status_code == 400
        r = post("/settings/cameras/add", {"role": "room", "name": "Dup", "source": 0})
        assert r.status_code == 400 and "already used by Entrance" in r.get_json()["error"]

    def test_failed_source_shows_error(self, live):
        _, m, _, post = live
        d = post("/settings/cameras/add", {"role": "room", "name": "Old", "source": "missing.mp4"}).get_json()
        row = {c["id"]: c for c in d["cameras"]}["room"]
        assert row["status"] == "error" and "not found" in row["error"]
        assert "not found" in d["warning"]

    def test_disable_enable_restart_remove(self, live):
        _, m, cfg, post = live
        post("/settings/cameras/add", {"role": "room", "name": "Bay", "source": 6})
        LOG.clear()
        d = post("/settings/cameras/room/action", {"action": "disable"}).get_json()
        assert not m.is_running("room") and LOG == [("pipeline.stop", "room"), ("capture.stop", "room")]
        assert {c["id"]: c for c in d["cameras"]}["room"]["status"] == "disabled"
        assert json.load(open(cfg["camera_setup_file"]))["cameras"][1]["enabled"] is False
        post("/settings/cameras/room/action", {"action": "enable"})
        assert m.is_running("room")
        d = post("/settings/cameras/room/action", {"action": "restart"}).get_json()
        assert d["message"] == "Bay is running."
        LOG.clear()
        post("/settings/cameras/room/action", {"action": "remove"})
        assert ("capture.stop", "room") in LOG and "room" not in m.states
        assert [c["id"] for c in json.load(open(cfg["camera_setup_file"]))["cameras"]] == ["entrance"]

    def test_update_source_restarts_only_that_camera(self, live):
        _, m, _, post = live
        post("/settings/cameras/add", {"role": "room", "name": "Bay", "source": 6})
        LOG.clear()
        post("/settings/cameras/room/update", {"name": "Bay A", "source": 8})
        assert m.running_cfg("room")["source"] == 8
        assert all(name == "room" for _, name in LOG)          # entrance untouched
        LOG.clear()
        post("/settings/cameras/room/update", {"name": "Bay B"})  # name only: no restart
        assert LOG == []
