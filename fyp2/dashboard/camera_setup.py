"""
Cameras page (admin only): which cameras exist and what each one does.

- /settings/cameras: one card per camera (picture, name, source, status),
  grouped into "Entrance inspection" and "Room monitoring". "+ Add Camera"
  asks for the purpose first (entrance or room), then name and source.
- /settings/cameras/<id>: that camera's own page. Entrance: name and
  source. Room: the same plus the Zones tab (see zone_editor.py).
- Card menu / page actions: Restart, Disable / Enable, Remove, Retry.

Any number of entrance and room cameras can be set up. With a
CameraManager (main.py passes one) every change applies straight away to
that ONE camera — its threads are stopped (pipeline first, then capture,
which releases the device) and started again; the other cameras keep
running. Whether another camera can start depends on the hardware: the
manager refuses with a reason when memory is short or
camera_capacity.max_active is reached, and the page warns when cameras
run below camera_capacity.min_fps.

Without a manager (tests, older setups) changes are only saved to
output/cameras.json and apply on the next start of main.py.
"""

from __future__ import annotations

import logging
import os
import threading

import cv2
from flask import Response, abort, jsonify, render_template, request

from core.camera_manager import CapacityError
from core.camera_registry import (MAX_CAMERAS, describe_source, detect_cameras, grab_frame,
                                  load_cameras, next_id, next_room_id, save_cameras,
                                  validate_cameras)
from core.zones import MAX_POINTS, MAX_ZONES, load_active_zones
from dashboard.auth import admin_required

logger = logging.getLogger(__name__)

_KEYS = ("id", "name", "enabled", "backend", "source")


def _plain(cams):
    return [{k: c.get(k) for k in _KEYS} for c in cams]


def _zone_counts(config, cid, zones_for, running) -> dict:
    try:
        zones = zones_for(cid) if running else load_active_zones(config, cid)[0]
    except (ValueError, KeyError, OSError):
        zones = []
    return {"hazard": sum(z.type == "hazard" for z in zones),
            "restricted": sum(z.type == "restricted" for z in zones)}


def init_camera_setup(app, config: dict, states: dict, running: list, on_restart,
                      zones_for=lambda cid: [], manager=None) -> None:
    started = [c for c in running if c.get("enabled", True)]
    running_by_src_static = {(c.get("backend", "opencv"), str(c.get("source"))): c["id"]
                             for c in started if "source" in c}

    def _running_by_src():
        if manager is None:
            return running_by_src_static
        out = {}
        for cid in manager.running_ids():
            cfg = manager.running_cfg(cid) or {}
            out[(cfg.get("backend", "opencv"), str(cfg.get("source")))] = cid
        return out

    def _is_running(cid):
        if manager is not None:
            return manager.is_running(cid)
        return states.get(cid) is not None

    def _status(c):
        if not c.get("enabled", True):
            return {"status": "disabled", "error": None}
        if manager is None:
            if states.get(c["id"]) is not None:
                return {"status": "running", "error": None}
            return {"status": "stopped", "error": "Starts when main.py is restarted"}
        st = manager.status(c["id"])
        if st["status"] == "stopped":
            return {"status": "error", "error": "Not running"}
        return st

    def _load():
        try:
            return load_cameras(config)
        except (ValueError, KeyError, OSError) as e:
            abort(Response(f'{{"ok": false, "error": "Saved camera file is broken: {e}"}}',
                           500, mimetype="application/json"))

    def _payload(extra=None):
        cams, source = _load()
        out = []
        for c in cams:
            row = {k: c.get(k) for k in _KEYS + ("role",)}
            row["source_info"] = describe_source(c.get("backend", "opencv"), c.get("source"))
            row.update(_status(c))
            if c["role"] == "room":
                row["zones"] = _zone_counts(config, c["id"], zones_for, _is_running(c["id"]))
            out.append(row)
        capacity = {"live": manager is not None, "add_problem": None, "slow": [], "max_cameras": MAX_CAMERAS}
        if manager is not None:
            capacity["add_problem"] = manager.capacity_problem()
            names = {c["id"]: c["name"] for c in cams}
            capacity["slow"] = [{**s, "name": names.get(s["id"], s["id"])} for s in manager.overload()]
            capacity["min_fps"] = manager.min_fps
        data = {"ok": True, "cameras": out, "from": source, "capacity": capacity,
                "running": [c["id"] for c in cams if _is_running(c["id"])],
                "next_ids": {"entrance": next_id(cams, "entrance"), "room": next_id(cams, "room")},
                "next_room_id": next_room_id(cams), "pending_restart": False}
        data.update(extra or {})
        return jsonify(data)

    def _err(msg, code=400):
        return jsonify({"ok": False, "error": msg}), code

    def _save(plain_list):
        """Validate + save; returns the saved list normalised (with sizes)."""
        save_cameras(config, validate_cameras(plain_list))
        return _load()[0]

    def _find(cams, cid):
        c = next((x for x in cams if x["id"] == cid), None)
        if c is None:
            abort(404)
        return c

    def _start(cfg):
        """Start / restart one camera; message for the page."""
        if manager is None:
            return {"message": "Saved. Restart main.py to apply."}
        try:
            st = manager.restart(cfg)
        except CapacityError as e:
            return {"warning": str(e)}
        if st["status"] == "error":
            return {"warning": f"{cfg['name']}: {st['error']}"}
        return {"message": f"{cfg['name']} is running."}

    # ── pages ──
    @app.route("/settings/cameras")
    @admin_required
    def settings_cameras():
        return render_template("cameras.html", live=manager is not None)

    @app.route("/settings/cameras/<cid>")
    @admin_required
    def settings_camera_edit(cid):
        try:
            cams, _ = load_cameras(config)
        except (ValueError, KeyError, OSError):
            cams = []
        cam = next((c for c in cams if c["id"] == cid), None) or \
            next((c for c in running if c["id"] == cid), None)
        if cam is None:
            abort(404)
        cam = {"role": "entrance" if cid.startswith("entrance") else "room", **cam}
        tab = request.args.get("tab", "general")
        if cam["role"] != "room" or tab not in ("general", "zones"):
            tab = "general"
        return render_template("camera_edit.html", cam={k: cam.get(k) for k in _KEYS + ("role",)},
                               tab=tab, running=_is_running(cid), live=manager is not None,
                               max_zones=MAX_ZONES, max_points=MAX_POINTS)

    # ── data ──
    @app.route("/settings/cameras/data", methods=["GET", "POST"])
    @admin_required
    def settings_cameras_data():
        if request.method == "POST":         # whole-list save (applies on restart)
            payload = request.get_json(silent=True) or {}
            try:
                cams = validate_cameras(payload.get("cameras"))
            except ValueError as e:
                return _err(str(e))
            path = save_cameras(config, cams)
            logger.info("Camera setup saved to %s: %s", path,
                        ", ".join(f"{c['id']}={c['name']}" for c in cams))
        return _payload()

    @app.route("/settings/cameras/add", methods=["POST"])
    @admin_required
    def settings_cameras_add():
        p = request.get_json(silent=True) or {}
        role = p.get("role")
        if role not in ("entrance", "room"):
            return _err("Choose what the camera is for: Entrance inspection or Room monitoring")
        cams, _ = _load()
        cid = next_id(cams, role)
        problem = manager.capacity_problem() if manager is not None else None
        new = {"id": cid, "name": str(p.get("name", "")).strip(), "enabled": problem is None,
               "backend": p.get("backend", "opencv"), "source": p.get("source")}
        try:
            cams = _save(_plain(cams) + [new])
        except ValueError as e:
            return _err(str(e))
        cfg = _find(cams, cid)
        logger.info("Camera added from dashboard: %s (%s) %s", cid, cfg["name"], role)
        if problem:
            extra = {"warning": f"{cfg['name']} was saved but is disabled. {problem}"}
        else:
            extra = _start(cfg)
        return _payload({"id": cid, **extra})

    @app.route("/settings/cameras/<cid>/update", methods=["POST"])
    @admin_required
    def settings_camera_update(cid):
        p = request.get_json(silent=True) or {}
        cams, _ = _load()
        old = _find(cams, cid)
        plain = _plain(cams)
        for c in plain:
            if c["id"] == cid:
                c["name"] = str(p.get("name", c["name"])).strip()
                c["backend"] = p.get("backend", c["backend"])
                c["source"] = p.get("source", c["source"])
        try:
            cams = _save(plain)
        except ValueError as e:
            return _err(str(e))
        cfg = _find(cams, cid)
        extra = {"message": "Saved."}
        source_changed = (cfg["backend"], str(cfg["source"])) != (old["backend"], str(old["source"]))
        if cfg["enabled"] and (source_changed or not _is_running(cid)):
            extra = _start(cfg)
        return _payload(extra)

    @app.route("/settings/cameras/<cid>/action", methods=["POST"])
    @admin_required
    def settings_camera_action(cid):
        action = (request.get_json(silent=True) or {}).get("action")
        cams, _ = _load()
        cfg = _find(cams, cid)
        plain = _plain(cams)
        if action in ("restart", "retry"):
            if not cfg["enabled"]:
                return _err("This camera is disabled. Enable it first.")
            return _payload(_start(cfg))
        if action == "disable":
            if manager is not None:
                manager.stop(cid)          # threads stopped, device released
            for c in plain:
                if c["id"] == cid:
                    c["enabled"] = False
            _save(plain)
            logger.info("Camera %s disabled from dashboard.", cid)
            return _payload({"message": f"{cfg['name']} is disabled."})
        if action == "enable":
            if manager is not None:
                problem = manager.capacity_problem()
                if problem:
                    return _err(problem, 409)
            for c in plain:
                if c["id"] == cid:
                    c["enabled"] = True
            try:
                cams = _save(plain)
            except ValueError as e:
                return _err(str(e))
            return _payload(_start(_find(cams, cid)))
        if action == "remove":
            if manager is not None:
                manager.remove(cid)        # stop threads and release the device first
            _save([c for c in plain if c["id"] != cid])
            logger.info("Camera %s (%s) removed from dashboard.", cid, cfg["name"])
            return _payload({"message": f"{cfg['name']} was removed."})
        return _err("Unknown action")

    @app.route("/settings/cameras/detect")
    @admin_required
    def settings_cameras_detect():
        found = detect_cameras()
        by_src = _running_by_src()
        for f in found:
            f["in_use_by"] = by_src.get((f["backend"], str(f["source"])))
        return jsonify({"cameras": found})

    @app.route("/settings/cameras/preview.jpg")
    @admin_required
    def settings_cameras_preview():
        backend = request.args.get("backend", "opencv")
        src = request.args.get("source", "")
        cid = _running_by_src().get((backend, src))
        if cid and states.get(cid) is not None and states[cid].get_frame() is not None:
            frame = states[cid].get_frame()          # running: use its live frame
        else:
            if src.lstrip("-").isdigit():
                source = int(src)
            elif os.path.isfile(src):                # a test video file
                source = src
            else:
                abort(404)                           # no URLs or odd paths
            frame = grab_frame(backend, source)
        if frame is None:
            abort(503)
        h, w = frame.shape[:2]
        if w > 480:
            frame = cv2.resize(frame, (480, int(h * 480 / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.route("/settings/cameras/restart", methods=["POST"])
    @admin_required
    def settings_cameras_restart():
        """Restart the whole program (fallback; single cameras restart live)."""
        if on_restart is None:
            return _err("Restart is not available here; restart main.py by hand.", 501)
        logger.warning("Restart of the whole program requested from the dashboard.")
        threading.Timer(0.5, on_restart).start()   # let this response go out first
        return jsonify({"ok": True})
