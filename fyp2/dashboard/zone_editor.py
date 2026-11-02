"""
Room camera zones: draw hazard / restricted zones on a room camera's image
in the browser (admin only). The editor lives on each room camera's own
page (Settings > Cameras > Edit > Zones); the entrance camera has no zones.

Saved zones go to a JSON file (config zones.file, default
output/zones.json, one list per camera id), which overrides config.yaml's
zones.areas. For a running camera they are swapped into its ZoneMonitor
straight away (no restart). A room camera that is set up but not running
yet (just added, or switched on) can have its zones drawn on a snapshot;
they start working when the camera starts. Coordinates are camera pixels.
"""

from __future__ import annotations

import logging

import cv2
from flask import Response, abort, jsonify, redirect, request, url_for

from core.camera_registry import grab_frame, load_cameras
from core.zones import load_active_zones, save_zones, zones_from_areas, zones_to_areas
from dashboard.auth import admin_required

logger = logging.getLogger(__name__)


def configured_room_cams(config: dict) -> list[dict]:
    """Room cameras in the saved camera setup (running or not)."""
    try:
        cams, _ = load_cameras(config)
    except (ValueError, KeyError, OSError):
        return []
    return [c for c in cams if c["role"] == "room"]


def init_zone_editor(app, config: dict, states: dict, running_room_ids, zones_for, apply_zones) -> None:
    """running_room_ids(): ids of the room cameras running right now."""

    def _camera_arg():
        running_ids = running_room_ids()
        cid = request.args.get("camera") or (running_ids[0] if running_ids else None)
        if cid in running_ids:
            return cid, None
        cam = next((c for c in configured_room_cams(config) if c["id"] == cid), None)
        if cam is None:
            abort(404)
        return cid, cam

    @app.route("/settings/zones")
    @admin_required
    def settings_zones():
        """Old address: zones now live on each room camera's page."""
        running_ids = running_room_ids()
        cid = request.args.get("camera") or (running_ids[0] if running_ids else None)
        if not cid:
            return redirect(url_for("settings_cameras"))
        return redirect(url_for("settings_camera_edit", cid=cid, tab="zones"))

    @app.route("/settings/zones/frame.jpg")
    @admin_required
    def settings_zones_frame():
        """The camera's current frame WITHOUT overlays, to draw on.
        Not running yet: one snapshot straight from the camera."""
        cid, cam = _camera_arg()
        if cam is None:
            frame = states[cid].get_frame()
        else:
            frame = grab_frame(cam.get("backend", "opencv"), cam.get("source"),
                               cam.get("width", 640), cam.get("height", 480))
        if frame is None:
            abort(503)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            abort(500)
        return Response(buf.tobytes(), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/settings/zones/data", methods=["GET", "POST"])
    @admin_required
    def settings_zones_data():
        cid, cam = _camera_arg()
        running = cam is None
        if request.method == "GET":
            zones = zones_for(cid) if running else load_active_zones(config, cid)[0]
            return jsonify({"camera": cid, "running": running, "zones": zones_to_areas(zones)})
        payload = request.get_json(silent=True) or {}
        try:
            zones = zones_from_areas(payload.get("zones"))
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        path = save_zones(config, zones, cid)
        if running:
            apply_zones(cid, zones)
        logger.info("Zones for %s updated from dashboard: %d zones saved to %s%s", cid, len(zones), path,
                    "" if running else " (camera not running yet)")
        return jsonify({"ok": True, "camera": cid, "running": running,
                        "zones": zones_to_areas(zones), "file": path})
