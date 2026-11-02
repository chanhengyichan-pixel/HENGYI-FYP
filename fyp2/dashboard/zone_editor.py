"""
Settings > Room zones: draw hazard / restricted zones on the room camera
image in the browser (admin only).

Saved zones go to a JSON file (config zones.file, default
output/zones.json), which overrides config.yaml's zones.areas, and are
swapped into the running ZoneMonitor straight away — no restart, no
config editing. Coordinates are room-camera pixels, the same as before.
"""

from __future__ import annotations

import logging

import cv2
from flask import Response, abort, jsonify, render_template, request

from core.zones import MAX_POINTS, MAX_ZONES, save_zones, zones_from_areas, zones_to_areas
from dashboard.auth import admin_required

logger = logging.getLogger(__name__)


def init_zone_editor(app, config: dict, room_state, get_zones, apply_zones) -> None:

    @app.route("/settings/zones")
    @admin_required
    def settings_zones():
        return render_template("zones.html", room_enabled=room_state is not None,
                               max_zones=MAX_ZONES, max_points=MAX_POINTS)

    @app.route("/settings/zones/frame.jpg")
    @admin_required
    def settings_zones_frame():
        """Current room frame WITHOUT overlays, to draw on."""
        if room_state is None:
            abort(404)
        frame = room_state.get_frame()
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
        if request.method == "GET":
            return jsonify({"zones": zones_to_areas(get_zones())})
        payload = request.get_json(silent=True) or {}
        try:
            zones = zones_from_areas(payload.get("zones"))
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        path = save_zones(config, zones)
        apply_zones(zones)
        logger.info("Room zones updated from dashboard: %d zones saved to %s", len(zones), path)
        return jsonify({"ok": True, "zones": zones_to_areas(zones), "file": path})
