import logging
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

from dashboard.auth import init_auth
from dashboard.zone_editor import init_zone_editor
from dashboard.camera_setup import init_camera_setup
from core.camera_registry import load_cameras

logger = logging.getLogger(__name__)

_FRAME_INTERVAL = 1.0 / 30  # cap stream at 30 FPS

# Bounding box colors (BGR).  Traffic-light convention:
#   green  — positive PPE confirmed (helmet, vest)
#   red    — violation detected (no-helmet, no-vest)
#   orange — neutral person box
#   grey   — fallback for unrecognised class labels
_BOX_COLORS = {
    "helmet":    (0, 200, 0),
    "vest":      (0, 200, 0),
    "no-helmet": (0, 0, 220),
    "no-vest":   (0, 0, 220),
    "person":    (0, 140, 255),
}
_DEFAULT_COLOR = (128, 128, 128)


_STATUS_COLOR = {
    "assigned": (0, 200, 0),          # green — PASS
    "rejected": (0, 0, 220),          # red   — FAIL
    "claimed_by_other": (0, 200, 220),  # yellow — valid candidate, lost tie-break
}
_REGION_COLOR = {
    "head_region": (255, 200, 0),
    "torso_region": (255, 140, 0),
}
_PERSON_BOX_COLOR = (0, 140, 255)

# Room overlay colours (BGR): hazard amber, restricted red.
_ZONE_COLOR = {
    "hazard":     (0, 170, 245),
    "restricted": (60, 60, 230),
}


def _draw_debug_overlay(frame, debug_info):
    """
    Draw DecisionEngine's per-frame calibration diagnostics onto `frame`.

    `debug_info` is FrameVerdict.debug_info (see core/decision_engine.py's
    module docstring for its exact structure) or None. A None debug_info
    is the normal case whenever config["decision_engine"]["debug"] is
    false (the default) — this function is a deliberate no-op in that
    case: zero cv2 calls, so enabling debug_overlay costs nothing when
    debug itself is off.

    Kept entirely separate from _draw_detections(): decision_engine.py
    does not draw (no cv2 import there, see its module docstring), so
    this function — not DecisionEngine — owns the "how do we visualise
    this" decision, consistent with the rest of the project's separation
    between AI/decision modules and the dashboard's rendering layer.
    """
    if debug_info is None:
        return

    for person in debug_info["persons"]:
        x1, y1, x2, y2 = person["box"]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      _PERSON_BOX_COLOR, 2)
        cv2.putText(
            frame, f"person {person['person_index']} ({person['confidence']:.2f})",
            (int(x1), max(int(y1) - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _PERSON_BOX_COLOR, 1,
        )

        for region_key in ("head_region", "torso_region"):
            y_lo, y_hi, x_lo, x_hi = person[region_key]
            cv2.rectangle(
                frame, (int(x_lo), int(y_lo)), (int(x_hi), int(y_hi)),
                _REGION_COLOR[region_key], 1,
            )

        for ppe_key in ("helmet", "vest"):
            for candidate in person[ppe_key]["candidates"]:
                cx, cy = candidate["centroid"]
                color = _STATUS_COLOR.get(candidate["status"], _DEFAULT_COLOR)
                cv2.drawMarker(
                    frame, (int(cx), int(cy)), color,
                    markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2,
                )
                containment = candidate["containment"]
                label = (
                    f"{candidate['class_name']} "
                    f"{containment:.2f}" if containment is not None
                    else f"{candidate['class_name']} (claimed)"
                )
                if candidate.get("wear_ok") is False:
                    label += " NOT WORN"
                cv2.putText(
                    frame, label, (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )


def _draw_detections(frame, detections):
    """
    Draw bounding boxes on frame (must be a writable copy — caller's responsibility).

    Frame and detections are read from SharedState in two separate get_ calls
    and may be very slightly out of sync (a detection from one pipeline cycle
    drawn on a frame from a fraction of a second later).  This is an accepted
    Level 1 approximation; exact pairing would require bundling frame and
    detections into a single SharedState field, which is a larger change
    deferred to a later milestone.
    """
    for d in detections:
        color = _BOX_COLORS.get(d.class_name, _DEFAULT_COLOR)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(
            frame, label, (d.x1, max(d.y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )


def _missing_text(violations) -> str:
    """['no-helmet', 'no-vest'] -> 'no helmet & vest'."""
    h, v = "no-helmet" in violations, "no-vest" in violations
    return "no helmet & vest" if h and v else "no helmet" if h else "no vest" if v else ""


def _zone_tag(frame, name, pts, color):
    """Zone name as a solid tag (zone colour, contrasting text) centred in
    the zone, so it stays readable on any background and labels of
    neighbouring zones do not run into each other."""
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.45, frame.shape[1] / 1100)
    thick = 1 if scale < 0.8 else 2
    zone_w = pts[:, 0].max() - pts[:, 0].min()
    (tw, th), base = cv2.getTextSize(name, font, scale, thick)
    if tw > 0.85 * zone_w:  # shrink to fit inside the zone
        scale = max(0.3, scale * 0.85 * zone_w / tw)
        (tw, th), base = cv2.getTextSize(name, font, scale, thick)
    # Centre of the bounding box, not the vertex average: a zone drawn with
    # extra or doubled corners would pull the vertex average to one side.
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
    pad = 3
    x1, y1 = int(cx - tw / 2) - pad, int(cy - th / 2) - pad
    x2, y2 = x1 + tw + 2 * pad, y1 + th + base + 2 * pad
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    # Dark text on light colours (amber), white text on dark ones (red).
    text = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(frame, name, (x1 + pad, y1 + pad + th), font, scale, text,
                thick, cv2.LINE_AA)


def _draw_zones(frame, zones, zone_status):
    """
    Room camera overlay: translucent zone polygons with their names, and
    each tracked person's ID plus seconds inside a zone. A person inside a
    zone who is breaking its rule is labelled in the zone's colour; a
    person correctly equipped inside a hazard zone is labelled green.
    """
    if zones:
        overlay = frame.copy()
        for z in zones:
            pts = np.array(z.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], _ZONE_COLOR.get(z.type, _DEFAULT_COLOR))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
        for z in zones:
            pts = np.array(z.polygon, dtype=np.int32)
            color = _ZONE_COLOR.get(z.type, _DEFAULT_COLOR)
            cv2.polylines(frame, [pts], True, color, 2)
            _zone_tag(frame, z.name, pts, color)

    for st in zone_status:
        st["_color"] = ((0, 200, 0) if st["zone_type"] == "hazard" and st["ppe_ok"]
                        else _ZONE_COLOR.get(st["zone_type"], _DEFAULT_COLOR))


def _draw_room_detections(frame, detections, zone_status):
    """Person boxes with track IDs; colour reflects zone status if any."""
    by_track = {st["track_id"]: st for st in zone_status}
    for d in detections:
        if d.class_name != "person":
            color = _BOX_COLORS.get(d.class_name, _DEFAULT_COLOR)
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 1)
            continue
        st = by_track.get(d.track_id)
        color = st["_color"] if st else _PERSON_BOX_COLOR
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        label = f"#{d.track_id}" if d.track_id is not None else "person"
        if st:
            missing = _missing_text(st.get("violations") or [])
            if st["zone_type"] == "hazard" and missing:
                label += f" {missing} {st['seconds']:.1f}s"
            elif st["zone_type"] != "hazard":
                label += f" NO ENTRY {st['seconds']:.1f}s"
        font, scale = cv2.FONT_HERSHEY_DUPLEX, max(0.4, frame.shape[1] / 1300)
        (tw, th), base = cv2.getTextSize(label, font, scale, 1)
        # Solid tag above the box; inside the box when it touches the top.
        y2 = d.y1 if d.y1 > th + base + 6 else d.y1 + th + base + 6
        y1 = y2 - th - base - 6
        cv2.rectangle(frame, (d.x1, y1), (d.x1 + tw + 6, y2), color, -1)
        text = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(frame, label, (d.x1 + 3, y2 - base - 3), font, scale, text, 1, cv2.LINE_AA)
        cv2.circle(frame, ((d.x1 + d.x2) // 2, d.y2), 4, color, -1)  # foot point


def create_app(config: dict, states: dict, event_store, zones=None, user_store=None,
               zone_monitor=None, zone_monitors=None, cameras=None, on_restart=None,
               manager=None):
    """
    states        : {"entrance": SharedState, "<room id>": SharedState, ...}
                    (a None value = that camera is off)
    zone_monitors : {"<room id>": ZoneMonitor} for the running room cameras;
                    zones are read from them every frame and the zone editor
                    updates them live. zone_monitor= is the older single-room
                    form (same as {"room": zone_monitor}).
    zones         : static zones for "room" when no ZoneMonitor is given (tests)
    cameras       : camera list from core.camera_registry (names for the
                    dashboard); derived from states when not given
    on_restart    : restarts the whole of main.py (kept as a fallback)
    manager       : core.camera_manager.CameraManager. When given, the
                    Cameras page starts / stops single cameras live, and
                    the camera lists below are re-read on every request
                    (states / zone_monitors are the manager's own dicts).
    user_store    : storage.user_store.UserStore (dashboard accounts)
    """
    app = Flask(__name__)
    if user_store is None:
        from storage.user_store import UserStore
        user_store = UserStore(config)
    init_auth(app, config, user_store)

    zone_monitors = dict(zone_monitors or {})
    if zone_monitor is not None:
        zone_monitors.setdefault("room", zone_monitor)
    static_zones = {"room": list(zones or [])}

    def zones_for(cid):
        zm = zone_monitors.get(cid)
        return zm.zones if zm is not None else static_zones.get(cid, [])

    def apply_zones(cid, new_zones):
        if cid in zone_monitors:
            zone_monitors[cid].set_zones(new_zones)
        else:
            static_zones[cid] = list(new_zones)

    entrance = states["entrance"]
    if cameras is None:
        cameras = [{"id": "entrance", "role": "entrance", "name": "Entrance", "enabled": True}]
        cameras += [{"id": cid, "role": "room", "enabled": True,
                     "name": "Room" if cid == "room" else f"Room {cid[4:]}"}
                    for cid, st in states.items() if cid != "entrance" and st is not None]

    def cams_now():
        """Current camera list (live with a manager: cameras come and go)."""
        if manager is not None:
            try:
                return load_cameras(config)[0]
            except (ValueError, KeyError, OSError):
                pass
        return cameras

    def live_cams(role):
        """Cameras of this role that have a video state (running or tried)."""
        return [{"id": c["id"], "name": c["name"]} for c in cams_now()
                if c["role"] == role and c.get("enabled", True) and states.get(c["id"]) is not None]

    def room_ids_now():
        return [c["id"] for c in live_cams("room")]

    def running_room_ids():
        if manager is not None:
            return list(zone_monitors)
        return room_ids_now()

    def cam_names():
        return {c["id"]: c["name"] for c in cams_now()}

    init_zone_editor(app, config, states, running_room_ids, zones_for, apply_zones)
    init_camera_setup(app, config, states, cameras, on_restart, zones_for, manager=manager)
    screenshot_dir = Path(config["alerts"]["screenshot_dir"]).resolve()

    # Calibration/diagnostic flag — see config.yaml's decision_engine
    # section and core/decision_engine.py's module docstring. Read once
    # at construction (config does not change at runtime), not per-frame.
    debug_overlay_enabled = bool(config["decision_engine"].get("debug_overlay", False))

    def _status_frame(frame, camera):
        """Dimmed last picture (or a blank one) with the camera's problem written on it."""
        info = manager.status(camera) if manager is not None else {"status": "starting", "error": None}
        title = {"error": "CAMERA ERROR", "disconnected": "CAMERA DISCONNECTED",
                 "stopped": "CAMERA STOPPED"}.get(info["status"], "WAITING FOR CAMERA")
        img = (frame // 4) if frame is not None else np.zeros((360, 640, 3), np.uint8)
        h, w = img.shape[:2]
        scale = w / 640
        for i, (text, size, color) in enumerate([(title, 0.9, (80, 80, 240)),
                                                 ((info.get("error") or "")[:70], 0.5, (220, 220, 220))]):
            if not text:
                continue
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, size * scale, 1)
            cv2.putText(img, text, ((w - tw) // 2, int(h / 2 + i * 34 * scale)),
                        cv2.FONT_HERSHEY_DUPLEX, size * scale, color, max(1, int(scale)), cv2.LINE_AA)
        return img

    def _generate_frames(camera: str):
        state = states[camera]
        waited = 0.0
        while True:
            frame = state.get_frame()
            running = manager is None or manager.status(camera)["status"] == "running"

            if frame is None or not running:
                # No (live) picture: after a short wait show why, once a second.
                if running and waited < 1.0:
                    time.sleep(0.05)
                    waited += 0.05
                    continue
                ok, buffer = cv2.imencode(".jpg", _status_frame(frame, camera))
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                time.sleep(1.0)
                continue
            waited = 0.0

            # Read detections separately; slight temporal mismatch is
            # acceptable (see _draw_detections docstring).
            detections = state.get_latest_detections()
            if not camera.startswith("entrance"):
                zone_status = state.get_zone_status()
                _draw_zones(frame, zones_for(camera), zone_status)
                _draw_room_detections(frame, detections, zone_status)
            else:
                _draw_detections(frame, detections)
                if debug_overlay_enabled:
                    _draw_debug_overlay(frame, state.get_latest_debug_info())

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
            time.sleep(_FRAME_INTERVAL)

    @app.route("/video_feed")
    @app.route("/video_feed/<camera>")
    def video_feed(camera="entrance"):
        if states.get(camera) is None:
            abort(404)
        return Response(
            _generate_frames(camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _entrance_block(cid, name):
        st = states[cid]
        session_state = st.get_session_state()
        return {
            "id": cid,
            "name": name,
            "session_state": session_state.value if session_state is not None else "unknown",
            "camera_status": _cam_status(cid),
            "fps": round(st.get_fps(), 1),
            "detections": [
                {"class_name": d.class_name, "confidence": round(d.confidence, 2)}
                for d in st.get_latest_detections()
            ],
            "events": event_store.get_recent(10, camera=cid),
        }

    def _cam_status(cid):
        if manager is not None:
            return manager.status(cid)["status"]
        return states[cid].get_camera_status()

    @app.route("/api/status")
    def api_status():
        names = cam_names()
        entrances = [_entrance_block(c["id"], c["name"]) for c in live_cams("entrance")]
        first = next((e for e in entrances if e["id"] == "entrance"), None) or \
            (entrances[0] if entrances else _entrance_block("entrance", names.get("entrance", "Entrance")))
        room_ids = room_ids_now()
        status = {
            # Top-level fields: the first entrance camera (older single-entrance form).
            "session_state": first["session_state"],
            "camera_status": first["camera_status"],
            "fps": first["fps"],
            "detections": first["detections"],
            "recent_events": event_store.get_recent(10),
            # Separate lists so the dashboard shows entrance decisions and
            # room violations in their own panels.
            "entrance_events": event_store.get_recent(
                10, camera=[e["id"] for e in entrances] or "entrance"),
            "room_events": event_store.get_recent(10, camera=room_ids),
            "entrances": entrances,
            "room": {"enabled": False},
            "rooms": [],
        }
        for cam in live_cams("room"):
            st = states[cam["id"]]
            dets = st.get_latest_detections()
            status["rooms"].append({
                "id": cam["id"],
                "name": cam["name"],
                "enabled": True,
                "camera_status": _cam_status(cam["id"]),
                "fps": round(st.get_fps(), 1),
                "persons": sum(1 for d in dets if d.class_name == "person"),
                "zone_status": st.get_zone_status(),
                "zones": [{"name": z.name, "type": z.type} for z in zones_for(cam["id"])],
                "events": event_store.get_recent(10, camera=cam["id"]),
            })
        if status["rooms"]:
            status["room"] = status["rooms"][0]   # older single-room field
        return jsonify(status)

    @app.route("/")
    def index():
        rooms = live_cams("room")
        return render_template("index.html", room_enabled=bool(rooms), room_cams=rooms,
                               entrance_cams=live_cams("entrance"))

    @app.route("/history")
    def history():
        return render_template("history.html")

    @app.route("/api/stats")
    def api_stats():
        return jsonify(event_store.get_stats())

    @app.route("/statistics")
    def statistics():
        return render_template("statistics.html", room_enabled=bool(live_cams("room")),
                               cam_names=cam_names())

    @app.route("/api/summary")
    def api_summary():
        # ?days=1 (today), 7, 30; 0 = all time
        try:
            days = max(0, min(365, int(request.args.get("days", 7))))
        except (TypeError, ValueError):
            days = 7
        return jsonify(event_store.get_summary(days or None))

    @app.route("/api/events")
    def api_events():
        # Pagination: ?page=N (1-based), optional ?type=HAZARD_PPE etc.
        # Page size fixed at 25.
        _PAGE_SIZE = 25
        try:
            page = max(1, int(request.args.get("page", 1)))
        except (TypeError, ValueError):
            page = 1
        event_type = request.args.get("type") or None
        offset = (page - 1) * _PAGE_SIZE
        result = event_store.get_all_events(limit=_PAGE_SIZE, offset=offset,
                                            event_type=event_type)
        return jsonify({
            "page":      page,
            "page_size": _PAGE_SIZE,
            "events":    result["events"],
            "has_more":  result["has_more"],
        })

    @app.route("/snapshots/<path:filename>")
    def snapshot(filename):
        # Only files inside the screenshot directory can be served.
        return send_from_directory(screenshot_dir, Path(filename).name)

    return app
