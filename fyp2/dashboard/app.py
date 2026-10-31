import logging
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

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
            x, y = pts[0]
            cv2.putText(frame, f"{z.name} ({z.type})", (int(x) + 4, int(y) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
            label += f" {st['zone']} {st['seconds']:.1f}s"
        # Box touching the top of the image: draw the label inside it.
        label_y = d.y1 - 6 if d.y1 > 16 else d.y1 + 16
        cv2.putText(frame, label, (d.x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.circle(frame, ((d.x1 + d.x2) // 2, d.y2), 4, color, -1)  # foot point


def create_app(config: dict, states: dict, event_store, zones=None):
    """
    states : {"entrance": SharedState, "room": SharedState | None}
    zones  : list of core.zones.Zone drawn on the room stream
    """
    app = Flask(__name__)
    zones = zones or []
    entrance = states["entrance"]
    room = states.get("room")
    screenshot_dir = Path(config["alerts"]["screenshot_dir"]).resolve()

    # Calibration/diagnostic flag — see config.yaml's decision_engine
    # section and core/decision_engine.py's module docstring. Read once
    # at construction (config does not change at runtime), not per-frame.
    debug_overlay_enabled = bool(config["decision_engine"].get("debug_overlay", False))

    def _generate_frames(camera: str):
        state = states[camera]
        while True:
            frame = state.get_frame()

            if frame is None:
                # No frame yet — wait briefly and retry rather than crashing.
                time.sleep(0.05)
                continue

            # Read detections separately; slight temporal mismatch is
            # acceptable (see _draw_detections docstring).
            detections = state.get_latest_detections()
            if camera == "room":
                zone_status = state.get_zone_status()
                _draw_zones(frame, zones, zone_status)
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

    @app.route("/api/status")
    def api_status():
        session_state = entrance.get_session_state()
        detections = entrance.get_latest_detections()
        status = {
            "session_state": session_state.value if session_state is not None else "unknown",
            "camera_status": entrance.get_camera_status(),
            "fps": round(entrance.get_fps(), 1),
            "recent_events": event_store.get_recent(10),
            "detections": [
                {"class_name": d.class_name, "confidence": round(d.confidence, 2)}
                for d in detections
            ],
            "room": {"enabled": False},
        }
        if room is not None:
            room_dets = room.get_latest_detections()
            status["room"] = {
                "enabled": True,
                "camera_status": room.get_camera_status(),
                "fps": round(room.get_fps(), 1),
                "persons": sum(1 for d in room_dets if d.class_name == "person"),
                "zone_status": room.get_zone_status(),
                "zones": [{"name": z.name, "type": z.type} for z in zones],
            }
        return jsonify(status)

    @app.route("/")
    def index():
        return render_template("index.html", room_enabled=room is not None)

    @app.route("/history")
    def history():
        return render_template("history.html")

    @app.route("/api/stats")
    def api_stats():
        return jsonify(event_store.get_stats())

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
