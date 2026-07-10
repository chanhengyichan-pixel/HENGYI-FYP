import logging
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request

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


def create_app(config: dict, shared_state, event_store):
    app = Flask(__name__)

    def _generate_frames():
        while True:
            frame = shared_state.get_frame()

            if frame is None:
                # No frame yet — wait briefly and retry rather than crashing.
                time.sleep(0.05)
                continue

            # Read detections separately; slight temporal mismatch is
            # acceptable at Level 1 (see _draw_detections docstring).
            detections = shared_state.get_latest_detections()
            _draw_detections(frame, detections)

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
    def video_feed():
        return Response(
            _generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/status")
    def api_status():
        session_state = shared_state.get_session_state()
        detections = shared_state.get_latest_detections()
        return jsonify({
            "session_state": session_state.value if session_state is not None else "unknown",
            "camera_status": shared_state.get_camera_status(),
            "fps": round(shared_state.get_fps(), 1),
            "recent_events": event_store.get_recent(10),
            "detections": [
                {"class_name": d.class_name, "confidence": round(d.confidence, 2)}
                for d in detections
            ],
        })

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/history")
    def history():
        return render_template("history.html")

    @app.route("/api/stats")
    def api_stats():
        return jsonify(event_store.get_stats())

    @app.route("/api/events")
    def api_events():
        # Pagination: ?page=N (1-based).  Page size fixed at 25.
        # Offset = (page - 1) * page_size, computed here so callers
        # only need to track a page number, not raw offsets.
        _PAGE_SIZE = 25
        try:
            page = max(1, int(request.args.get("page", 1)))
        except (TypeError, ValueError):
            page = 1
        offset = (page - 1) * _PAGE_SIZE
        result = event_store.get_all_events(limit=_PAGE_SIZE, offset=offset)
        return jsonify({
            "page":      page,
            "page_size": _PAGE_SIZE,
            "events":    result["events"],
            "has_more":  result["has_more"],
        })

    return app
