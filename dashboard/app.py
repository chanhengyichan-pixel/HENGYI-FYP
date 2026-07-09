import logging
import time

import cv2
from flask import Flask, Response, render_template

logger = logging.getLogger(__name__)


_FRAME_INTERVAL = 1.0 / 30  # cap stream at 30 FPS


def create_app(config: dict, shared_state):
    app = Flask(__name__)

    def _generate_frames():
        while True:
            frame = shared_state.get_frame()

            if frame is None:
                # No frame yet — wait briefly and retry rather than crashing.
                time.sleep(0.05)
                continue

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

    @app.route("/")
    def index():
        return render_template("index.html")

    return app
