import logging
import sys
import threading
from pathlib import Path

from core.config_loader import load_config
from core.shared_state import SharedState
from core.camera import CameraCapture
from core.detector import Detector
from core.decision_engine import DecisionEngine
from core.session_controller import SessionController
from core.pipeline import Pipeline
from alerts.alert_manager import AlertManager
from alerts.screenshot_handler import ScreenshotHandler
from dashboard.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("output/logs/app.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    shared_state = SharedState()

    camera = CameraCapture(config, shared_state)
    detector = Detector(config)
    decision_engine = DecisionEngine()
    session_controller = SessionController(config)
    alert_manager = AlertManager([ScreenshotHandler(config)])
    pipeline = Pipeline(config, shared_state, camera, detector,
                        decision_engine, session_controller, alert_manager)

    try:
        camera.start()
    except RuntimeError as e:
        logger.error("Failed to start camera: %s", e)
        sys.exit(1)

    # Pipeline starts after camera so the camera thread has time to produce
    # frames, but the pipeline's None-check means startup order is actually
    # safe either way — it will simply spin on sleep(0.005) until a frame
    # arrives.
    pipeline.start()

    app = create_app(config, shared_state)

    dash = config["dashboard"]
    logger.info("Starting dashboard on http://%s:%d", dash["host"], dash["port"])

    # Flask runs in a daemon thread rather than the main thread (changed from
    # Step 4's approach). Step 4 ran Flask in the main thread, which meant
    # KeyboardInterrupt was caught cleanly — but only if no MJPEG streaming
    # connection was open. An open /video_feed connection holds Flask's
    # dev server in its generator loop indefinitely, blocking the main thread
    # from reaching the finally block and preventing camera.stop() from being
    # called. Moving Flask to a daemon thread and polling flask_thread.join()
    # with a 1-second timeout lets the main thread respond to KeyboardInterrupt
    # immediately regardless of open streaming connections.
    flask_thread = threading.Thread(
        target=lambda: app.run(host=dash["host"], port=dash["port"], threaded=True),
        daemon=True,
        name="FlaskThread",
    )
    flask_thread.start()

    try:
        while flask_thread.is_alive():
            flask_thread.join(timeout=1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        # Stop pipeline before camera — pipeline depends on camera being
        # alive while it runs, not the other way around.
        pipeline.stop()
        camera.stop()
        logger.info("Done.")


if __name__ == "__main__":
    main()
