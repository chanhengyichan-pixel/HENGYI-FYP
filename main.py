import logging
import sys
import threading
from pathlib import Path

from core.config_loader import load_config
from core.shared_state import SharedState
from core.camera import CameraCapture
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

    try:
        camera.start()
    except RuntimeError as e:
        logger.error("Failed to start camera: %s", e)
        sys.exit(1)

    app = create_app(config, shared_state)

    dash = config["dashboard"]
    logger.info("Starting dashboard on http://%s:%d", dash["host"], dash["port"])

    # Flask runs in a daemon thread so that KeyboardInterrupt on the main
    # thread triggers the finally block immediately without waiting for Flask
    # to finish serving active streaming connections (which would hang).
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
        camera.stop()
        logger.info("Done.")


if __name__ == "__main__":
    main()
