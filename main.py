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
from core.face_recognizer import FaceRecognizer
from alerts.alert_manager import AlertManager
from alerts.screenshot_handler import ScreenshotHandler
from alerts.voice_handler import VoiceHandler
from alerts.event_log_handler import EventLogHandler
from storage.event_store import EventStore
from storage.worker_store import WorkerStore
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


def run_dashboard(config: dict) -> None:
    """Start the full access-control dashboard and block until shutdown (Ctrl+C).

    Extracted verbatim from the original main() body — no logic changes.
    Constructs every component, starts the camera + pipeline + Flask thread,
    and runs the clean shutdown sequence on KeyboardInterrupt.

    Called from _menu() for Choice 1.  When it returns, the caller exits
    the program entirely — stopping the dashboard is shutting down the whole
    system, not something you'd relaunch from a menu immediately.
    """
    shared_state = SharedState()
    event_store = EventStore(config)
    worker_store = WorkerStore(config)

    camera = CameraCapture(config, shared_state)
    detector = Detector(config)
    decision_engine = DecisionEngine(config)
    session_controller = SessionController(config)
    face_recognizer = FaceRecognizer(config, worker_store)
    alert_manager = AlertManager([
        ScreenshotHandler(config),
        VoiceHandler(config),
        EventLogHandler(event_store),
    ])
    pipeline = Pipeline(config, shared_state, camera, detector,
                        decision_engine, session_controller, alert_manager,
                        face_recognizer=face_recognizer)

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

    app = create_app(config, shared_state, event_store)

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
        event_store.close()
        worker_store.close()
        logger.info("Done.")


def _menu(config: dict) -> None:
    """Interactive CLI menu shown before any hardware is touched.

    Config is loaded once by the caller before entering here — fail-fast on
    a bad config file before the user has picked anything.

    Choice 1 — Dashboard: calls run_dashboard() and exits when it returns.
      Stopping the dashboard is shutting down the whole access-control system;
      we do not loop back to the menu afterward.

    Choice 2 — Register Worker: prompts for a name, calls enroll_worker(),
      prints a clear success/failure line, then loops back.  Enrolling
      multiple workers in one sitting is an expected workflow.

    Choice 3 — Exit: exits immediately.

    The webcam is never opened until the user explicitly picks a mode.  This
    prevents device contention between CameraCapture (dashboard) and the
    direct cv2.VideoCapture in enroll_worker.py — both compete for the same
    /dev/video0 device and will fail or produce garbled frames if run
    simultaneously.
    """
    # Deferred import: enroll_worker.py inserts its parent directory into
    # sys.path at module level.  Importing here (rather than at the top of
    # main.py) keeps the import side-effect close to where it's needed and
    # avoids loading cv2/numpy until the user actually requests enrollment.
    from scripts.enroll_worker import enroll_worker

    while True:
        print("\n=== Safety Access Control System ===")
        print("  1. Start Dashboard")
        print("  2. Register New Worker")
        print("  3. Exit")

        choice = input("Choice: ").strip()

        if choice == "1":
            run_dashboard(config)
            sys.exit(0)

        elif choice == "2":
            name = input("Worker name: ").strip()
            if not name:
                print("Name cannot be empty.")
                continue
            success = enroll_worker(name, config)
            if success:
                print(f"\nEnrollment successful: '{name}' is now registered.")
            else:
                print(f"\nEnrollment failed for '{name}'. See errors above.")

        elif choice == "3":
            sys.exit(0)

        else:
            print("Invalid choice — enter 1, 2, or 3.")


if __name__ == "__main__":
    config = load_config()
    _menu(config)
