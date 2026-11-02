"""
Edge AI Industrial Safety Monitoring — entry point.

Starts, from config/config.yaml:
  - entrance camera + Pipeline       (PPE ALLOW/DENY check point)
  - room camera + RoomPipeline       (hazard / restricted zones), if enabled
  - alert handlers shared by both    (screenshot, voice, LEDs, event log)
  - Flask dashboard                  (both streams, events, history)

Run from the fyp2/ directory:
    python main.py
    python main.py --config config/config_pi.yaml
"""

import argparse
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
from core.pose_wear_check import PoseWearChecker
from core.room_pipeline import RoomPipeline
from core.zone_monitor import ZoneMonitor
from core.zones import load_active_zones
from alerts.alert_manager import AlertManager
from alerts.screenshot_handler import ScreenshotHandler
from alerts.voice_handler import VoiceHandler
from alerts.led_handler import LedHandler
from alerts.event_log_handler import EventLogHandler
from storage.event_store import EventStore
from dashboard.app import create_app

logger = logging.getLogger(__name__)


def run(config: dict) -> None:
    """Construct every component, start all threads, block until Ctrl+C."""
    cams = config["cameras"]
    entrance_cfg = cams["entrance"]
    room_cfg = cams.get("room", {"enabled": False})

    event_store = EventStore(config)
    decision_engine = DecisionEngine(config)

    handlers = [
        # ScreenshotHandler BEFORE EventLogHandler: it fills event.snapshot
        # for zone events, which EventLogHandler then stores.
        ScreenshotHandler(config),
        VoiceHandler(config),
        EventLogHandler(event_store),
    ]
    led_handler = None
    if config.get("leds", {}).get("enabled", False):
        led_handler = LedHandler(config)
        handlers.append(led_handler)
    alert_manager = AlertManager(handlers)

    states = {"entrance": SharedState(), "room": None}
    cameras = []
    pipelines = []

    if entrance_cfg.get("enabled", True):
        cam = CameraCapture(entrance_cfg, states["entrance"], name="entrance")
        cameras.append(cam)
        pose_checker = None
        if config["decision_engine"].get("pose_check", {}).get("enabled", False):
            pose_checker = PoseWearChecker(config)
        pipelines.append(Pipeline(
            config, states["entrance"], cam, Detector(config), decision_engine,
            SessionController(config), alert_manager, pose_checker=pose_checker,
        ))

    zone_monitor = None
    if room_cfg.get("enabled", False):
        states["room"] = SharedState()
        zones, source = load_active_zones(config)
        logger.info("Room zones: %d loaded from %s", len(zones), source)
        zone_monitor = ZoneMonitor(config, zones=zones)
        cam = CameraCapture(room_cfg, states["room"], name="room")
        cameras.append(cam)
        # Separate Detector instance: tracker state must not mix cameras.
        pipelines.append(RoomPipeline(
            config, states["room"], Detector(config), decision_engine,
            zone_monitor, alert_manager,
        ))

    if not cameras:
        logger.error("No camera enabled in config — nothing to do.")
        sys.exit(1)

    for cam in cameras:
        try:
            cam.start()
        except (RuntimeError, ImportError, ValueError) as e:
            logger.error("Failed to start camera: %s", e)
            for started in cameras:
                started.stop()
            sys.exit(1)

    # Pipelines start after cameras; their frame checks make the order safe
    # either way (they spin on a short sleep until a frame arrives).
    for p in pipelines:
        p.start()

    app = create_app(config, states, event_store, zone_monitor=zone_monitor)

    dash = config["dashboard"]
    logger.info("Starting dashboard on http://%s:%d", dash["host"], dash["port"])

    # Flask runs in a daemon thread and the main thread polls join() with a
    # timeout, so Ctrl+C is handled immediately even while an MJPEG stream
    # connection is open (an open /video_feed would otherwise keep Flask's
    # dev server in its generator loop and block shutdown).
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
        # Stop pipelines before cameras — pipelines read camera frames.
        for p in pipelines:
            p.stop()
        for cam in cameras:
            cam.stop()
        if led_handler is not None:
            led_handler.close()
        event_store.close()
        logger.info("Done.")


def _setup_logging(config: dict) -> None:
    # Root log level is tied to decision_engine.debug: DEBUG-level
    # diagnostic logs from DecisionEngine are filtered out unless the root
    # level is itself DEBUG.
    debug_enabled = config.get("decision_engine", {}).get("debug", False)
    log_path = Path("output/logs/app.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug_enabled else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    # Flask logs every 1 s /api/status poll at INFO — keep the log readable.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge AI safety monitor")
    parser.add_argument("--config", default="config/config.yaml",
                        help="config file (default: config/config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    _setup_logging(config)
    logger.info("Config loaded from %s — mode=%s", args.config, config.get("mode", "laptop"))
    run(config)
