"""
Edge AI Industrial Safety Monitoring — entry point.

Starts, from config/config.yaml:
  - entrance camera(s) + Pipeline    (PPE ALLOW/DENY check point), one each
  - room camera(s) + RoomPipeline    (hazard / restricted zones), one each
  Each camera is started / stopped on its own by core/camera_manager.py,
  also while running (dashboard Cameras page: add, restart, disable, remove).
  Which camera does what: output/cameras.json (Settings > Cameras in the
  dashboard), else the cameras: section of the config file.
  - alert handlers shared by both    (screenshot, voice, LEDs, event log)
  - Flask dashboard                  (both streams, events, history)

Run from the fyp2/ directory:
    python main.py
    python main.py --config config/config_pi.yaml
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

from core.config_loader import load_config
from core.detector import Detector
from core.decision_engine import DecisionEngine
from core.session_controller import SessionController
from core.pipeline import Pipeline
from core.pose_wear_check import PoseWearChecker
from core.room_pipeline import RoomPipeline
from core.zone_monitor import ZoneMonitor
from core.zones import load_active_zones
from core.camera_registry import load_cameras
from core.camera_manager import CameraManager, CapacityError
from alerts.alert_manager import AlertManager
from alerts.screenshot_handler import ScreenshotHandler
from alerts.voice_handler import VoiceHandler
from alerts.speaker import Speaker
from alerts.led_handler import LedHandler
from alerts.event_log_handler import EventLogHandler
from storage.event_store import EventStore
from dashboard.app import create_app

logger = logging.getLogger(__name__)


def run(config: dict) -> bool:
    """Construct every component, start all threads, block until Ctrl+C.
    Returns True when the dashboard asked for a restart (new camera setup)."""
    camera_list, cam_source = load_cameras(config)
    logger.info("Cameras (%s): %s", cam_source,
                ", ".join(f"{c['id']}={c['name']} [{c['backend']}:{c['source']}]"
                          + ("" if c["enabled"] else " (off)") for c in camera_list) or "none")

    event_store = EventStore(config)
    decision_engine = DecisionEngine(config)

    # One USB speaker shared by all cameras: prompts are queued, never overlap.
    speaker = Speaker(config["voice"].get("max_wait_seconds", 8.0))

    handlers = [
        # ScreenshotHandler BEFORE EventLogHandler: it fills event.snapshot
        # for zone events, which EventLogHandler then stores.
        ScreenshotHandler(config),
        VoiceHandler(config, speaker=speaker),
        EventLogHandler(event_store),
    ]
    led_handler = None
    if config.get("leds", {}).get("enabled", False):
        led_handler = LedHandler(config)
        handlers.append(led_handler)
    alert_manager = AlertManager(handlers)

    pose_enabled = config["decision_engine"].get("pose_check", {}).get("enabled", False)

    def make_pipeline(cam_cfg, state, capture):
        """AI pipeline for one camera (called by CameraManager on every start)."""
        cid = cam_cfg["id"]
        if cam_cfg["role"] == "entrance":
            if cid == "entrance":
                am = alert_manager
            else:
                # Each extra entrance camera gets its own screenshot / voice /
                # log handlers (they remember the last decision to debounce);
                # the LEDs and the speaker are shared hardware.
                extra = [ScreenshotHandler(config, camera=cid), VoiceHandler(config, speaker=speaker),
                         EventLogHandler(event_store, camera=cid)]
                am = AlertManager(extra + ([led_handler] if led_handler else []))
            pipeline = Pipeline(
                config, state, capture, Detector(config), decision_engine,
                SessionController(config), am,
                pose_checker=PoseWearChecker(config) if pose_enabled else None, name=cid,
            )
            return pipeline, None
        zones, source = load_active_zones(config, cid)
        logger.info("Zones for %s: %d loaded from %s", cid, len(zones), source)
        zone_monitor = ZoneMonitor(config, zones=zones, camera=cid)
        # Separate Detector per camera: tracker state must not mix cameras.
        pipeline = RoomPipeline(config, state, Detector(config), decision_engine,
                                zone_monitor, alert_manager, name=cid)
        return pipeline, zone_monitor

    # Starts / stops each camera on its own, also later from the dashboard.
    manager = CameraManager(config, make_pipeline)
    for cam_cfg in camera_list:
        if not cam_cfg["enabled"]:
            continue
        try:
            result = manager.start(cam_cfg)
        except CapacityError as e:
            logger.error("Camera %s not started: %s", cam_cfg["id"], e)
            manager.note_error(cam_cfg["id"], f"Not started: {e}")
            continue
        if result["status"] == "error":
            logger.error("Camera %s: %s. Fix it in the dashboard (Cameras), or delete "
                         "output/cameras.json to go back to config.yaml.", cam_cfg["id"], result["error"])
    if not manager.running_ids():
        logger.warning("No camera is running. Add or fix cameras in the dashboard (Cameras page).")

    restart = threading.Event()
    app = create_app(config, manager.states, event_store, zone_monitors=manager.zone_monitors,
                     cameras=camera_list, on_restart=restart.set, manager=manager)

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
        while flask_thread.is_alive() and not restart.is_set():
            flask_thread.join(timeout=1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Restarting..." if restart.is_set() else "Shutting down...")
        # Each camera: pipeline first, then capture (releases the device).
        manager.stop_all()
        if led_handler is not None:
            led_handler.close()
        speaker.close()
        event_store.close()
        logger.info("Done.")
    return restart.is_set()


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
    if run(config):
        # Camera setup changed in the dashboard: start this program again
        # from scratch (cameras released above), with the same arguments.
        # Flask's dev server marks its listening socket inheritable and
        # records it in WERKZEUG_SERVER_FD; close it, or the new process
        # finds the port still taken.
        fd = os.environ.pop("WERKZEUG_SERVER_FD", None)
        if fd is not None:
            try:
                os.close(int(fd))
            except (OSError, ValueError):
                pass
        time.sleep(1)
        os.execv(sys.executable, [sys.executable] + sys.argv)
