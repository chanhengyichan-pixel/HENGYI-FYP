"""
Worker enrollment orchestration — browser-based guided capture.

Usage (standalone):
    cd /path/to/fyp2
    source venv/bin/activate
    python scripts/enroll_worker.py "Worker Name"

Prints a URL; open it in a browser to see the live guidance overlay
(face box, instruction text, progress bar) and complete enrollment.
The terminal-only capture loop from the previous version of this
script has been replaced by scripts/enrollment_server.py's guided
Flask/MJPEG flow — see that module's docstring for the full design
rationale (guidance priority order, capture cooldown, port reuse).

This script is STANDALONE — it does NOT use CameraCapture or
SharedState. It opens the webcam directly for the enrollment session
and releases it when done, following the same start()/stop() discipline
as core/camera.py and core/pipeline.py.

Do NOT run this while main.py's dashboard (Choice 1) is running — both
would compete for the same webcam device AND the same dashboard port.
main.py's CLI menu structure guarantees these are mutually exclusive
entry points (see enrollment_server.py's "Port reuse" docstring
section for why reusing config["dashboard"]["port"] here is safe).

Manual verification after enrolling multiple workers
-----------------------------------------------------
After enrolling Worker A then Worker B, run main.py and stand in front
of the camera as each worker in turn. The dashboard's worker name field
should show the correct name for each person — if it shows Worker B's
name while Worker A is present (or vice versa), the model is corrupted
and re-enrollment is needed. This is the exact scenario the train()/
update() bug (preserved-fixed in EnrollmentSession._finish_success)
would have broken silently.
"""

import sys
import threading
import webbrowser
from pathlib import Path

# Allow imports from the project root whether run standalone or imported.
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from werkzeug.serving import make_server

from core.config_loader import load_config
from storage.worker_store import WorkerStore
from scripts.enrollment_server import (
    EnrollmentSession, EnrollmentState, create_enrollment_app, validate_pose_stages,
)

_CASCADE_PATH = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")


def enroll_worker(worker_name: str, config: dict) -> bool:
    """Run a full browser-guided enrollment session for `worker_name`.

    Parameters
    ----------
    worker_name : str
        Display name for the worker. Must be non-empty (caller's responsibility).
    config : dict
        Loaded application config (same object passed to all other components).

    Returns
    -------
    bool
        True if enrollment completed successfully (target sample count
        reached and the model was trained/updated and written), False on
        any failure — webcam unavailable, or the session timed out
        without reaching the sample target (see
        config["enrollment"]["session_timeout_seconds"]).

    Side effects
    ------------
    - Writes/updates models/face_recognizer.yml (path from config["face"]["model_path"]).
    - Inserts a new row in the workers database (config["workers"]["db_path"]).
    Both writes are permanent; a False return means the model write did
    NOT happen (WorkerStore entry IS written before capture so that
    label_id is reserved — same ordering as the previous implementation
    — but the model is only written on a successful finish).

    Lifecycle
    ---------
    Starts a temporary Flask server (werkzeug.serving.make_server, not
    app.run() — needed for an explicit, externally-triggered .shutdown(),
    which app.run() has no clean way to provide) on a background thread,
    reusing config["dashboard"]["port"]. Blocks on a threading.Event
    that EnrollmentSession sets when the session finishes (success or
    failure), then shuts the server down, joins its thread, and releases
    the webcam before returning — mirroring core/camera.py's
    start()/stop() discipline so a leaked server or open device can
    never carry over into the next enrollment or dashboard launch.
    """
    enroll_cfg = config["enrollment"]
    # Fail fast on a misconfigured pose_stages/num_samples mismatch —
    # before the webcam, WorkerStore, or Flask server are touched at all.
    validate_pose_stages(enroll_cfg)

    model_path = Path(config["face"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    worker_store = WorkerStore(config)
    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    model_existed = model_path.exists()
    if model_existed:
        recognizer.read(str(model_path))
        print(f"Existing model loaded from {model_path} — will update with new worker.")
    else:
        print(f"No existing model found — creating new model at {model_path}.")

    label_id = worker_store.add_worker(worker_name)
    print(f"\nRegistered '{worker_name}' with label_id={label_id}.")

    cap = cv2.VideoCapture(config["camera"]["source"])
    if not cap.isOpened():
        print("Error: could not open webcam.")
        worker_store.close()
        return False

    state = EnrollmentState(
        target_samples=int(enroll_cfg["num_samples"]),
        stage_count=len(enroll_cfg["pose_stages"]),
    )
    completed_event = threading.Event()
    session = EnrollmentSession(
        cap=cap, cascade=cascade, recognizer=recognizer,
        model_path=model_path, model_existed=model_existed,
        label_id=label_id, cfg=enroll_cfg,
        state=state, completed_event=completed_event,
    )

    app = create_enrollment_app(session, worker_name)
    dash_cfg = config["dashboard"]
    # threaded=True is required, not optional — see enrollment_server.py's
    # "Threaded server — required, not optional" docstring section. Without
    # it, GET /enroll/status can't be answered while the long-lived
    # GET /enroll/feed stream is open, so the page never learns the
    # session finished until the stream itself drops.
    server = make_server(dash_cfg["host"], dash_cfg["port"], app, threaded=True)
    server_thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="EnrollmentServerThread"
    )
    server_thread.start()

    display_host = "127.0.0.1" if dash_cfg["host"] == "0.0.0.0" else dash_cfg["host"]
    url = f"http://{display_host}:{dash_cfg['port']}/enroll"
    print(f"\nOpen {url} in your browser to continue enrollment.")
    try:
        webbrowser.open(url)
    except Exception:
        pass  # best-effort convenience only; the printed URL above is authoritative

    # Hard backstop beyond the in-session no-progress timeout
    # (config["enrollment"]["session_timeout_seconds"]): that timeout only
    # starts counting once frames are being pulled from /enroll/feed by a
    # connected browser. If no browser ever connects at all, nothing would
    # otherwise end the wait — this bounds it regardless.
    session_timeout = float(enroll_cfg["session_timeout_seconds"])
    completed = completed_event.wait(timeout=session_timeout + 30)
    if not completed:
        state.finish(False, "Enrollment timed out — no browser connected.")

    server.shutdown()
    server_thread.join(timeout=5)
    cap.release()
    worker_store.close()

    result = state.snapshot()
    success = result["status"] == "success"
    if success:
        print(f"\nEnrollment complete: '{worker_name}' (label_id={label_id}) is now registered.")
    else:
        print(f"\nEnrollment failed for '{worker_name}': {result['message']}")
    return success


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts/enroll_worker.py "Worker Name"')
        sys.exit(1)

    name = sys.argv[1].strip()
    if not name:
        print("Error: worker name cannot be empty.")
        sys.exit(1)

    cfg = load_config()
    success = enroll_worker(name, cfg)
    sys.exit(0 if success else 1)
