"""
Browser-based guided enrollment: webcam capture, live guidance overlay,
multi-pose LBPH sample capture, and MJPEG streaming for
scripts/enroll_worker.py.

This module owns everything that touches the webcam and Flask for
enrollment: the capture loop, the per-frame guidance overlay, the pose
stage sequencing, the Flask routes, and the capture-cooldown timing. It
does NOT touch LBPH training itself beyond calling
recognizer.train()/update() at the end of a successful session — the
train()-vs-update() choice (train() only for a brand new model,
update() for every subsequent enrollment, so an existing model's other
workers are never discarded) is preserved exactly as it was in the
original terminal-loop implementation.

Why a separate module from scripts/enroll_worker.py:
enroll_worker.py is the orchestration entry point (webcam lifecycle,
WorkerStore, the werkzeug server's start/stop discipline) that main.py's
CLI menu calls. This module is the guided-capture *mechanism* it
delegates to — splitting them keeps the pure, unit-testable guidance
logic (evaluate_capture_guidance, CaptureCooldown, validate_pose_stages)
importable without pulling in WorkerStore or the CLI orchestration,
mirroring how core/decision_engine.py stays independent of
core/pipeline.py.

Port reuse
----------
The enrollment Flask server reuses config["dashboard"]["port"]. This is
safe because main.py's CLI menu makes CameraCapture (the main dashboard)
and this module's own cv2.VideoCapture mutually exclusive entry points
— Choice 1 (dashboard) and Choice 2 (enrollment) can never run at the
same time, so the port is never contended.

Threaded server — required, not optional
------------------------------------------
enroll_worker.py's make_server(...) call passes threaded=True. This was
a real bug found in live testing: the dev server's default
(threaded=False) handles exactly one request at a time. The long-lived
GET /enroll/feed MJPEG stream is itself a single request that stays
open for the whole session — with threaded=False, the page's periodic
GET /enroll/status polls queue up behind it and never receive a
response until the stream connection closes. The completion banner
therefore never appeared: the browser had no way to learn the session
had finished until the feed dropped, by which point there was nothing
useful left to show. threaded=True lets /enroll/status be answered
concurrently with the in-progress /enroll/feed stream.

Guidance check priority
------------------------
evaluate_capture_guidance() runs a fixed sequence of checks and returns
on the FIRST one that fails, so the user is always given exactly one,
unambiguous instruction — never two contradictory messages at once
(e.g. "too dark" and "move closer" simultaneously). Order, from most to
least fundamental: no face -> multiple faces -> lighting -> too far ->
too close -> off-center -> (all pass) the active pose stage's directive.

Capture cooldown
----------------
Even while a frame continuously satisfies every guidance check, samples
are not captured on every single valid frame — that would produce
near-identical frames captured within a fraction of a second, which is
poor training diversity for LBPH. CaptureCooldown enforces a minimum
wall-clock interval between captures (config-driven,
capture_cooldown_seconds) so samples are spread across a few seconds of
natural micro-movement instead. Uses the same injectable-clock pattern
as core/session_controller.py.

Multi-pose capture
-------------------
Samples are captured across an ordered sequence of pose stages
(config["enrollment"]["pose_stages"] — center, slight_left,
slight_right, slight_down by default) instead of one flat capture
count. This improves LBPH training diversity beyond what natural
micro-movement within a single pose provides, without adding a second
detector: the same frontal Haar cascade is used throughout, and poses
are deliberately "slight" — within what a frontal-face cascade reliably
still detects. Every sample across every stage trains/updates the SAME
label; stages are a capture-diversity technique, not separate
identities. validate_pose_stages() enforces at startup that the stages'
sample counts sum to exactly num_samples, so the two numbers can never
silently drift apart.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

logger = logging.getLogger(__name__)

_SAMPLE_SIZE = (100, 100)
_MIN_FACE_SIZE = (60, 60)
# Number of frames after a capture during which the face box is drawn in
# the "flash" color with a "Captured" label — gives the user unambiguous
# visual confirmation that a sample actually registered, not just a
# fleeting log line they'd have to be watching the terminal to see.
_FLASH_FRAMES = 6
_FLASH_COLOR = (0, 200, 220)   # cyan — matches dashboard's accent color


def validate_pose_stages(cfg: dict) -> None:
    """Fail fast if enrollment.pose_stages' sample counts don't sum to
    enrollment.num_samples.

    Pure/standalone so it can run before any webcam or Flask resource is
    touched (enroll_worker.py calls this first) and so it's directly
    unit-testable without constructing a full EnrollmentSession.

    Raises
    ------
    ValueError : if the stage sample counts don't sum to num_samples,
                 or pose_stages is empty.
    """
    stages = cfg["pose_stages"]
    if not stages:
        raise ValueError("enrollment.pose_stages must not be empty.")
    total = sum(int(stage["samples"]) for stage in stages)
    expected = int(cfg["num_samples"])
    if total != expected:
        raise ValueError(
            f"enrollment config is inconsistent: pose_stages samples sum to "
            f"{total}, but enrollment.num_samples is {expected} — these must "
            f"match exactly. Fix config/config.yaml before enrolling."
        )


@dataclass(frozen=True)
class GuidanceResult:
    """One instruction for the user, plus whether the current frame is a
    valid capture candidate. `valid=True` means `message` is the current
    pose stage's directive (e.g. "Turn slightly LEFT and hold still") —
    every other case is a specific corrective instruction (see module
    docstring's priority order)."""
    valid: bool
    message: str


def evaluate_capture_guidance(
    face_count: int,
    face_box: tuple[int, int, int, int] | None,
    frame_width: int,
    frame_height: int,
    brightness: float,
    cfg: dict,
    stage_instruction: str = "Hold still",
) -> GuidanceResult:
    """Pure function: given detection results for one frame, decide what
    to tell the user and whether this frame is a valid capture candidate.

    Parameters
    ----------
    face_count : number of faces the Haar cascade found in this frame.
    face_box   : (x, y, w, h) of the face used for the remaining checks
                 (the largest detected face — see EnrollmentSession).
                 Only read when face_count == 1; may be None otherwise.
    frame_width, frame_height : frame dimensions in pixels.
    brightness : mean pixel intensity of the face ROI, 0-255.
    cfg        : config["enrollment"] dict — reads min_face_area_ratio,
                 max_face_area_ratio, center_tolerance_ratio,
                 brightness_threshold.
    stage_instruction : the active pose stage's directive text, returned
                 verbatim when every check passes. Defaults to a generic
                 "Hold still" so this function stays usable without a
                 caller that tracks pose stages.

    No camera, no cv2 calls — takes already-computed scalars so it can
    be unit tested with synthetic inputs.
    """
    if face_count == 0:
        return GuidanceResult(False, "No face detected — please face the camera")
    if face_count > 1:
        return GuidanceResult(False, "Multiple faces detected — only one person should be in frame")

    if brightness < cfg["brightness_threshold"]:
        return GuidanceResult(False, "Lighting is too dark — move to a brighter area")

    x, y, w, h = face_box
    face_area_ratio = (w * h) / (frame_width * frame_height)
    if face_area_ratio < cfg["min_face_area_ratio"]:
        return GuidanceResult(False, "Move closer to the camera")
    if face_area_ratio > cfg["max_face_area_ratio"]:
        return GuidanceResult(False, "Move back from the camera")

    # Offset normalized independently per axis (as a fraction of each
    # half-dimension) so a non-square frame doesn't bias tolerance toward
    # one axis; the larger of the two axis offsets governs the check.
    face_cx = x + w / 2.0
    face_cy = y + h / 2.0
    offset_x = abs(face_cx - frame_width / 2.0) / (frame_width / 2.0)
    offset_y = abs(face_cy - frame_height / 2.0) / (frame_height / 2.0)
    if max(offset_x, offset_y) > cfg["center_tolerance_ratio"]:
        return GuidanceResult(False, "Move to the center of the frame")

    return GuidanceResult(True, stage_instruction)


class CaptureCooldown:
    """Enforces a minimum wall-clock interval between sample captures.

    Injectable clock (default time.monotonic), same pattern as
    core/session_controller.py.SessionController — lets tests advance
    time deterministically without real sleeps.
    """

    def __init__(self, cooldown_seconds: float, clock: Callable[[], float] = time.monotonic):
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._last_capture: float | None = None

    def ready(self) -> bool:
        """True if enough time has passed since the last capture (or none yet)."""
        if self._last_capture is None:
            return True
        return (self._clock() - self._last_capture) >= self._cooldown_seconds

    def mark_captured(self) -> None:
        self._last_capture = self._clock()


class EnrollmentState:
    """Thread-safe progress/outcome holder, polled by GET /enroll/status.

    Written by the capture loop (a Flask request-handling thread, since
    the generator runs inside the /enroll/feed request), read by the
    /enroll/status request (a different, concurrently-served thread —
    see the module docstring's "Threaded server" section for why that
    concurrency requires threaded=True). Same lock-protected
    single-resource pattern as core/shared_state.py.

    Tracks both overall progress (captured/target samples) and current
    pose-stage progress, so the page can show e.g. "Stage 2/4: Slight
    Left — 3/5" without re-deriving stage math client-side.
    """

    def __init__(self, target_samples: int, stage_count: int):
        self._lock = threading.Lock()
        self._target = target_samples
        self._stage_count = stage_count
        self._captured = 0
        self._stage_index = 1
        self._stage_label = ""
        self._stage_captured = 0
        self._stage_target = 0
        self._status = "in_progress"  # "in_progress" | "success" | "failure"
        self._message = ""

    def update_progress(self, captured: int, stage_index: int, stage_label: str,
                         stage_captured: int, stage_target: int) -> None:
        with self._lock:
            self._captured = captured
            self._stage_index = stage_index
            self._stage_label = stage_label
            self._stage_captured = stage_captured
            self._stage_target = stage_target

    def finish(self, success: bool, message: str) -> None:
        with self._lock:
            self._status = "success" if success else "failure"
            self._message = message

    def is_finished(self) -> bool:
        with self._lock:
            return self._status != "in_progress"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "captured": self._captured,
                "target": self._target,
                "stage_index": self._stage_index,
                "stage_count": self._stage_count,
                "stage_label": self._stage_label,
                "stage_captured": self._stage_captured,
                "stage_target": self._stage_target,
                "message": self._message,
            }


def _draw_guidance_overlay(frame, faces, guidance: GuidanceResult,
                            captured: int, target: int, flash: bool,
                            stage_display: str) -> None:
    """Draw the face box(es), instruction text, pose-stage progress line,
    and overall progress bar directly onto `frame` (mutated in place —
    caller's frame, not a copy).

    This is where nearly all of the enrollment UX lives, per design: the
    browser page itself is minimal chrome around this stream.
    """
    h, w = frame.shape[:2]

    if len(faces) == 0:
        cv2.putText(
            frame, "No face detected", (max(w // 2 - 150, 10), h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2,
        )
    else:
        box_color = _FLASH_COLOR if flash else ((0, 200, 0) if guidance.valid else (0, 0, 220))
        for (x, y, fw, fh) in faces:
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), box_color, 2)
        if flash:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            cv2.putText(
                frame, "Captured", (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _FLASH_COLOR, 2,
            )

    # Instruction + stage banner — top of frame, two lines.
    cv2.rectangle(frame, (0, 0), (w, 58), (17, 24, 33), -1)
    text_color = (0, 200, 0) if guidance.valid else (0, 0, 220)
    cv2.putText(
        frame, guidance.message, (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2,
    )
    cv2.putText(
        frame, stage_display, (10, 47),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 216), 1,
    )

    # Overall progress bar — bottom of frame.
    margin, bar_h = 10, 16
    bar_y = h - bar_h - 8
    bar_w = w - 2 * margin
    frac = min(1.0, captured / target) if target else 0.0
    cv2.rectangle(frame, (margin, bar_y), (margin + bar_w, bar_y + bar_h), (37, 51, 68), -1)
    cv2.rectangle(frame, (margin, bar_y), (margin + int(bar_w * frac), bar_y + bar_h), (0, 180, 216), -1)
    cv2.putText(
        frame, f"{captured} / {target}", (margin + 6, bar_y + bar_h - 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
    )


class EnrollmentSession:
    """Owns the webcam capture loop for one enrollment: Haar detection,
    guidance evaluation, cooldown-gated multi-pose sample capture,
    overlay drawing, MJPEG frame encoding, and — on reaching the sample
    target — train()/update() + writing the model file.

    Does NOT open or release the camera (enroll_worker.py owns that
    lifecycle, matching core/camera.py's start()/stop() discipline) and
    does NOT touch WorkerStore — label_id is reserved by the caller
    before construction.
    """

    def __init__(self, cap, cascade, recognizer, model_path: Path,
                 model_existed: bool, label_id: int, cfg: dict,
                 state: EnrollmentState, completed_event: threading.Event):
        validate_pose_stages(cfg)  # defense in depth — enroll_worker.py already validated first

        self._cap = cap
        self._cascade = cascade
        self._recognizer = recognizer
        self._model_path = model_path
        self._model_existed = model_existed
        self._label_id = label_id
        self._cfg = cfg
        self._state = state
        self._completed_event = completed_event

        self._target = int(cfg["num_samples"])
        self._cooldown = CaptureCooldown(float(cfg["capture_cooldown_seconds"]))
        self._session_timeout_seconds = float(cfg["session_timeout_seconds"])

        self._stages = cfg["pose_stages"]
        self._stage_idx = 0          # 0-based index into self._stages
        self._stage_captured = 0     # samples captured within the current stage

        self._samples: list[np.ndarray] = []
        self._labels: list[int] = []

    def frame_generator(self):
        """Generator yielding MJPEG multipart chunks — the request handler
        for GET /enroll/feed. Runs the full capture loop: each frame pulled
        from the webcam is detected, guided against the active pose
        stage's directive, possibly captured, annotated, and streamed,
        until every stage's sample target is reached or the session times
        out from lack of progress.
        """
        flash_remaining = 0
        last_progress_time = time.monotonic()

        while not self._state.is_finished():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=_MIN_FACE_SIZE
            )
            face_count = len(faces)
            face_box = None
            brightness = 0.0
            roi = None

            if face_count >= 1:
                # Largest face drives guidance/capture — same "reduce
                # background false-positive noise" heuristic as
                # core/face_recognizer.py's detect_face().
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                roi = gray[y:y + fh, x:x + fw]
                brightness = float(roi.mean()) if roi.size else 0.0
                face_box = (int(x), int(y), int(fw), int(fh))

            stage = self._stages[self._stage_idx]
            stage_target = int(stage["samples"])
            guidance = evaluate_capture_guidance(
                face_count, face_box, frame.shape[1], frame.shape[0],
                brightness, self._cfg, stage_instruction=stage["instruction"],
            )

            flash = flash_remaining > 0
            if flash_remaining > 0:
                flash_remaining -= 1

            if (guidance.valid and roi is not None and self._cooldown.ready()
                    and len(self._samples) < self._target):
                self._samples.append(cv2.resize(roi, _SAMPLE_SIZE))
                self._labels.append(self._label_id)
                self._cooldown.mark_captured()
                self._stage_captured += 1
                flash_remaining = _FLASH_FRAMES
                flash = True
                last_progress_time = time.monotonic()
                logger.info(
                    "Enrollment sample captured (%d/%d overall, stage '%s' %d/%d)",
                    len(self._samples), self._target, stage["name"],
                    self._stage_captured, stage_target,
                )
                # Advance to the next stage once this one's quota is met
                # (unless this is already the last stage — stay put so
                # display/state reflect the final stage correctly while
                # the overall-target check below ends the session).
                if self._stage_captured >= stage_target and self._stage_idx < len(self._stages) - 1:
                    self._stage_idx += 1
                    self._stage_captured = 0
                    stage = self._stages[self._stage_idx]
                    stage_target = int(stage["samples"])

            self._state.update_progress(
                captured=len(self._samples),
                stage_index=self._stage_idx + 1,
                stage_label=stage["label"],
                stage_captured=self._stage_captured,
                stage_target=stage_target,
            )

            if len(self._samples) >= self._target:
                self._finish_success()
            elif time.monotonic() - last_progress_time > self._session_timeout_seconds:
                self._finish_failure(
                    f"No progress for {self._session_timeout_seconds:.0f}s — enrollment timed out."
                )

            stage_display = (
                f"Stage {self._stage_idx + 1}/{len(self._stages)}: "
                f"{stage['label']} — {self._stage_captured}/{stage_target}"
            )
            _draw_guidance_overlay(
                frame, faces, guidance, len(self._samples), self._target, flash, stage_display
            )

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    def status(self) -> dict:
        return self._state.snapshot()

    def _finish_success(self) -> None:
        logger.info("Training LBPH model on %d samples for label_id=%d", len(self._samples), self._label_id)
        # train() only for a brand new model; update() for every
        # subsequent enrollment so existing workers' data is never
        # discarded — preserved exactly from the prior terminal-loop
        # implementation (see scripts/enroll_worker.py's docstring).
        if self._model_existed:
            self._recognizer.update(self._samples, np.array(self._labels))
        else:
            self._recognizer.train(self._samples, np.array(self._labels))
        self._recognizer.write(str(self._model_path))
        self._state.finish(True, "Enrollment complete.")
        self._completed_event.set()

    def _finish_failure(self, message: str) -> None:
        logger.warning("Enrollment failed: %s", message)
        self._state.finish(False, message)
        self._completed_event.set()


_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Worker Enrollment</title>
  <style>
    :root {
      --bg: #0f1419; --bg-card: #161d27; --bg-header: #111820;
      --border: #1e2d3d; --cyan: #00b4d8; --green: #22c55e;
      --red: #ef4444; --text: #cbd5e1; --text-dim: #64748b;
      --radius: 8px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      height: 100%; background: var(--bg); color: var(--text);
      font-family: system-ui, "Segoe UI", "Inter", sans-serif;
    }
    body {
      display: flex; flex-direction: column; align-items: center;
      padding: 24px 16px;
    }
    /* Video panel scales with the viewport (not a small fixed pixel
       box) — width fills up to a generous cap, height is viewport-
       relative, and object-fit:contain keeps the 4:3 webcam feed
       un-distorted inside that larger box regardless of window size. */
    #card {
      background: var(--bg-card); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 20px;
      width: min(1200px, 95vw);
    }
    h1 { font-size: 16px; color: var(--cyan); margin-bottom: 12px; }
    #worker-name { color: var(--text-dim); font-size: 13px; margin-bottom: 10px; }
    #video-wrap {
      width: 100%; height: 72vh; min-height: 320px;
      background: #000; border-radius: var(--radius); border: 1px solid var(--border);
      display: flex; align-items: center; justify-content: center; overflow: hidden;
    }
    #video-wrap img {
      width: 100%; height: 100%; object-fit: contain; display: block;
    }
    #banner {
      margin-top: 14px; padding: 12px 16px; border-radius: var(--radius);
      font-size: 15px; display: none;
    }
    #banner.success { display: block; background: #052010; color: var(--green); border: 1px solid #166534; }
    #banner.failure { display: block; background: #1f0505; color: var(--red); border: 1px solid #7f1d1d; }
  </style>
</head>
<body>
  <div id="card">
    <h1>Worker Enrollment</h1>
    <div id="worker-name">Enrolling: {{ worker_name }}</div>
    <div id="video-wrap">
      <img id="video" src="/enroll/feed" alt="enrollment camera stream">
    </div>
    <div id="banner"></div>
  </div>
  <script>
    const banner = document.getElementById("banner");
    const video = document.getElementById("video");
    const workerName = {{ worker_name | tojson }};

    function finish(kind, text) {
      // Stop the MJPEG stream from trying to keep displaying/reconnect —
      // the server has already shut the feed connection down at this
      // point (see enrollment_server.py's completion handling), so
      // leaving the <img> pointed at /enroll/feed would just show a
      // frozen last frame with no indication anything happened.
      video.removeAttribute("src");
      document.getElementById("video-wrap").style.display = "none";
      banner.className = kind;
      banner.textContent = text;
    }

    const poll = setInterval(() => {
      fetch("/enroll/status").then(r => r.json()).then(data => {
        if (data.status === "success") {
          clearInterval(poll);
          finish("success", "Enrollment complete for " + workerName +
                 " — you can close this tab and return to the terminal.");
        } else if (data.status === "failure") {
          clearInterval(poll);
          finish("failure", "Enrollment failed: " + data.message);
        }
      }).catch(() => {});
    }, 1000);
  </script>
</body>
</html>
"""


def create_enrollment_app(session: EnrollmentSession, worker_name: str) -> Flask:
    """Build the self-contained Flask app for one enrollment session.

    Three routes, mirroring dashboard/app.py's page/stream/status split:
      GET /enroll         - the guided HTML page (minimal chrome; the
                             overlay drawn into the stream carries the UX)
      GET /enroll/feed     - the annotated MJPEG stream
      GET /enroll/status   - JSON progress/outcome, polled by the page

    Must be served with threaded=True (see enroll_worker.py's
    make_server() call and this module's "Threaded server" docstring
    section) so /enroll/status can be answered while /enroll/feed's
    long-lived stream is still open.
    """
    app = Flask(__name__)

    @app.route("/enroll")
    def enroll_page():
        return render_template_string(_PAGE_TEMPLATE, worker_name=worker_name)

    @app.route("/enroll/feed")
    def enroll_feed():
        return Response(
            session.frame_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/enroll/status")
    def enroll_status():
        return jsonify(session.status())

    return app
