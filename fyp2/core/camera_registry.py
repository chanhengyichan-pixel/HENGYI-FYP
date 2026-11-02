"""
Camera list: which physical camera does which job.

A fresh Raspberry Pi cannot know which camera is at the entrance and which
watches the room, so the admin sets this once in the dashboard
(Settings > Cameras). The list is saved to output/cameras.json and takes
precedence over the `cameras:` section of config.yaml. Changes apply on
the next start of the monitoring program (the dashboard has a Restart
button), so the running detection threads are never rewired live.

Each camera:
    id       stable key used in events and zone files: entrance cameras are
             "entrance", "entrance2", ...; room cameras "room", "room2", ...
             (never renamed, so old events keep pointing at the right camera)
    role     "entrance" (PPE check, ALLOW/DENY) or "room" (zones)
    name     what people see on the dashboard, e.g. "Workshop A"
    enabled  false = configured but not started
    backend  "opencv" (USB webcam / video file) or "picamera2" (Pi CSI)
    source   device index, video path, or CSI camera number

There is no fixed number of entrance or room cameras: how many can run at
once depends on the machine (every camera adds one YOLO pass). The
CameraManager (core/camera_manager.py) refuses to start another camera
when memory runs low or camera_capacity.max_active is reached, and the
Cameras page warns when the frame rate drops. MAX_CAMERAS is only a
sanity cap on the saved list.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

ENTRANCE = "entrance"
ROOM = "room"
MAX_CAMERAS = 12          # sanity cap on the saved list, not a hardware limit
BACKENDS = ("opencv", "picamera2")
_DEFAULTS = {"width": 640, "height": 480, "max_failed_reads": 10, "loop_video": False}
_ID_RE = re.compile(r"^(entrance|room)(\d*)$")


def role_of(cid: str) -> str:
    return ENTRANCE if str(cid).startswith(ENTRANCE) else ROOM


def default_name(cid: str) -> str:
    m = _ID_RE.match(cid)
    base = "Entrance" if m.group(1) == ENTRANCE else "Room"
    return f"{base} {m.group(2)}" if m.group(2) else base


def _sort_key(c):
    m = _ID_RE.match(c["id"])
    return (m.group(1) != ENTRANCE, int(m.group(2) or 1))


def describe_source(backend: str, source) -> dict:
    """What kind of source this is, in words: {"kind": usb|csi|file, "label"}."""
    if backend == "picamera2":
        return {"kind": "csi", "label": f"Raspberry Pi Camera – CSI {source}"}
    if isinstance(source, int) or (isinstance(source, str) and source.strip().lstrip("-").isdigit()):
        return {"kind": "usb", "label": f"USB Camera – /dev/video{int(source)}"}
    if source in (None, ""):
        return {"kind": "none", "label": "Not chosen"}
    return {"kind": "file", "label": f"Video File – {source}"}


def cameras_file(config: dict) -> str:
    return config.get("camera_setup_file", "output/cameras.json")


def _normalise(cam: dict, config_defaults: dict) -> dict:
    out = {**_DEFAULTS, **config_defaults, **cam}
    out["role"] = role_of(out["id"])
    out.setdefault("name", default_name(out["id"]))
    out["enabled"] = bool(out.get("enabled", True))
    return out


def _from_config(config: dict) -> list[dict]:
    """Camera list from config.yaml (cameras.entrance, cameras.room, room2...)."""
    out = []
    for cid, cfg in (config.get("cameras") or {}).items():
        if not _ID_RE.match(cid):
            continue
        name = cfg.get("name") or default_name(cid)
        out.append(_normalise({"id": cid, **cfg, "name": name}, {}))
    out.sort(key=_sort_key)
    return out


def load_cameras(config: dict) -> tuple[list[dict], str]:
    """(camera list, where it came from). cameras.json wins over config.yaml.
    Size/read settings missing from the file are taken from config.yaml."""
    path = cameras_file(config)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)["cameras"]
        base = {c["id"]: c for c in _from_config(config)}
        cams = []
        for c in validate_cameras(saved):
            b = base.get(c["id"]) or base.get(ROOM if c["role"] == ROOM else ENTRANCE) or {}
            defaults = {k: b[k] for k in ("width", "height", "max_failed_reads", "loop_video") if k in b}
            cams.append(_normalise(c, defaults))
        return cams, path
    return _from_config(config), "config.yaml"


def validate_cameras(cams) -> list[dict]:
    """Check a camera list from the Cameras page. Raises ValueError."""
    if not isinstance(cams, list):
        raise ValueError("Bad camera list")
    if len(cams) > MAX_CAMERAS:
        raise ValueError(f"At most {MAX_CAMERAS} cameras in one setup")
    seen_ids, seen_src, out = set(), set(), []
    for i, c in enumerate(cams, 1):
        if not isinstance(c, dict):
            raise ValueError(f"Camera {i}: bad entry")
        cid = str(c.get("id", ""))
        if not _ID_RE.match(cid):
            raise ValueError(f"Camera {i}: bad id '{cid}'")
        if cid in seen_ids:
            raise ValueError(f"Camera id '{cid}' is used twice")
        seen_ids.add(cid)
        name = str(c.get("name", "")).strip()
        if not name or len(name) > 40:
            raise ValueError(f"Camera {i}: name must be 1-40 characters")
        backend = c.get("backend", "opencv")
        if backend not in BACKENDS:
            raise ValueError(f"{name}: backend must be opencv or picamera2")
        src = c.get("source")
        if isinstance(src, str) and src.strip().lstrip("-").isdigit():
            src = int(src.strip())
        if src is None or src == "":
            raise ValueError(f"{name}: choose a camera source")
        enabled = bool(c.get("enabled", True))
        if enabled:
            key = (backend, str(src))
            if key in seen_src:
                other = next(o["name"] for o in out if o["enabled"] and (o["backend"], str(o["source"])) == key)
                raise ValueError(f"{name}: this camera is already used by {other}")
            seen_src.add(key)
        entry = {"id": cid, "role": role_of(cid), "name": name,
                 "enabled": enabled, "backend": backend, "source": src}
        for k in ("width", "height"):
            if k in c:
                entry[k] = int(c[k])
        out.append(entry)
    out.sort(key=_sort_key)
    return out


def next_id(cams, role: str = ROOM) -> str:
    """First free id for a new camera of this role: room, room2, room3..."""
    ids = {c["id"] for c in cams}
    if role not in ids:
        return role
    n = 2
    while f"{role}{n}" in ids:
        n += 1
    return f"{role}{n}"


def next_room_id(cams) -> str:
    return next_id(cams, ROOM)


def save_cameras(config: dict, cams) -> str:
    path = cameras_file(config)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"cameras": validate_cameras(cams)}, f, indent=2)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Finding cameras on this machine
# ---------------------------------------------------------------------------

def detect_cameras() -> list[dict]:
    """Cameras this machine can see: [{backend, source, label}].
    CSI cameras via Picamera2 (Pi only), USB cameras via /dev/video*."""
    found = []
    try:
        from picamera2 import Picamera2  # Pi only
        for i, info in enumerate(Picamera2.global_camera_info()):
            model = info.get("Model", "camera")
            found.append({"backend": "picamera2", "source": i, "kind": "csi",
                          "label": f"Raspberry Pi Camera – CSI {i} ({model})"})
    except Exception:  # not a Pi, or libcamera missing
        pass
    for dev in sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        idx = int(re.sub(r"\D", "", dev) or 0)
        name = ""
        try:
            with open(f"/sys/class/video4linux/video{idx}/name", encoding="utf-8") as f:
                name = f.read().strip()
        except OSError:
            pass
        # Pi CSI cameras and codecs also show up as /dev/video*; skip those.
        if any(s in name.lower() for s in ("unicam", "bcm2835", "rp1-cfe", "pispbe", "codec", "isp")):
            continue
        found.append({"backend": "opencv", "source": idx, "kind": "usb",
                      "label": f"USB Camera – /dev/video{idx}" + (f" ({name})" if name else "")})
    return found


def grab_frame(backend: str, source, width: int = 640, height: int = 480):
    """One frame from a camera that is NOT in use, for a preview. None on failure."""
    try:
        if backend == "opencv":
            import cv2
            cap = cv2.VideoCapture(source)
            try:
                if not cap.isOpened():
                    return None
                frame = None
                for _ in range(5):  # first frames are often dark
                    ok, f = cap.read()
                    if ok:
                        frame = f
                return frame
            finally:
                cap.release()
        if backend == "picamera2":
            from picamera2 import Picamera2
            cam = Picamera2(int(source))
            try:
                cam.configure(cam.create_preview_configuration(
                    main={"size": (width, height), "format": "RGB888"}))
                cam.start()
                return cam.capture_array()
            finally:
                cam.stop()
                cam.close()
    except Exception as e:  # busy, unplugged, not a Pi...
        logger.info("Preview of %s %s failed: %s", backend, source, e)
    return None
