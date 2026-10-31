"""CameraCapture with the opencv backend reading a small generated video file."""

import time

import cv2
import numpy as np

from core.camera import CameraCapture
from core.shared_state import SharedState


def _write_video(path, frames=5, fps=50):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    writer.release()


def _cfg(path, loop):
    return {"backend": "opencv", "source": str(path), "width": 64, "height": 48,
            "max_failed_reads": 3, "loop_video": loop}


def test_video_file_loops_instead_of_disconnecting(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video)
    state = SharedState()
    cam = CameraCapture(_cfg(video, loop=True), state, name="room")
    cam.start()
    time.sleep(0.4)                      # 5 frames at 50 FPS = 0.1 s, so it looped
    _, frame_id = state.get_frame_with_id()
    status = state.get_camera_status()
    cam.stop()
    assert status == "running"
    assert frame_id > 5


def test_video_file_without_loop_ends_as_disconnected(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video)
    state = SharedState()
    cam = CameraCapture(_cfg(video, loop=False), state, name="room")
    cam.start()
    time.sleep(0.5)
    status = state.get_camera_status()
    cam.stop()
    assert status == "disconnected"
