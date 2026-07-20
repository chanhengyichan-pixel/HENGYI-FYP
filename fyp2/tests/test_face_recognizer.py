"""
Unit tests for core/face_recognizer.py.

All cv2 dependencies are stubbed via DI (_recognizer and _cascade
constructor params).  No real webcam, no real model file, no LBPH
training required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from core.face_recognizer import FaceRecognizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "face": {
        "model_path": "/nonexistent/face_model.yml",  # does not exist — no-model path
        "match_threshold": 75.0,
    }
}

_FAKE_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
_FAKE_ROI = np.zeros((60, 60), dtype=np.uint8)


def _cascade_detecting(x=10, y=10, w=80, h=80):
    """Stub cascade that 'detects' one face."""
    stub = MagicMock()
    stub.detectMultiScale.return_value = np.array([[x, y, w, h]])
    return stub


def _cascade_no_face():
    """Stub cascade that detects nothing."""
    stub = MagicMock()
    stub.detectMultiScale.return_value = ()
    return stub


def _recognizer_stub(label_id: int, distance: float):
    """Stub LBPH recognizer with fixed predict() output."""
    stub = MagicMock()
    stub.predict.return_value = (label_id, distance)
    return stub


def _worker_store_stub(name: str | None):
    """Stub WorkerStore.get_name()."""
    stub = MagicMock()
    stub.get_name.return_value = name
    return stub


def make_recognizer(
    config=None,
    recognizer_stub=None,
    cascade_stub=None,
    worker_store=None,
):
    cfg = config or BASE_CONFIG
    ws = worker_store or _worker_store_stub("Alice")
    rec = recognizer_stub or _recognizer_stub(0, 50.0)
    cas = cascade_stub or _cascade_detecting()
    return FaceRecognizer(cfg, ws, _recognizer=rec, _cascade=cas)


# ---------------------------------------------------------------------------
# No-model path (graceful degradation)
# ---------------------------------------------------------------------------

class TestNoModelPath:
    def test_identify_returns_none_when_no_model(self, tmp_path):
        """identify() must return None when model file doesn't exist — no crash."""
        fr = make_recognizer()
        # _trained is False because model_path doesn't exist
        result = fr.identify(_FAKE_FRAME)
        assert result is None

    def test_detect_face_still_works_without_model(self):
        """detect_face() doesn't need the LBPH model — only the cascade."""
        fr = make_recognizer()
        roi = fr.detect_face(_FAKE_FRAME)
        # cascade stub says face is present, so ROI should be non-None
        assert roi is not None


# ---------------------------------------------------------------------------
# detect_face
# ---------------------------------------------------------------------------

class TestDetectFace:
    def test_returns_none_when_no_face(self):
        fr = make_recognizer(cascade_stub=_cascade_no_face())
        assert fr.detect_face(_FAKE_FRAME) is None

    def test_returns_roi_when_face_detected(self):
        fr = make_recognizer(cascade_stub=_cascade_detecting(x=0, y=0, w=60, h=60))
        roi = fr.detect_face(_FAKE_FRAME)
        assert roi is not None
        assert roi.ndim == 2  # grayscale

    def test_picks_largest_face_when_multiple(self):
        """If two faces detected, the larger one's ROI should be returned."""
        stub = MagicMock()
        # small face: 30×30=900, large face: 100×100=10000
        stub.detectMultiScale.return_value = np.array([[5, 5, 30, 30], [0, 0, 100, 100]])
        fr = make_recognizer(cascade_stub=stub)
        roi = fr.detect_face(_FAKE_FRAME)
        assert roi is not None
        assert roi.shape == (100, 100)


# ---------------------------------------------------------------------------
# identify — threshold logic
# ---------------------------------------------------------------------------

class TestIdentifyThreshold:
    def _make_trained(self, label_id, distance, worker_name="Alice"):
        """Return a FaceRecognizer whose _trained flag is forced True."""
        rec = _recognizer_stub(label_id, distance)
        ws = _worker_store_stub(worker_name)
        cas = _cascade_detecting()
        fr = FaceRecognizer(BASE_CONFIG, ws, _recognizer=rec, _cascade=cas)
        fr._trained = True  # force — model_path doesn't exist in test env
        return fr

    def test_returns_name_when_distance_below_threshold(self):
        fr = self._make_trained(0, 50.0)  # 50 < 75 → match
        result = fr.identify(_FAKE_FRAME)
        assert result is not None
        name, dist = result
        assert name == "Alice"
        assert dist == pytest.approx(50.0)

    def test_returns_none_when_distance_above_threshold(self):
        fr = self._make_trained(0, 90.0)  # 90 > 75 → no match
        assert fr.identify(_FAKE_FRAME) is None

    def test_returns_match_when_distance_equals_threshold(self):
        # Code uses `distance > threshold`, so equal means distance is NOT exceeded → match.
        fr = self._make_trained(0, 75.0)
        result = fr.identify(_FAKE_FRAME)
        assert result is not None
        assert result[0] == "Alice"

    def test_returns_none_when_no_face_detected(self):
        rec = _recognizer_stub(0, 50.0)
        ws = _worker_store_stub("Alice")
        fr = FaceRecognizer(BASE_CONFIG, ws, _recognizer=rec, _cascade=_cascade_no_face())
        fr._trained = True
        assert fr.identify(_FAKE_FRAME) is None

    def test_returns_none_when_label_id_not_in_worker_store(self):
        """label_id from recognizer has no matching worker — log warning, return None."""
        rec = _recognizer_stub(99, 30.0)
        ws = _worker_store_stub(None)  # get_name returns None
        cas = _cascade_detecting()
        fr = FaceRecognizer(BASE_CONFIG, ws, _recognizer=rec, _cascade=cas)
        fr._trained = True
        assert fr.identify(_FAKE_FRAME) is None

    def test_custom_threshold_respected(self):
        config = {"face": {"model_path": "/nonexistent/m.yml", "match_threshold": 40.0}}
        rec = _recognizer_stub(0, 45.0)  # 45 > 40 → no match
        ws = _worker_store_stub("Alice")
        cas = _cascade_detecting()
        fr = FaceRecognizer(config, ws, _recognizer=rec, _cascade=cas)
        fr._trained = True
        assert fr.identify(_FAKE_FRAME) is None
