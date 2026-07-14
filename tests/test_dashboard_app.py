"""
Unit tests for dashboard/app.py's _draw_debug_overlay().

No Flask app construction, no camera — a plain numpy frame array and a
debug_info dict/None shaped like FrameVerdict.debug_info
(core/decision_engine.py). Confirms the overlay is a genuine no-op (no
drawing) when debug_info is None, and that a well-formed debug_info does
not raise.
"""

from unittest.mock import patch

import numpy as np

from dashboard.app import _draw_debug_overlay


def _fake_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _sample_debug_info():
    return {
        "persons": [
            {
                "person_index": 0,
                "box": (100, 100, 300, 500),
                "confidence": 0.9,
                "head_region": (100.0, 200.0, 145.0, 255.0),
                "torso_region": (200.0, 400.0, 120.0, 280.0),
                "helmet": {
                    "assigned": "helmet",
                    "candidates": [
                        {
                            "class_name": "helmet",
                            "box": (150, 100, 250, 190),
                            "centroid": (200.0, 145.0),
                            "containment": 1.0,
                            "containment_ok": True,
                            "x_ok": True,
                            "y_ok": True,
                            "region_ok": True,
                            "claimed_by": None,
                            "status": "assigned",
                            "reasons": [],
                        },
                    ],
                },
                "vest": {
                    "assigned": None,
                    "candidates": [
                        {
                            "class_name": "vest",
                            "box": (130, 230, 270, 370),
                            "centroid": (200.0, 300.0),
                            "containment": 0.4,
                            "containment_ok": False,
                            "x_ok": True,
                            "y_ok": True,
                            "region_ok": True,
                            "claimed_by": None,
                            "status": "rejected",
                            "reasons": ["containment below threshold (0.40 < 0.60)"],
                        },
                    ],
                },
                "person_status": "NON_COMPLIANT",
            }
        ],
        "frame": {
            "person_count": 1,
            "helmets_assigned": 1,
            "vests_assigned": 0,
            "status": "non_compliant",
            "violations": ["no-vest"],
        },
    }


class TestDebugOverlayNoOp:
    def test_none_debug_info_draws_nothing(self):
        frame = _fake_frame()
        with patch("dashboard.app.cv2") as mock_cv2:
            _draw_debug_overlay(frame, None)
        mock_cv2.rectangle.assert_not_called()
        mock_cv2.putText.assert_not_called()
        mock_cv2.drawMarker.assert_not_called()


class TestDebugOverlayDrawing:
    def test_valid_debug_info_does_not_raise(self):
        frame = _fake_frame()
        _draw_debug_overlay(frame, _sample_debug_info())  # must not raise

    def test_valid_debug_info_draws_person_box_and_regions(self):
        frame = _fake_frame()
        with patch("dashboard.app.cv2") as mock_cv2:
            _draw_debug_overlay(frame, _sample_debug_info())
        # person box + head region + torso region = 3 rectangle calls
        assert mock_cv2.rectangle.call_count == 3
        # one marker per PPE candidate (1 helmet + 1 vest)
        assert mock_cv2.drawMarker.call_count == 2
