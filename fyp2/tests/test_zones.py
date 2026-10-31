"""Unit tests for core/zones.py — pure geometry, no OpenCV."""

from types import SimpleNamespace

import pytest

from core.zones import Zone, foot_point, load_zones, point_in_polygon

SQUARE = ((0, 0), (10, 0), (10, 10), (0, 10))


class TestPointInPolygon:
    def test_inside(self):
        assert point_in_polygon((5, 5), SQUARE)

    def test_outside(self):
        assert not point_in_polygon((15, 5), SQUARE)
        assert not point_in_polygon((5, -1), SQUARE)

    def test_concave_polygon_notch_is_outside(self):
        # U shape: the notch between the arms is outside.
        u = ((0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10))
        assert point_in_polygon((1, 8), u)
        assert not point_in_polygon((5, 8), u)


class TestFootPoint:
    def test_bottom_centre_of_box(self):
        det = SimpleNamespace(x1=100, y1=50, x2=200, y2=400)
        assert foot_point(det) == (150.0, 400.0)

    def test_upper_body_over_zone_but_feet_outside_is_not_inside(self):
        zone = Zone("floor", "hazard", ((0, 300), (640, 300), (640, 480), (0, 480)))
        # Box overlaps the zone vertically only above y=300? No — feet at y=290.
        det = SimpleNamespace(x1=100, y1=0, x2=200, y2=290)
        assert not zone.contains(foot_point(det))


class TestLoadZones:
    def test_loads_config_areas(self):
        cfg = {"zones": {"areas": [
            {"name": "A", "type": "hazard", "polygon": [[0, 0], [1, 0], [1, 1]]},
            {"name": "B", "type": "restricted", "polygon": [[0, 0], [2, 0], [2, 2]]},
        ]}}
        zones = load_zones(cfg)
        assert [z.name for z in zones] == ["A", "B"]
        assert zones[1].type == "restricted"

    def test_bad_type_rejected(self):
        cfg = {"zones": {"areas": [{"name": "A", "type": "danger",
                                    "polygon": [[0, 0], [1, 0], [1, 1]]}]}}
        with pytest.raises(ValueError):
            load_zones(cfg)

    def test_too_few_points_rejected(self):
        cfg = {"zones": {"areas": [{"name": "A", "type": "hazard",
                                    "polygon": [[0, 0], [1, 0]]}]}}
        with pytest.raises(ValueError):
            load_zones(cfg)
