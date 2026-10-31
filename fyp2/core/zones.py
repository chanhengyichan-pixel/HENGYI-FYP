"""
Zone geometry for the room camera.

A Zone is a named polygon in room-camera pixel coordinates with a type:
  hazard     — PPE required inside
  restricted — nobody allowed inside

A person is considered inside a zone when their FOOT POINT — the
bottom-centre of their bounding box — is inside the polygon. The foot
point is used (not the box centre) because the camera looks down at an
angle: a person standing just outside a floor zone can have their
upper body drawn over it, but their feet are where they actually stand.

Pure Python, no cv2 — unit-testable without OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass

HAZARD = "hazard"
RESTRICTED = "restricted"
_ZONE_TYPES = (HAZARD, RESTRICTED)


@dataclass(frozen=True)
class Zone:
    name: str
    type: str
    polygon: tuple[tuple[float, float], ...]

    def contains(self, point: tuple[float, float]) -> bool:
        return point_in_polygon(point, self.polygon)


def foot_point(det) -> tuple[float, float]:
    """Bottom-centre of a detection box (x1, y1, x2, y2 attributes)."""
    return ((det.x1 + det.x2) / 2.0, float(det.y2))


def point_in_polygon(point, polygon) -> bool:
    """Ray-casting test. Points exactly on an edge may go either way."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def load_zones(config: dict) -> list[Zone]:
    """Build Zone objects from config["zones"]["areas"]. Fails fast on bad input."""
    zones = []
    for area in config["zones"]["areas"]:
        ztype = area["type"]
        if ztype not in _ZONE_TYPES:
            raise ValueError(f"Zone '{area['name']}': type must be one of "
                             f"{_ZONE_TYPES}, got '{ztype}'")
        polygon = tuple((float(x), float(y)) for x, y in area["polygon"])
        if len(polygon) < 3:
            raise ValueError(f"Zone '{area['name']}': polygon needs at least 3 points")
        zones.append(Zone(name=str(area["name"]), type=ztype, polygon=polygon))
    return zones
