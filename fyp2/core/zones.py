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


# ---------------------------------------------------------------------------
# Zones drawn on the dashboard (Settings > Room zones) are saved to a JSON
# file, which then takes precedence over config.yaml's zones.areas. This
# lets an admin redraw zones for a new camera position without editing
# config files or restarting.
# ---------------------------------------------------------------------------

MAX_ZONES = 10
MAX_POINTS = 20


def zones_file(config: dict) -> str:
    return config.get("zones", {}).get("file", "output/zones.json")


def zones_from_areas(areas) -> list[Zone]:
    """Validate a list of {"name", "type", "polygon"} dicts (from the web
    editor or a saved file). Raises ValueError with a readable message."""
    if not isinstance(areas, list):
        raise ValueError("Zones must be a list")
    if len(areas) > MAX_ZONES:
        raise ValueError(f"At most {MAX_ZONES} zones")
    zones, names = [], set()
    for i, area in enumerate(areas, 1):
        try:
            name = str(area["name"]).strip()
            ztype = area["type"]
            pts = area["polygon"]
        except (KeyError, TypeError):
            raise ValueError(f"Zone {i}: needs name, type and polygon") from None
        if not name or len(name) > 40:
            raise ValueError(f"Zone {i}: name must be 1-40 characters")
        if name.lower() in names:
            raise ValueError(f"Zone name '{name}' is used twice")
        names.add(name.lower())
        if ztype not in _ZONE_TYPES:
            raise ValueError(f"Zone '{name}': type must be hazard or restricted")
        try:
            polygon = tuple((round(float(x), 1), round(float(y), 1)) for x, y in pts)
        except (TypeError, ValueError):
            raise ValueError(f"Zone '{name}': bad polygon points") from None
        if not 3 <= len(polygon) <= MAX_POINTS:
            raise ValueError(f"Zone '{name}': needs 3-{MAX_POINTS} points")
        zones.append(Zone(name=name, type=ztype, polygon=polygon))
    return zones


def zones_to_areas(zones) -> list[dict]:
    return [{"name": z.name, "type": z.type, "polygon": [list(p) for p in z.polygon]}
            for z in zones]


def load_active_zones(config: dict) -> tuple[list[Zone], str]:
    """Zones the system should use, and where they came from:
    the saved web-editor file if it exists, else config.yaml."""
    import json
    import os
    path = zones_file(config)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return zones_from_areas(json.load(f)["zones"]), path
    return load_zones(config), "config.yaml"


def save_zones(config: dict, zones) -> str:
    """Write zones to the JSON file atomically; return the path."""
    import json
    import os
    path = zones_file(config)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"zones": zones_to_areas(zones)}, f, indent=2)
    os.replace(tmp, path)
    return path
