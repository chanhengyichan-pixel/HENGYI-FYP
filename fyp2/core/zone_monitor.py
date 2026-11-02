"""
ZoneMonitor — time-based hazard / restricted zone rules for the room camera.

Input per frame
---------------
  detections : tracked detections from Detector.track() (persons carry a
               track_id once ByteTrack has confirmed them)
  verdict    : FrameVerdict from DecisionEngine.evaluate() on the same
               detections — its person_results say, per person box,
               whether helmet and vest are correctly worn

Rules (per person track, per zone)
----------------------------------
  hazard     : foot point inside AND (no helmet OR no vest) continuously
               for hazard_seconds  -> HAZARD_PPE event
  restricted : foot point inside continuously for restricted_seconds
               -> RESTRICTED_ENTRY event (highest severity)

"Continuously" tolerates short dropouts: a person may disappear (missed
detection, occlusion, a PPE flicker) for up to grace_seconds without their
timer restarting. Longer than that and the timer starts again from zero.

Flicker filter (hazard)
-----------------------
Far from the camera a worn helmet can drop out of detection for a frame
or two. The grace period bridges those gaps, so on its own a flickering
helmet could add up to a false HAZARD_PPE. hazard_min_missing_ratio
(e.g. 0.7) requires PPE to be missing in at least that share of the
frames since the timer started. 0 (default) disables the filter.

Cooldown
--------
After an event fires, the same track does not raise the same event type
again for cooldown_seconds — someone standing in a zone for a minute gets
one alarm, not one per frame.

Persons without a track_id (tracker not yet confirmed) are ignored: a
timer needs a stable identity to accumulate against. ByteTrack confirms
a new person within a few frames, well inside the 1-2 s thresholds.

Time comes from an injectable clock (default time.monotonic), as in
SessionController, so tests can drive time explicitly.
"""

from __future__ import annotations

import time as _time_module
from dataclasses import dataclass, field
from typing import Callable

from core.zones import HAZARD, RESTRICTED, Zone, foot_point, load_zones

HAZARD_PPE = "HAZARD_PPE"
RESTRICTED_ENTRY = "RESTRICTED_ENTRY"

SEVERITY = {HAZARD_PPE: "high", RESTRICTED_ENTRY: "critical"}


@dataclass
class ZoneEvent:
    """One fired zone violation. Mutable: ScreenshotHandler fills `snapshot`
    before EventLogHandler stores the event."""
    event_type: str
    zone: str
    track_id: int
    violations: list[str] = field(default_factory=list)
    camera: str = "room"
    snapshot: str | None = None

    @property
    def severity(self) -> str:
        return SEVERITY[self.event_type]


@dataclass
class _Timer:
    start: float
    last_seen: float
    bad: int = 0    # frames in this run where the condition held (PPE missing / inside)
    good: int = 0   # hazard only: frames in this run where PPE looked fine


class ZoneMonitor:
    def __init__(self, config: dict, clock: Callable[[], float] = _time_module.monotonic,
                 zones: list[Zone] | None = None):
        zcfg = config["zones"]
        self._hazard_seconds = float(zcfg["hazard_seconds"])
        self._restricted_seconds = float(zcfg["restricted_seconds"])
        self._grace_seconds = float(zcfg["grace_seconds"])
        self._cooldown_seconds = float(zcfg["cooldown_seconds"])
        # Hazard: share of frames in the run that must show PPE missing.
        # Far from the camera a worn helmet can flicker out for a frame or
        # two; without this, those flickers add up to a false alarm.
        self._min_missing_ratio = float(zcfg.get("hazard_min_missing_ratio", 0.0))
        self._zones = zones if zones is not None else load_zones(config)
        self._clock = clock

        # (track_id, zone_name) -> _Timer for the condition currently held
        self._timers: dict[tuple[int, str], _Timer] = {}
        # (track_id, event_type) -> clock time the event last fired
        self._last_fired: dict[tuple[int, str], float] = {}

    @property
    def zones(self) -> list[Zone]:
        return list(self._zones)

    def set_zones(self, zones: list[Zone]) -> None:
        """Swap in new zones (from the dashboard editor) while running.
        Timers belong to the old zones, so they are dropped; cooldowns are
        kept so a person standing still does not get a second alert."""
        self._zones = list(zones)
        self._timers = {}

    def update(self, detections, verdict) -> tuple[list[ZoneEvent], list[dict]]:
        """
        Advance all zone timers by one frame.

        Returns (events, status):
          events — ZoneEvents that fired on THIS frame (usually empty)
          status — one dict per tracked person currently inside a zone:
                   {"track_id", "zone", "zone_type", "seconds", "ppe_ok",
                    "violations"} — for the dashboard overlay
        """
        now = self._clock()
        ppe_by_box = self._ppe_by_person_box(verdict)
        zones = self._zones  # one consistent set for this frame (editor may swap it)

        events: list[ZoneEvent] = []
        status: list[dict] = []

        for det in detections:
            if det.class_name != "person" or det.track_id is None:
                continue
            violations = ppe_by_box.get(_box_key(det), ["no-helmet", "no-vest"])
            ppe_ok = not violations
            point = foot_point(det)

            for zone in zones:
                if not zone.contains(point):
                    continue

                if zone.type == HAZARD and ppe_ok:
                    # Inside a hazard zone but correctly equipped: fine. A
                    # running "missing PPE" timer counts this frame against it.
                    t = self._timers.get((det.track_id, zone.name))
                    if t is not None and now - t.last_seen <= self._grace_seconds:
                        t.good += 1
                    status.append(self._status(det, zone, 0.0, ppe_ok, violations))
                    continue

                key = (det.track_id, zone.name)
                timer = self._timers.get(key)
                if timer is None or now - timer.last_seen > self._grace_seconds:
                    timer = _Timer(start=now, last_seen=now)
                    self._timers[key] = timer
                timer.last_seen = now
                timer.bad += 1
                held = now - timer.start
                status.append(self._status(det, zone, held, ppe_ok, violations))

                if zone.type == HAZARD:
                    event_type, needed = HAZARD_PPE, self._hazard_seconds
                else:
                    event_type, needed = RESTRICTED_ENTRY, self._restricted_seconds

                ratio_ok = (zone.type != HAZARD
                            or timer.bad / (timer.bad + timer.good) >= self._min_missing_ratio)
                if held >= needed and ratio_ok and self._cooldown_over(det.track_id, event_type, now):
                    self._last_fired[(det.track_id, event_type)] = now
                    events.append(ZoneEvent(
                        event_type=event_type,
                        zone=zone.name,
                        track_id=det.track_id,
                        violations=list(violations) if zone.type == HAZARD else [],
                    ))

        self._prune(now)
        return events, status

    def _cooldown_over(self, track_id, event_type, now) -> bool:
        last = self._last_fired.get((track_id, event_type))
        return last is None or now - last >= self._cooldown_seconds

    def _prune(self, now) -> None:
        """Forget timers past their grace period and expired cooldowns, so
        memory does not grow with every person who ever walked past."""
        self._timers = {k: t for k, t in self._timers.items()
                        if now - t.last_seen <= self._grace_seconds}
        self._last_fired = {k: t for k, t in self._last_fired.items()
                            if now - t < self._cooldown_seconds}

    @staticmethod
    def _ppe_by_person_box(verdict) -> dict:
        """Map person box -> violations list, from DecisionEngine person_results."""
        out = {}
        for r in getattr(verdict, "person_results", []) or []:
            box = r.get("box")
            if box is not None:
                out[tuple(box)] = list(r["violations"])
        return out

    @staticmethod
    def _status(det, zone, seconds, ppe_ok, violations) -> dict:
        return {
            "track_id": det.track_id,
            "zone": zone.name,
            "zone_type": zone.type,
            "seconds": round(seconds, 1),
            "ppe_ok": ppe_ok,
            "violations": list(violations),
        }


def _box_key(det) -> tuple:
    return (det.x1, det.y1, det.x2, det.y2)
