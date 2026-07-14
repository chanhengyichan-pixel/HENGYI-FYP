"""
Single-frame PPE compliance decision engine — per-person, geometry-aware.

Input
-----
A list of detection objects, each exposing (via duck typing —
core.detector.Detection or any object with the same attributes,
e.g. a SimpleNamespace stub in tests):
  .class_name : str
  .confidence : float
  .x1, .y1, .x2, .y2 : box coordinates (pixels), x1<x2, y1<y2

Does NOT import detector.py or ultralytics — no camera/model dependency.

Model schema (exp01_roboflow)
-----------------------------
  0: helmet    — PPE present (positive signal)
  1: no-helmet — PPE absent  (negative / violation signal)
  2: no-vest   — PPE absent  (negative / violation signal)
  3: person
  4: vest      — PPE present (positive signal)

Why replace the count-based rule
---------------------------------
The previous version only compared counts (helmet_count >= person_count)
and never looked at geometry. That has two concrete failure modes with
multiple people in frame:
  1. A helmet detected anywhere in the frame — even lying on a table, or
     worn by a person who isn't the one it gets counted against — still
     satisfied the count check. The count rule cannot tell "whose PPE is
     whose".
  2. With N people and M<N helmets, the rule correctly flags a shortfall,
     but cannot say WHICH person is missing PPE, and worse, an
     over-threshold set of detections (e.g. two helmets, only one person
     wearing either, one held in-hand) could pass by coincidence of count
     without either person actually being compliant.
This module fixes both by spatially attributing each PPE detection to
(at most) one specific person before any pass/fail judgment is made.

Algorithm — claim-and-remove PPE-to-person assignment
------------------------------------------------------
1. Partition detections into persons and ppe_items (everything else).
   Empty persons -> NO_PERSON immediately (unchanged from prior behavior).
2. Sort persons by detection confidence, descending. This is the
   deterministic tie-break for step 3/4: when a PPE box is a valid
   candidate for more than one person's region (e.g. two people standing
   close together), the higher-confidence person detection gets first
   claim. Confidence is used (rather than, say, box size or position)
   because it is the detector's own signal of which person-box is more
   trustworthy, and it's already available on every detection with no
   extra computation.
3. For each (person, ppe_item) pair, a PPE item is a valid candidate for
   a person only if BOTH:
     a. containment ratio (fraction of the PPE box's area inside the
        person's box) >= containment_threshold. Containment, not IoU, is
        used deliberately: PPE boxes (helmet, vest) are much smaller than
        a person box, so an IoU score is dominated by the size mismatch
        and would stay low even for a perfectly-worn item. Containment
        directly answers the "is this part physically within this whole"
        question that matters here.
     b. the PPE item's box CENTROID falls within the expected body region
        for its class (head region for helmet/no-helmet, torso region for
        vest/no-vest), both vertically and horizontally, using ratios of
        the person's own box. The CENTROID (not full-box containment) is
        tested for region validity because a correctly-worn item's box can
        straddle a region boundary by a few pixels due to ordinary
        detection noise (e.g. a helmet box extending slightly below the
        nominal head/torso boundary) — requiring the whole box to sit
        inside the region would produce false negatives on well-worn PPE.
        The centroid is a robust single point that reflects the item's
        actual physical location regardless of minor box-size noise.
4. Per person (sorted order from step 2), claim-and-remove:
     a. Among items still unclaimed with class_name in
        {"helmet", "no-helmet"}, pick the one with the highest containment
        score that is also region-valid for this person (if any) and
        remove it from the pool — no other person can claim it afterward.
        This prevents one PPE box being double-counted for two different
        people, which the previous count-based rule was vulnerable to.
     b. Repeat independently for {"vest", "no-vest"}.
     c. Outcome per person:
          assigned "helmet"    -> has_helmet = True
          assigned "no-helmet" -> explicit violation
          nothing assigned     -> ambiguous -> fail-safe treats as violation
                                   (same philosophy as the previous
                                   single-person no-signal rule: ambiguity
                                   never grants access)
        Same logic, independently, for vest.
5. Aggregation: NON_COMPLIANT if ANY person fails either check, COMPLIANT
   only if every person passes both. `violations` is the deduplicated
   union of violation strings ("no-helmet", "no-vest") across all
   non-compliant people — the SAME flat list[str] shape as before, so
   EventLogHandler's JSON storage, the dashboard, and get_stats()'s
   json_each() aggregation require zero changes downstream.

Additive field: person_results
-------------------------------
FrameVerdict gains `person_results: list[dict]`, defaulting to an empty
list. Each entry is
  {"person_index": int, "helmet_ok": bool, "vest_ok": bool,
   "violations": list[str]}
in the same order as the confidence-sorted persons list. This exists for
future per-person dashboard/dissertation presentation; nothing existing
reads it and SessionController only ever reads `.status`, so this is a
purely additive change to the public contract.

Configuration
-------------
DecisionEngine now requires the full project config dict (breaking
change to the constructor — it previously took no arguments). Expected
shape, under config["decision_engine"]:

  decision_engine:
    containment_threshold: 0.60
    head:
      top_ratio: 0.25
      width_ratio: 0.55
    torso:
      top_ratio: 0.25
      bottom_ratio: 0.75
      width_ratio: 0.80

Fails fast (KeyError) on any missing key, consistent with every other
config-driven module in this project (Detector, FaceRecognizer,
SessionController).

CALIBRATION NOTE: the head/torso ratios above are reasonable starting
points based on typical human body proportions, not measurements from
real camera footage. Like face.match_threshold (see
core/face_recognizer.py), these values require calibration once real
deployment footage is available — camera angle, mounting height, and
lens distortion all affect where PPE actually appears relative to a
person's bounding box.

No-person rule
--------------
Returns NO_PERSON — not COMPLIANT. The session controller decides
whether to stay IDLE, extend a timeout, etc. The engine does not make
that policy call.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ComplianceStatus(enum.Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NO_PERSON = "no_person"


@dataclass
class FrameVerdict:
    """Result returned by DecisionEngine.evaluate()."""
    status: ComplianceStatus
    violations: list[str] = field(default_factory=list)
    person_count: int = 0
    person_results: list[dict] = field(default_factory=list)

    def is_compliant(self) -> bool:
        return self.status is ComplianceStatus.COMPLIANT


_PERSON_CLASS = "person"
_HELMET_CLASSES = ("helmet", "no-helmet")
_VEST_CLASSES = ("vest", "no-vest")


def _box_area(d) -> float:
    return max(0.0, d.x2 - d.x1) * max(0.0, d.y2 - d.y1)


def _containment_ratio(item, person) -> float:
    """Fraction of `item`'s box area that falls inside `person`'s box.

    Deliberately containment, not IoU — see module docstring. Returns 0.0
    for a zero-area item box rather than raising, since a degenerate box
    should simply fail the threshold, not crash evaluation.
    """
    item_area = _box_area(item)
    if item_area == 0.0:
        return 0.0

    ix1 = max(item.x1, person.x1)
    iy1 = max(item.y1, person.y1)
    ix2 = min(item.x2, person.x2)
    iy2 = min(item.y2, person.y2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    return inter_area / item_area


def _centroid(d) -> tuple[float, float]:
    return ((d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0)


class DecisionEngine:
    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : full project config dict; reads config["decision_engine"].

        Fails fast (KeyError) on any missing key — same convention as
        Detector, FaceRecognizer, and SessionController.
        """
        de_cfg = config["decision_engine"]
        self._containment_threshold = float(de_cfg["containment_threshold"])

        head_cfg = de_cfg["head"]
        self._head_top_ratio = float(head_cfg["top_ratio"])
        self._head_width_ratio = float(head_cfg["width_ratio"])

        torso_cfg = de_cfg["torso"]
        self._torso_top_ratio = float(torso_cfg["top_ratio"])
        self._torso_bottom_ratio = float(torso_cfg["bottom_ratio"])
        self._torso_width_ratio = float(torso_cfg["width_ratio"])

    def evaluate(self, detections) -> FrameVerdict:
        """
        Evaluate a single frame's detections and return a FrameVerdict.

        Parameters
        ----------
        detections : iterable of objects with .class_name: str,
                     .confidence: float, .x1/.y1/.x2/.y2 box coords.

        Returns
        -------
        FrameVerdict
        """
        detections = list(detections)
        persons = [d for d in detections if d.class_name == _PERSON_CLASS]
        ppe_pool = [d for d in detections if d.class_name != _PERSON_CLASS]

        if not persons:
            return FrameVerdict(status=ComplianceStatus.NO_PERSON)

        # Deterministic tie-break for claim-and-remove: higher-confidence
        # person detections resolve PPE ownership ambiguity first.
        persons = sorted(persons, key=lambda p: p.confidence, reverse=True)

        person_results = []
        all_violations: list[str] = []

        for idx, person in enumerate(persons):
            helmet_label, ppe_pool = self._claim(
                person, ppe_pool, _HELMET_CLASSES, self._is_head_region_valid
            )
            vest_label, ppe_pool = self._claim(
                person, ppe_pool, _VEST_CLASSES, self._is_torso_region_valid
            )

            violations = []
            helmet_ok = helmet_label == "helmet"
            if not helmet_ok:
                violations.append("no-helmet")

            vest_ok = vest_label == "vest"
            if not vest_ok:
                violations.append("no-vest")

            person_results.append({
                "person_index": idx,
                "helmet_ok": helmet_ok,
                "vest_ok": vest_ok,
                "violations": violations,
            })
            all_violations.extend(violations)

        deduped_violations = list(dict.fromkeys(all_violations))

        if deduped_violations:
            return FrameVerdict(
                status=ComplianceStatus.NON_COMPLIANT,
                violations=deduped_violations,
                person_count=len(persons),
                person_results=person_results,
            )

        return FrameVerdict(
            status=ComplianceStatus.COMPLIANT,
            person_count=len(persons),
            person_results=person_results,
        )

    def _claim(self, person, pool, classes, region_validator):
        """Find the best candidate in `pool` for `person` among `classes`.

        Returns (class_name_or_None, new_pool) where new_pool has the
        claimed item removed (claim-and-remove: once assigned, an item is
        no longer available to any other person).
        """
        best_item = None
        best_score = -1.0

        for item in pool:
            if item.class_name not in classes:
                continue
            score = _containment_ratio(item, person)
            if score < self._containment_threshold:
                continue
            if not region_validator(item, person):
                continue
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            return None, pool

        new_pool = [d for d in pool if d is not best_item]
        return best_item.class_name, new_pool

    def _is_head_region_valid(self, item, person) -> bool:
        cx, cy = _centroid(item)
        person_height = person.y2 - person.y1
        person_width = person.x2 - person.x1
        center_x = (person.x1 + person.x2) / 2.0

        y_ok = person.y1 <= cy <= person.y1 + self._head_top_ratio * person_height
        half_band = (self._head_width_ratio / 2.0) * person_width
        x_ok = (center_x - half_band) <= cx <= (center_x + half_band)
        return y_ok and x_ok

    def _is_torso_region_valid(self, item, person) -> bool:
        cx, cy = _centroid(item)
        person_height = person.y2 - person.y1
        person_width = person.x2 - person.x1
        center_x = (person.x1 + person.x2) / 2.0

        y_top = person.y1 + self._torso_top_ratio * person_height
        y_bottom = person.y1 + self._torso_bottom_ratio * person_height
        y_ok = y_top <= cy <= y_bottom

        half_band = (self._torso_width_ratio / 2.0) * person_width
        x_ok = (center_x - half_band) <= cx <= (center_x + half_band)
        return y_ok and x_ok
