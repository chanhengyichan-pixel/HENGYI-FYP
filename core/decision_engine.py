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

Debug diagnostic mode (calibration tool — NOT for normal operation)
---------------------------------------------------------------------
evaluate() accepts an optional `debug: bool = False` parameter (also
exposed as config["decision_engine"]["debug"], wired through main.py
and Pipeline). This is purely diagnostic — it never alters which PPE
item gets assigned to which person, only what gets logged/returned
alongside the FrameVerdict. When debug=False (the default), the code
path is identical to a build with no debug support at all: no extra
condition evaluation, no log calls, no debug_info construction.

When debug=True, four kinds of information are produced:

1. Per-candidate-pair logging. For every (person, ppe_item) pair
   considered during claim-and-remove, a "claim-check" DEBUG log line
   reports containment and region validity — and, unlike a plain
   pass/fail, the SPECIFIC reason(s) a candidate was rejected: the
   containment score can fail independently of the region check, and
   the region check's X-band and Y-band are evaluated and reported
   independently of each other (both, not just whichever the
   short-circuited boolean would have stopped at) so e.g. "centroid X
   outside head region" and "centroid Y outside head region" are always
   distinguishable in the log, even when both fail simultaneously. An
   item that was already claimed by an earlier person (in an earlier
   iteration of the outer per-person loop) is reported separately as
   "claimed by person N — skipped", since for that item neither
   containment nor region validity was the reason it's unavailable here.
2. Per-class assignment-outcome logging, once all candidates for a
   given person+class have been considered: which candidate (if any)
   was assigned, and a summary of what was rejected and why.
3. A per-person summary line once both helmet and vest are resolved for
   that person (e.g. "Person 0: helmet=PASS, vest=FAIL -> NON_COMPLIANT").
4. A final frame-level summary line once every person has been
   processed: person count, total helmets/vests assigned, and the
   final FrameVerdict status + violations.

In addition, when debug=True, FrameVerdict.debug_info (see below) is
populated with the same underlying data in structured form, for
programmatic consumption (e.g. the dashboard's diagnostic overlay)
rather than log-scraping.

This exists to answer, for one specific real-world frame captured
during calibration, "why did this PPE item get assigned/not-assigned to
this person" without having to re-derive the geometry by hand. It is
meant to be switched on temporarily while tuning containment_threshold
and the head/torso ratios against real camera footage, then switched
off again. Leaving it on during normal/demo operation will flood the
log at the pipeline's inference rate (observed ~170fps on capable
hardware) — config["decision_engine"]["debug"] therefore defaults to
false.

Additive field: debug_info
---------------------------
FrameVerdict gains `debug_info: dict | None`, defaulting to None.
Non-None ONLY when evaluate() was called with debug=True — never
populated otherwise, so this is a zero-shape-change addition for every
existing caller. Structure (all coordinates are the same pixel-space
box/centroid coordinates evaluate() received):

  {
    "persons": [
      {
        "person_index": int,
        "box": (x1, y1, x2, y2), "confidence": float,
        "head_region": (y_lo, y_hi, x_lo, x_hi),
        "torso_region": (y_lo, y_hi, x_lo, x_hi),
        "helmet": {"assigned": "helmet" | "no-helmet" | None,
                   "candidates": [<candidate dict>, ...]},
        "vest":   {"assigned": "vest" | "no-vest" | None,
                   "candidates": [<candidate dict>, ...]},
        "person_status": "COMPLIANT" | "NON_COMPLIANT",  # this person only
      },
      ...
    ],
    "frame": {
      "person_count": int, "helmets_assigned": int, "vests_assigned": int,
      "status": str, "violations": list[str],
    },
  }

  <candidate dict> = {
    "class_name": str, "box": (x1, y1, x2, y2), "centroid": (cx, cy),
    "containment": float, "containment_ok": bool,
    "x_ok": bool, "y_ok": bool, "region_ok": bool,
    "claimed_by": int | None,  # person_index that already holds this item
    "status": "assigned" | "rejected" | "claimed_by_other",
    "reasons": list[str],  # human-readable rejection reasons, empty if assigned
  }

This module deliberately does NOT draw anything (no cv2 import, no
drawing calls) — consistent with the separation of concerns already
established across the project (detector.py returns Detection objects
and lets dashboard/app.py draw them; this module returns FrameVerdict
and lets dashboard/app.py draw debug_info via _draw_debug_overlay()).

No-person rule
--------------
Returns NO_PERSON — not COMPLIANT. The session controller decides
whether to stay IDLE, extend a timeout, etc. The engine does not make
that policy call.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    debug_info: dict | None = None

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

    def evaluate(self, detections, debug: bool = False) -> FrameVerdict:
        """
        Evaluate a single frame's detections and return a FrameVerdict.

        Parameters
        ----------
        detections : iterable of objects with .class_name: str,
                     .confidence: float, .x1/.y1/.x2/.y2 box coords.
        debug : if True, log detailed per-candidate diagnostics at DEBUG
                level and populate FrameVerdict.debug_info — see the
                "Debug diagnostic mode" section of the module docstring.
                Diagnostic only; never changes the result.

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

        # Immutable snapshot of every PPE item, used only for debug
        # diagnostics so that items already claimed by an earlier person
        # can still be reported (they're gone from the mutating `ppe_pool`
        # by the time a later person is processed). `claim_history` maps
        # id(item) -> the person_index that claimed it. Both are no-ops
        # when debug=False.
        all_ppe_items = list(ppe_pool) if debug else None
        claim_history: dict[int, int] = {} if debug else None

        person_results = []
        debug_persons = [] if debug else None
        all_violations: list[str] = []
        total_helmets_assigned = 0
        total_vests_assigned = 0

        for idx, person in enumerate(persons):
            helmet_label, ppe_pool, helmet_debug = self._claim(
                idx, person, ppe_pool, _HELMET_CLASSES,
                "head", self._head_region_bounds, self._is_head_region_valid,
                debug, all_ppe_items, claim_history,
            )
            vest_label, ppe_pool, vest_debug = self._claim(
                idx, person, ppe_pool, _VEST_CLASSES,
                "torso", self._torso_region_bounds, self._is_torso_region_valid,
                debug, all_ppe_items, claim_history,
            )

            violations = []
            helmet_ok = helmet_label == "helmet"
            if helmet_ok:
                total_helmets_assigned += 1
            else:
                violations.append("no-helmet")

            vest_ok = vest_label == "vest"
            if vest_ok:
                total_vests_assigned += 1
            else:
                violations.append("no-vest")

            person_results.append({
                "person_index": idx,
                "helmet_ok": helmet_ok,
                "vest_ok": vest_ok,
                "violations": violations,
            })
            all_violations.extend(violations)

            if debug:
                person_status = "COMPLIANT" if (helmet_ok and vest_ok) else "NON_COMPLIANT"
                logger.debug(
                    "Person %d: helmet=%s, vest=%s -> %s",
                    idx, "PASS" if helmet_ok else "FAIL",
                    "PASS" if vest_ok else "FAIL", person_status,
                )
                debug_persons.append({
                    "person_index": idx,
                    "box": (person.x1, person.y1, person.x2, person.y2),
                    "confidence": person.confidence,
                    "head_region": self._head_region_bounds(person),
                    "torso_region": self._torso_region_bounds(person),
                    "helmet": helmet_debug,
                    "vest": vest_debug,
                    "person_status": person_status,
                })

        deduped_violations = list(dict.fromkeys(all_violations))
        status = (
            ComplianceStatus.NON_COMPLIANT if deduped_violations
            else ComplianceStatus.COMPLIANT
        )

        debug_info = None
        if debug:
            debug_info = {
                "persons": debug_persons,
                "frame": {
                    "person_count": len(persons),
                    "helmets_assigned": total_helmets_assigned,
                    "vests_assigned": total_vests_assigned,
                    "status": status.value,
                    "violations": deduped_violations,
                },
            }
            logger.debug(
                "Frame summary: persons=%d, helmets_assigned=%d, vests_assigned=%d, "
                "status=%s, violations=%s",
                len(persons), total_helmets_assigned, total_vests_assigned,
                status.value, deduped_violations,
            )

        return FrameVerdict(
            status=status,
            violations=deduped_violations,
            person_count=len(persons),
            person_results=person_results,
            debug_info=debug_info,
        )

    def _claim(self, person_idx, person, pool, classes, region_name,
               bounds_fn, region_validator, debug, all_ppe_items, claim_history):
        """Find the best candidate in `pool` for `person` among `classes`.

        Returns (class_name_or_None, new_pool, debug_entry_or_None) where
        new_pool has the claimed item removed (claim-and-remove: once
        assigned, an item is no longer available to any other person).
        debug_entry is only built when debug=True; see the module
        docstring's "Additive field: debug_info" section for its shape.

        The actual accept/reject decision (containment_ok and region_ok)
        is computed identically regardless of `debug` — debug=True only
        adds extra, otherwise-unused independent X/Y-axis evaluation for
        richer logging, it never changes which item wins.
        """
        best_item = None
        best_score = -1.0
        debug_candidates = [] if debug else None
        # Parallel list of item object references, same indices as
        # debug_candidates, so the winning candidate's dict can be found by
        # identity (not by value — two candidates could share identical
        # box/class coordinates) once best_item is known.
        debug_items = [] if debug else None

        if debug:
            y_lo, y_hi, x_lo, x_hi = bounds_fn(person)

            # Report items already claimed by an earlier person first —
            # they're no longer in `pool`, so the main loop below (which
            # iterates `pool`) would otherwise never mention them.
            for item in all_ppe_items:
                if item.class_name not in classes:
                    continue
                claimer = claim_history.get(id(item))
                if claimer is None:
                    continue
                cx, cy = _centroid(item)
                logger.debug(
                    "claim-check person[%d] %s item=%s box=(%s,%s,%s,%s): "
                    "already claimed by person %d — skipped for person %d",
                    person_idx, region_name, item.class_name,
                    item.x1, item.y1, item.x2, item.y2, claimer, person_idx,
                )
                debug_candidates.append({
                    "class_name": item.class_name,
                    "box": (item.x1, item.y1, item.x2, item.y2),
                    "centroid": (cx, cy),
                    "containment": None,
                    "containment_ok": None,
                    "x_ok": None,
                    "y_ok": None,
                    "region_ok": None,
                    "claimed_by": claimer,
                    "status": "claimed_by_other",
                    "reasons": [f"already claimed by person {claimer}"],
                })

        for item in pool:
            if item.class_name not in classes:
                continue
            score = _containment_ratio(item, person)
            containment_ok = score >= self._containment_threshold
            # Actual decision — unchanged regardless of `debug`.
            region_ok = region_validator(item, person)
            is_candidate = containment_ok and region_ok

            if debug:
                cx, cy = _centroid(item)
                # Independent X/Y evaluation purely for diagnostic detail —
                # does not replace or short-circuit the real `region_ok`
                # decision above.
                x_ok = x_lo <= cx <= x_hi
                y_ok = y_lo <= cy <= y_hi

                reasons = []
                if not containment_ok:
                    reasons.append(
                        f"containment below threshold "
                        f"({score:.2f} < {self._containment_threshold:.2f})"
                    )
                if not x_ok:
                    reasons.append(
                        f"centroid X outside {region_name} region "
                        f"(cx={cx:.1f}, valid=[{x_lo:.1f},{x_hi:.1f}])"
                    )
                if not y_ok:
                    reasons.append(
                        f"centroid Y outside {region_name} region "
                        f"(cy={cy:.1f}, valid=[{y_lo:.1f},{y_hi:.1f}])"
                    )

                logger.debug(
                    "claim-check person[%d] box=(%s,%s,%s,%s) conf=%.4f | "
                    "item=%s box=(%s,%s,%s,%s) centroid=(%.1f,%.1f) | "
                    "%s_region y=[%.1f,%.1f] x=[%.1f,%.1f] | "
                    "containment=%.4f (threshold=%.2f) | outcome=%s%s",
                    person_idx, person.x1, person.y1, person.x2, person.y2,
                    person.confidence,
                    item.class_name, item.x1, item.y1, item.x2, item.y2,
                    cx, cy,
                    region_name, y_lo, y_hi, x_lo, x_hi,
                    score, self._containment_threshold,
                    "candidate" if is_candidate else "rejected",
                    "" if is_candidate else f" ({'; '.join(reasons)})",
                )

                debug_candidates.append({
                    "class_name": item.class_name,
                    "box": (item.x1, item.y1, item.x2, item.y2),
                    "centroid": (cx, cy),
                    "containment": score,
                    "containment_ok": containment_ok,
                    "x_ok": x_ok,
                    "y_ok": y_ok,
                    "region_ok": region_ok,
                    "claimed_by": None,
                    # "status"/"reasons" for the currently-best candidate are
                    # corrected to "assigned"/[] below, once known.
                    "status": "rejected" if not is_candidate else "candidate",
                    "reasons": reasons,
                })
                debug_items.append(item)

            if not is_candidate:
                continue
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            if debug:
                logger.debug(
                    "claim-outcome person[%d] %s: no candidate assigned "
                    "(%d rejected)",
                    person_idx, region_name,
                    sum(1 for c in debug_candidates if c["status"] == "rejected"),
                )
                return None, pool, {"assigned": None, "candidates": debug_candidates}
            return None, pool, None

        if debug:
            for i, item in enumerate(debug_items):
                if item is best_item:
                    debug_candidates[i]["status"] = "assigned"
                    debug_candidates[i]["reasons"] = []
                    break
            claim_history[id(best_item)] = person_idx
            rejected = [c for c in debug_candidates if c["status"] == "rejected"]
            logger.debug(
                "claim-outcome person[%d] %s: assigned %s (containment=%.4f), "
                "%d rejected: %s",
                person_idx, region_name, best_item.class_name, best_score,
                len(rejected),
                "; ".join(f"{c['class_name']} [{', '.join(c['reasons'])}]" for c in rejected)
                or "none",
            )

        new_pool = [d for d in pool if d is not best_item]
        debug_entry = (
            {"assigned": best_item.class_name, "candidates": debug_candidates}
            if debug else None
        )
        return best_item.class_name, new_pool, debug_entry

    def _head_region_bounds(self, person) -> tuple[float, float, float, float]:
        """Return (y_lo, y_hi, x_lo, x_hi) head-region bounds for `person`."""
        person_height = person.y2 - person.y1
        person_width = person.x2 - person.x1
        center_x = (person.x1 + person.x2) / 2.0
        half_band = (self._head_width_ratio / 2.0) * person_width
        return (
            person.y1, person.y1 + self._head_top_ratio * person_height,
            center_x - half_band, center_x + half_band,
        )

    def _torso_region_bounds(self, person) -> tuple[float, float, float, float]:
        """Return (y_lo, y_hi, x_lo, x_hi) torso-region bounds for `person`."""
        person_height = person.y2 - person.y1
        person_width = person.x2 - person.x1
        center_x = (person.x1 + person.x2) / 2.0
        half_band = (self._torso_width_ratio / 2.0) * person_width
        return (
            person.y1 + self._torso_top_ratio * person_height,
            person.y1 + self._torso_bottom_ratio * person_height,
            center_x - half_band, center_x + half_band,
        )

    def _is_head_region_valid(self, item, person) -> bool:
        cx, cy = _centroid(item)
        y_lo, y_hi, x_lo, x_hi = self._head_region_bounds(person)
        return (y_lo <= cy <= y_hi) and (x_lo <= cx <= x_hi)

    def _is_torso_region_valid(self, item, person) -> bool:
        cx, cy = _centroid(item)
        y_lo, y_hi, x_lo, x_hi = self._torso_region_bounds(person)
        return (y_lo <= cy <= y_hi) and (x_lo <= cx <= x_hi)
