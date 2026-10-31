"""
Pose-based wear check — is the helmet/vest WORN, or only held?

Why this exists
---------------
DecisionEngine only sees boxes. A vest held at waist height in front of
the body, or held up in front of the camera, produces a vest box in the
same place as a worn vest, and the shape rules in DecisionEngine
("wear_check") cannot tell them apart — confirmed on real screenshots
from the laptop webcam.

A small pose model (YOLOv8n-pose, 17 COCO body keypoints) tells us where
the shoulders, wrists and nose are. Holding something needs the hands:
a worn vest leaves the arms hanging at the sides, a held vest puts the
wrists on the vest, up at chest height or above.

Rules (sw = shoulder width in pixels, sy = shoulder line y)
-----------------------------------------------------------
Vest (only when DecisionEngine accepted a vest for this person):
  - hand holding vest: a wrist ON the vest box (+- shoulder_margin*sw)
                       AND less than hand_hold_ratio * sw below the
                       shoulders (arms hanging down put the wrists
                       ~1.3-1.6 sw below the shoulders; holding a vest
                       in front bends the elbow). Hands up at the head,
                       above the vest (adjusting a helmet), do not count.
  - shoulders outside: a shoulder keypoint outside the vest box
                       (+- shoulder_margin * sw)
  - vest too low     : vest top more than vest_top_max_below_shoulder*sw
                       below the shoulder line
Helmet (only when DecisionEngine accepted a helmet for this person):
  - hand on helmet   : a wrist on/around the helmet box (+- helmet_hand_pad
                       * helmet width). A helmet held up over the head or in
                       front of the face has the same box as a worn one, so
                       at the checkpoint the hands must be down.
  - above the head   : helmet bottom more than helmet_max_gap_above_eyes*sw
                       above the eyes (worn helmets measured -0.15..+0.01)
  - beside head      : helmet centre more than helmet_max_offset*sw
                       left/right of the nose
  - too low          : helmet bottom more than helmet_max_bottom_below_nose
                       * sw below the nose

A failed rule turns that item into a violation (no-vest / no-helmet), so
the session votes DENY and the voice names the right item.

If the needed keypoints are not visible (low confidence), or no pose
matches the person, the item is left as DecisionEngine decided — the
check only ever removes PPE that is clearly being held.

Checkpoint rule for the demo: stand with arms down at the sides.

Cost: the pose model runs only on frames where DecisionEngine already
accepted a helmet or vest, so frames with nothing to verify cost nothing.
"""

from __future__ import annotations

import dataclasses
import logging

from core.decision_engine import ComplianceStatus

logger = logging.getLogger(__name__)

# COCO keypoint indices
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10

_DEFAULTS = {
    "weights": "yolov8n-pose.pt",
    "imgsz": 320,
    "keypoint_conf": 0.5,
    "hand_hold_ratio": 1.0,
    "shoulder_margin": 0.15,
    "vest_top_max_below_shoulder": 0.35,
    "helmet_max_offset": 0.5,
    "helmet_max_bottom_below_nose": 0.15,
    "helmet_hand_pad": 0.4,
    "helmet_max_gap_above_eyes": 0.06,
}


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


class PoseWearChecker:
    """Downgrade a FrameVerdict when pose shows PPE is held, not worn.

    `pose_fn(frame) -> list[(box, keypoints)]` can be injected for tests;
    box is (x1, y1, x2, y2), keypoints a list of 17 (x, y, conf). By
    default a YOLO pose model is loaded from config.
    """

    def __init__(self, config: dict, pose_fn=None):
        cfg = {**_DEFAULTS, **(config.get("decision_engine", {}).get("pose_check") or {})}
        self._kp_conf = float(cfg["keypoint_conf"])
        self._hand_hold = float(cfg["hand_hold_ratio"])
        self._sh_margin = float(cfg["shoulder_margin"])
        self._vest_top = float(cfg["vest_top_max_below_shoulder"])
        self._helmet_offset = float(cfg["helmet_max_offset"])
        self._helmet_bottom = float(cfg["helmet_max_bottom_below_nose"])
        self._helmet_hand_pad = float(cfg["helmet_hand_pad"])
        self._helmet_gap_eyes = float(cfg["helmet_max_gap_above_eyes"])
        self._last_logged: tuple = ()

        if pose_fn is None:
            from ultralytics import YOLO  # imported lazily: tests inject pose_fn
            model = YOLO(cfg["weights"])
            imgsz = int(cfg["imgsz"])
            logger.info("Pose wear check: loaded %s (imgsz %d)", cfg["weights"], imgsz)

            def pose_fn(frame):
                r = model(frame, imgsz=imgsz, verbose=False)[0]
                if r.keypoints is None or r.boxes is None:
                    return []
                return list(zip(r.boxes.xyxy.tolist(), r.keypoints.data.tolist()))
        self._pose_fn = pose_fn

    # ------------------------------------------------------------------
    def check(self, frame, verdict):
        """Return verdict, or a copy with held PPE turned into violations."""
        results = verdict.person_results or []
        if not any(r.get("helmet_ok") or r.get("vest_ok") for r in results):
            return verdict  # nothing accepted, nothing to verify

        poses = self._pose_fn(frame)
        if not poses:
            return verdict

        new_results = []
        changed = False
        all_reasons = []
        for r in results:
            pose = self._match(r["box"], poses, single=len(results) == 1)
            reasons = self.reasons_for(r, pose[1]) if pose is not None else {}
            if not reasons:
                new_results.append(r)
                continue
            r = dict(r)
            violations = list(r["violations"])
            if "helmet" in reasons:
                r["helmet_ok"] = False
                r["helmet_box"] = None
                violations.append("no-helmet")
            if "vest" in reasons:
                r["vest_ok"] = False
                r["vest_box"] = None
                violations.append("no-vest")
            r["violations"] = sorted(set(violations), key=["no-helmet", "no-vest"].index)
            r["pose_reasons"] = reasons
            new_results.append(r)
            changed = True
            all_reasons.extend(f"{k}: {v}" for k, v in reasons.items())

        self._log(all_reasons)
        if not changed:
            return verdict

        violations = []
        for r in new_results:
            violations.extend(r["violations"])
        status = (ComplianceStatus.NON_COMPLIANT if violations
                  else ComplianceStatus.COMPLIANT)
        return dataclasses.replace(verdict, status=status,
                                   violations=violations,
                                   person_results=new_results)

    # ------------------------------------------------------------------
    def reasons_for(self, result, kps) -> dict:
        """{'helmet': reason, 'vest': reason} for items that look held.
        Empty dict = nothing wrong (or not enough keypoints to judge)."""
        ok = lambda i: kps[i][2] >= self._kp_conf  # noqa: E731
        if not (ok(L_SHOULDER) and ok(R_SHOULDER)):
            return {}
        ls, rs = kps[L_SHOULDER], kps[R_SHOULDER]
        sw = abs(ls[0] - rs[0])
        if sw < 10:
            return {}  # side-on or tiny: shoulder width meaningless
        sy = (ls[1] + rs[1]) / 2.0
        wrists = [kps[i] for i in (L_WRIST, R_WRIST) if ok(i)]
        hb = result.get("helmet_box") if result.get("helmet_ok") else None
        nose = kps[NOSE] if ok(NOSE) else None
        out = {}

        vb = result.get("vest_box")
        if result.get("vest_ok") and vb is not None:
            m = self._sh_margin * sw
            # Hands at the head (adjusting a worn helmet) are not holding
            # the vest, so they are left out of the holding rule.
            body_wrists = [w for w in wrists if not self._at_head(w, hb, nose, sw)]
            if any(vb[0] - m <= w[0] <= vb[2] + m and w[1] >= vb[1] - m
                   and (w[1] - sy) / sw < self._hand_hold
                   for w in body_wrists):
                out["vest"] = "hand holding the vest"
            elif not all(vb[0] - m <= s[0] <= vb[2] + m for s in (ls, rs)):
                out["vest"] = "vest does not cover the shoulders"
            elif (vb[1] - sy) / sw > self._vest_top:
                out["vest"] = "vest top below the shoulders"

        if hb is not None:
            hw = hb[2] - hb[0]
            pad = self._helmet_hand_pad * hw
            eyes = [kps[i] for i in (L_EYE, R_EYE) if ok(i)]
            if any(hb[0] - pad <= w[0] <= hb[2] + pad and w[1] <= hb[3] + 0.5 * hw
                   for w in wrists):
                # Holding it up above the head or in front of the face looks
                # like a worn helmet from the box alone; at the checkpoint
                # the hands must be down, so a hand on it means "held".
                out["helmet"] = "hand on the helmet (keep hands down)"
            elif eyes and (sum(e[1] for e in eyes) / len(eyes) - hb[3]) / sw > self._helmet_gap_eyes:
                out["helmet"] = "helmet held above the head"
            elif nose is not None:
                if abs((hb[0] + hb[2]) / 2.0 - nose[0]) / sw > self._helmet_offset:
                    out["helmet"] = "helmet beside the head"
                elif (hb[3] - nose[1]) / sw > self._helmet_bottom:
                    out["helmet"] = "helmet below head height"
        return out

    @staticmethod
    def _at_head(w, helmet_box, nose, sw) -> bool:
        """Wrist up at the head: around the worn helmet, or near the face."""
        if helmet_box is not None:
            pad = 0.5 * (helmet_box[2] - helmet_box[0])
            return (helmet_box[0] - pad <= w[0] <= helmet_box[2] + pad
                    and w[1] <= helmet_box[3] + pad)
        if nose is not None:
            return abs(w[0] - nose[0]) <= 0.5 * sw and w[1] <= nose[1] + 0.5 * sw
        return False

    @staticmethod
    def _match(person_box, poses, single: bool):
        best, best_iou = None, 0.0
        for p in poses:
            v = _iou(person_box, p[0])
            if v > best_iou:
                best, best_iou = p, v
        if best_iou >= 0.3:
            return best
        # Entrance: one person at the checkpoint. The detector's person box
        # can be distorted by a held vest, so fall back to the only pose.
        if single and len(poses) == 1:
            return poses[0]
        return best if best_iou >= 0.1 else None

    def _log(self, reasons: list[str]) -> None:
        key = tuple(sorted(reasons))
        if key != self._last_logged:
            if reasons:
                logger.info("Pose wear check: not worn — %s", "; ".join(key))
            self._last_logged = key
