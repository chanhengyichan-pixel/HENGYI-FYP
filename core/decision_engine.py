"""
Single-frame PPE compliance decision engine.

Input
-----
A list of detection objects with a .class_name attribute (str).
Uses duck typing — accepts core.detector.Detection or any object that
exposes .class_name.  Does NOT import detector.py or ultralytics.

Model schema (exp01_roboflow)
-----------------------------
  0: helmet   — PPE present (positive signal)
  1: no-helmet — PPE absent  (negative / violation signal)
  2: no-vest   — PPE absent  (negative / violation signal)
  3: person
  4: vest     — PPE present (positive signal)

Compliance rule
---------------
A frame is COMPLIANT iff ALL of the following hold:
  1. At least one person is detected.
  2. No 'no-helmet' box is present.
  3. No 'no-vest' box is present.
  4. helmet_count >= person_count (every person has a confirmed helmet).
  5. vest_count   >= person_count (every person has a confirmed vest).

Conditions 4 and 5 use counts, not just presence.  Checking "at least
one helmet exists" is insufficient with multiple people: one compliant
person's helmet box would mask a second person who has no PPE signal
at all, producing a false COMPLIANT verdict.  Requiring
helmet_count >= person_count ensures the positive-signal coverage
scales with the number of people detected.

If either count falls short, it is treated as NON_COMPLIANT under the
same fail-safe rule as the single-person no-signal case: ambiguity
never grants access.

Multi-person rule
-----------------
The frame is NON_COMPLIANT if ANY explicit violation is detected, OR if
the count of confirmed PPE boxes does not cover every detected person.
One non-compliant person blocks the gate.

No-person rule
--------------
Returns NO_PERSON — not COMPLIANT.  The session controller decides
whether to stay IDLE, extend a timeout, etc.  The engine does not make
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

    def is_compliant(self) -> bool:
        return self.status is ComplianceStatus.COMPLIANT


_VIOLATION_CLASSES = {"no-helmet", "no-vest"}
_REQUIRED_PPE = {"helmet", "vest"}
_PERSON_CLASS = "person"


class DecisionEngine:
    def evaluate(self, detections) -> FrameVerdict:
        """
        Evaluate a single frame's detections and return a FrameVerdict.

        Parameters
        ----------
        detections : iterable of objects with a .class_name: str attribute

        Returns
        -------
        FrameVerdict
        """
        labels = [d.class_name for d in detections]

        person_count = labels.count(_PERSON_CLASS)

        if person_count == 0:
            return FrameVerdict(status=ComplianceStatus.NO_PERSON)

        violations = [lbl for lbl in labels if lbl in _VIOLATION_CLASSES]

        # Require one confirmed PPE box per detected person (count comparison,
        # not just presence) so that one compliant person cannot mask another
        # person who has no PPE signal at all.
        insufficient_ppe = [
            f"no-{ppe}"
            for ppe in _REQUIRED_PPE
            if labels.count(ppe) < person_count
        ]

        # Missing positive signal is also a violation (fail-safe ambiguity rule).
        all_violations = list(dict.fromkeys(violations + insufficient_ppe))

        if all_violations:
            return FrameVerdict(
                status=ComplianceStatus.NON_COMPLIANT,
                violations=all_violations,
                person_count=person_count,
            )

        return FrameVerdict(
            status=ComplianceStatus.COMPLIANT,
            person_count=person_count,
        )
