"""
Tests for core/pose_wear_check.py. No model: pose_fn is injected and
returns hand-made keypoints. Geometry is taken from real webcam frames
(640x480): shoulders ~140 px apart, arms down puts wrists ~1.5 shoulder
widths below the shoulders.
"""

from core.decision_engine import ComplianceStatus, FrameVerdict
from core.pose_wear_check import PoseWearChecker

CONFIG = {"decision_engine": {"pose_check": {"enabled": True}}}

PERSON = (220, 0, 460, 480)
HELMET = (254, 4, 374, 84)
VEST = (234, 146, 441, 427)


def _kps(nose=(298, 90), lsho=(403, 184), rsho=(263, 191),
         lwri=(402, 390), rwri=(249, 399), conf=0.9):
    k = [[0.0, 0.0, 0.0] for _ in range(17)]
    k[0] = [*nose, conf]
    k[5] = [*lsho, conf]
    k[6] = [*rsho, conf]
    if lwri is not None:
        k[9] = [*lwri, conf]
    if rwri is not None:
        k[10] = [*rwri, conf]
    return k


def _verdict(helmet_box=HELMET, vest_box=VEST, person_box=PERSON):
    return FrameVerdict(
        status=ComplianceStatus.COMPLIANT, violations=[], person_count=1,
        person_results=[{
            "person_index": 0, "box": person_box,
            "helmet_ok": helmet_box is not None, "vest_ok": vest_box is not None,
            "violations": [], "helmet_box": helmet_box, "vest_box": vest_box,
        }],
    )


def _check(kps, verdict=None, pose_box=PERSON):
    calls = []

    def pose_fn(frame):
        calls.append(frame)
        return [(pose_box, kps)]
    checker = PoseWearChecker(CONFIG, pose_fn=pose_fn)
    return checker.check("frame", verdict or _verdict()), calls


def test_worn_arms_down_stays_compliant():
    v, _ = _check(_kps())
    assert v.status is ComplianceStatus.COMPLIANT
    assert v.violations == []


def test_vest_held_at_waist_rejected():
    # Real frame: wrists bent up on the vest, < 1 shoulder width below shoulders.
    v, _ = _check(_kps(lwri=(400, 300), rwri=(270, 280)))
    assert v.status is ComplianceStatus.NON_COMPLIANT
    assert v.violations == ["no-vest"]
    assert v.person_results[0]["pose_reasons"]["vest"] == "hand holding the vest"
    assert v.person_results[0]["vest_ok"] is False
    assert v.person_results[0]["helmet_ok"] is True


def test_vest_held_up_beside_body_rejected():
    # Vest held out to the side: shoulders are not under the vest box.
    v, _ = _check(_kps(), _verdict(vest_box=(0, 130, 230, 480)))
    assert "no-vest" in v.violations


def test_hands_on_helmet_reject_helmet_not_vest():
    # Wrists up at the helmet: the vest is not blamed, but a hand on the
    # helmet means it may be held above the head -> helmet not accepted.
    v, _ = _check(_kps(lwri=(380, 100), rwri=(250, 95)))
    assert v.violations == ["no-helmet"]


def test_helmet_hovering_above_head_rejected():
    # Helmet bottom (60) well above the eyes (y=75), hands not detected.
    k = _kps(lwri=None, rwri=None)
    k[1] = [315, 75, 0.9]
    k[2] = [287, 77, 0.9]
    v, _ = _check(k, _verdict(helmet_box=(254, -20, 374, 50)))
    assert v.violations == ["no-helmet"]


def test_worn_helmet_with_eyes_visible_ok():
    k = _kps()
    k[1] = [315, 75, 0.9]
    k[2] = [287, 77, 0.9]
    v, _ = _check(k)
    assert v.status is ComplianceStatus.COMPLIANT


def test_helmet_held_beside_head_rejected():
    v, _ = _check(_kps(), _verdict(helmet_box=(400, 60, 500, 140)))
    assert v.violations == ["no-helmet"]


def test_helmet_held_in_front_of_face_rejected():
    # Helmet bottom well below the nose.
    v, _ = _check(_kps(), _verdict(helmet_box=(250, 60, 360, 170)))
    assert v.violations == ["no-helmet"]


def test_no_shoulders_visible_leaves_verdict_alone():
    v, _ = _check(_kps(lwri=(400, 300), rwri=(270, 280), conf=0.2))
    assert v.status is ComplianceStatus.COMPLIANT


def test_pose_not_run_when_nothing_accepted():
    verdict = FrameVerdict(status=ComplianceStatus.NON_COMPLIANT,
                           violations=["no-helmet", "no-vest"], person_count=1,
                           person_results=[{"person_index": 0, "box": PERSON,
                                            "helmet_ok": False, "vest_ok": False,
                                            "violations": ["no-helmet", "no-vest"],
                                            "helmet_box": None, "vest_box": None}])
    v, calls = _check(_kps(), verdict)
    assert calls == []
    assert v is verdict


def test_single_person_matched_even_if_person_box_is_off():
    # The detector's person box can be pulled sideways by a held vest.
    v, _ = _check(_kps(lwri=(400, 300), rwri=(270, 280)),
                  _verdict(person_box=(0, 0, 230, 480)), pose_box=PERSON)
    assert "no-vest" in v.violations


def test_no_person_verdict_untouched():
    verdict = FrameVerdict(status=ComplianceStatus.NO_PERSON)
    v, calls = _check(_kps(), verdict)
    assert v is verdict and calls == []
