"""seam②: the URDF limit parser."""
import xml.etree.ElementTree as ET

import pytest

from homebody.backends.kuavo.limits import arm_limits, load_limits
from homebody.backends.kuavo.profile import ARM_JOINTS, urdf_path


def _write_urdf(tmp_path, entries):
    root = ET.Element("robot", {"name": "t"})
    for name, lo, hi, jtype in entries:
        joint = ET.SubElement(root, "joint", {"name": name, "type": jtype})
        if jtype == "revolute":
            ET.SubElement(joint, "limit", {"lower": str(lo), "upper": str(hi)})
    path = tmp_path / "mini.urdf"
    ET.ElementTree(root).write(path)
    return path


def test_real_urdf_yields_14_limits():
    limits = load_limits(urdf_path())
    assert len(limits) == 14
    for side in ("left", "right"):
        assert {limits[n] for n in ARM_JOINTS[side]} <= set(limits.values())


def test_real_urdf_left_right_mirroring():
    limits = load_limits(urdf_path())
    # l2/r2 and l6/r6 are mirrored; the rest match.
    for i in (1, 3, 4, 5, 7):
        assert limits[f"zarm_l{i}_joint"] == limits[f"zarm_r{i}_joint"]
    l2, r2 = limits["zarm_l2_joint"], limits["zarm_r2_joint"]
    assert l2[0] == pytest.approx(-r2[1]) and l2[1] == pytest.approx(-r2[0])
    l6, r6 = limits["zarm_l6_joint"], limits["zarm_r6_joint"]
    assert l6[0] == pytest.approx(-r6[1]) and l6[1] == pytest.approx(-r6[0])


def test_arm_limits_returns_side_tuples():
    sides = arm_limits(urdf_path())
    assert set(sides) == {"left", "right"}
    for side in sides:
        assert len(sides[side]) == 7
        assert all(lo < hi for lo, hi in sides[side])


def test_fixed_and_unknown_joints_skipped(tmp_path):
    path = _write_urdf(tmp_path, [
        ("zarm_l1_joint", -1.0, 1.0, "revolute"),
        ("zarm_l7_end_effector_joint", None, None, "fixed"),
        ("wheel_joint", -1.0, 1.0, "continuous"),
    ])
    assert set(load_limits(path)) == {"zarm_l1_joint"}


def test_missing_arm_limit_raises(tmp_path):
    path = _write_urdf(tmp_path, [("zarm_l1_joint", -1.0, 1.0, "revolute")])
    with pytest.raises(ValueError, match="zarm_l2"):
        arm_limits(path)
