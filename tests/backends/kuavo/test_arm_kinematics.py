"""Arm kinematics for the kuavo chain: zarm_l1..7 from waist_yaw_link to the fixed
zarm_l7_end_effector, on the vendored biped_s62 URDF. The G1 constructor keeps its
default arguments, so every existing call site stays put."""
from pathlib import Path

import numpy as np
import pytest

from homebody.helpers.kinematics import Arm

ROOT = Path(__file__).resolve().parents[3]
KUAVO_URDF = ROOT / "assets/robot/kuavo_5w/biped_s62.urdf"
G1_URDF = ROOT / "assets/robot/g1_29dof_with_hand.urdf"

LEFT_ZERO_PALM = np.array([0.0, 0.2527, -0.4662])  # from waist_yaw_link, measured in the URDF


def test_kuavo_arm_chain_is_the_seven_zarm_joints():
    arm = Arm(KUAVO_URDF, "left", palm_link="zarm_l7_end_effector", root_link="waist_yaw_link")
    assert arm.names == (
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint",
        "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
    )


def test_kuavo_arm_forward_at_zero_joints_puts_the_palm_below_the_shoulder():
    arm = Arm(KUAVO_URDF, "left", palm_link="zarm_l7_end_effector", root_link="waist_yaw_link")
    pose, jacobian = arm.forward(np.zeros(7))
    assert pose[:3, 3] == pytest.approx(LEFT_ZERO_PALM, abs=1e-6)
    assert jacobian.shape == (6, 7)


def test_kuavo_arm_limits_come_from_the_urdf():
    arm = Arm(KUAVO_URDF, "left", palm_link="zarm_l7_end_effector", root_link="waist_yaw_link")
    lo, hi = arm.lower[0], arm.upper[0]
    assert lo == pytest.approx(-np.pi) and hi == pytest.approx(1.5707963267949, rel=1e-6)
    assert (arm.lower <= arm.upper).all() and len(arm.lower) == 7


def test_kuavo_arm_ik_round_trip_converges():
    arm = Arm(KUAVO_URDF, "left", palm_link="zarm_l7_end_effector", root_link="waist_yaw_link")
    rng = np.random.default_rng(7)
    for _ in range(4):
        joints = rng.uniform(arm.lower + 0.1, arm.upper - 0.1)
        pose, _ = arm.forward(joints)
        solved = arm.solve(pose, np.zeros(7))
        if solved is None:
            continue
        reached, _ = arm.forward(solved)
        assert reached[:3, 3] == pytest.approx(pose[:3, 3], abs=5e-3)
        assert reached[:3, :3] == pytest.approx(pose[:3, :3], abs=1e-2)


def test_kuavo_arm_without_a_mesh_directory_has_no_collision_shapes():
    arm = Arm(KUAVO_URDF, "left", palm_link="zarm_l7_end_effector", root_link="waist_yaw_link")
    assert arm.collision is None


def test_g1_arm_default_arguments_are_unchanged():
    arm = Arm(G1_URDF, "left")
    assert arm.names == (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    )
    assert arm.collision is not None


def test_kuavo_arm_rejects_an_unknown_palm_link():
    with pytest.raises(ValueError, match="no_such_link"):
        Arm(KUAVO_URDF, "left", palm_link="no_such_link", root_link="waist_yaw_link")
