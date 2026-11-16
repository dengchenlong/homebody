"""Kuavo arm collision meshes: the vendored STL copies under assets/robot/kuavo_5w/biped_s62/meshes
carry the link names and baked visual origins, so the Arm built with a mesh directory gets
link boxes for the seven chain links the vendor named differently."""
from pathlib import Path

import numpy as np
import pytest

from homebody.helpers.kinematics import Arm

ROOT = Path(__file__).resolve().parents[3]
KUAVO_URDF = ROOT / "assets/robot/kuavo_5w/biped_s62.urdf"
KUAVO_MESHES = ROOT / "assets/robot/kuavo_5w/biped_s62/meshes"


@pytest.fixture(scope="module")
def left_arm():
    return Arm(KUAVO_URDF, "left", palm_link="zarm_l7_end_effector",
               root_link="waist_yaw_link", meshes=KUAVO_MESHES)


def test_vendored_meshes_carry_the_link_names():
    for side, prefix in (("l", "zarm_l"), ("r", "zarm_r")):
        for index in range(1, 8):
            path = KUAVO_MESHES / f"{prefix}{index}_link.STL"
            assert path.is_file(), f"missing vendored mesh {path}"


def test_kuavo_arm_with_meshes_has_collision(left_arm):
    assert left_arm.collision is not None
    assert left_arm.collision.links == (
        "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link",
        "zarm_l5_link", "zarm_l6_link", "zarm_l7_link", "zarm_l7_end_effector",
    )


def test_kuavo_collision_has_one_box_per_meshed_link(left_arm):
    assert set(left_arm.collision.bounds) == {f"zarm_l{i}_link" for i in range(1, 8)}


def test_kuavo_collision_has_no_finger_joints_on_the_v1_gripper_less_arm(left_arm):
    assert left_arm.collision.hand_names == []


def test_kuavo_arm_boxes_track_the_chain_at_zero_joints(left_arm):
    """The baked visual origin puts the pitch link's box around the joint chain: at zero
    joints the l1 box must sit near the l1 joint origin (y 0.2527 from waist_yaw_link)."""
    joints = np.zeros(7)
    boxes = left_arm.collision.boxes(left_arm, joints)
    centers = [box[0] for box in boxes]
    # zarm_l1_link's mesh is centered near its joint origin; its baked offset keeps it
    # beside the shoulder, not shifted 0.081 m across the torso.
    assert centers[0][1] == pytest.approx(0.2527, abs=0.05)
    assert centers[0][2] == pytest.approx(0.242, abs=0.05)
    assert centers[1][1] == pytest.approx(0.2527, abs=0.05)


def test_kuavo_arm_boxes_move_with_the_joints(left_arm):
    """Rotating l1 about its axis sweeps the forearm box away from the zero pose."""
    boxes = left_arm.collision.boxes(left_arm, np.zeros(7))
    zero_forearm = np.array(boxes[3][0])
    turned = left_arm.collision.boxes(left_arm, np.array([np.pi / 2, 0, 0, 0, 0, 0, 0]))
    assert np.linalg.norm(turned[3][0] - zero_forearm) > 0.05


def test_kuavo_arm_boxes_stay_beside_the_torso_at_zero_joints(left_arm):
    """At zero joints every left-arm box sits on the left side of the midline; a shifted
    or unshifted l1 bake would put a box across the torso."""
    from homebody.helpers.collision import contains
    boxes = left_arm.collision.boxes(left_arm, np.zeros(7))
    centers = np.array([box[0] for box in boxes])
    assert (centers[:, 1] > 0.05).all()
    # Each box contains its own center point (sanity of the bounds arithmetic).
    for box, center in zip(boxes, centers):
        assert bool(contains(box, center.reshape(1, 3))[0])


def test_right_arm_mirrors_the_left_meshes():
    arm = Arm(KUAVO_URDF, "right", palm_link="zarm_r7_end_effector",
              root_link="waist_yaw_link", meshes=KUAVO_MESHES)
    boxes = arm.collision.boxes(arm, np.zeros(7))
    assert boxes[0][0][1] == pytest.approx(-0.2527, abs=0.05)
    assert set(arm.collision.bounds) == {f"zarm_r{i}_link" for i in range(1, 8)}


def test_a_missing_mesh_skips_its_link_without_failing_the_chain():
    """The end-effector dummy has no STL; CollisionGeometry skips links without a mesh
    instead of refusing the whole arm."""
    arm = Arm(KUAVO_URDF, "left", palm_link="zarm_l7_end_effector",
              root_link="waist_yaw_link", meshes=KUAVO_MESHES)
    assert "zarm_l7_end_effector" not in arm.collision.bounds


def test_g1_arm_keeps_its_collision_with_the_new_skip_rule():
    arm = Arm(ROOT / "assets/robot/g1_29dof_with_hand.urdf", "left")
    assert arm.collision is not None
    assert "left_hand_palm_link" in arm.collision.bounds
