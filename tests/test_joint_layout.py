"""JointLayout: the per-robot joint layout the observation records read instead of the
hard-coded 29-joint G1 assumption. The G1 default must leave every existing number put."""
import numpy as np
import pytest

from homebody.primitives.observations import (
    Frame,
    HandState,
    JointLayout,
    Measurement,
    NavigationMap,
    g1_layout,
    kuavo_layout,
)


def hand():
    return {side: HandState(np.zeros(7), 0.0, np.zeros(3), np.zeros(3))
            for side in ("left", "right")}


def navigation():
    return NavigationMap(np.zeros((40, 40), dtype=bool), np.diag([0.1, 0.1, 1]), 0.1)


def measurement(layout, joints):
    return Measurement(1, 1, 0.0, np.eye(4), np.eye(4), np.zeros(3), joints, hand(),
                       layout=layout)


def frame(layout, joints):
    return Frame(1, 1, 0.0, np.zeros((6, 8, 3), dtype=np.uint8), np.ones((6, 8)),
                 np.array([[10, 0, 4], [0, 10, 3], [0, 0, 1]]), np.eye(4), np.eye(4),
                 np.zeros(3), joints, hand(), np.ones((6, 8), dtype=int), navigation(),
                 layout=layout)


def test_g1_layout_is_the_default_and_matches_the_old_constants():
    layout = g1_layout()
    assert layout.joints == 29
    assert layout.arms == slice(15, 29)
    assert layout.arm["left"] == slice(15, 22)
    assert layout.arm["right"] == slice(22, 29)


def test_measurement_and_frame_default_to_the_g1_layout():
    """No layout passed: the G1 slices hold, so every existing construction path is unchanged."""
    joints = np.arange(29, dtype=float)
    for packet in (measurement(None, joints), frame(None, joints)):
        assert packet.joints.shape == (29,)
        assert packet.arm_joints("left").tolist() == joints[15:22].tolist()
        assert packet.arm_joints("right").tolist() == joints[22:29].tolist()


def test_measurement_and_frame_reject_a_wrong_joint_count():
    with pytest.raises(ValueError):
        measurement(None, np.zeros(20))
    with pytest.raises(ValueError):
        frame(None, np.zeros(20))
    with pytest.raises(ValueError):
        measurement(kuavo_layout(), np.zeros(29))


def test_kuavo_layout_slices_the_twenty_sensor_vector():
    """The 20-sensor order: lift 4, left arm 4..10, right arm 11..17, head 18..19."""
    layout = kuavo_layout()
    assert layout.joints == 20
    assert layout.arms == slice(4, 18)
    assert layout.arm["left"] == slice(4, 11)
    assert layout.arm["right"] == slice(11, 18)
    joints = np.arange(20, dtype=float)
    packet = measurement(layout, joints)
    assert packet.arm_joints("left").tolist() == joints[4:11].tolist()
    assert packet.arm_joints("right").tolist() == joints[11:18].tolist()
    assert frame(layout, joints).arm_joints("left").shape == (7,)


def test_layouts_are_comparable_value_objects():
    assert g1_layout() == g1_layout()
    assert kuavo_layout() != g1_layout()
    assert JointLayout(joints=29, arms=slice(15, 29),
                       arm={"left": slice(15, 22), "right": slice(22, 29)}) == g1_layout()
