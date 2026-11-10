"""seam①: the pure joint math — unit conversion and the wire/sensor order mapping."""
import math

import pytest

from homebody.backends.kuavo.joints import (
    deg_to_rad,
    from_sensor_radians,
    rad_to_deg,
    to_wire_degrees,
)
from homebody.backends.kuavo.profile import ARM_JOINTS, SENSOR_JOINTS, WIRE_JOINT_ORDER


def test_rad_deg_round_trip():
    for value in (0.0, 0.1, -2.5, math.pi / 2, 1.0):
        assert deg_to_rad(rad_to_deg(value)) == pytest.approx(value)
        assert rad_to_deg(deg_to_rad(value)) == pytest.approx(value)


def test_rad_to_deg_known_values():
    assert rad_to_deg(math.pi) == pytest.approx(180.0)
    assert rad_to_deg(-math.pi / 6) == pytest.approx(-30.0)


def test_wire_order_matches_urdf_order():
    """homebody speaks the URDF's zarm order; the wire order must be pinned identical."""
    assert WIRE_JOINT_ORDER["left"] == tuple(f"zarm_l{i}_joint" for i in range(1, 8))
    assert WIRE_JOINT_ORDER["right"] == tuple(f"zarm_r{i}_joint" for i in range(1, 8))
    assert all(name in SENSOR_JOINTS for names in WIRE_JOINT_ORDER.values() for name in names)


def test_to_wire_degrees_shape_and_units():
    joints = [0.1 * i for i in range(7)]
    out = to_wire_degrees("left", joints)
    assert len(out) == 7
    assert out == pytest.approx([math.degrees(v) for v in joints])


def test_to_wire_degrees_rejects_wrong_length():
    with pytest.raises(ValueError):
        to_wire_degrees("left", [0.0] * 6)
    with pytest.raises(ValueError):
        to_wire_degrees("right", [0.0] * 8)


def test_sensor_vector_layout():
    """The 20-sensor order: 4 lift, then left arm 7, right arm 7, head 2."""
    assert SENSOR_JOINTS[4:11] == ARM_JOINTS["left"]
    assert SENSOR_JOINTS[11:18] == ARM_JOINTS["right"]


def test_from_sensor_radians_splits_arms():
    q = [0.01 * i for i in range(20)]
    arms = from_sensor_radians(q)
    assert set(arms) == {"left", "right"}
    assert arms["left"] == tuple(q[4:11])
    assert arms["right"] == tuple(q[11:18])


def test_from_sensor_radians_rejects_wrong_length():
    with pytest.raises(ValueError):
        from_sensor_radians([0.0] * 14)
