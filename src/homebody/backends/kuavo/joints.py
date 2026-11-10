"""Pure joint math for the Kuavo port: unit conversion and wire-order mapping.

The arm trajectory topic speaks DEGREES (the controller converts back internally); the
sensor stream speaks radians. Every crossing goes through this module so seam tests can
pin the conversion once.
"""
from __future__ import annotations

import math

from .profile import ARM_JOINTS, WIRE_JOINT_ORDER

__all__ = ["deg_to_rad", "from_sensor_radians", "rad_to_deg", "to_wire_degrees"]


def rad_to_deg(value: float) -> float:
    return value * 180.0 / math.pi


def deg_to_rad(value: float) -> float:
    return value * math.pi / 180.0


def to_wire_degrees(side: str, joints) -> list[float]:
    """One arm's 7 joint positions in radians, homebody order -> the 14-vector's degrees.

    Returns only that side's block (7 degrees, wire order), in the position the wire
    vector expects; the sender concatenates left then right.
    """
    order = WIRE_JOINT_ORDER[side]
    if len(joints) != len(order):
        raise ValueError(f"{side} arm needs {len(order)} joints, got {len(joints)}")
    return [rad_to_deg(float(value)) for value in joints]


def from_sensor_radians(sensor_q) -> dict[str, tuple[float, ...]]:
    """The 20-sensor vector in radians -> one dict of the two arms' tuples, homebody order."""
    q = list(sensor_q)
    if len(q) != 20:
        raise ValueError(f"sensor vector needs 20 joints, got {len(q)}")
    names = list(ARM_JOINTS["left"]) + list(ARM_JOINTS["right"])
    return {
        "left": tuple(float(q[i]) for i in range(4, 11)),
        "right": tuple(float(q[i]) for i in range(11, 18)),
    } if names else {}
