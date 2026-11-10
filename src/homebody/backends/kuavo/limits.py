"""Arm joint limits parsed from the biped_s62 URDF."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .profile import ARM_JOINTS

__all__ = ["arm_limits", "load_limits"]


def load_limits(path) -> dict[str, tuple[float, float]]:
    """{joint name: (lower, upper) rad} for every revolute zarm joint in the URDF."""
    root = ET.parse(path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.get("name") or ""
        if not name.startswith("zarm_") or not name.endswith("_joint") or "end_effector" in name:
            continue
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


def arm_limits(path) -> dict[str, tuple[tuple[float, float], ...]]:
    """{side: 7 (lo, hi) tuples} in homebody's per-arm order; every arm joint must exist."""
    table = load_limits(path)
    missing = [name for names in ARM_JOINTS.values() for name in names if name not in table]
    if missing:
        raise ValueError(f"URDF at {path} lacks limits for {missing}")
    return {side: tuple(table[name] for name in names) for side, names in ARM_JOINTS.items()}
