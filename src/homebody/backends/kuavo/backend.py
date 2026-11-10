"""The Kuavo 5W backend: epoch-guarded wheel-base and arm commands over KuavoPort.

Signatures align with the Robot protocol, but the data classes stay kuavo-native until
Phase 2 parameterizes the 29-joint assumption; snapshot() and grip() refuse until then.
"""
from __future__ import annotations

import math
import threading
import time as _time

import numpy as np

from homebody.primitives.actions import ActionRejected

from .joints import from_sensor_radians, to_wire_degrees
from .limits import arm_limits
from .port import KuavoPort
from .profile import (
    ARM_EXTERNAL,
    BASE_MAX_ANGULAR_RPS,
    BASE_MAX_LINEAR_MPS,
    DEADMAN_S,
    MPC_ARM_ONLY,
)
from .safety import ArmGate

__all__ = ["KuavoBackend"]


class KuavoBackend:
    """One commander at a time: an epoch guards every command, a deadman stops the base,
    the arm gates clamp every target before it reaches the wire."""

    def __init__(self, port: KuavoPort | None = None, urdf=None, deadman_s: float = DEADMAN_S):
        self._port = port if port is not None else KuavoPort()
        self._deadman_s = float(deadman_s)
        limits = arm_limits(urdf) if urdf is not None else None
        self._limits: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        self._gates: dict[str, ArmGate] = {}
        self._last_arm_time: dict[str, float] = {}
        self._epoch = 0
        self._arm_active = False
        self._last_cmd_vel = 0.0
        self._stop_lock = threading.Lock()
        self._closed = False
        if limits is not None:
            for side, pairs in limits.items():
                lower = tuple(lo for lo, _ in pairs)
                upper = tuple(hi for _, hi in pairs)
                self._limits[side] = (lower, upper)
                self._gates[side] = ArmGate(lower, upper)

    # --- the Robot protocol's surface, minus Frame/Measurement (Phase 2)
    @property
    def epoch(self) -> int:
        return self._epoch

    def new_epoch(self) -> int:
        """An operator stop starts a new generation and refuses older commands."""
        self._epoch += 1
        self._stop_base()
        self._arm_active = False
        return self._epoch

    def _check_epoch(self, requested: int) -> None:
        if type(requested) is not int or requested != self._epoch:
            raise ActionRejected("Command belongs to an expired execution generation")

    def base_velocity(self, epoch: int, forward: float, lateral: float,
                      yaw_rate: float) -> None:
        self._check_epoch(epoch)
        values = (float(forward), float(lateral), float(yaw_rate))
        caps = (BASE_MAX_LINEAR_MPS, BASE_MAX_LINEAR_MPS, BASE_MAX_ANGULAR_RPS)
        if any(not math.isfinite(v) for v in values) or any(
                abs(v) > cap for v, cap in zip(values, caps)):
            raise ActionRejected("Base velocity exceeds configured limits")
        self._port.publish_cmd_vel(*values)
        self._last_cmd_vel = _time.monotonic()

    def arm_target(self, epoch: int, side: str, joints) -> None:
        self._check_epoch(epoch)
        if side not in self._gates:
            raise ActionRejected("Hand must be left or right")
        values = np.asarray(joints, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ActionRejected("Command requires 7 finite numbers")
        lower, upper = self._limits[side]
        if np.any(values < lower) or np.any(values > upper):
            raise ActionRejected("Arm target exceeds joint limits")
        measured = self._measured_arm(side)
        gate = self._gates[side]
        now = self._port.clock()
        dt = self._dt_since(side, now)
        q_cmd, following = gate(values, measured, dt)
        if not following:
            raise ActionRejected("The arm is not following its commands")
        self._send_arm(side, q_cmd)
        self._last_arm_time[side] = now

    def advance(self, epoch: int, seconds: float) -> None:
        self._check_epoch(epoch)
        result = float(seconds)
        if not 0 < result <= 1:
            raise ActionRejected("Advance duration must be between zero and one second")
        _time.sleep(result)

    def stop(self, epoch: int) -> None:
        self._check_epoch(epoch)
        self._stop_base()

    # --- internals
    def _stop_base(self) -> None:
        with self._stop_lock:
            if not self._closed:
                self._port.publish_cmd_vel(0.0, 0.0, 0.0)
        self._last_cmd_vel = _time.monotonic()

    def _measured_arm(self, side: str) -> np.ndarray:
        sensors = self._port.sensors()
        if sensors is None:
            raise ActionRejected("No sensor frame yet")
        arms = from_sensor_radians(sensors["joint_q"])
        return np.asarray(arms[side], dtype=float)

    def _dt_since(self, side: str, now: float) -> float:
        last = self._last_arm_time.get(side)
        return self._deadman_s if last is None else min(max(now - last, 0.0), self._deadman_s)

    def _send_arm(self, side: str, q_cmd: np.ndarray) -> None:
        """Send one side's clamped block; the wire takes both blocks, degrees."""
        left = self._gates["left"]._q_cmd if side == "right" else q_cmd
        right = q_cmd if side == "right" else self._gates["right"]._q_cmd
        self._port.publish_arm_traj(to_wire_degrees("left", left),
                                    to_wire_degrees("right", right))

    # --- the protocol's reset/close pair
    def reset(self) -> None:
        self.new_epoch()

    def close(self) -> None:
        self._closed = True
        self._stop_base()
        self._port.close()

    # --- Phase 2 placeholders, explicit about what is missing
    def snapshot(self):
        raise RuntimeError("Kuavo camera access lands in M3 (Phase 2)")

    def grip(self, epoch: int, side: str, closure: float) -> None:
        raise ActionRejected("The kuavo 5W v1 has no gripper channel")

    def ensure_arm_control(self) -> None:
        """The MPC mode and external-control switch, in the order the controller wants."""
        if not self._port.set_mpc_ctrl_mode(MPC_ARM_ONLY):
            raise ActionRejected("The controller refused the manipulation MPC mode")
        if not self._port.set_arm_ctrl_mode(ARM_EXTERNAL):
            raise ActionRejected("The controller refused the external-control mode")
        self._arm_active = True
