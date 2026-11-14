"""The Kuavo 5W backend: a full Robot implementation over KuavoPort.

One commander at a time: an epoch guards every command, a deadman stops the base, the
arm gates clamp every target before it reaches the wire. Observations translate at this
boundary into homebody-native records under the Kuavo joint layout; the camera is not
wired yet, so snapshot() returns a valid blank frame (no depth, no labels) and grip()
refuses — the 5W v1 has no gripper channel.
"""
from __future__ import annotations

import math
import threading
import time as _time

import numpy as np

from homebody.helpers.geometry import transform
from homebody.primitives.actions import ActionRejected
from homebody.primitives.observations import (
    Frame,
    HandState,
    JointLayout,
    Measurement,
    NavigationMap,
    kuavo_layout,
)

from .joints import from_sensor_radians
from .limits import arm_limits
from .port import KuavoPort
from .profile import (
    ARM_EXTERNAL,
    BASE_MAX_ANGULAR_RPS,
    BASE_MAX_LINEAR_MPS,
    DEADMAN_S,
    MPC_BASE_ARM,
)
from .safety import ArmGate

__all__ = ["KuavoBackend"]

# The session reads this before touching the scorer's evaluation surface: a real-robot
# backend records its own sensors, never simulator-only physics state.
EVALUATES = False

_BLANK_INTRINSICS = np.diag([1.0, 1.0, 1.0]).astype(float)  # no camera: pixels carry no metric
# Until a scanned map is wired in (M2's on-site work), navigate plans over the same
# open fallback raster the MuJoCo backend uses when no scene package is given.
_BLANK_MAP = NavigationMap(np.zeros((200, 200), dtype=bool),
                           np.array([[.025, 0, -2.5], [0, -.025, 2.5], [0, 0, 1]]), .025)


def _blank_hand() -> HandState:
    """The gripper-less hand: closed-empty zero state; retention evidence never appears."""
    return HandState(np.zeros(7), 0.0, np.zeros(3), np.zeros(3))


class KuavoBackend:
    """One commander at a time: an epoch guards every command, a deadman stops the base,
    the arm gates clamp every target before it reaches the wire."""

    def __init__(self, port: KuavoPort | None = None, urdf=None, deadman_s: float = DEADMAN_S,
                 layout: JointLayout | None = None):
        self._port = port if port is not None else KuavoPort()
        self._deadman_s = float(deadman_s)
        self.layout = kuavo_layout() if layout is None else layout
        limits = arm_limits(urdf) if urdf is not None else None
        self._limits: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        self._gates: dict[str, ArmGate] = {}
        self._last_arm_time: dict[str, float] = {}
        self._epoch = 0
        self._sequence = 0
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

    # --- the Robot protocol's observation surface
    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def time(self) -> float:
        """This backend runs in real time: the monotonic wall clock stamps every frame."""
        return _time.monotonic()

    def new_epoch(self) -> int:
        """An operator stop starts a new generation and refuses older commands."""
        self._epoch += 1
        self._sequence = 0
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

    def snapshot(self) -> Frame:
        """One synchronized frame. When the port has a decoded camera frame — real rgb,
        depth and intrinsics, the aligned-depth-to-rgb pair camera.md pins — the images
        pass through; otherwise the images hold nothing (black rgb, no metric depth, no
        labels): navigation still routes, and any skill that needs a click is refused at
        its own argument check."""
        sensors = self._port.sensors()
        if sensors is None:
            raise ActionRejected("No sensor frame yet")
        joints = np.asarray(sensors["joint_q"], dtype=float)
        blank = np.full((8, 8), np.nan, dtype=float)
        camera_frame = self._camera_frame()
        if camera_frame is not None:
            rgb, depth, intrinsics = camera_frame
        else:
            rgb = np.zeros((*blank.shape, 3), dtype=np.uint8)
            depth, intrinsics = blank, _BLANK_INTRINSICS
        return Frame(self.epoch, self._sequence, self.time,
                     rgb, depth, intrinsics,
                     transform(self._camera_position()),
                     transform([0.0, 0.0, 0.0]),
                     self._base_pose(), joints,
                     {side: _blank_hand() for side in ("left", "right")},
                     np.zeros(depth.shape, dtype=np.int32), _BLANK_MAP, layout=self.layout)

    def _camera_frame(self):
        """The port's camera dict as (rgb, depth, intrinsics), or None until the real
        triple is complete — a frame without depth has no click geometry."""
        camera = self._port.camera()
        if camera is None:
            return None
        rgb, depth, intrinsics = camera.get("rgb"), camera.get("depth"), camera.get("intrinsics")
        if rgb is None or depth is None or intrinsics is None:
            return None
        return rgb, depth, intrinsics

    def measure(self) -> Measurement:
        sensors = self._port.sensors()
        if sensors is None:
            raise ActionRejected("No sensor frame yet")
        self._sequence += 1
        joints = np.asarray(sensors["joint_q"], dtype=float)
        return Measurement(self.epoch, self._sequence, self.time,
                           transform(self._camera_position()), transform([0.0, 0.0, 0.0]),
                           self._base_pose(), joints,
                           {side: _blank_hand() for side in ("left", "right")},
                           layout=self.layout)

    def _camera_position(self):
        """The head camera sits ahead and above the base by the biped_s62 mounting offset;
        without a wired camera its exact pose is placeholder, so keep it at the base."""
        return [0.0, 0.0, 0.0]

    def _base_pose(self):
        """[x, y, yaw] on the map from the odometry's position and orientation."""
        odom = self._port.odom()
        if odom is None:
            return np.zeros(3)
        qx, qy, qz, qw = (float(v) for v in odom["orientation"])
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        position = odom["position"]
        return np.array([float(position[0]), float(position[1]), yaw])

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
        """Send one side's clamped block; the wire takes both blocks, radians in, and
        the port converts to degrees at the only crossing."""
        left = self._gates["left"]._q_cmd if side == "right" else q_cmd
        right = q_cmd if side == "right" else self._gates["right"]._q_cmd
        self._port.publish_arm_traj(left, right)

    # --- the protocol's reset/close pair
    def reset(self) -> None:
        self.new_epoch()

    def close(self) -> None:
        self._closed = True
        self._stop_base()
        self._port.close()

    def grip(self, epoch: int, side: str, closure: float) -> None:
        raise ActionRejected("The kuavo 5W v1 has no gripper channel")

    def ensure_arm_control(self) -> None:
        """The MPC mode and external-control switch, in the order the controller wants.

        BASE_ARM, not ARM_ONLY: the demo drives the base and the arms together, and a
        permission the controller lacks shows up as success with motionless arms.
        """
        if not self._port.set_mpc_ctrl_mode(MPC_BASE_ARM):
            raise ActionRejected("The controller refused the manipulation MPC mode")
        if not self._port.set_arm_ctrl_mode(ARM_EXTERNAL):
            raise ActionRejected("The controller refused the external-control mode")
        self._arm_active = True
