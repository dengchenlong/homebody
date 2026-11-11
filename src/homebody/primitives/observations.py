"""Immutable, synchronized sensor records a backend fills and skills read."""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from homebody.helpers.geometry import points_in, unproject


def frozen_array(value, shape=None):
    array = np.array(value, copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"Expected shape {shape}, received {array.shape}")
    array.setflags(write=False)
    return array


class ObservationError(ValueError):
    """Missing or unusable sensor evidence."""


@dataclass
class FreshEvidence:
    """Counts consecutive equal values over strictly newer frames of one epoch."""
    epoch: int
    sequence: int = -1
    time: float = -float("inf")
    value: object = None
    count: int = 0

    def update(self, frame, value):
        if (frame.epoch != self.epoch or frame.sequence <= self.sequence or
                frame.time <= self.time):
            return None
        self.sequence, self.time = frame.sequence, frame.time
        self.count = (self.count + 1 if value == self.value else 1) if value else 0
        self.value = value
        return self.count


MIN_TARGET_PIXELS = 12
ARMS = slice(15, 29)
ARM = {"left": slice(15, 22), "right": slice(22, 29)}


@dataclass(frozen=True)
class JointLayout:
    """One robot's joint vector: its length and each arm's slice inside it.

    Backends supply the layout their wire speaks; the observation records read it instead
    of hard-coded offsets. The G1 layout (ARMS/ARM above) stays the default.
    """
    joints: int
    arms: slice
    arm: Mapping[str, slice]

    def __post_init__(self):
        if self.joints <= 0 or self.arms.stop > self.joints:
            raise ValueError("Layout arms must fall inside the joint vector")
        sides = tuple(self.arm)
        if sides != ("left", "right"):
            raise ValueError("Layout arms must be exactly left and right")
        for side, arm_slice in self.arm.items():
            if arm_slice.stop - arm_slice.start != 7 or arm_slice.stop > self.joints:
                raise ValueError(f"The {side} arm slice must cover seven joints")


def g1_layout() -> JointLayout:
    """The G1's 29 joints in controller order, arms at 15..28."""
    return JointLayout(29, ARMS, dict(ARM))


def kuavo_layout() -> JointLayout:
    """The Kuavo 5W's 20-sensor vector: lift 4, arms 4..17, head 2."""
    return JointLayout(20, slice(4, 18), {"left": slice(4, 11), "right": slice(11, 18)})

@dataclass(frozen=True)
class HandState:
    """One hand's closure in [0, 1] and per-finger evidence ordered thumb, index, middle.

    Gaps are fractions of the calibrated closure ray, not distances.
    """
    joints: np.ndarray
    closure: float
    finger_effort_fraction: np.ndarray
    finger_closure_gap: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "joints", frozen_array(self.joints, (7,)))
        for name in ("finger_effort_fraction", "finger_closure_gap"):
            array = frozen_array(getattr(self, name), (3,))
            if not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError("Finger evidence must be finite and nonnegative")
            object.__setattr__(self, name, array)

    @property
    def effort_fraction(self):
        return float(min(self.finger_effort_fraction[0], max(self.finger_effort_fraction[1:])))

    @property
    def closure_gap(self):
        return float(min(self.finger_closure_gap[0], max(self.finger_closure_gap[1:])))

    def retained(self, settings):
        paired = ((self.finger_effort_fraction > settings.hold_effort_fraction) &
                  (self.finger_closure_gap > settings.hold_closure_gap))
        return bool(paired[0] and np.any(paired[1:]))


@dataclass(frozen=True)
class NavigationMap:
    occupied: np.ndarray
    T_map_px: np.ndarray
    resolution: float

    def __post_init__(self):
        object.__setattr__(self, "occupied", frozen_array(self.occupied).astype(bool))
        self.occupied.setflags(write=False)
        object.__setattr__(self, "T_map_px", frozen_array(self.T_map_px, (3, 3)))
        if self.occupied.ndim != 2 or self.resolution <= 0:
            raise ValueError("Invalid navigation raster")


class RobotState:
    """Fields Measurement and Frame share: epoch, sequence, time, poses, joints and hands."""

    @property
    def frame_id(self):
        return self.epoch, self.sequence

    def arm_joints(self, side):
        return self.joints[self.layout.arm[side]]

    def palm(self, arm, side):
        """World pose of this side's palm from the measured arm joints."""
        return self.world_torso @ arm.forward(self.arm_joints(side))[0]


@dataclass(frozen=True)
class Measurement(RobotState):
    """The robot's joints, hands and poses at one instant, without the camera image."""
    epoch: int
    sequence: int
    time: float
    world_camera: np.ndarray
    world_torso: np.ndarray
    base_pose: np.ndarray
    joints: np.ndarray
    hands: Mapping[str, HandState]
    layout: JointLayout | None = None

    def __post_init__(self):
        if self.layout is None:
            object.__setattr__(self, "layout", g1_layout())
        for name, shape in (("world_camera", (4, 4)), ("world_torso", (4, 4)), ("base_pose", (3,)),
                            ("joints", (self.layout.joints,))):
            object.__setattr__(self, name, frozen_array(getattr(self, name), shape))
        object.__setattr__(self, "hands", MappingProxyType(dict(self.hands)))


@dataclass(frozen=True)
class Frame(RobotState):
    """One synchronized sensor packet.

    rgb, depth (metres along the optical axis, NaN where invalid) and labels (a positive
    integer per scene object, -1 on the robot, 0 elsewhere) share one image grid.
    world_camera is x right, y down, z ahead. base_pose is [x, y, yaw] on the map. joints
    are the G1's 29 in controller order, arms at ARMS. hands has "left" and "right".
    """
    epoch: int
    sequence: int
    time: float
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    world_camera: np.ndarray
    world_torso: np.ndarray
    base_pose: np.ndarray
    joints: np.ndarray
    hands: Mapping[str, HandState]
    labels: np.ndarray
    navigation: NavigationMap
    layout: JointLayout | None = None

    def __post_init__(self):
        if self.layout is None:
            object.__setattr__(self, "layout", g1_layout())
        height, width = np.shape(self.depth)
        for name, shape in (("rgb", (height, width, 3)), ("depth", (height, width)),
                            ("labels", (height, width)), ("intrinsics", (3, 3)),
                            ("world_camera", (4, 4)), ("world_torso", (4, 4)),
                            ("base_pose", (3,)), ("joints", (self.layout.joints,))):
            object.__setattr__(self, name, frozen_array(getattr(self, name), shape))
        object.__setattr__(self, "hands", MappingProxyType(dict(self.hands)))

    def normalized_pixel(self, point):
        if (not isinstance(point, (list, tuple)) or len(point) != 2 or
            any(type(value) not in (int, float) or
                not 0 <= value <= 1000 for value in point)):
            raise ObservationError("point_normalized_1000 requires two finite numbers in [0,1000]")
        height, width = self.depth.shape
        return tuple(np.floor(np.asarray(point) * [width - 1, height - 1] / 1000).astype(int).tolist())

    def point(self, u, v):
        if type(u) is not int or type(v) is not int:
            raise ObservationError("Click coordinates must be integer image pixels")
        if not (0 <= v < self.depth.shape[0] and 0 <= u < self.depth.shape[1]):
            raise ObservationError("Click is outside the captured image")
        z = float(self.depth[v, u])
        if not np.isfinite(z) or z <= 0:
            raise ObservationError("Click has no metric depth")
        p = [(u - self.intrinsics[0, 2]) * z / self.intrinsics[0, 0],
             (v - self.intrinsics[1, 2]) * z / self.intrinsics[1, 1], z]
        return points_in(self.world_camera, p)

    def select(self, u, v):
        self.point(u, v)
        label = int(self.labels[v, u])
        if label <= 0:
            raise ObservationError("Click is background; no visible target selected")
        return label

    def visible(self, label):
        """True when the label has enough valid depth pixels for geometry."""
        return len(unproject(self.depth, self.intrinsics, self.labels == label)) >= MIN_TARGET_PIXELS

    def target_points(self, label):
        cloud = unproject(self.depth, self.intrinsics, self.labels == label)
        if len(cloud) < MIN_TARGET_PIXELS:
            raise ObservationError("Selected target is no longer sufficiently visible")
        return points_in(self.world_camera, cloud)
