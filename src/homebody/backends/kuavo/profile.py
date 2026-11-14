"""The Kuavo 5W wheel-arm: topic names, joint tables and motion limits.

Facts are pinned from `kuavo_ws/src/kuavo_assets` (biped_s62, ROBOT_VERSION=62) and the
leju SDK; the arm trajectory topic speaks degrees, the sensor stream radians.
"""
from __future__ import annotations

from pathlib import Path

# --- topics and services (same on sim and real, per the leju SDK core/ros/control.py)
TOPIC_SENSORS = "/sensors_data_raw"  # kuavo_msgs/sensorsData: IMU + 20 joint states
TOPIC_ODOM = "/odom"  # nav_msgs/Odometry
TOPIC_CMD_VEL = "/cmd_vel"  # geometry_msgs/Twist; the controller stops the base 1 s after the last one
TOPIC_ARM_TRAJ = "/kuavo_arm_traj"  # sensor_msgs/JointState, 14 joints, DEGREES
TOPIC_ARM_REACH = "/lb_arm_joint_reach_time/"  # + joint name -> Float32, the Ruckig plan's duration
SERVICE_ARM_CTRL_MODE = "/wheel_arm_change_arm_ctrl_mode"  # 0 fixed / 1 auto-swing / 2 external
SERVICE_MPC_CTRL_MODE = "/mobile_manipulator_mpc_control"  # 1 arm-only / 3 base+arm
SERVICE_ENABLE_CONTROL = "/enable_control"  # std_srvs/Trigger, master enable

# --- head camera (real robot; assets/robot/kuavo_5w/camera.md — the sim publishes none)
TOPIC_CAMERA_RGB = "/camera/color/image_raw"  # sensor_msgs/Image, rgb8/bgr8
TOPIC_DEPTH_ALIGNED = "/camera/aligned_depth_to_color/image_raw"  # shares the rgb grid
TOPIC_CAMERA_COLOR_INFO = "/camera/color/camera_info"  # RealSense factory calibration

ARM_FIXED, ARM_AUTO_SWING, ARM_EXTERNAL = 0, 1, 2
MPC_NO_CONTROL, MPC_ARM_ONLY, MPC_BASE_ONLY, MPC_BASE_ARM = 0, 1, 2, 3

CONTROL_HZ = 100  # Hz; the controller's arm trajectory tick
BASE_MAX_LINEAR_MPS = 1.2  # kuavo_v62 VelocityLimit
BASE_MAX_ANGULAR_RPS = 1.0  # rad/s cap kept well under the VelocityLimit's 50 deg/s
DEADMAN_S = 0.2  # s between renewals before the gate stops publishing

# --- joint tables
# The 20-sensor order: lift column (4), arms (14), head (2); waist is 0-dof on the 5W.
SENSOR_JOINTS: tuple[str, ...] = (
    "knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint",
    *[f"zarm_l{i}_joint" for i in range(1, 8)],
    *[f"zarm_r{i}_joint" for i in range(1, 8)],
    "zhead_1_joint", "zhead_2_joint",
)
assert len(SENSOR_JOINTS) == 20
ARM_SLICE = slice(4, 18)  # the 14 arm joints inside the 20-sensor vector
LEFT_ARM_JOINTS: tuple[str, ...] = tuple(f"zarm_l{i}_joint" for i in range(1, 8))
RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(f"zarm_r{i}_joint" for i in range(1, 8))
ARM_JOINTS: dict[str, tuple[str, ...]] = {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS}
assert len(LEFT_ARM_JOINTS) == len(RIGHT_ARM_JOINTS) == 7

# homebody's arm order is the URDF order; the wire order matches it one-to-one, so the
# mapping is the identity. Kept as an explicit table so seam tests pin it.
WIRE_JOINT_ORDER: dict[str, tuple[str, ...]] = {
    "left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS,
}


def urdf_path() -> Path:
    """The biped_s62 URDF the limits module parses; overridable through the mount point."""
    return Path("/home/dev_dcl/kuavo_ws/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf")


__all__ = [
    "ARM_AUTO_SWING", "ARM_EXTERNAL", "ARM_FIXED", "ARM_JOINTS", "ARM_SLICE",
    "BASE_MAX_ANGULAR_RPS", "BASE_MAX_LINEAR_MPS", "CONTROL_HZ", "DEADMAN_S",
    "LEFT_ARM_JOINTS", "MPC_ARM_ONLY", "MPC_BASE_ARM", "MPC_BASE_ONLY", "MPC_NO_CONTROL",
    "RIGHT_ARM_JOINTS", "SENSOR_JOINTS", "SERVICE_ARM_CTRL_MODE", "SERVICE_ENABLE_CONTROL",
    "SERVICE_MPC_CTRL_MODE", "TOPIC_ARM_REACH", "TOPIC_ARM_TRAJ", "TOPIC_CAMERA_COLOR_INFO",
    "TOPIC_CAMERA_RGB", "TOPIC_CMD_VEL", "TOPIC_DEPTH_ALIGNED", "TOPIC_ODOM",
    "TOPIC_SENSORS", "WIRE_JOINT_ORDER", "urdf_path",
]
