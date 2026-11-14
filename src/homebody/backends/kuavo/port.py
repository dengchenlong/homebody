"""The Kuavo ROS1 port: topic and service access, isolated behind a lazy rospy import.

The machine this runs on has no ROS1; the port is constructed inside the kuavo
controller container (kuavo_opensource_mpc_wbc_img), where rospy and kuavo_msgs exist.
Every import failure names the container instead of a traceback.
"""
from __future__ import annotations

import os
import threading

from . import camera
from .joints import to_wire_degrees
from .profile import (
    SERVICE_ARM_CTRL_MODE,
    SERVICE_MPC_CTRL_MODE,
    TOPIC_ARM_REACH,
    TOPIC_ARM_TRAJ,
    TOPIC_CAMERA_COLOR_INFO,
    TOPIC_CAMERA_RGB,
    TOPIC_CMD_VEL,
    TOPIC_DEPTH_ALIGNED,
    TOPIC_ODOM,
    TOPIC_SENSORS,
)

__all__ = ["KuavoPort"]


def _raise_ros_missing() -> None:
    raise ImportError(
        "KuavoPort needs rospy and kuavo_msgs; run inside the kuavo controller container"
        " (kuavo_opensource_mpc_wbc_img), not on this host"
    )


def _patch_rospy_findcaller() -> None:
    """rosgraph 1.17's RospyLogger.findCaller spins forever on Python >= 3.11: once the
    walk reaches the top frame, `if f.f_back` never advances and the `hasattr(f, "f_code")`
    loop never exits, so init_node burns a core inside its first log call. Python 3.8's
    logging walks a different stack shape and never hits it (the M1 demo ran clean).
    The patched walk exits when the chain ends, exactly the upstream intent."""
    import sys

    if (sys.version_info.major, sys.version_info.minor) < (3, 11):
        return
    import inspect

    try:
        import rosgraph.roslogging
    except ImportError:
        return

    if getattr(rosgraph.roslogging.RospyLogger.findCaller, "_homebody_patched", False):
        return

    def findCaller(self, *args, **kwargs):
        file_name, lineno, func_name = super(rosgraph.roslogging.RospyLogger, self).findCaller(
            *args, **kwargs)[:3]
        file_name = os.path.normcase(file_name)
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
        while hasattr(frame, "f_code"):
            code = frame.f_code
            if os.path.normcase(code.co_filename) == file_name \
                    and frame.f_lineno == lineno and code.co_name == func_name:
                break
            if frame.f_back is None:
                break
            frame = frame.f_back
        code = frame.f_code if frame is not None else None
        name = code.co_name if code is not None else func_name
        try:
            class_name = frame.f_locals["self"].__class__.__name__
            name = f"{class_name}.{name}"
        except (KeyError, TypeError):
            pass
        return code.co_filename if code is not None else file_name, \
            frame.f_lineno if frame is not None else lineno, name, None

    findCaller._homebody_patched = True
    rosgraph.roslogging.RospyLogger.findCaller = findCaller


class KuavoPort:
    """The one ROS1 surface the backend speaks through. Construction attaches to the
    running controller; every call afterwards is plain publish/subscribe."""

    def __init__(self) -> None:
        try:
            import rospy
            from geometry_msgs.msg import Twist
            from kuavo_msgs.msg import sensorsData
            from kuavo_msgs.srv import changeArmCtrlMode, changeTorsoCtrlMode
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import CameraInfo, Image, JointState
            from std_msgs.msg import Float32
            from std_srvs.srv import Empty, Trigger
        except ImportError:
            _raise_ros_missing()
            raise  # unreachable; keeps the names defined for type checkers
        _patch_rospy_findcaller()
        # Every publish/subscribe needs a node; the port owns the name so callers
        # never touch rospy. disable_signals keeps this safe in threaded contexts.
        rospy.init_node("homebody_kuavo", disable_signals=True, anonymous=True)
        self._rospy = rospy
        self._Twist, self._JointState, self._Odometry, self._Float32 = (
            Twist, JointState, Odometry, Float32)
        self._sensorsData, self._Trigger, self._Empty = sensorsData, Trigger, Empty
        self._changeArmCtrlMode, self._changeTorsoCtrlMode = changeArmCtrlMode, changeTorsoCtrlMode
        self._lock = threading.Lock()
        self._sensors: dict | None = None
        self._odom: dict | None = None
        self._camera: dict | None = None
        self._camera_stamp: float | None = None
        self._reach: dict[str, float] = {}
        self._subs = [
            rospy.Subscriber(TOPIC_SENSORS, sensorsData, self._on_sensors, queue_size=1,
                             tcp_nodelay=True),
            rospy.Subscriber(TOPIC_ODOM, Odometry, self._on_odom, queue_size=1, tcp_nodelay=True),
            rospy.Subscriber(TOPIC_CAMERA_RGB, Image, self._on_rgb, queue_size=1,
                             tcp_nodelay=True),
            rospy.Subscriber(TOPIC_DEPTH_ALIGNED, Image, self._on_depth, queue_size=1,
                             tcp_nodelay=True),
            rospy.Subscriber(TOPIC_CAMERA_COLOR_INFO, CameraInfo, self._on_color_info,
                             queue_size=1),
        ]
        self._pub_cmd_vel = rospy.Publisher(TOPIC_CMD_VEL, Twist, queue_size=1, tcp_nodelay=True)
        self._pub_arm_traj = rospy.Publisher(TOPIC_ARM_TRAJ, JointState, queue_size=1,
                                             tcp_nodelay=True)

    # --- callbacks
    @staticmethod
    def _vector3(v) -> list[float]:
        return [float(v.x), float(v.y), float(v.z)]

    def _on_sensors(self, msg) -> None:
        quat = msg.imu_data.quat  # geometry_msgs/Quaternion: x y z w attributes
        with self._lock:
            self._sensors = {
                "stamp": msg.header.stamp.to_sec() if msg.header.stamp else msg.sensor_time.to_sec(),
                "joint_q": [float(v) for v in msg.joint_data.joint_q],
                "joint_v": [float(v) for v in msg.joint_data.joint_v],
                "joint_torque": [float(v) for v in msg.joint_data.joint_torque],
                "imu_quat": [float(quat.x), float(quat.y), float(quat.z), float(quat.w)],
                "imu_gyro": self._vector3(msg.imu_data.gyro),
            }

    def _on_odom(self, msg) -> None:
        with self._lock:
            self._odom = {
                "stamp": msg.header.stamp.to_sec(),
                "position": [msg.pose.pose.position.x, msg.pose.pose.position.y,
                             msg.pose.pose.position.z],
                "orientation": [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w],
                "linear": [msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                           msg.twist.twist.linear.z],
                "angular": [msg.twist.twist.angular.x, msg.twist.twist.angular.y,
                            msg.twist.twist.angular.z],
            }

    def _on_reach(self, name):
        def callback(msg) -> None:
            with self._lock:
                self._reach[name] = float(msg.data)
        return callback

    # --- camera callbacks: the RealSense streams, decoded to plain arrays (camera.md).
    # A frame that fails to decode never replaces the last good one.
    def _on_rgb(self, msg) -> None:
        rgb = camera.decode_rgb(msg)
        if rgb is None:
            return
        with self._lock:
            self._camera = {**(self._camera or {}), "rgb": rgb}

    def _on_depth(self, msg) -> None:
        depth = camera.decode_depth(msg)
        if depth is None:
            return
        with self._lock:
            self._camera = {**(self._camera or {}), "depth": depth}
            self._camera_stamp = self._rospy.Time.now().to_sec()

    def _on_color_info(self, msg) -> None:
        intrinsics = camera.decode_intrinsics(msg)
        if intrinsics is None:
            return
        with self._lock:
            self._camera = {**(self._camera or {}), "intrinsics": intrinsics}

    # --- reads
    def sensors(self) -> dict | None:
        """The newest sensor frame, or None before the first one arrives."""
        with self._lock:
            return self._sensors

    def camera(self) -> dict | None:
        """The newest decoded camera frame {rgb, depth, intrinsics}, or None before the
        first one arrives. One dict per call; a later callback cannot mutate it."""
        with self._lock:
            return dict(self._camera) if self._camera is not None else None

    def camera_stamp(self) -> float | None:
        """When the newest depth frame arrived on the ROS clock, or None."""
        with self._lock:
            return self._camera_stamp

    def odom(self) -> dict | None:
        """The newest odometry, or None before the first one arrives."""
        with self._lock:
            return self._odom

    def arm_reach_time(self, joint_name: str) -> float | None:
        """The Ruckig plan's duration for one arm joint, or None when unknown."""
        return self._reach.get(joint_name)

    def watch_reach(self, joint_names: tuple[str, ...]) -> None:
        """Subscribe to the reach-time topic for each named arm joint."""
        for name in joint_names:
            self._rospy.Subscriber(f"{TOPIC_ARM_REACH}{name}", self._Float32,
                                   self._on_reach(name), queue_size=1)

    def clock(self) -> float:
        return self._rospy.Time.now().to_sec()

    # --- writes
    def publish_cmd_vel(self, forward: float, lateral: float, yaw_rate: float) -> None:
        msg = self._Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = forward, lateral, yaw_rate
        self._pub_cmd_vel.publish(msg)

    def publish_arm_traj(self, left_rad, right_rad) -> None:
        """One 14-position JointState: left 7 then right 7. The wire speaks degrees, so
        the radians cross through joints.to_wire_degrees here, the only crossing."""
        msg = self._JointState()
        msg.name = [f"zarm_l{i}_joint" for i in range(1, 8)] \
            + [f"zarm_r{i}_joint" for i in range(1, 8)]
        msg.position = to_wire_degrees("left", left_rad) + to_wire_degrees("right", right_rad)
        self._pub_arm_traj.publish(msg)

    # --- services
    def set_arm_ctrl_mode(self, mode: int) -> bool:
        """0 fixed / 1 auto-swing / 2 external, through the wheel-arm service."""
        response = self._service(SERVICE_ARM_CTRL_MODE, self._changeArmCtrlMode, mode)
        return bool(response.result)

    def set_mpc_ctrl_mode(self, mode: int) -> bool:
        """1 arm-only / 3 base+arm, the permission the arm obeys."""
        response = self._service(SERVICE_MPC_CTRL_MODE, self._changeTorsoCtrlMode, mode)
        return bool(response.result)

    def _service(self, name: str, srv_type, mode: int):
        self._rospy.wait_for_service(name, timeout=5.0)
        proxy = self._rospy.ServiceProxy(name, srv_type)
        request = srv_type._request_class()
        request.control_mode = mode
        return proxy(request)

    def enable_control(self) -> bool:
        """The master enable, a std_srvs/Trigger."""
        return bool(self._service("/enable_control", self._Trigger,
                                  self._Trigger._request_class()).success)

    def close(self) -> None:
        for sub in getattr(self, "_subs", []):
            sub.unregister()
        self._subs = []
