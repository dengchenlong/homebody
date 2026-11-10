"""The Kuavo ROS1 port: topic and service access, isolated behind a lazy rospy import.

The machine this runs on has no ROS1; the port is constructed inside the kuavo
controller container (kuavo_opensource_mpc_wbc_img), where rospy and kuavo_msgs exist.
Every import failure names the container instead of a traceback.
"""
from __future__ import annotations

import threading

from .joints import rad_to_deg
from .profile import (
    SERVICE_ARM_CTRL_MODE,
    SERVICE_MPC_CTRL_MODE,
    TOPIC_ARM_REACH,
    TOPIC_ARM_TRAJ,
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    TOPIC_SENSORS,
)

__all__ = ["KuavoPort"]


def _raise_ros_missing() -> None:
    raise ImportError(
        "KuavoPort needs rospy and kuavo_msgs; run inside the kuavo controller container"
        " (kuavo_opensource_mpc_wbc_img), not on this host"
    )


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
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float32
            from std_srvs.srv import Empty, Trigger
        except ImportError:
            _raise_ros_missing()
            raise  # unreachable; keeps the names defined for type checkers
        self._rospy = rospy
        self._Twist, self._JointState, self._Odometry, self._Float32 = (
            Twist, JointState, Odometry, Float32)
        self._sensorsData, self._Trigger, self._Empty = sensorsData, Trigger, Empty
        self._changeArmCtrlMode, self._changeTorsoCtrlMode = changeArmCtrlMode, changeTorsoCtrlMode
        self._lock = threading.Lock()
        self._sensors: dict | None = None
        self._odom: dict | None = None
        self._reach: dict[str, float] = {}
        self._subs = [
            rospy.Subscriber(TOPIC_SENSORS, sensorsData, self._on_sensors, queue_size=1,
                             tcp_nodelay=True),
            rospy.Subscriber(TOPIC_ODOM, Odometry, self._on_odom, queue_size=1, tcp_nodelay=True),
        ]
        self._pub_cmd_vel = rospy.Publisher(TOPIC_CMD_VEL, Twist, queue_size=1, tcp_nodelay=True)
        self._pub_arm_traj = rospy.Publisher(TOPIC_ARM_TRAJ, JointState, queue_size=1,
                                             tcp_nodelay=True)

    # --- callbacks
    def _on_sensors(self, msg) -> None:
        with self._lock:
            self._sensors = {
                "stamp": msg.header.stamp.to_sec() if msg.header.stamp else msg.sensor_time.to_sec(),
                "joint_q": [float(v) for v in msg.joint_data.joint_q],
                "joint_v": [float(v) for v in msg.joint_data.joint_v],
                "joint_torque": [float(v) for v in msg.joint_data.joint_torque],
                "imu_quat": [float(v) for v in msg.imu_data.quat],
                "imu_gyro": [float(v) for v in msg.imu_data.gyroscope],
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

    # --- reads
    def sensors(self) -> dict | None:
        """The newest sensor frame, or None before the first one arrives."""
        with self._lock:
            return self._sensors

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

    def publish_arm_traj(self, left_deg, right_deg) -> None:
        """One 14-degree JointState: left 7 then right 7, both in degrees."""
        msg = self._JointState()
        msg.name = [f"zarm_l{i}_joint" for i in range(1, 8)] \
            + [f"zarm_r{i}_joint" for i in range(1, 8)]
        msg.position = [rad_to_deg(v) if abs(v) < 3.5 else float(v) for v in left_deg + right_deg]
        self._pub_arm_traj.publish(msg)

    # --- services
    def call_service(self, name: str, request) -> object:
        self._rospy.wait_for_service(name, timeout=5.0)
        proxy = self._rospy.ServiceProxy(name, type(request._response_class)
                                         if hasattr(request, "_response_class") else None)
        return proxy

    def set_arm_ctrl_mode(self, mode: int) -> bool:
        """0 fixed / 1 auto-swing / 2 external, through the wheel-arm service."""
        request = self._changeArmCtrlMode._request_class()
        request.control_mode = mode
        response = self._service(SERVICE_ARM_CTRL_MODE, self._changeArmCtrlMode, request)
        return bool(response.result)

    def set_mpc_ctrl_mode(self, mode: int) -> bool:
        """1 arm-only / 3 base+arm, the permission the arm obeys."""
        request = self._changeTorsoCtrlMode._request_class()
        request.control_mode = mode
        response = self._service(SERVICE_MPC_CTRL_MODE, self._changeTorsoCtrlMode, request)
        return bool(response.result)

    def _service(self, name: str, srv_type, request):
        self._rospy.wait_for_service(name, timeout=5.0)
        proxy = self._rospy.ServiceProxy(name, srv_type)
        return proxy(request)

    def enable_control(self) -> bool:
        """The master enable, a std_srvs/Trigger."""
        return bool(self._service("/enable_control", self._Trigger,
                                  self._Trigger._request_class()).success)

    def close(self) -> None:
        for sub in getattr(self, "_subs", []):
            sub.unregister()
        self._subs = []
