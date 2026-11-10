"""seam④: the port's message-construction contract, rospy and kuavo_msgs stubbed out.

Pins what reaches the wire: degrees on /kuavo_arm_traj (14, left then right), body-frame
twist on /cmd_vel, and the mode services called before any arm command.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from homebody.backends.kuavo.backend import KuavoBackend
from homebody.backends.kuavo.port import KuavoPort
from homebody.backends.kuavo.profile import (
    SERVICE_ARM_CTRL_MODE,
    SERVICE_MPC_CTRL_MODE,
    urdf_path,
)


class RosStub:
    """Just enough rospy: record publishers, serve the two mode services."""

    def __init__(self):
        self.published: dict[str, list] = {}
        self.service_calls: list[tuple[str, int]] = []
        self.services = {SERVICE_ARM_CTRL_MODE, SERVICE_MPC_CTRL_MODE}

    class Time:
        @staticmethod
        def now():
            return SimpleNamespace(to_sec=lambda: 0.0)

    def Subscriber(self, topic, msg_type, callback, queue_size=1, tcp_nodelay=False):
        return SimpleNamespace(topic=topic, unregister=lambda: None)

    def Publisher(self, topic, msg_type, queue_size=1, tcp_nodelay=False):
        records = self.published.setdefault(topic, [])
        return SimpleNamespace(
            publish=lambda msg, _t=topic: records.append(msg))

    def wait_for_service(self, name, timeout=5.0):
        if name not in self.services:
            raise RuntimeError(f"no service {name}")

    def ServiceProxy(self, name, srv_type):
        def proxy(request):
            self.service_calls.append((name, request.control_mode))
            return SimpleNamespace(result=True, success=True)
        return proxy


class MsgStub:
    """Just enough of the message shapes the port touches."""

    class Twist:
        def __init__(self):
            self.linear = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.angular = SimpleNamespace(x=0.0, y=0.0, z=0.0)

    class JointState:
        def __init__(self):
            self.name, self.position = [], []

    class sensorsData:
        class header:
            stamp = None
        sensor_time = SimpleNamespace(to_sec=lambda: 0.0)

    class changeArmCtrlMode:
        @staticmethod
        def _request_class():
            return SimpleNamespace(control_mode=0)

    class changeTorsoCtrlMode:
        @staticmethod
        def _request_class():
            return SimpleNamespace(control_mode=0)


def install_stubs(monkeypatch):
    ros = RosStub()
    monkeypatch.setitem(__import__("sys").modules, "rospy", ros)
    monkeypatch.setitem(__import__("sys").modules, "geometry_msgs.msg",
                        SimpleNamespace(Twist=MsgStub.Twist))
    monkeypatch.setitem(__import__("sys").modules, "nav_msgs.msg",
                        SimpleNamespace(Odometry=object))
    monkeypatch.setitem(__import__("sys").modules, "sensor_msgs.msg",
                        SimpleNamespace(JointState=MsgStub.JointState))
    monkeypatch.setitem(__import__("sys").modules, "std_msgs.msg",
                        SimpleNamespace(Float32=object))
    monkeypatch.setitem(__import__("sys").modules, "std_srvs.srv",
                        SimpleNamespace(Empty=object, Trigger=object))
    monkeypatch.setitem(__import__("sys").modules, "kuavo_msgs.msg",
                        SimpleNamespace(sensorsData=MsgStub.sensorsData))
    monkeypatch.setitem(__import__("sys").modules, "kuavo_msgs.srv",
                        SimpleNamespace(changeArmCtrlMode=MsgStub.changeArmCtrlMode,
                                        changeTorsoCtrlMode=MsgStub.changeTorsoCtrlMode))
    return ros


def make_port(monkeypatch):
    ros = install_stubs(monkeypatch)
    port = KuavoPort()
    return port, ros


def feed_sensors(port, joint_q):
    """Push one sensor frame through the port's real callback, as rospy would."""
    frame = SimpleNamespace(
        header=SimpleNamespace(stamp=None),
        sensor_time=SimpleNamespace(to_sec=lambda: 0.0),
        joint_data=SimpleNamespace(joint_q=list(joint_q), joint_v=[0.0] * 20,
                                   joint_torque=[0.0] * 20),
        imu_data=SimpleNamespace(quat=[0.0] * 4, gyroscope=[0.0] * 3),
    )
    port._on_sensors(frame)


def test_arm_traj_reaches_the_wire_in_degrees(monkeypatch):
    port, ros = make_port(monkeypatch)
    backend = KuavoBackend(port=port, urdf=urdf_path())
    backend.new_epoch()
    feed_sensors(port, [0.0] * 20)
    backend.arm_target(1, "left", [0.2, 0.0, -0.5, -0.3, 0.0, 0.0, 0.0])
    msgs = ros.published["/kuavo_arm_traj"]
    assert len(msgs) == 1
    msg = msgs[0]
    assert len(msg.position) == 14  # 7 left + 7 right
    assert msg.name[:7] == [f"zarm_l{i}_joint" for i in range(1, 8)]
    assert msg.name[7:] == [f"zarm_r{i}_joint" for i in range(1, 8)]
    # 0.2 rad was converted to degrees, not sent raw; the far target is rate-clamped
    assert msg.position[0] == pytest.approx(math.degrees(0.2))
    assert msg.position[2] == pytest.approx(math.degrees(-0.21), abs=1e-3)  # one step's worth
    assert msg.position[7:] == pytest.approx([0.0] * 7)  # the idle right block rides along


def test_cmd_vel_is_body_frame_twist(monkeypatch):
    port, ros = make_port(monkeypatch)
    backend = KuavoBackend(port=port, urdf=urdf_path())
    backend.new_epoch()
    backend.base_velocity(1, 0.2, 0.05, -0.3)
    msg = ros.published["/cmd_vel"][-1]
    assert (msg.linear.x, msg.linear.y, msg.angular.z) == pytest.approx((0.2, 0.05, -0.3))


def test_mode_services_precede_the_first_arm_command(monkeypatch):
    port, ros = make_port(monkeypatch)
    backend = KuavoBackend(port=port, urdf=urdf_path())
    backend.new_epoch()
    feed_sensors(port, [0.0] * 20)
    backend.arm_target(1, "left", [0.0] * 7)
    backend.ensure_arm_control()
    modes = {name: mode for name, mode in ros.service_calls}
    assert modes[SERVICE_MPC_CTRL_MODE] == 3  # BASE_ARM, or the arm silently won't move
    assert modes[SERVICE_ARM_CTRL_MODE] == 2  # external control
    # the first external-control switch must come before the first arm publish would
    backend.arm_target(1, "left", [0.0] * 7)
    backend.arm_target(1, "left", [0.0] * 7)


def test_arm_ctrl_mode_refusal_raises(monkeypatch):
    port, ros = make_port(monkeypatch)
    ros.services.discard(SERVICE_ARM_CTRL_MODE)
    backend = KuavoBackend(port=port, urdf=urdf_path())
    backend.new_epoch()
    with pytest.raises(RuntimeError, match="service"):
        backend.ensure_arm_control()


def test_ensure_arm_control_is_idempotent_enough(monkeypatch):
    port, ros = make_port(monkeypatch)
    backend = KuavoBackend(port=port, urdf=urdf_path())
    backend.new_epoch()
    backend.ensure_arm_control()
    before = len(ros.service_calls)
    backend.ensure_arm_control()
    assert len(ros.service_calls) == before + 2  # re-asserted, not cached away
