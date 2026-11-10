"""seam③: backend behavior with a stubbed port — epoch refusal, deadman, clamping."""
import math

import pytest

from homebody.backends.kuavo.backend import KuavoBackend
from homebody.backends.kuavo.limits import arm_limits
from homebody.backends.kuavo.profile import urdf_path
from homebody.primitives.actions import ActionRejected

LEFT_LIMITS = arm_limits(urdf_path())["left"]


class FakePort:
    """Records what the backend sends; hands back a canned sensor frame."""

    def __init__(self, joint_q):
        self.joint_q = list(joint_q)
        self.cmd_vel = []
        self.arm_traj = []
        self.clock_value = 100.0
        self.closed = False

    def sensors(self):
        return {"joint_q": self.joint_q, "joint_v": [0.0] * 20}

    def clock(self):
        return self.clock_value

    def publish_cmd_vel(self, forward, lateral, yaw_rate):
        self.cmd_vel.append((forward, lateral, yaw_rate))

    def publish_arm_traj(self, left_deg, right_deg):
        self.arm_traj.append((tuple(left_deg), tuple(right_deg)))

    def close(self):
        self.closed = True


def make_backend(joint_q=None):
    port = FakePort(joint_q=[0.0] * 20 if joint_q is None else joint_q)
    backend = KuavoBackend(port=port, urdf=urdf_path())
    return backend, port


def test_epoch_starts_at_zero_and_new_epoch_advances():
    backend, _ = make_backend()
    assert backend.epoch == 0
    backend.new_epoch()
    assert backend.epoch == 1


def test_stale_epoch_commands_are_refused():
    backend, port = make_backend()
    with pytest.raises(ActionRejected):
        backend.base_velocity(1, 0.1, 0.0, 0.0)
    backend.new_epoch()
    stops = port.cmd_vel.count((0.0, 0.0, 0.0))  # new_epoch stops the base first
    backend.base_velocity(1, 0.1, 0.0, 0.0)
    assert port.cmd_vel[-1] == (0.1, 0.0, 0.0)
    with pytest.raises(ActionRejected):
        backend.base_velocity(0, 0.1, 0.0, 0.0)
    assert len(port.cmd_vel) == stops + 1


def test_non_int_epoch_is_refused():
    backend, _ = make_backend()
    with pytest.raises(ActionRejected):
        backend.base_velocity(True, 0.1, 0.0, 0.0)


def test_base_velocity_caps():
    backend, _ = make_backend()
    backend.new_epoch()
    backend.base_velocity(1, 1.2, 0.0, 0.0)
    with pytest.raises(ActionRejected):
        backend.base_velocity(1, 1.3, 0.0, 0.0)
    with pytest.raises(ActionRejected):
        backend.base_velocity(1, 0.0, 0.0, 1.1)
    with pytest.raises(ActionRejected):
        backend.base_velocity(1, math.nan, 0.0, 0.0)


def test_stop_zeros_the_base():
    backend, port = make_backend()
    backend.new_epoch()
    backend.base_velocity(1, 0.2, 0.0, 0.0)
    backend.stop(1)
    assert port.cmd_vel[-1] == (0.0, 0.0, 0.0)


def test_stop_refuses_stale_epoch():
    backend, port = make_backend()
    backend.new_epoch()
    with pytest.raises(ActionRejected):
        backend.stop(0)
    # the only zero commands so far came from new_epoch itself
    assert all(v != (0.0, 0.0, 0.0) for v in port.cmd_vel[1:])


def test_arm_target_over_limit_is_refused():
    backend, port = make_backend()
    backend.new_epoch()
    _, hi = LEFT_LIMITS[0]
    with pytest.raises(ActionRejected, match="joint limits"):
        backend.arm_target(1, "left", [hi + 0.1, *([0.0] * 6)])
    assert port.arm_traj == []


def test_arm_target_within_limits_clamps_to_rate_and_margin():
    backend, port = make_backend()
    backend.new_epoch()
    _, hi = LEFT_LIMITS[0]
    backend.arm_target(1, "left", [hi, *([0.0] * 6)])
    assert port.arm_traj, "a legal target must reach the wire"
    left_deg, _ = port.arm_traj[-1]
    # the first step is rate-limited from the primed zero pose, and stays off the stop
    assert left_deg[0] < math.degrees(hi)


def test_arm_target_nan_refused():
    backend, _ = make_backend()
    backend.new_epoch()
    with pytest.raises(ActionRejected):
        backend.arm_target(1, "left", [float("nan"), *([0.0] * 6)])
    with pytest.raises(ActionRejected):
        backend.arm_target(1, "left", [0.0] * 6)


def test_arm_target_before_any_sensor_frame_refused():
    backend, port = make_backend()
    stub = type("NoSensorPort", (), {
        "sensors": lambda self: None, "clock": lambda self: 0.0,
        "publish_cmd_vel": lambda self, *v: port.cmd_vel.append(v),
        "publish_arm_traj": lambda self, l, r: port.arm_traj.append((l, r)),
        "close": lambda self: None,
    })()
    backend._port = stub
    backend.new_epoch()
    with pytest.raises(ActionRejected, match="sensor"):
        backend.arm_target(1, "left", [0.0] * 7)


def test_grip_refused_on_v1():
    backend, _ = make_backend()
    backend.new_epoch()
    with pytest.raises(ActionRejected):
        backend.grip(1, "left", 0.5)


def test_snapshot_refused_until_m3():
    backend, _ = make_backend()
    with pytest.raises(RuntimeError, match="M3"):
        backend.snapshot()


def test_reset_starts_new_epoch():
    backend, port = make_backend()
    backend.new_epoch()
    backend.reset()
    assert backend.epoch == 2
    assert port.cmd_vel[-1] == (0.0, 0.0, 0.0)


def test_close_stops_and_closes_port():
    backend, port = make_backend()
    backend.new_epoch()
    backend.close()
    assert port.closed
    assert port.cmd_vel[-1] == (0.0, 0.0, 0.0)
