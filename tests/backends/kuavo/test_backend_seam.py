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
        self.odom_value = None
        self.camera_value = None

    def sensors(self):
        return {"joint_q": self.joint_q, "joint_v": [0.0] * 20}

    def camera(self):
        return self.camera_value

    def odom(self):
        return self.odom_value

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


def test_snapshot_returns_a_blank_frame_until_a_camera_is_wired():
    """M1's placeholder refused; M3 keeps the loop alive with a valid, empty frame."""
    backend, _ = make_backend()
    frame = backend.snapshot()
    assert frame.joints.shape == (20,)


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


def test_gate_tolerates_a_slow_arm_that_keeps_converging():
    """The demo failure: the sim arm tracks the 1.05 rad/s command ramp with lag, so the
    gap grows by (almost exactly) max_delta per tick and the strict `<` flagged a
    healthy arm as not-following. Convergence counts: only sustained divergence refuses."""
    import numpy as np

    from homebody.backends.kuavo.safety import ArmGate

    gate = ArmGate(lower=[-1.0] * 7, upper=[1.0] * 7)
    measured = np.array([0.262, 0.001, 0.0, -0.524, 0.0, 0.0, 0.006])
    preset = np.array([0.0, 0.3, 0.0, -0.6, 0.0, -0.3, 0.0])
    gate.prime(measured)
    gate.arm(ramp_s=2.0)
    arm = measured.copy()
    dt = 0.2
    for _ in range(200):
        _, following = gate(preset, arm, dt)
        assert following, f"refused while the arm was converging (weight={gate.weight})"
        arm = arm + 0.5 * (gate._q_cmd - arm)  # a lagging but honest arm
        dt = 0.01


def test_gate_flags_a_stuck_arm_once_the_gap_diverges():
    """The inverse: an arm that never moves while commands march away must refuse."""
    import numpy as np

    from homebody.backends.kuavo.safety import ArmGate

    gate = ArmGate(lower=[-1.0] * 7, upper=[1.0] * 7)
    measured = np.array([0.262, 0.001, 0.0, -0.524, 0.0, 0.0, 0.006])
    preset = np.array([0.0, 0.3, 0.0, -0.6, 0.0, -0.3, 0.0])
    gate.prime(measured)
    gate.arm(ramp_s=2.0)
    arm = measured.copy()
    stuck = None
    for tick in range(200):
        _, following = gate(preset, arm, 0.01)
        if not following:
            stuck = tick
            break
        # dead arm: nothing moves
    assert stuck is not None, "a motionless arm was never flagged"
    assert stuck < 100


def test_gate_survives_the_deadman_step_boundary():
    """The demo death: tick0 uses the deadman dt (0.2 s) so the command jumps 0.21 rad;
    tick1 adds one 0.0105 step. If the sim arm has not measurably moved yet, the gap
    equals last_gap + max_delta exactly and the strict `<` refused a healthy arm. The
    judge must compare against the best gap seen, with divergence slack, not a strict
    per-tick inequality."""
    import numpy as np

    from homebody.backends.kuavo.safety import ArmGate

    gate = ArmGate(lower=[-1.0] * 7, upper=[1.0] * 7)
    measured = np.array([0.262, 0.001, 0.0, -0.524, 0.0, 0.0, 0.006])
    preset = np.array([0.0, 0.3, 0.0, -0.6, 0.0, -0.3, 0.0])
    gate.prime(measured)
    gate.arm(ramp_s=2.0)
    arm = measured.copy()
    gate(preset, arm, 0.2)  # tick0: the deadman-sized step
    _, following = gate(preset, arm, 0.01)  # tick1: arm not moved yet
    assert following, "the step boundary must not kill a converging arm"
