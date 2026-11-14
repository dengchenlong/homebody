"""seam: the Kuavo backend conforms to the primitives Robot protocol. Only external
behaviour is asserted — typed results, typed refusals, stale-epoch refusal — over a
stubbed port, with no ROS daemon and no MuJoCo."""
import numpy as np
import pytest
from test_backend_seam import FakePort

from homebody.backends.kuavo.backend import KuavoBackend
from homebody.backends.kuavo.profile import urdf_path
from homebody.primitives.actions import ActionRejected
from homebody.primitives.observations import Frame
from homebody.primitives.robot import Robot


def make_backend(joint_q=None, odom=None):
    port = FakePort(joint_q=[0.0] * 20 if joint_q is None else joint_q)
    port.odom_value = odom or {
        "stamp": 100.0, "position": [1.0, 2.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, 0.0],
    }
    backend = KuavoBackend(port=port, urdf=urdf_path())
    return backend, port


def test_the_kuavo_backend_is_a_robot():
    assert isinstance(KuavoBackend, Robot)


def test_time_is_a_monotonic_clock():
    backend, port = make_backend()
    before = backend.time
    assert port.clock_value == 100.0  # the port clock is not this backend's time
    assert backend.time >= before
    backend.new_epoch()
    assert backend.epoch == 1


def test_snapshot_without_a_camera_returns_a_valid_blank_frame():
    """The session loop consumes snapshot() unconditionally: without a camera it must
    return a valid frame whose images hold nothing, not refuse."""
    backend, _ = make_backend()
    frame = backend.snapshot()
    assert isinstance(frame, Frame)
    assert frame.joints.shape == (20,)
    assert frame.arm_joints("left").shape == (7,)
    assert frame.base_pose.shape == (3,)
    assert not np.isfinite(frame.depth).any()
    assert not frame.labels.any()


def test_snapshot_positions_the_robot_from_odometry():
    backend, port = make_backend()
    yaw = np.pi / 2
    port.odom_value["orientation"] = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
    frame = backend.snapshot()
    assert frame.base_pose[:2] == pytest.approx([1.0, 2.0])
    assert frame.base_pose[2] == pytest.approx(yaw, abs=1e-9)


def test_measure_matches_snapshot_without_the_camera():
    backend, _ = make_backend()
    measured = backend.measure()
    assert measured.joints.shape == (20,)
    assert measured.base_pose.tolist() == backend.snapshot().base_pose.tolist()


def test_advance_stamps_a_monotonic_clock():
    backend, _ = make_backend()
    before = backend.time
    backend.advance(backend.epoch, 0.2)
    assert backend.time > before


def test_stale_epoch_commands_are_refused_after_stop_and_new_epoch():
    backend, _ = make_backend()
    old = backend.epoch
    backend.new_epoch()
    with pytest.raises(ActionRejected):
        backend.base_velocity(old, 0.1, 0.0, 0.0)
    with pytest.raises(ActionRejected):
        backend.arm_target(old, "left", np.zeros(7))
    with pytest.raises(ActionRejected):
        backend.stop(old)


def test_grip_refuses_with_a_typed_error():
    backend, _ = make_backend()
    with pytest.raises(ActionRejected):
        backend.grip(backend.epoch, "right", 0.5)


def test_protocol_reset_and_close():
    backend, port = make_backend()
    backend.reset()
    assert backend.epoch == 1
    backend.close()
    assert port.closed
    backend.stop(backend.epoch)  # after close the base publishes nothing more


def test_snapshot_uses_the_camera_when_one_is_wired():
    """When the port hands back a decoded camera frame, snapshot builds the Frame from
    it: the rgb and depth arrays pass through, the intrinsics come along, and the
    depth click geometry works."""
    backend, port = make_backend()
    port.camera_value = {
        "rgb": np.full((4, 6, 3), 7, dtype=np.uint8),
        "depth": np.where(np.arange(24).reshape(4, 6) % 2 == 0,
                          (np.arange(24).reshape(4, 6) / 10.0) + 0.5, np.nan),
        "intrinsics": np.array([[400.0, 0.0, 3.0], [0.0, 400.0, 2.0], [0.0, 0.0, 1.0]]),
    }
    frame = backend.snapshot()
    assert frame.rgb.shape == (4, 6, 3)
    assert frame.rgb[0, 0, 0] == 7
    assert frame.depth.shape == (4, 6)
    assert frame.intrinsics[0, 0] == 400.0
    world = frame.point(2, 0)  # pixel index 2: even, so its depth is finite
    assert np.isfinite(world).all()


def test_snapshot_falls_back_to_blank_when_the_camera_has_no_frames():
    """The sim publishes no image topics: camera() is None and the blank frame stays."""
    backend, _ = make_backend()
    frame = backend.snapshot()
    assert not np.isfinite(frame.depth).any()
    assert frame.intrinsics[0, 0] == 1.0


def test_snapshot_keeps_blank_until_both_streams_arrive():
    """An rgb-only frame is not a usable camera frame: no depth means no click geometry,
    so the blank fallback holds until depth is in."""
    backend, port = make_backend()
    port.camera_value = {
        "rgb": np.zeros((4, 6, 3), dtype=np.uint8),
        "intrinsics": np.diag([400.0, 400.0, 1.0]).astype(float),
    }
    frame = backend.snapshot()
    assert not np.isfinite(frame.depth).any()


def test_snapshot_camera_depth_with_identity_intrinsics_is_a_real_frame():
    """The port dict always carries the full triple once depth is in; identity
    intrinsics are a legitimate calibration, so the frame passes through."""
    backend, port = make_backend()
    port.camera_value = {
        "rgb": np.zeros((4, 6, 3), dtype=np.uint8),
        "depth": np.full((4, 6), 1.5),
        "intrinsics": np.eye(3),
    }
    frame = backend.snapshot()
    assert np.isfinite(frame.depth).all()
