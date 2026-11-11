"""seam: the session loop runs skills on the kuavo backend. Navigate prepares against
the blank frame the camera-less backend supplies; a click skill is refused at its own
argument check. One integration level above the protocol test, still ROS-free."""
import pytest
from test_backend_seam import FakePort

from homebody.backends.kuavo.backend import KuavoBackend
from homebody.backends.kuavo.profile import urdf_path
from homebody.primitives.observations import ObservationError
from homebody.session.settings import Settings
from homebody.skills.contract import Planning, SkillFailure
from homebody.skills.navigate import prepare as navigate_prepare


def make_backend():
    port = FakePort(joint_q=[0.0] * 20)
    port.set_mpc_ctrl_mode = lambda mode: True
    port.set_arm_ctrl_mode = lambda mode: True
    return KuavoBackend(port=port, urdf=urdf_path())


def planning():
    return Planning({}, {}, Settings(), {})


def test_navigate_prepares_on_the_blank_frame():
    backend = make_backend()
    epoch = backend.new_epoch()
    backend.ensure_arm_control()
    frame = backend.snapshot()
    assert frame.base_pose.shape == (3,)
    plan = navigate_prepare({"x": 0.0, "y": 0.0}, frame, frame, planning())
    assert plan.goal.shape == (2,) and plan.epoch == epoch


def test_navigate_reports_a_domain_refusal_outside_the_fallback_map():
    backend = make_backend()
    backend.new_epoch()
    frame = backend.snapshot()
    with pytest.raises(SkillFailure) as failure:
        navigate_prepare({"x": 40.0, "y": 40.0}, frame, frame, planning())
    assert failure.value.code == "NO_ROUTE"


def test_click_skills_cannot_select_on_a_frame_without_depth():
    backend = make_backend()
    backend.new_epoch()
    frame = backend.snapshot()
    with pytest.raises(ObservationError):
        frame.point(3, 3)
    with pytest.raises(ObservationError):
        frame.select(3, 3)
