"""测试 Sim2Real 模块。"""
import numpy as np
import torch
from robocore.sim2real.domain_randomization import DomainRandomizer, RandomizationConfig
from robocore.sim2real.real_robot import RealRobotInterface, RealRobotConfig
from robocore.sim2real.calibration import CameraCalibration


def test_domain_randomizer_image():
    dr = DomainRandomizer()
    img = torch.rand(2, 3, 64, 64)
    out = dr.randomize_image(img)
    assert out.shape == img.shape
    assert (out >= 0).all() and (out <= 1).all()


def test_domain_randomizer_state():
    dr = DomainRandomizer()
    state = torch.randn(2, 10)
    out = dr.randomize_state(state)
    assert out.shape == state.shape
    assert not torch.allclose(out, state)


def test_domain_randomizer_forward():
    dr = DomainRandomizer()
    obs = {"image": torch.rand(2, 3, 64, 64), "state": torch.randn(2, 10)}
    out = dr(obs)
    assert "image" in out and "state" in out


def test_domain_randomizer_physics():
    dr = DomainRandomizer()
    params = dr.randomize_physics()
    assert "friction" in params
    assert 0.5 <= params["friction"] <= 1.5


def test_real_robot_interface():
    robot = RealRobotInterface(RealRobotConfig(control_freq=1000))
    robot.connect()
    obs = robot.get_observation()
    assert "state" in obs
    robot.execute_action(np.zeros(7))
    robot.disconnect()


def test_camera_calibration():
    cal = CameraCalibration()
    K = cal.intrinsic_matrix
    assert K.shape == (3, 3)
    assert K[0, 0] == cal.fx

    T = cal.extrinsic_matrix
    assert T.shape == (4, 4)

    # 往返测试
    point = np.array([0.1, 0.2, 0.5])
    u, v = cal.world_to_pixel(point)
    recovered = cal.pixel_to_world(u, v, 0.5)
    # 不完全精确因为外参变换
    assert recovered.shape == (3,)


def test_camera_calibration_serialization():
    cal = CameraCalibration(fx=600, fy=600)
    d = cal.to_dict()
    cal2 = CameraCalibration.from_dict(d)
    assert cal2.fx == 600
