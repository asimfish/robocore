"""测试 3D 感知模块。"""
import torch
from robocore.perception3d.pointcloud import PointCloudProcessor, FarthestPointSampling, PointNetEncoder
from robocore.perception3d.voxel import VoxelEncoder
from robocore.perception3d.depth import DepthToPointCloud


def test_fps():
    fps = FarthestPointSampling(num_points=64)
    pts = torch.randn(2, 256, 3)
    out = fps(pts)
    assert out.shape == (2, 64, 3)


def test_pointcloud_processor():
    proc = PointCloudProcessor(num_points=64, normalize=True, downsample_method="random")
    pts = torch.randn(2, 256, 3)
    out = proc(pts)
    assert out.shape == (2, 64, 3)


def test_pointnet_encoder():
    enc = PointNetEncoder(input_dim=3, output_dim=64)
    pts = torch.randn(2, 128, 3)
    out = enc(pts)
    assert out.shape == (2, 64)


def test_pointnet_tnet():
    enc = PointNetEncoder(input_dim=3, output_dim=64, use_tnet=True)
    pts = torch.randn(2, 128, 3)
    out = enc(pts)
    assert out.shape == (2, 64)


def test_voxel_encoder():
    enc = VoxelEncoder(voxel_size=16, output_dim=64, channels=(16, 32))
    pts = torch.randn(2, 256, 3).clamp(-1, 1)
    out = enc(pts)
    assert out.shape == (2, 64)


def test_depth_to_pointcloud():
    d2pc = DepthToPointCloud(fx=100, fy=100, cx=32, cy=24, depth_scale=1.0)
    depth = torch.rand(2, 48, 64) * 2
    pc = d2pc(depth)
    assert pc.shape == (2, 48 * 64, 3)


def test_depth_to_pointcloud_rgb():
    d2pc = DepthToPointCloud(fx=100, fy=100, cx=32, cy=24, depth_scale=1.0)
    depth = torch.rand(2, 48, 64) * 2
    rgb = torch.randint(0, 255, (2, 3, 48, 64))
    pc = d2pc(depth, rgb)
    assert pc.shape == (2, 48 * 64, 6)
