"""相机标定工具。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraCalibration:
    """相机标定参数。

    存储相机内参和外参，用于：
    - 深度图 → 点云转换
    - 手眼标定
    - 多相机融合
    """

    # 内参
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    # 外参 (camera-to-world)
    position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # quaternion wxyz

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """3x3 内参矩阵。"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """4x4 外参矩阵 (camera-to-world)。"""
        w, x, y, z = self.rotation
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = self.position
        return T

    def pixel_to_world(self, u: float, v: float, depth: float) -> np.ndarray:
        """像素坐标 + 深度 → 世界坐标。"""
        z = depth
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        point_cam = np.array([x, y, z, 1.0])
        point_world = self.extrinsic_matrix @ point_cam
        return point_world[:3]

    def world_to_pixel(self, point: np.ndarray) -> tuple[float, float]:
        """世界坐标 → 像素坐标。"""
        T_inv = np.linalg.inv(self.extrinsic_matrix)
        point_cam = T_inv @ np.append(point, 1.0)
        u = self.fx * point_cam[0] / point_cam[2] + self.cx
        v = self.fy * point_cam[1] / point_cam[2] + self.cy
        return float(u), float(v)

    def to_dict(self) -> dict:
        return {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "position": list(self.position),
            "rotation": list(self.rotation),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CameraCalibration:
        return cls(
            fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
            width=d.get("width", 640), height=d.get("height", 480),
            position=tuple(d.get("position", (0, 0, 1))),
            rotation=tuple(d.get("rotation", (1, 0, 0, 0))),
        )
