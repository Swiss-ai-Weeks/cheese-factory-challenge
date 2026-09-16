"""Camera calibration and pixel-to-conveyor projection math."""

from __future__ import annotations

import math

import numpy as np


def quaternion_matrix_wxyz(quaternion: tuple[float, float, float, float] | list[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("camera quaternion cannot be zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pixel_to_plane(
    pixel_xy: tuple[float, float],
    resolution_hw: tuple[int, int] | list[int],
    horizontal_fov_deg: float,
    camera_position: tuple[float, float, float] | list[float],
    camera_orientation_wxyz: tuple[float, float, float, float] | list[float],
    plane_z: float,
    camera_axes: str = "usd",
) -> np.ndarray:
    """Intersect an OpenCV pixel ray with a horizontal conveyor plane.

    USD cameras point along local -Z and image +Y points down. Pixel centers are
    used so the projection is consistent at even and odd resolutions.
    """
    height, width = (int(v) for v in resolution_hw)
    u, v = (float(value) for value in pixel_xy)
    focal = width / (2.0 * math.tan(math.radians(float(horizontal_fov_deg)) / 2.0))
    cx, cy = (width - 1.0) / 2.0, (height - 1.0) / 2.0
    if camera_axes == "usd":
        ray_camera = np.asarray([(u - cx) / focal, -(v - cy) / focal, -1.0], dtype=np.float64)
    elif camera_axes == "isaac":
        # Isaac camera world axes: +X forward, +Y left, +Z up.
        ray_camera = np.asarray([1.0, -(u - cx) / focal, -(v - cy) / focal], dtype=np.float64)
    else:
        raise ValueError(f"unsupported camera axes: {camera_axes}")
    ray_world = quaternion_matrix_wxyz(camera_orientation_wxyz) @ ray_camera
    origin = np.asarray(camera_position, dtype=np.float64)
    if abs(ray_world[2]) < 1e-9:
        raise ValueError("camera ray is parallel to conveyor plane")
    distance = (float(plane_z) - origin[2]) / ray_world[2]
    if distance <= 0:
        raise ValueError("conveyor plane is behind camera")
    return origin + distance * ray_world
