import numpy as np

from sim.factory.geometry import pixel_to_plane


def test_center_pixel_hits_point_below_overhead_camera():
    hit = pixel_to_plane(
        (319.5, 239.5),
        (480, 640),
        42.0,
        (0.5, 0.0, 1.3),
        (1.0, 0.0, 0.0, 0.0),
        0.055,
    )
    np.testing.assert_allclose(hit, [0.5, 0.0, 0.055], atol=1e-9)


def test_pixel_lateral_offsets_are_not_hardcoded():
    left = pixel_to_plane((250, 239.5), (480, 640), 42.0, (0.5, 0.0, 1.3), (1, 0, 0, 0), 0.055)
    right = pixel_to_plane((390, 239.5), (480, 640), 42.0, (0.5, 0.0, 1.3), (1, 0, 0, 0), 0.055)
    assert left[0] < 0.5 < right[0]
    np.testing.assert_allclose(abs(left[0] - 0.5), abs(right[0] - 0.5), atol=0.002)


def test_isaac_camera_axes_project_downward():
    orientation = (-0.5, 0.5, -0.5, -0.5)
    hit = pixel_to_plane(
        (319.5, 239.5),
        (480, 640),
        42.0,
        (0.5, 0.0, 1.3),
        orientation,
        0.055,
        camera_axes="isaac",
    )
    np.testing.assert_allclose(hit, [0.5, 0.0, 0.055], atol=1e-9)
