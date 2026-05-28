from __future__ import annotations

from apps.coordinates import image_to_robot_point


def test_image_to_robot_point_passes_portrait_coordinates_through():
    assert image_to_robot_point(0.0, 0.0) == (0.0, 0.0)
    assert image_to_robot_point(1.0, 0.0) == (1.0, 0.0)
    assert image_to_robot_point(0.0, 1.0) == (0.0, 1.0)
    assert image_to_robot_point(1.0, 1.0) == (1.0, 1.0)
    assert image_to_robot_point(0.25, 0.5) == (0.25, 0.5)
