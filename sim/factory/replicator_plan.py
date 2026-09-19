"""Deterministic parameter plans consumed by the Isaac Replicator capture."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReplicatorPlan:
    frame_index: int
    object_position: tuple[float, float, float]
    object_rotation_deg: float
    object_size_fraction: float
    camera_position: tuple[float, float, float]
    focal_length_mm: float
    key_intensity: float
    key_color: tuple[float, float, float]
    work_light_intensity: float
    dome_intensity: float
    floor_color: tuple[float, float, float]
    plate_color: tuple[float, float, float]
    plate_roughness: float
    clutter_positions: tuple[tuple[float, float, float], ...]
    clutter_scales: tuple[tuple[float, float, float], ...]
    clutter_colors: tuple[tuple[float, float, float], ...]
    occluder_enabled: bool
    occluder_position: tuple[float, float, float]
    occluder_scale: tuple[float, float, float]

    def as_dict(self) -> dict:
        return asdict(self)


def _triple(generator, low: float, high: float) -> tuple[float, float, float]:
    return tuple(float(generator.uniform(low, high)) for _ in range(3))


def sample_plan(generator, frame_index: int, focal_length_mm: float) -> ReplicatorPlan:
    """Draw one bounded plan from a NumPy-compatible Replicator generator."""
    object_x = float(generator.uniform(0.480, 0.520))
    object_y = float(generator.uniform(-0.025, 0.025))
    camera_position = (
        float(generator.uniform(0.480, 0.520)),
        float(generator.uniform(-0.025, 0.025)),
        float(generator.uniform(1.24, 1.36)),
    )

    clutter_positions = []
    clutter_scales = []
    clutter_colors = []
    for index in range(3):
        side = -1.0 if index % 2 == 0 else 1.0
        clutter_positions.append((
            float(0.50 + side * generator.uniform(0.075, 0.105)),
            float(generator.uniform(-0.070, 0.070)),
            float(generator.uniform(0.044, 0.065)),
        ))
        clutter_scales.append((
            float(generator.uniform(0.010, 0.026)),
            float(generator.uniform(0.010, 0.035)),
            float(generator.uniform(0.012, 0.035)),
        ))
        clutter_colors.append(_triple(generator, 0.08, 0.75))

    occluder_enabled = bool(generator.uniform(0.0, 1.0) < 0.65)
    occluder_side = -1.0 if generator.uniform(0.0, 1.0) < 0.5 else 1.0
    return ReplicatorPlan(
        frame_index=frame_index,
        object_position=(object_x, object_y, 0.02925),
        object_rotation_deg=float(generator.uniform(0.0, 360.0)),
        object_size_fraction=float(generator.uniform(0.50, 0.76)),
        camera_position=camera_position,
        focal_length_mm=float(focal_length_mm * generator.uniform(0.94, 1.06)),
        key_intensity=float(generator.uniform(2600.0, 6200.0)),
        key_color=(1.0, float(generator.uniform(0.82, 1.0)), float(generator.uniform(0.72, 1.0))),
        work_light_intensity=float(generator.uniform(280.0, 980.0)),
        dome_intensity=float(generator.uniform(90.0, 360.0)),
        floor_color=_triple(generator, 0.075, 0.155),
        plate_color=_triple(generator, 0.78, 0.98),
        plate_roughness=float(generator.uniform(0.18, 0.72)),
        clutter_positions=tuple(clutter_positions),
        clutter_scales=tuple(clutter_scales),
        clutter_colors=tuple(clutter_colors),
        occluder_enabled=occluder_enabled,
        occluder_position=(
            float(0.50 + occluder_side * generator.uniform(0.035, 0.052)),
            float(generator.uniform(-0.030, 0.030)),
            0.075,
        ),
        occluder_scale=(
            float(generator.uniform(0.008, 0.015)),
            float(generator.uniform(0.030, 0.060)),
            float(generator.uniform(0.060, 0.105)),
        ),
    )
