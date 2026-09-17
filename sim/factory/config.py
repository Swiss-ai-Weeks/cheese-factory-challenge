"""Validated, machine-independent configuration for the factory demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class FactoryConfig:
    raw: dict[str, Any]
    source: Path

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def bins(self) -> dict[str, tuple[float, float, float]]:
        return {name: tuple(float(v) for v in xyz) for name, xyz in self.raw["bins"].items()}

    @property
    def model_path(self) -> Path:
        value = Path(self.raw["perception"]["model_path"])
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def output_root(self) -> Path:
        return PROJECT_ROOT / "outputs" / "factory"

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]


def _validate(data: dict[str, Any]) -> None:
    required = {"seed", "physics_hz", "belt", "camera", "perception", "robot", "bins", "reject_position"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"factory config is missing: {', '.join(missing)}")
    expected_bins = {"bin_hard", "bin_semi_hard", "bin_soft", "bin_fresh", "bin_blue"}
    if set(data["bins"]) != expected_bins:
        raise ValueError(f"bins must be exactly {sorted(expected_bins)}")
    resolution = data["camera"]["resolution"]
    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError("camera.resolution must be [height, width]")
    orientation = data["belt"]["object_orientation_wxyz"]
    if len(orientation) != 4:
        raise ValueError("belt.object_orientation_wxyz must be [w, x, y, z]")
    if not 0.0 <= float(data["perception"]["min_confidence"]) <= 1.0:
        raise ValueError("perception.min_confidence must be between 0 and 1")


def load_config(path: str | Path | None = None) -> FactoryConfig:
    source = Path(path) if path else DEFAULT_CONFIG
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    _validate(data)
    return FactoryConfig(raw=data, source=source.resolve())
