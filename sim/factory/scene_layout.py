"""Validated, declarative authoring data for the static factory shell.

This module intentionally has no Isaac Sim imports.  It can be tested and edited
with an ordinary Python interpreter; :mod:`sim.factory.scene` is the small
adapter that turns the declarations into OpenUSD prims at runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


DEFAULT_SCENE_LAYOUT = Path(__file__).with_name("scene_layout.json")
LAYOUT_ENVIRONMENT_VARIABLE = "CHEESE_FACTORY_SCENE_LAYOUT"


@dataclass(frozen=True)
class SceneLayout:
    raw: dict[str, Any]
    source: Path

    @property
    def materials(self) -> dict[str, dict[str, Any]]:
        return self.raw["materials"]

    @property
    def boxes(self) -> list[dict[str, Any]]:
        return self.raw["boxes"]

    @property
    def lights(self) -> list[dict[str, Any]]:
        return self.raw["lights"]

    @property
    def overview_camera(self) -> dict[str, Any]:
        return self.raw["overview_camera"]


def _vector(value: Any, length: int, field: str) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} numbers")
    try:
        [float(component) for component in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only numbers") from exc


def _absolute_prim_path(value: Any, field: str) -> str:
    path = str(value)
    if not path.startswith("/World/") or "//" in path or path.endswith("/"):
        raise ValueError(f"{field} must be an absolute prim path below /World")
    return path


def _validate(data: dict[str, Any]) -> None:
    required = {"schema_version", "materials", "boxes", "lights", "overview_camera"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"scene layout is missing: {', '.join(missing)}")
    if data["schema_version"] != 1:
        raise ValueError("scene layout schema_version must be 1")
    if not isinstance(data["materials"], dict) or not data["materials"]:
        raise ValueError("scene layout requires at least one material")

    for name, material in data["materials"].items():
        if not name or not isinstance(material, dict):
            raise ValueError("every material requires a non-empty name and object value")
        _vector(material.get("color"), 3, f"materials.{name}.color")
        for field in ("roughness", "metallic"):
            value = float(material.get(field, 0.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"materials.{name}.{field} must be between 0 and 1")

    seen_paths: set[str] = set()
    for index, box in enumerate(data["boxes"]):
        path = _absolute_prim_path(box.get("path"), f"boxes[{index}].path")
        if path in seen_paths:
            raise ValueError(f"duplicate scene prim path: {path}")
        seen_paths.add(path)
        _vector(box.get("size"), 3, f"boxes[{index}].size")
        _vector(box.get("position"), 3, f"boxes[{index}].position")
        _vector(box.get("rotation_xyz_deg", [0.0, 0.0, 0.0]), 3, f"boxes[{index}].rotation_xyz_deg")
        if box.get("material") not in data["materials"]:
            raise ValueError(f"{path} refers to unknown material {box.get('material')!r}")
        if box.get("collision", "none") not in {"none", "static"}:
            raise ValueError(f"{path}.collision must be 'none' or 'static'")
        if not str(box.get("role", "")).strip():
            raise ValueError(f"{path}.role must explain the prim's purpose")

    for index, light in enumerate(data["lights"]):
        path = _absolute_prim_path(light.get("path"), f"lights[{index}].path")
        if path in seen_paths:
            raise ValueError(f"duplicate scene prim path: {path}")
        seen_paths.add(path)
        if light.get("type") not in {"dome", "rect", "sphere"}:
            raise ValueError(f"{path}.type must be dome, rect, or sphere")
        if float(light.get("intensity", 0.0)) <= 0.0:
            raise ValueError(f"{path}.intensity must be positive")
        if "position" in light:
            _vector(light["position"], 3, f"lights[{index}].position")
        if "color" in light:
            _vector(light["color"], 3, f"lights[{index}].color")

    camera = data["overview_camera"]
    _absolute_prim_path(camera.get("path"), "overview_camera.path")
    _vector(camera.get("position"), 3, "overview_camera.position")
    _vector(camera.get("look_at"), 3, "overview_camera.look_at")
    if float(camera.get("focal_length_mm", 0.0)) <= 0.0:
        raise ValueError("overview_camera.focal_length_mm must be positive")


def load_scene_layout(path: str | Path | None = None) -> SceneLayout:
    """Load the checked-in layout or an explicit developer override.

    Relative paths are resolved from the repository root so launch behaviour is
    independent of the caller's current working directory.
    """

    selected = path or os.environ.get(LAYOUT_ENVIRONMENT_VARIABLE) or DEFAULT_SCENE_LAYOUT
    source = Path(selected)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    _validate(data)
    return SceneLayout(raw=data, source=source.resolve())
