"""Isaac Sim scene authoring and camera acquisition."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT, FactoryConfig
from .controller import create_perception_pick_place_task
from .perception import DEVELOPMENT_PALETTE


BIN_COLORS = {
    "bin_hard": (0.90, 0.72, 0.18),
    "bin_semi_hard": (0.92, 0.42, 0.10),
    "bin_soft": (0.84, 0.84, 0.76),
    "bin_fresh": (0.45, 0.78, 0.92),
    "bin_blue": (0.18, 0.48, 0.72),
}


class IsaacFactoryScene:
    def __init__(self, config: FactoryConfig, classifier_mode: str = "model"):
        self.config = config
        self.classifier_mode = classifier_mode
        robot = config.section("robot")
        pick = tuple(float(v) for v in robot["pick_position"])
        first_bin = config.bins["bin_hard"]
        self.task = create_perception_pick_place_task(
            robot_path=robot["prim_path"],
            cube_path="/World/ActiveObject",
            pick_position=pick,
            place_position=first_bin,
        )
        self.camera = None
        self._camera_authoring = None
        self._camera_resolution = None
        self._authored: list[Any] = []
        self._carrier_root = None
        self._carrier_transform = None
        self._carrier_texture = None
        self._carrier_piece_transform = None
        self._held_out_textures: dict[str, list[Path]] = {}
        self._texture_indices: dict[str, int] = defaultdict(int)

    def setup_scene(self) -> None:
        from isaacsim.core.experimental.objects import Cube
        from isaacsim.core.experimental.utils import stage as stage_utils
        from isaacsim.sensors.experimental.rtx import RtxCamera
        from pxr import UsdGeom, UsdPhysics

        print("FACTORY_SCENE robot_reference", flush=True)
        self.task.setup_scene()
        print("FACTORY_SCENE robot_reference_done", flush=True)
        belt = self.config.section("belt")
        self._authored.append(
            Cube(
                "/World/Conveyor",
                positions=[0.50, -0.30, float(belt["plane_z"]) / 2.0],
                sizes=1.0,
                scales=[0.38, 1.45, float(belt["plane_z"])],
                colors=[0.12, 0.16, 0.19],
            )
        )
        # The rendered belt is wider than the cargo lane and visually passes
        # under two bins. Use a narrow invisible collider under only the cargo
        # path so cuMotion does not interpret remote bin approaches as blocked.
        self._authored.append(
            Cube(
                "/World/ConveyorCollider",
                positions=[0.50, -0.30, float(belt["plane_z"]) / 2.0],
                sizes=1.0,
                scales=[0.14, 1.45, float(belt["plane_z"])],
            )
        )
        conveyor_collider = stage_utils.get_current_stage().GetPrimAtPath("/World/ConveyorCollider")
        UsdPhysics.CollisionAPI.Apply(conveyor_collider)
        UsdGeom.Imageable(conveyor_collider).MakeInvisible()
        for name, position in self.config.bins.items():
            self._create_bin(name, position, BIN_COLORS[name])
        reject = tuple(float(v) for v in self.config.raw["reject_position"])
        self._authored.append(
            Cube(
                "/World/RejectChute",
                positions=[reject[0], reject[1], 0.015],
                sizes=1.0,
                scales=[0.20, 0.22, 0.03],
                colors=[0.42, 0.08, 0.08],
            )
        )
        UsdPhysics.CollisionAPI.Apply(stage_utils.get_current_stage().GetPrimAtPath("/World/RejectChute"))
        if self.classifier_mode == "model":
            self._create_model_visual(stage_utils.get_current_stage())
        camera = self.config.section("camera")
        authoring = RtxCamera(
            camera["prim_path"],
            tick_rate=float(self.config.raw["render_hz"]),
            positions=camera["position"],
            orientations=camera["orientation_wxyz"],
        )
        authoring.camera.set_clipping_ranges(0.05, 100.0)
        # The projection module consumes this same calibrated horizontal FOV.
        # The experimental Camera API exposes the default USD 20.955 mm
        # aperture as 2.0955 (it converts to USD tenths internally).
        import math

        focal_mm = 2.0955 / (2.0 * math.tan(math.radians(float(camera["horizontal_fov_deg"])) / 2.0))
        authoring.camera.set_focal_lengths(focal_mm)
        self._camera_authoring = authoring
        self._camera_resolution = tuple(int(v) for v in camera["resolution"])
        print("FACTORY_SCENE authored", flush=True)

    def _create_model_visual(self, stage) -> None:
        """Author the same carrier/cutout representation used for training."""
        from pxr import Gf, UsdGeom, UsdShade
        from sim.usd_kit import assiette, materiau_texture, materiau_uni, quad

        manifest = PROJECT_ROOT / "data" / "processed" / "manifest_sim.csv"
        cutouts = PROJECT_ROOT / "data" / "processed" / "cutouts" / "manifest.csv"
        if not manifest.exists() or not cutouts.exists():
            raise FileNotFoundError("model-mode scene requires processed cutouts and manifest_sim.csv")
        held_out_groups = {
            row["group"]
            for row in csv.DictReader(manifest.open())
            if row["split"] == "test"
        }
        textures: dict[str, list[Path]] = defaultdict(list)
        for row in csv.DictReader(cutouts.open()):
            if row["uid"] not in held_out_groups:
                continue
            label = "not_cheese" if row["bin"] == "not_cheese" else row["label"]
            textures[label].append(PROJECT_ROOT / "data" / "processed" / row["path"])
        required = set(self.config.raw["evaluation_sequence"])
        missing = sorted(label for label in required if not textures[label])
        if missing:
            raise RuntimeError(f"held-out evaluation textures are missing for: {missing}")
        self._held_out_textures = {label: sorted(paths) for label, paths in textures.items()}

        root = UsdGeom.Xform.Define(stage, "/World/InspectionCarrier")
        self._carrier_root = root
        self._carrier_transform = UsdGeom.Xformable(root).AddTransformOp()
        plate = assiette(stage, "/World/InspectionCarrier/Plate", rayon=0.090,
                         hauteur=0.045, epaisseur=0.008)
        plate_material = materiau_uni(
            stage, "/World/InspectionCarrier/PlateMaterial", (0.92, 0.92, 0.90), rugosite=0.25
        )
        UsdShade.MaterialBindingAPI(plate).Bind(plate_material)
        piece = quad(stage, "/World/InspectionCarrier/Cheese")
        piece_material, self._carrier_texture = materiau_texture(
            stage, "/World/InspectionCarrier/CheeseMaterial"
        )
        UsdShade.MaterialBindingAPI(piece).Bind(piece_material)
        self._carrier_piece_transform = UsdGeom.Xformable(piece).AddTransformOp()
        UsdGeom.Imageable(root).MakeInvisible()
        print(
            "FACTORY_SCENE model_visual "
            f"held_out_groups={len(held_out_groups)} labels={len(self._held_out_textures)}",
            flush=True,
        )

    def _set_model_visual(
        self,
        class_name: str,
        *,
        texture_path: Path | None = None,
        size_fraction: float = 0.65,
        rotation_deg: float = 0.0,
    ) -> None:
        from PIL import Image
        from pxr import Gf, UsdGeom

        if texture_path is None:
            paths = self._held_out_textures[class_name]
            index = self._texture_indices[class_name] % len(paths)
            self._texture_indices[class_name] += 1
            path = paths[index]
        else:
            path = Path(texture_path)
        self._carrier_texture.GetInput("file").Set(str(path))
        with Image.open(path) as image:
            aspect = image.width / image.height
        size = float(size_fraction) * (2.0 * 0.082)
        width, height = (size * aspect, size) if aspect >= 1.0 else (size, size / aspect)
        diagonal = 0.5 * (width * width + height * height) ** 0.5
        limit = 0.92 * 0.082
        if diagonal > limit:
            scale = limit / diagonal
            width, height = width * scale, height * scale
        self._carrier_piece_transform.Set(
            Gf.Matrix4d().SetScale(Gf.Vec3d(width, height, 1.0))
            * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(rotation_deg)))
            * Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, 0.010))
        )
        UsdGeom.Imageable(self._carrier_root).MakeVisible()

    def _create_bin(self, name: str, position: tuple[float, float, float], color: tuple[float, float, float]) -> None:
        from isaacsim.core.experimental.objects import Cube
        from isaacsim.core.experimental.utils import stage as stage_utils

        x, y, _ = position
        root = f"/World/Bins/{name}"
        self._authored.append(Cube(f"{root}/Floor", positions=[x, y, 0.012], sizes=1.0, scales=[0.18, 0.18, 0.024], colors=color))
        for suffix, offset, scale in (
            ("Left", (-0.10, 0.0, 0.07), (0.02, 0.22, 0.14)),
            ("Right", (0.10, 0.0, 0.07), (0.02, 0.22, 0.14)),
            ("Back", (0.0, 0.10, 0.07), (0.22, 0.02, 0.14)),
        ):
            self._authored.append(
                Cube(
                    f"{root}/{suffix}",
                    positions=[x + offset[0], y + offset[1], offset[2]],
                    sizes=1.0,
                    scales=scale,
                    colors=color,
                )
            )
        stage = stage_utils.get_current_stage()
        stage.GetPrimAtPath(f"{root}/Floor").SetCustomDataByKey("factory_bin", name)

    async def initialize(self) -> None:
        # The cargo lane and reject chute have static collision geometry. Bin
        # walls stay visual so the maintained Franka task has a feasible
        # descent corridor; placement is still validated from object settling.
        from isaacsim.sensors.experimental.rtx import CameraSensor
        import isaacsim.core.experimental.utils.app as app_utils

        # CameraSensor owns its render product and does not need a UI viewport.
        # This is important for the no-window headless experience.
        app_utils.stop(commit=True)
        await app_utils.update_app_async()
        self.camera = CameraSensor(
            self._camera_authoring,
            resolution=self._camera_resolution,
            annotators=["rgb", "distance_to_image_plane"],
        )
        # Match NVIDIA's 6.1 CameraSensor tests: set the view after the runtime
        # sensor has wrapped the prim, because construction standardizes xforms.
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera_view(
            self.camera.camera.paths[0],
            eye=self.config.section("camera")["position"],
            target=[0.50, 0.00, float(self.config.section("belt")["plane_z"])],
        )
        app_utils.play(commit=True)
        await app_utils.update_app_async()
        camera_position, camera_orientation = self.camera_pose()
        print(
            f"FACTORY_CAMERA pose={np.asarray(camera_position).tolist()} orientation={np.asarray(camera_orientation).tolist()}",
            flush=True,
        )
        print("FACTORY_CAMERA experimental_runtime_initialized", flush=True)
        print("FACTORY_CONTROLLER initialize", flush=True)
        self.task.initialize()
        # Conveyor cargo remains a dynamic, collidable rigid body for the
        # gripper, but gravity is disabled to prevent tunnelling while scripted
        # belt poses are applied in the full streaming update loop.
        self.task.cubes[0].set_enabled_gravities([False])
        print("FACTORY_CONTROLLER initialized", flush=True)

    def reset_robot(self) -> None:
        self.task.reset_robot()
        self.task.cubes[0].set_enabled_gravities([False])

    def set_object(
        self,
        object_id: str,
        class_name: str,
        position: tuple[float, float, float],
        *,
        model_texture_path: Path | None = None,
        model_size_fraction: float = 0.65,
        model_rotation_deg: float = 0.0,
    ) -> None:
        from isaacsim.core.experimental.utils import stage as stage_utils
        from pxr import Gf, UsdGeom

        cube = self.task.cubes[0]
        cube.set_enabled_gravities([False])
        cube.set_world_poses(
            positions=np.asarray([position], dtype=np.float32),
            orientations=np.asarray([self.config.section("belt")["object_orientation_wxyz"]], dtype=np.float32),
        )
        prim = stage_utils.get_current_stage().GetPrimAtPath(self.task.cube_paths[0])
        prim.SetCustomDataByKey("factory_object_id", object_id)
        prim.SetCustomDataByKey("factory_ground_truth_class", class_name)
        if self.classifier_mode == "model":
            UsdGeom.Imageable(prim).MakeInvisible()
            self._set_model_visual(
                class_name,
                texture_path=model_texture_path,
                size_fraction=model_size_fraction,
                rotation_deg=model_rotation_deg,
            )
            self._sync_model_visual(position)
        else:
            UsdGeom.Imageable(prim).MakeVisible()
            rgb = DEVELOPMENT_PALETTE[class_name]
            UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*(value / 255.0 for value in rgb))])

    def set_object_texture(
        self,
        object_id: str,
        class_name: str,
        position: tuple[float, float, float],
        texture_path: Path,
        *,
        size_fraction: float,
        rotation_deg: float,
    ) -> None:
        """Place one explicitly selected source cutout for domain adaptation."""
        self.set_object(
            object_id,
            class_name,
            position,
            model_texture_path=texture_path,
            model_size_fraction=size_fraction,
            model_rotation_deg=rotation_deg,
        )

    def _sync_model_visual(self, position) -> None:
        if self._carrier_transform is None:
            return
        from pxr import Gf

        # PickPlaceTask's cube position is its centre; the training carrier's
        # local origin is its base on the belt.
        self._carrier_transform.Set(
            Gf.Matrix4d().SetTranslate(
                Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]) - 0.02575)
            )
        )

    def sync_object_visual(self) -> None:
        if self.classifier_mode == "model":
            self._sync_model_visual(self.object_position())

    def move_object(self, position: tuple[float, float, float]) -> None:
        cube = self.task.cubes[0]
        cube.set_world_poses(positions=np.asarray([position], dtype=np.float32))
        self._sync_model_visual(position)

    def set_object_gravity(self, enabled: bool) -> None:
        self.task.cubes[0].set_enabled_gravities([enabled])

    def object_position(self) -> np.ndarray:
        return np.asarray(self.task.cubes[0].get_world_poses()[0].numpy()[0], dtype=np.float64)

    def camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = self._camera_authoring.get_world_poses()
        return (
            np.asarray(positions.numpy()[0], dtype=np.float64),
            np.asarray(orientations.numpy()[0], dtype=np.float64),
        )

    def get_camera_frame(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.camera is None:
            return None, None
        rgb, _ = self.camera.get_data("rgb")
        depth, _ = self.camera.get_data("distance_to_image_plane")
        if rgb is None:
            return None, None
        return rgb.numpy(), None if depth is None else depth.numpy()

    def cleanup(self) -> None:
        self.task.cleanup()
        self.camera = None
        self._camera_authoring = None
        self._carrier_root = None
        self._carrier_transform = None
        self._carrier_texture = None
        self._carrier_piece_transform = None
        self._authored.clear()
