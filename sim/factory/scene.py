"""Isaac Sim scene authoring and camera acquisition."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT, FactoryConfig
from .controller import create_perception_pick_place_task
from .layout import BELT_CENTER_XY, BELT_SIZE_XY, RECEIVER_SIZE_XY
from .perception import DEVELOPMENT_PALETTE
from .scene_layout import load_scene_layout


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
        self.scene_layout = load_scene_layout()
        self._overview_camera_path = str(self.scene_layout.overview_camera["path"])

    def setup_scene(self) -> None:
        from isaacsim.core.experimental.objects import Cube
        from isaacsim.core.experimental.utils import stage as stage_utils
        from isaacsim.sensors.experimental.rtx import RtxCamera
        from pxr import UsdGeom, UsdPhysics

        print("FACTORY_SCENE robot_reference", flush=True)
        self.task.setup_scene()
        print("FACTORY_SCENE robot_reference_done", flush=True)
        belt = self.config.section("belt")
        stage = stage_utils.get_current_stage()
        self._create_factory_shell(stage)
        self._authored.append(
            Cube(
                "/World/Conveyor",
                positions=[*BELT_CENTER_XY, float(belt["plane_z"]) / 2.0],
                sizes=1.0,
                scales=[*BELT_SIZE_XY, float(belt["plane_z"])],
                colors=[0.055, 0.075, 0.09],
            )
        )
        # Keep collision confined to the cargo lane so the arm has a feasible
        # descent corridor to every receiver. The visible belt uses the same
        # non-overlapping footprint; there is no hidden visual/physics mismatch.
        self._authored.append(
            Cube(
                "/World/ConveyorCollider",
                positions=[*BELT_CENTER_XY, float(belt["plane_z"]) / 2.0],
                sizes=1.0,
                scales=[0.12, BELT_SIZE_XY[1], float(belt["plane_z"])],
            )
        )
        conveyor_collider = stage.GetPrimAtPath("/World/ConveyorCollider")
        UsdPhysics.CollisionAPI.Apply(conveyor_collider)
        UsdGeom.Imageable(conveyor_collider).MakeInvisible()
        for name, position in self.config.bins.items():
            self._create_bin(stage, name, position, BIN_COLORS[name])
        reject = tuple(float(v) for v in self.config.raw["reject_position"])
        self._create_bin(stage, "reject", reject, (0.72, 0.10, 0.08))
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Bins/reject/Floor"))
        if self.classifier_mode == "model":
            self._create_model_visual(stage)
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

    def _create_factory_shell(self, stage) -> None:
        """Author human-editable declarations as a stable OpenUSD hierarchy."""
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
        from sim.usd_kit import boite_orientee, materiau_uni, viser

        materials = {}
        for name, declaration in self.scene_layout.materials.items():
            materials[name] = materiau_uni(
                stage,
                f"/World/FactoryMaterials/{name}",
                tuple(float(value) for value in declaration["color"]),
                float(declaration.get("roughness", 0.5)),
                float(declaration.get("metallic", 0.0)),
            )

        for declaration in self.scene_layout.boxes:
            box = boite_orientee(
                stage,
                declaration["path"],
                tuple(float(value) for value in declaration["size"]),
                tuple(float(value) for value in declaration["position"]),
                materials[declaration["material"]],
                tuple(float(value) for value in declaration.get("rotation_xyz_deg", (0.0, 0.0, 0.0))),
            )
            prim = box.GetPrim()
            prim.SetCustomDataByKey("factory_role", declaration["role"])
            prim.SetCustomDataByKey("factory_source", str(self.scene_layout.source))
            prim.SetCustomDataByKey("factory_collision", declaration.get("collision", "none"))
            if declaration.get("collision", "none") == "static":
                UsdPhysics.CollisionAPI.Apply(prim)

        for declaration in self.scene_layout.lights:
            light_type = declaration["type"]
            if light_type == "sphere":
                light = UsdLux.SphereLight.Define(stage, declaration["path"])
                light.CreateRadiusAttr(float(declaration["radius"]))
            elif light_type == "rect":
                light = UsdLux.RectLight.Define(stage, declaration["path"])
                light.CreateWidthAttr(float(declaration["width"]))
                light.CreateHeightAttr(float(declaration["height"]))
            else:
                light = UsdLux.DomeLight.Define(stage, declaration["path"])
            light.CreateIntensityAttr(float(declaration["intensity"]))
            if "color" in declaration:
                light.CreateColorAttr(Gf.Vec3f(*[float(value) for value in declaration["color"]]))
            if "position" in declaration:
                UsdGeom.Xformable(light).AddTranslateOp().Set(
                    Gf.Vec3d(*[float(value) for value in declaration["position"]])
                )
            light.GetPrim().SetCustomDataByKey("factory_role", declaration["role"])
            light.GetPrim().SetCustomDataByKey("factory_source", str(self.scene_layout.source))

        declaration = self.scene_layout.overview_camera
        camera = UsdGeom.Camera.Define(stage, declaration["path"])
        camera.CreateFocalLengthAttr(float(declaration["focal_length_mm"]))
        camera.CreateClippingRangeAttr(Gf.Vec2f(*[float(value) for value in declaration["clipping_range"]]))
        viser(
            UsdGeom.Xformable(camera).AddTransformOp(),
            Gf.Vec3d(*[float(value) for value in declaration["position"]]),
            Gf.Vec3d(*[float(value) for value in declaration["look_at"]]),
        )
        camera.GetPrim().SetCustomDataByKey("factory_role", declaration["role"])
        camera.GetPrim().SetCustomDataByKey("factory_source", str(self.scene_layout.source))
        print(f"FACTORY_LAYOUT source={self.scene_layout.source}", flush=True)

    def _create_bin(self, stage, name: str, position: tuple[float, float, float], color: tuple[float, float, float]) -> None:
        from isaacsim.core.experimental.objects import Cube
        from pxr import Gf, UsdGeom, UsdShade
        from sim.usd_kit import boite, materiau_texture, materiau_uni, quad

        x, y, _ = position
        root = f"/World/Bins/{name}"
        tray_color = tuple(max(0.045, component * 0.28) for component in color)
        accent = materiau_uni(stage, f"{root}/AccentMaterial", color, 0.38, 0.18)
        steel = materiau_uni(stage, f"{root}/SteelMaterial", (0.32, 0.35, 0.38), 0.32, 0.72)
        dark = materiau_uni(stage, f"{root}/DarkMaterial", (0.07, 0.08, 0.095), 0.78, 0.06)

        self._authored.append(Cube(f"{root}/Floor", positions=[x, y, 0.012], sizes=1.0, scales=[0.14, 0.14, 0.024], colors=tray_color))
        for suffix, offset, scale in (
            ("Left", (-0.075, 0.0, 0.065), (0.015, 0.16, 0.13)),
            ("Right", (0.075, 0.0, 0.065), (0.015, 0.16, 0.13)),
            ("Back", (0.0, 0.075, 0.065), (0.16, 0.015, 0.13)),
        ):
            self._authored.append(Cube(f"{root}/{suffix}", positions=[x + offset[0], y + offset[1], offset[2]], sizes=1.0, scales=scale, colors=tray_color))

        # Compact stainless receiver frame with a replaceable dark tote.
        boite(stage, f"{root}/Station/base", (RECEIVER_SIZE_XY[0], RECEIVER_SIZE_XY[1], 0.026), (x, y, -0.015), steel)
        for index, (dx, dy) in enumerate(((-0.072, -0.072), (-0.072, 0.072), (0.072, -0.072), (0.072, 0.072))):
            boite(stage, f"{root}/Station/leg_{index}", (0.018, 0.018, 0.16), (x + dx, y + dy, -0.095), steel)
            boite(stage, f"{root}/Station/foot_{index}", (0.042, 0.042, 0.012), (x + dx, y + dy, -0.18), dark)
        boite(stage, f"{root}/Station/accent", (0.155, 0.018, 0.035), (x, y - 0.084, 0.045), accent)
        boite(stage, f"{root}/Station/sign_post", (0.018, 0.018, 0.31), (x, y + 0.095, 0.19), steel)

        label_path = PROJECT_ROOT / "sim" / "labels" / f"{name}.png"
        label = quad(stage, f"{root}/Station/label")
        label_material, _ = materiau_texture(stage, f"{root}/Station/LabelMaterial", str(label_path))
        UsdShade.MaterialBindingAPI(label).Bind(label_material)
        UsdGeom.Xformable(label).AddTransformOp().Set(
            Gf.Matrix4d().SetScale(Gf.Vec3d(0.17, 0.055, 1.0))
            * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0))
            * Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y + 0.087, 0.30))
        )
        stage.GetPrimAtPath(f"{root}/Floor").SetCustomDataByKey("factory_bin", name)

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
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(self._overview_camera_path)
            print(f"FACTORY_OVERVIEW camera={self._overview_camera_path}", flush=True)
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
