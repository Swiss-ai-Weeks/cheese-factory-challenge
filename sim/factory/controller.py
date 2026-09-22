"""Isaac 6.1 camera-localized wrapper around NVIDIA's cuMotion task."""

from __future__ import annotations

from typing import Any

import numpy as np


PHASE_TO_FACTORY_STATE = {
    "APPROACH_PICK": "APPROACHING",
    "DESCEND_PICK": "DESCENDING",
    "GRASP": "GRASPING",
    "LIFT": "LIFTING",
    "APPROACH_PLACE": "MOVING_TO_BIN",
    "DESCEND_PLACE": "MOVING_TO_BIN",
    "RELEASE": "RELEASING",
    "RETREAT": "RETURNING_HOME",
    "DONE": "COMPLETE",
    "FAILED": "RECOVERY",
}


def create_perception_pick_place_task(
    robot_path: str,
    cube_path: str,
    pick_position: tuple[float, float, float],
    place_position: tuple[float, float, float],
    planning_obstacle_paths: tuple[str, ...] = (),
) -> Any:
    """Build a task whose commanded pick comes from calibrated camera pixels.

    Isaac imports intentionally live inside this factory so pure-Python tests can
    import the rest of the package outside Isaac Sim.
    """
    import isaacsim.robot_motion.experimental.motion_generation as mg
    import warp as wp
    from isaacsim.robot_motion.examples.manipulation.pick_place_task import PickPlaceTask
    from isaacsim.robot_motion.examples.manipulation.robots import SurfaceGripperConfig

    class PerceptionPickPlaceTask(PickPlaceTask):
        def __init__(self) -> None:
            super().__init__(
                robot_path=robot_path,
                cube_path=cube_path,
                cube_positions=[pick_position],
                place_position=place_position,
                robot_name="franka",
            )
            self.perception_pick_position = np.asarray(pick_position, dtype=np.float32)
            self.planning_obstacle_paths = tuple(planning_obstacle_paths)

        def initialize(self, exclude_prim_paths=()) -> None:
            """Bind the world and prove every declared safety obstacle is tracked.

            NVIDIA's task discovers PhysX CollisionAPI prims while creating its
            cuMotion world. Re-enabling the named roots is an intentional
            registration check: it raises instead of running if a collider was
            renamed, omitted, or excluded from the planner world.
            """
            super().initialize(exclude_prim_paths)
            if self.planning_obstacle_paths:
                self.scenario.set_planning_obstacles_enabled(
                    self.planning_obstacle_paths,
                    True,
                )
            print(
                "FACTORY_CUMOTION_OBSTACLES "
                + ",".join(self.planning_obstacle_paths),
                flush=True,
            )

        def set_camera_goal(self, pick_xyz, place_xyz) -> None:
            self.perception_pick_position = np.asarray(pick_xyz, dtype=np.float32)
            self.place_position = np.asarray(place_xyz, dtype=np.float32)

        def _capture_setpoint(self):
            # Deliberately do not call cube.get_world_poses(): this is the critical
            # boundary proving that the pick XY originates in rendered pixels.
            pick = self.perception_pick_position.copy()
            place = self.place_position.copy()
            if isinstance(self.scenario.robot_config.gripper, SurfaceGripperConfig):
                pick[2] += 0.02575
                place[2] += 0.02575
            return mg.RobotState(
                sites=mg.SpatialState.from_name(
                    spatial_space=["pick", "place"],
                    positions=(
                        ["pick", "place"],
                        wp.array([pick, place], dtype=wp.float32, device="cpu"),
                    ),
                )
            )

    return PerceptionPickPlaceTask()
