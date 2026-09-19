"""Launch and evaluate the autonomous camera-to-bin Isaac Sim loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback

import numpy as np

from sim.factory.config import FactoryConfig, load_config
from sim.factory.controller import PHASE_TO_FACTORY_STATE
from sim.factory.geometry import pixel_to_plane
from sim.factory.hud import FactoryHud, configure_presentation_workspace
from sim.factory.perception import (
    BIN_OF_TYPE,
    ForegroundDetector,
    annotate_frame,
    make_sorter,
    showcase_sort_result,
)
from sim.factory.runtime_status import RuntimeStatus
from sim.factory.state_machine import FactoryState, FactoryStateMachine


def _next_state(machine: FactoryStateMachine, target_name: str) -> None:
    target = FactoryState[target_name]
    if machine.state is target:
        return
    # A physics update may cross short controller phases between observations.
    path = [
        FactoryState.APPROACHING,
        FactoryState.DESCENDING,
        FactoryState.GRASPING,
        FactoryState.LIFTING,
        FactoryState.MOVING_TO_BIN,
        FactoryState.RELEASING,
        FactoryState.RETURNING_HOME,
        FactoryState.COMPLETE,
    ]
    if machine.state in path and target in path:
        for state in path[path.index(machine.state) + 1 : path.index(target) + 1]:
            machine.transition(state)
    elif target in {FactoryState.RECOVERY, FactoryState.FAILED}:
        machine.transition(target, "Isaac controller reported failure")


def _print_progress(records: list[dict]) -> None:
    objects = [record for record in records if record["ground_truth"] != "empty"]
    successful = sum(bool(record.get("end_to_end_success")) for record in objects)
    rejected = sum(record.get("status") in {"not_cheese", "uncertain", "empty"} for record in objects)
    latest = objects[-1]
    print(
        "FACTORY_PROGRESS "
        f"completed={len(objects)} successful={successful} rejected={rejected} "
        f"object={latest['object_id']} status={latest['status']} "
        f"failure={latest.get('failure_reason') or '-'}",
        flush=True,
    )


class CheeseFactorySample:
    """BaseSample-compatible owner of scene resources."""

    def __init__(self, config: FactoryConfig, classifier_mode: str):
        from isaacsim.examples.base import BaseSample

        class _Sample(BaseSample):
            pass

        self._sample = _Sample()
        from sim.factory.scene import IsaacFactoryScene

        self.scene = IsaacFactoryScene(config, classifier_mode=classifier_mode)
        self._sample.setup_scene = self.scene.setup_scene

        async def post_load():
            await self.scene.initialize()

        async def post_reset():
            self.scene.reset_robot()

        async def post_clear():
            self.scene.cleanup()

        self._sample.setup_post_load = post_load
        self._sample.setup_post_reset = post_reset
        self._sample.setup_post_clear = post_clear

    def __getattr__(self, name):
        return getattr(self._sample, name)


async def _wait_frames(scene, app_utils, count: int) -> tuple[np.ndarray, np.ndarray | None]:
    latest_rgb = None
    latest_depth = None
    for _ in range(max(1, count)):
        await app_utils.update_app_async()
        rgb, depth = scene.get_camera_frame()
        if rgb is not None:
            latest_rgb, latest_depth = rgb, depth
    if latest_rgb is None:
        raise RuntimeError("RTX camera produced no RGB frame after bounded warm-up")
    return latest_rgb, latest_depth


async def run(
    factory_config: FactoryConfig,
    classifier_mode: str,
    max_objects: int | None = None,
    *,
    cleanup: bool = True,
) -> dict:
    import isaacsim.core.experimental.utils.app as app_utils

    sample = CheeseFactorySample(factory_config, classifier_mode)
    scene = sample.scene
    np.random.seed(factory_config.seed)
    output = factory_config.output_root
    frames_dir = output / "frames"
    output.mkdir(parents=True, exist_ok=True)
    runtime_status = RuntimeStatus.from_environment(output, classifier_mode)
    runtime_status.update("starting", max_objects=max_objects)
    sorter = None if classifier_mode == "showcase" else make_sorter(factory_config, classifier_mode)
    perception = factory_config.section("perception")
    camera_config = factory_config.section("camera")
    detector = ForegroundDetector(
        threshold=perception["background_threshold"],
        minimum_area_px=perception["minimum_area_px"],
        padding_px=perception["crop_padding_px"],
        roi=camera_config.get("roi"),
    )
    records: list[dict] = []
    sequence = list(factory_config.raw["evaluation_sequence"])
    limit = max_objects if max_objects is not None else int(factory_config.raw["max_objects"])
    total_objects = min(limit, len(sequence))
    hud = FactoryHud(
        classifier_mode,
        os.environ.get("CHEESE_SCENARIO", "default-evaluation"),
        total_objects,
    )
    try:
        print("FACTORY_STAGE loading", flush=True)
        hud.update(phase="LOADING FACTORY")
        runtime_status.update("loading", max_objects=max_objects)
        await sample.load_world_async()
        print("FACTORY_STAGE loaded", flush=True)
        runtime_status.update("loaded", max_objects=max_objects)
        await sample.reset_async()
        print("FACTORY_STAGE reset", flush=True)
        runtime_status.update("reset", max_objects=max_objects)
        hidden_windows = configure_presentation_workspace()
        print(f"FACTORY_PRESENTATION hidden_windows={','.join(hidden_windows)}", flush=True)
        hud.update(phase="READY · CAMERA CALIBRATED")
        app_utils.play(commit=True)

        spawn = tuple(float(v) for v in factory_config.section("belt")["spawn_position"])
        scene.set_object("background-parking", "not_cheese", spawn)
        background, _ = await _wait_frames(scene, app_utils, int(camera_config["warmup_frames"]))
        print(
            f"FACTORY_CAMERA background_rgb_minmax={int(background.min())},{int(background.max())}",
            flush=True,
        )
        detector.set_background(background)
        empty_detection = detector.detect(background)
        records.append(
            {
                "object_id": "empty-interval-000",
                "ground_truth": "empty",
                "status": "empty",
                "detected": empty_detection is not None,
                "pick_attempted": False,
                "pick_success": False,
                "correct_bin": True,
                "end_to_end_success": empty_detection is None,
                "cycle_time_s": 0.0,
            }
        )
        annotate_frame(background, None, None, FactoryState.WAITING.name, frames_dir / "empty-interval-000.png")

        runtime_status.update("running", completed_objects=0, total_objects=total_objects)
        for index, ground_truth in enumerate(sequence[:limit]):
            started = time.perf_counter()
            object_id = f"object-{index:03d}"
            machine = FactoryStateMachine()
            hud.update(
                phase="CONVEYOR → INSPECTION",
                object_id=object_id,
                ground_truth=ground_truth,
                prediction="analyzing…",
                confidence="—",
                destination="—",
            )
            scene.set_object(object_id, ground_truth, spawn)
            # The Franka's collision-aware retreat pose is not bit-identical to
            # its initial home pose. Refresh the empty pick-zone reference with
            # the current robot pose and the next object parked off camera.
            cycle_background, _ = await _wait_frames(scene, app_utils, 3)
            detector.set_background(cycle_background)
            belt = factory_config.section("belt")
            pick_y = float(belt["pick_line_y"])
            position = np.asarray(spawn, dtype=np.float64)
            steps = max(1, int(abs(pick_y - position[1]) / float(belt["speed_mps"]) * factory_config.raw["physics_hz"]))
            for step in range(steps):
                position[1] = spawn[1] + (pick_y - spawn[1]) * ((step + 1) / steps)
                scene.move_object(tuple(position))
                await app_utils.update_app_async()
            position[1] = pick_y
            scene.move_object(tuple(position))
            frame, depth = await _wait_frames(scene, app_utils, 5)
            depth_summary = "none" if depth is None else f"{float(np.nanmin(depth)):.4f},{float(np.nanmax(depth)):.4f}"
            print(
                f"FACTORY_OBJECT id={object_id} pose={scene.object_position().tolist()} "
                f"rgb_minmax={int(frame.min())},{int(frame.max())} depth_minmax={depth_summary}",
                flush=True,
            )
            detection = detector.detect(frame)
            if detection is None:
                annotate_frame(frame, None, None, "DETECTION_FAILED", frames_dir / f"{object_id}-detection-failed.png")
                records.append(
                    {
                        "object_id": object_id,
                        "ground_truth": ground_truth,
                        "status": "detection_failed",
                        "detected": False,
                        "pick_attempted": False,
                        "pick_success": False,
                        "correct_bin": False,
                        "end_to_end_success": False,
                        "failure_reason": "no foreground component in inspection ROI",
                        "cycle_time_s": time.perf_counter() - started,
                    }
                )
                _print_progress(records)
                hud.update(
                    phase="FAULT · DETECTION FAILED",
                    prediction="no foreground detected",
                    failures=hud.snapshot.failures + 1,
                    completed=index + 1,
                )
                runtime_status.update(
                    "running",
                    completed_objects=index + 1,
                    total_objects=min(limit, len(sequence)),
                    current_object=object_id,
                    last_status="detection_failed",
                )
                continue
            machine.begin(object_id)
            result = (
                showcase_sort_result(ground_truth)
                if classifier_mode == "showcase"
                else sorter.predict(detection.crop)
            )
            machine.classification(result.status)
            confidence_text = (
                "SCRIPTED"
                if classifier_mode == "showcase"
                else f"{100.0 * result.bin_confidence:.1f}%"
            )
            decision_text = f"{result.status} · {result.cheese_type}"
            destination_text = result.bin or "reject"
            if classifier_mode == "model" and not result.agreement:
                decision_text = f"{result.status} · type {result.cheese_type} · route {result.route_label}"
                destination_text = "SAFE HOLD · MODEL DISAGREEMENT"
            hud.update(
                phase="DECISION READY",
                prediction=decision_text,
                confidence=confidence_text,
                destination=destination_text,
            )
            camera_position, camera_orientation = scene.camera_pose()
            estimated = pixel_to_plane(
                detection.centroid,
                camera_config["resolution"],
                camera_config["horizontal_fov_deg"],
                camera_position,
                camera_orientation,
                float(position[2]),
                camera_axes="usd",
            )
            actual = scene.object_position()
            localization_error = float(np.linalg.norm(estimated[:2] - actual[:2]))
            annotate_frame(frame, detection, result, machine.state.name, frames_dir / f"{object_id}.png")

            pick_attempted = False
            pick_success = False
            failure_reason = None
            if result.status == "ok":
                pick_attempted = True
                scene.task.set_camera_goal(estimated, factory_config.bins[result.bin])
                scene.task.reset()
                machine.transition(FactoryState.APPROACHING)
                timeout_steps = int(factory_config.section("robot")["motion_timeout_s"] * factory_config.raw["physics_hz"])
                release_gravity_enabled = False
                last_hud_phase = None
                for _ in range(timeout_steps):
                    alive = scene.task.step(1.0 / float(factory_config.raw["physics_hz"]))
                    scene.sync_object_visual()
                    await app_utils.update_app_async()
                    status = scene.task.status()
                    if status.get("phase") == "RELEASE" and not release_gravity_enabled:
                        scene.set_object_gravity(True)
                        release_gravity_enabled = True
                    phase_target = PHASE_TO_FACTORY_STATE.get(str(status.get("phase")))
                    controller_phase = str(status.get("phase", "MOVING"))
                    if controller_phase != last_hud_phase:
                        hud.update(phase=f"ROBOT · {controller_phase.replace('_', ' ')}")
                        last_hud_phase = controller_phase
                    if phase_target and (phase_target != "COMPLETE" or scene.task.is_done):
                        _next_state(machine, phase_target)
                    if scene.task.is_done or not alive:
                        break
                status = scene.task.status()
                print(
                    f"FACTORY_CONTROLLER object={object_id} status={status} final_pose={scene.object_position().tolist()}",
                    flush=True,
                )
                pick_success = bool(scene.task.is_done and not scene.task.failed)
                if pick_success and machine.state is not FactoryState.COMPLETE:
                    _next_state(machine, "COMPLETE")
                if not pick_success:
                    failure_reason = str(status.get("failure_reason") or "motion timeout")
                    if machine.state not in {FactoryState.RECOVERY, FactoryState.COMPLETE}:
                        machine.transition(FactoryState.RECOVERY, failure_reason)
                    scene.reset_robot()
            else:
                # Rejected and uncertain objects go only to the red reject chute.
                scene.move_object(tuple(float(v) for v in factory_config.raw["reject_position"]))
                await app_utils.update_app_async(steps=5)
                machine.transition(FactoryState.COMPLETE)

            expected_bin = BIN_OF_TYPE.get(ground_truth)
            correct_classification = result.cheese_type == ground_truth
            correct_bin = (result.bin == expected_bin and pick_success) if expected_bin else result.status != "ok" and not pick_attempted
            end_to_end = bool(correct_bin and (pick_success if expected_bin else True))
            records.append(
                {
                    "object_id": object_id,
                    "ground_truth": ground_truth,
                    "detected": True,
                    "status": result.status,
                    "predicted_type": result.cheese_type,
                    "predicted_bin": result.bin,
                    "raw_route_label": result.route_label,
                    "type_implied_bin": result.type_implied_bin,
                    "decision_policy": result.decision_policy,
                    "cross_model_agreement": result.agreement,
                    "decision_reason": result.decision_reason,
                    "confidence": result.bin_confidence,
                    "inference_latency_ms": result.latency_ms,
                    "localization_error_m": localization_error,
                    "depth_m_at_centroid": None if depth is None else float(depth[int(detection.centroid[1]), int(detection.centroid[0]), 0]),
                    "classification_correct": correct_classification,
                    "pick_attempted": pick_attempted,
                    "pick_success": pick_success,
                    "correct_bin": correct_bin,
                    "end_to_end_success": end_to_end,
                    "failure_reason": failure_reason,
                    "state_history": machine.history,
                    "cycle_time_s": time.perf_counter() - started,
                }
            )
            _print_progress(records)
            hud.update(
                phase="CYCLE COMPLETE" if end_to_end else "CYCLE FAILED",
                completed=index + 1,
                successful=hud.snapshot.successful + int(end_to_end),
                rejected=hud.snapshot.rejected + int(result.status in {"not_cheese", "uncertain", "empty"}),
                failures=hud.snapshot.failures + int(not end_to_end),
            )
            runtime_status.update(
                "running",
                completed_objects=index + 1,
                total_objects=min(limit, len(sequence)),
                current_object=object_id,
                last_status=result.status,
            )

        object_records = [record for record in records if record["ground_truth"] != "empty"]
        detected_records = [record for record in object_records if record.get("detected")]
        cheese_records = [record for record in object_records if record["ground_truth"] in BIN_OF_TYPE]
        metrics = {
            "classifier_mode": classifier_mode,
            "development_classifier": classifier_mode == "development",
            "trained_model": classifier_mode == "model",
            "showcase_ground_truth_routing": classifier_mode == "showcase",
            "objects": len(object_records),
            "empty_intervals": 1,
            "detection_success": sum(bool(r.get("detected")) for r in object_records) / max(1, len(object_records)),
            "classification_accuracy": sum(bool(r.get("classification_correct")) for r in detected_records) / max(1, len(detected_records)),
            "pick_success": sum(bool(r.get("pick_success")) for r in cheese_records) / max(1, len(cheese_records)),
            "correct_bin_rate": sum(bool(r.get("correct_bin")) for r in cheese_records) / max(1, len(cheese_records)),
            "end_to_end_success": sum(bool(r.get("end_to_end_success")) for r in object_records) / max(1, len(object_records)),
            "mean_cycle_time_s": sum(float(r.get("cycle_time_s", 0.0)) for r in object_records) / max(1, len(object_records)),
        }
        report = {"metrics": metrics, "records": records}
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        runtime_status.update("complete", metrics=metrics)
        hud.update(phase="RUN COMPLETE")
        print("FACTORY_RESULTS", json.dumps(metrics, sort_keys=True), flush=True)
        return report
    except BaseException as exc:
        runtime_status.update("fatal", error_type=type(exc).__name__, error=str(exc))
        hud.update(phase=f"FATAL · {type(exc).__name__}", failures=hud.snapshot.failures + 1)
        raise
    finally:
        if cleanup:
            hud.destroy()
            await sample.clear_async()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--classifier", choices=("model", "development", "showcase"), default="model")
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"FACTORY_START classifier={args.classifier} headless={args.headless} max_objects={args.max_objects}", flush=True)
    from isaacsim import SimulationApp

    # The default *.python.kit and base no-window experiences produced empty
    # Replicator camera buffers with the 6.1 container on this host. The full
    # streaming experience uses the renderer stack already verified by the
    # remote GUI while remaining no-window when headless=True.
    simulation_app = SimulationApp(
        {"headless": args.headless},
        experience="/isaac-sim/apps/isaacsim.exp.full.streaming.kit",
    )
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.robot_motion.examples", True)
    try:
        # Standalone Kit coroutines advance only when SimulationApp pumps the
        # application. Python's run_until_complete alone deadlocks on
        # create_new_stage_async()/next_update_async().
        future = asyncio.ensure_future(run(load_config(args.config), args.classifier, args.max_objects))
        while not future.done() and simulation_app.is_running():
            simulation_app.update()
        if not future.done():
            raise RuntimeError("Isaac Sim stopped before the factory coroutine completed")
        future.result()
    except BaseException as exc:
        print(f"FACTORY_FATAL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
