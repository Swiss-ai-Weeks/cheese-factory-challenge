"""Rendered-pixel detector, classifier adapters, and debug annotation."""

from __future__ import annotations

import io
import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from .config import PROJECT_ROOT, FactoryConfig
from .timing import ObservationContext


BIN_OF_TYPE = {
    "hard_cheese": "bin_hard",
    "emmental_cheese": "bin_hard",
    "semi_hard_cheese": "bin_semi_hard",
    "raclette_cheese": "bin_semi_hard",
    "soft_cheese": "bin_soft",
    "goat_cheese_soft": "bin_soft",
    "processed_cheese": "bin_soft",
    "fresh_cheese": "bin_fresh",
    "cottage_cheese": "bin_fresh",
    "cream_cheese": "bin_fresh",
    "blue_mould_cheese": "bin_blue",
}

# Visually distinct proxy materials. This is appearance calibration, not spawn
# metadata: the development classifier sees only crop pixels.
DEVELOPMENT_PALETTE: dict[str, tuple[int, int, int]] = {
    "hard_cheese": (234, 184, 52),
    "emmental_cheese": (250, 224, 84),
    "semi_hard_cheese": (184, 112, 42),
    "raclette_cheese": (241, 137, 62),
    "soft_cheese": (232, 222, 190),
    "goat_cheese_soft": (196, 214, 230),
    "processed_cheese": (245, 161, 31),
    "fresh_cheese": (72, 214, 102),
    "cottage_cheese": (153, 78, 220),
    "cream_cheese": (250, 215, 218),
    "blue_mould_cheese": (96, 157, 135),
    "not_cheese": (205, 45, 55),
}

# RTX tone mapping under the factory lighting is not a pure transfer function.
# Keep the one materially ambiguous proxy calibrated to the median pixels from
# the rendered inspection crop. This remains pixel-only inference: the sorter
# is never told which logical object was spawned.
DEVELOPMENT_RENDERED_OVERRIDES: dict[str, tuple[int, int, int]] = {
    "fresh_cheese": (167, 220, 187),
}


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area_px: int
    crop: np.ndarray


@dataclass(frozen=True)
class DevelopmentSortResult:
    status: str
    bin: str | None
    bin_confidence: float
    cheese_type: str
    type_confidence: float
    topk_types: list[tuple[str, float]]
    latency_ms: float
    route_label: str | None = None
    type_implied_bin: str | None = None
    decision_policy: str = "fine_type_aggregation_v1"
    agreement: bool = True
    decision_reason: str = "fine_type_aggregation"
    item_id: str | None = None
    observation_sequence: int | None = None
    request_id: str | None = None
    frame_sha256: str | None = None
    observed_at_epoch: float | None = None
    server_received_at_epoch: float | None = None
    decision_at_epoch: float | None = None

    @property
    def actionable(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict:
        return asdict(self)


def bind_local_decision(
    result: DevelopmentSortResult, observation: ObservationContext,
) -> DevelopmentSortResult:
    """Attach trusted correlation metadata to an in-process decision."""

    decided_at = time.time()
    return replace(
        result,
        item_id=observation.item_id,
        observation_sequence=observation.sequence,
        request_id=observation.request_id,
        frame_sha256=observation.frame_sha256,
        observed_at_epoch=observation.captured_at_epoch,
        server_received_at_epoch=observation.captured_at_epoch,
        decision_at_epoch=decided_at,
    )


def showcase_sort_result(
    ground_truth: str, observation: ObservationContext | None = None,
) -> DevelopmentSortResult:
    """Return the scripted demo route with an explicit evidence boundary.

    Showcase mode still exercises rendered detection, camera localization and
    physical manipulation. Only class/bin selection is supplied by the known
    scripted scenario, so this result must never be reported as model inference.
    """
    if ground_truth == "not_cheese":
        result = DevelopmentSortResult(
            "not_cheese", None, 1.0, "not_cheese", 1.0, [("not_cheese", 1.0)], 0.0,
            route_label="not_cheese", decision_policy="scripted_showcase_v1",
            decision_reason="scripted_reject",
        )
        return bind_local_decision(result, observation) if observation else result
    target = BIN_OF_TYPE.get(ground_truth)
    if target is None:
        raise ValueError(f"showcase scenario has no route for {ground_truth!r}")
    result = DevelopmentSortResult(
        "ok", target, 1.0, ground_truth, 1.0, [(ground_truth, 1.0)], 0.0,
        route_label=target, type_implied_bin=target,
        decision_policy="scripted_showcase_v1", decision_reason="scripted_route",
    )
    return bind_local_decision(result, observation) if observation else result


class Sorter(Protocol):
    def predict(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int] | None = None,
        *,
        observation: ObservationContext | None = None,
    ): ...


class RemoteModelSorter:
    """Fail-closed client for the host-side trained perception service.

    Isaac Sim's runtime intentionally does not carry the PyTorch/timm training
    stack. Keeping inference on the host also mirrors an industrial deployment:
    the simulator/controller consumes a small versioned decision contract.
    """

    development_only = False
    valid_statuses = {"ok", "empty", "not_cheese", "uncertain"}
    valid_bins = set(BIN_OF_TYPE.values())

    def __init__(self, base_url: str, timeout_s: float = 60.0, decision_max_age_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.decision_max_age_s = float(decision_max_age_s)
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=min(10.0, self.timeout_s)) as response:
            health = json.loads(response.read())
        if health.get("ok") is not True:
            raise RuntimeError(f"perception service is not healthy: {health}")
        if health.get("contract_version") != 2:
            raise RuntimeError(f"perception service contract version is incompatible: {health.get('contract_version')!r}")
        if health.get("timing_contract_version") != 1:
            raise RuntimeError("perception service lacks item-bound timing contract v1")
        self.decision_policy = str(health.get("decision_policy", ""))
        types = set(health.get("types", []))
        unknown = types - set(BIN_OF_TYPE) - {"empty", "not_cheese"}
        if unknown or not {"empty", "not_cheese"}.issubset(types):
            raise RuntimeError(f"perception service has an incompatible label space: {sorted(types)}")

    def predict(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int] | None = None,
        *,
        observation: ObservationContext | None = None,
    ) -> DevelopmentSortResult:
        if observation is None:
            raise RuntimeError("production inference requires an observation context")
        image = Image.fromarray(_rgb(frame))
        if box is not None:
            image = image.crop(tuple(int(v) for v in box))
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        request = urllib.request.Request(
            f"{self.base_url}/predict",
            data=payload.getvalue(),
            headers={"Content-Type": "image/png", **observation.request_headers()},
            method="POST",
        )
        request_timeout = min(self.timeout_s, self.decision_max_age_s)
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            result = json.loads(response.read())
        status = str(result.get("status", ""))
        target = result.get("bin")
        if status not in self.valid_statuses:
            raise RuntimeError(f"perception service returned invalid status: {status!r}")
        if target is not None and target not in self.valid_bins:
            raise RuntimeError(f"perception service returned invalid bin: {target!r}")
        if (status == "ok") != (target is not None):
            raise RuntimeError(f"unsafe perception decision contract: status={status!r}, bin={target!r}")
        cheese_type = str(result["cheese_type"])
        if cheese_type not in BIN_OF_TYPE and cheese_type not in {"empty", "not_cheese"}:
            raise RuntimeError(f"perception service returned invalid cheese type: {cheese_type!r}")
        route_label = str(result["route_label"])
        if route_label not in self.valid_bins | {"empty", "not_cheese"}:
            raise RuntimeError(f"perception service returned invalid raw route: {route_label!r}")
        type_implied_bin = result.get("type_implied_bin")
        expected_type_bin = BIN_OF_TYPE.get(cheese_type)
        if type_implied_bin != expected_type_bin:
            raise RuntimeError("perception service returned inconsistent type-implied bin")
        agreement = bool(result["agreement"])
        decision_policy = str(result["decision_policy"])
        if (
            decision_policy == "route_authoritative_fail_closed_v1"
            and status == "ok"
            and (not agreement or target != route_label or target != type_implied_bin)
        ):
            raise RuntimeError("unsafe perception decision contract: actionable cross-model conflict")
        if decision_policy != self.decision_policy:
            raise RuntimeError("perception service changed decision policy")
        timing_fields = {
            "item_id": result.get("item_id"),
            "observation_sequence": result.get("observation_sequence"),
            "request_id": result.get("request_id"),
            "frame_sha256": result.get("frame_sha256"),
            "observed_at_epoch": result.get("observed_at_epoch"),
            "server_received_at_epoch": result.get("server_received_at_epoch"),
            "decision_at_epoch": result.get("decision_at_epoch"),
        }
        return DevelopmentSortResult(
            status=status,
            bin=target,
            bin_confidence=float(result["bin_confidence"]),
            cheese_type=cheese_type,
            type_confidence=float(result["type_confidence"]),
            topk_types=[(str(label), float(score)) for label, score in result.get("topk_types", [])],
            latency_ms=float(result["latency_ms"]),
            route_label=route_label,
            type_implied_bin=type_implied_bin,
            decision_policy=decision_policy,
            agreement=agreement,
            decision_reason=str(result["decision_reason"]),
            **timing_fields,
        )


class ForegroundDetector:
    """Single-object background subtraction detector for the stopped pick zone."""

    def __init__(self, threshold: int, minimum_area_px: int, padding_px: int, roi: list[int] | None = None):
        self.threshold = int(threshold)
        self.minimum_area_px = int(minimum_area_px)
        self.padding_px = int(padding_px)
        self.roi = tuple(int(v) for v in roi) if roi else None
        self._background: np.ndarray | None = None

    def set_background(self, frame: np.ndarray) -> None:
        self._background = _rgb(frame).copy()

    def detect(self, frame: np.ndarray) -> Detection | None:
        image = _rgb(frame)
        if self._background is None:
            raise RuntimeError("capture an empty-belt background before detection")
        if image.shape != self._background.shape:
            raise ValueError("frame and background dimensions differ")
        height, width = image.shape[:2]
        x0, y0, x1, y1 = self.roi or (0, 0, width, height)
        difference = np.max(np.abs(image.astype(np.int16) - self._background.astype(np.int16)), axis=2)
        mask = difference >= self.threshold
        roi_mask = np.zeros_like(mask)
        roi_mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
        ys, xs = np.nonzero(mask & roi_mask)
        if len(xs) < self.minimum_area_px:
            return None
        left = max(0, int(xs.min()) - self.padding_px)
        top = max(0, int(ys.min()) - self.padding_px)
        right = min(width, int(xs.max()) + self.padding_px + 1)
        bottom = min(height, int(ys.max()) + self.padding_px + 1)
        return Detection(
            box=(left, top, right, bottom),
            centroid=(float(xs.mean()), float(ys.mean())),
            area_px=int(len(xs)),
            crop=image[top:bottom, left:right].copy(),
        )


def _rgb(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 camera pixels")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max(initial=0) <= 1.0 else image, 0, 255).astype(np.uint8)
    return image


class DevelopmentColorSorter:
    """Explicit opt-in proxy classifier used only when trained weights are absent."""

    development_only = True

    def __init__(self, min_confidence: float = 0.55):
        self.min_confidence = float(min_confidence)
        self.labels = list(DEVELOPMENT_PALETTE)
        palette = np.asarray([DEVELOPMENT_PALETTE[label] for label in self.labels], dtype=np.float64) / 255.0
        # displayColor is authored linearly; RTX output is sRGB-like.
        palette = np.power(palette, 1.0 / 2.2)
        self.palette_rgb = palette * 255.0
        for label, rendered_rgb in DEVELOPMENT_RENDERED_OVERRIDES.items():
            self.palette_rgb[self.labels.index(label)] = rendered_rgb

    def predict(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int] | None = None,
        *,
        observation: ObservationContext | None = None,
    ) -> DevelopmentSortResult:
        started = time.perf_counter()
        image = _rgb(frame)
        if box is not None:
            x0, y0, x1, y1 = box
            image = image[y0:y1, x0:x1]
        if image.size == 0:
            result = self._result("empty", 1.0, started)
            return bind_local_decision(result, observation) if observation else result
        # The center half excludes most padded conveyor background while still
        # sampling the rendered object rather than any ground-truth property.
        height, width = image.shape[:2]
        center = image[height // 4 : max(height // 4 + 1, 3 * height // 4), width // 4 : max(width // 4 + 1, 3 * width // 4)]
        pixels = center.reshape(-1, 3).astype(np.float64)
        color = np.median(pixels, axis=0)
        norm = np.linalg.norm(color)
        if norm < 15:
            result = self._result("empty", 1.0, started)
            return bind_local_decision(result, observation) if observation else result
        distances = np.linalg.norm(self.palette_rgb - color, axis=1)
        order = np.argsort(distances)
        best = int(order[0])
        confidence = float(np.clip(1.0 - float(distances[best]) / 180.0, 0.0, 0.99))
        if len(order) > 1:
            margin = float(distances[int(order[1])] - distances[best])
            confidence = min(confidence, float(np.clip(0.55 + margin / 80.0, 0.0, 0.99)))
        label = self.labels[best]
        similarities = 1.0 - distances / (255.0 * np.sqrt(3.0))
        result = self._result(label, confidence, started, order[:3], similarities)
        return bind_local_decision(result, observation) if observation else result

    def _result(self, label: str, confidence: float, started: float, order=None, similarity=None) -> DevelopmentSortResult:
        latency = (time.perf_counter() - started) * 1000.0
        if label == "empty":
            return DevelopmentSortResult("empty", None, confidence, label, confidence, [(label, confidence)], latency)
        top = []
        if order is not None and similarity is not None:
            top = [(self.labels[int(i)], float(np.clip(similarity[int(i)], 0.0, 1.0))) for i in order]
        status = "not_cheese" if label == "not_cheese" else "ok"
        target = BIN_OF_TYPE.get(label)
        if confidence < self.min_confidence:
            status, target = "uncertain", None
        if status != "ok":
            target = None
        return DevelopmentSortResult(status, target, confidence, label, confidence, top, latency)


def make_sorter(config: FactoryConfig, mode: str) -> Sorter:
    threshold = float(config.section("perception")["min_confidence"])
    if mode == "development":
        return DevelopmentColorSorter(threshold)
    if mode != "model":
        raise ValueError(f"unknown classifier mode: {mode}")
    checkpoint = config.model_path
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"trained cheese checkpoint is missing: {checkpoint}. "
            "Supply/rebuild it before production use, or explicitly pass --classifier development for integration testing."
        )
    perception = config.section("perception")
    service_url = os.environ.get("CHEESE_SORTER_URL", perception["service_url"])
    return RemoteModelSorter(
        service_url,
        timeout_s=float(perception.get("service_timeout_s", 60.0)),
        decision_max_age_s=float(perception["decision_max_age_s"]),
    )


def annotate_frame(frame: np.ndarray, detection: Detection | None, result, state: str, output: Path) -> None:
    image = Image.fromarray(_rgb(frame))
    draw = ImageDraw.Draw(image)
    if detection is not None:
        draw.rectangle(detection.box, outline=(0, 255, 80), width=3)
    lines = [f"state: {state}"]
    if result is not None:
        lines.extend(
            [
                f"type: {result.cheese_type}",
                f"target: {result.bin or '-'}",
                f"confidence: {result.bin_confidence:.3f}",
                f"status: {result.status}",
                f"latency: {result.latency_ms:.1f} ms",
            ]
        )
    draw.rectangle((6, 6, 270, 20 + 18 * len(lines)), fill=(0, 0, 0))
    draw.multiline_text((12, 12), "\n".join(lines), fill=(255, 255, 255), spacing=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
