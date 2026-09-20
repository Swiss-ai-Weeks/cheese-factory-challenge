"""Item-bound observation and decision timing for fail-closed robot control."""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Mapping

import numpy as np


ITEM_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class DecisionValidationError(RuntimeError):
    """A decision that cannot safely authorize an actuator command."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ObservationContext:
    item_id: str
    sequence: int
    request_id: str
    captured_at_epoch: float
    captured_at_monotonic: float
    frame_sha256: str

    @classmethod
    def capture(cls, item_id: str, sequence: int, frame: np.ndarray) -> "ObservationContext":
        if not ITEM_ID_PATTERN.fullmatch(str(item_id)):
            raise ValueError("item_id contains unsupported characters")
        if int(sequence) < 1:
            raise ValueError("observation sequence must be positive")
        pixels = np.ascontiguousarray(frame)
        return cls(
            item_id=str(item_id),
            sequence=int(sequence),
            request_id=str(uuid.uuid4()),
            captured_at_epoch=time.time(),
            captured_at_monotonic=time.monotonic(),
            frame_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        )

    def request_headers(self) -> dict[str, str]:
        return {
            "X-Cheese-Item-ID": self.item_id,
            "X-Cheese-Observation-Sequence": str(self.sequence),
            "X-Cheese-Request-ID": self.request_id,
            "X-Cheese-Observed-At": repr(self.captured_at_epoch),
            "X-Cheese-Frame-SHA256": self.frame_sha256,
        }


@dataclass(frozen=True)
class DecisionTiming:
    item_id: str
    sequence: int
    request_id: str
    frame_sha256: str
    observed_at_epoch: float
    server_received_at_epoch: float
    decided_at_epoch: float
    validated_at_epoch: float
    age_ms: float


def parse_request_headers(headers: Mapping[str, str]) -> dict[str, object]:
    """Parse and validate the correlation envelope received by the model service."""

    try:
        item_id = str(headers["X-Cheese-Item-ID"])
        sequence = int(headers["X-Cheese-Observation-Sequence"])
        request_id = str(headers["X-Cheese-Request-ID"])
        observed_at = float(headers["X-Cheese-Observed-At"])
        frame_sha256 = str(headers["X-Cheese-Frame-SHA256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid correlation header: {exc}") from exc
    if not ITEM_ID_PATTERN.fullmatch(item_id):
        raise ValueError("invalid item ID")
    if sequence < 1:
        raise ValueError("invalid observation sequence")
    try:
        uuid.UUID(request_id)
    except ValueError as exc:
        raise ValueError("invalid request ID") from exc
    if not math.isfinite(observed_at) or observed_at <= 0.0:
        raise ValueError("invalid observation timestamp")
    if not SHA256_PATTERN.fullmatch(frame_sha256):
        raise ValueError("invalid frame digest")
    return {
        "item_id": item_id,
        "observation_sequence": sequence,
        "request_id": request_id,
        "observed_at_epoch": observed_at,
        "frame_sha256": frame_sha256,
    }


def validate_decision(
    observation: ObservationContext,
    result,
    *,
    current_item_id: str | None,
    max_age_s: float,
    now_monotonic: float | None = None,
    now_epoch: float | None = None,
) -> DecisionTiming:
    """Reject missing, mismatched, future, or stale inference metadata."""

    required = {
        "item_id": getattr(result, "item_id", None),
        "sequence": getattr(result, "observation_sequence", None),
        "request_id": getattr(result, "request_id", None),
        "frame_sha256": getattr(result, "frame_sha256", None),
        "observed_at_epoch": getattr(result, "observed_at_epoch", None),
        "server_received_at_epoch": getattr(result, "server_received_at_epoch", None),
        "decided_at_epoch": getattr(result, "decision_at_epoch", None),
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise DecisionValidationError("missing_metadata", f"inference metadata missing: {', '.join(missing)}")
    if current_item_id != observation.item_id or str(required["item_id"]) != observation.item_id:
        raise DecisionValidationError("item_mismatch", "stale inference item mismatch")
    try:
        sequence = int(required["sequence"])
    except (TypeError, ValueError) as exc:
        raise DecisionValidationError("malformed_sequence", "inference sequence is malformed") from exc
    if sequence != observation.sequence:
        raise DecisionValidationError("sequence_mismatch", "stale inference sequence mismatch")
    if str(required["request_id"]) != observation.request_id:
        raise DecisionValidationError("request_mismatch", "stale inference request mismatch")
    if str(required["frame_sha256"]) != observation.frame_sha256:
        raise DecisionValidationError("frame_mismatch", "stale inference frame digest mismatch")
    try:
        observed = float(required["observed_at_epoch"])
        received = float(required["server_received_at_epoch"])
        decided = float(required["decided_at_epoch"])
    except (TypeError, ValueError) as exc:
        raise DecisionValidationError("malformed_timestamp", "inference timestamp is malformed") from exc
    if not all(math.isfinite(value) for value in (observed, received, decided)):
        raise DecisionValidationError("malformed_timestamp", "inference timestamp is not finite")
    if abs(observed - observation.captured_at_epoch) > 1e-6:
        raise DecisionValidationError("timestamp_mismatch", "inference observation timestamp mismatch")
    if received < observed or decided < received:
        raise DecisionValidationError("timestamp_order", "inference timestamps are out of order")

    monotonic_now = time.monotonic() if now_monotonic is None else float(now_monotonic)
    epoch_now = time.time() if now_epoch is None else float(now_epoch)
    age_s = monotonic_now - observation.captured_at_monotonic
    if age_s < 0.0:
        raise DecisionValidationError("clock_regression", "inference monotonic clock moved backwards")
    if decided > epoch_now + 1.0:
        raise DecisionValidationError("future_timestamp", "inference decision timestamp is in the future")
    if age_s > float(max_age_s):
        raise DecisionValidationError(
            "decision_timeout", f"stale inference exceeded {float(max_age_s):.3f}s deadline"
        )
    return DecisionTiming(
        item_id=observation.item_id,
        sequence=observation.sequence,
        request_id=observation.request_id,
        frame_sha256=observation.frame_sha256,
        observed_at_epoch=observation.captured_at_epoch,
        server_received_at_epoch=received,
        decided_at_epoch=decided,
        validated_at_epoch=epoch_now,
        age_ms=age_s * 1000.0,
    )
