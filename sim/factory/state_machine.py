"""Explicit safety state machine for autonomous sorting cycles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class FactoryState(Enum):
    WAITING = auto()
    OBJECT_DETECTED = auto()
    CLASSIFYING = auto()
    STOPPING_BELT = auto()
    APPROACHING = auto()
    DESCENDING = auto()
    GRASPING = auto()
    LIFTING = auto()
    MOVING_TO_BIN = auto()
    RELEASING = auto()
    RETURNING_HOME = auto()
    REJECTING = auto()
    RECOVERY = auto()
    COMPLETE = auto()
    FAILED = auto()


ALLOWED_TRANSITIONS: dict[FactoryState, set[FactoryState]] = {
    FactoryState.WAITING: {FactoryState.OBJECT_DETECTED, FactoryState.WAITING, FactoryState.FAILED},
    FactoryState.OBJECT_DETECTED: {FactoryState.CLASSIFYING, FactoryState.RECOVERY},
    FactoryState.CLASSIFYING: {FactoryState.STOPPING_BELT, FactoryState.REJECTING, FactoryState.RECOVERY},
    FactoryState.STOPPING_BELT: {FactoryState.APPROACHING, FactoryState.RECOVERY},
    FactoryState.APPROACHING: {FactoryState.DESCENDING, FactoryState.RECOVERY},
    FactoryState.DESCENDING: {FactoryState.GRASPING, FactoryState.RECOVERY},
    FactoryState.GRASPING: {FactoryState.LIFTING, FactoryState.RECOVERY},
    FactoryState.LIFTING: {FactoryState.MOVING_TO_BIN, FactoryState.RECOVERY},
    FactoryState.MOVING_TO_BIN: {FactoryState.RELEASING, FactoryState.RECOVERY},
    FactoryState.RELEASING: {FactoryState.RETURNING_HOME, FactoryState.RECOVERY},
    FactoryState.RETURNING_HOME: {FactoryState.COMPLETE, FactoryState.RECOVERY},
    FactoryState.REJECTING: {FactoryState.COMPLETE, FactoryState.RECOVERY},
    FactoryState.RECOVERY: {FactoryState.WAITING, FactoryState.FAILED},
    FactoryState.COMPLETE: {FactoryState.WAITING},
    FactoryState.FAILED: {FactoryState.RECOVERY},
}


@dataclass
class FactoryStateMachine:
    state: FactoryState = FactoryState.WAITING
    object_id: str | None = None
    elapsed_s: float = 0.0
    failure_reason: str | None = None
    history: list[str] = field(default_factory=lambda: [FactoryState.WAITING.name])

    def transition(self, target: FactoryState, reason: str | None = None) -> None:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"unsafe transition {self.state.name} -> {target.name}")
        self.state = target
        self.elapsed_s = 0.0
        self.failure_reason = reason if target in {FactoryState.RECOVERY, FactoryState.FAILED} else None
        self.history.append(target.name)

    def tick(self, dt: float, timeout_s: float | None = None) -> bool:
        self.elapsed_s += max(0.0, float(dt))
        if timeout_s is not None and self.elapsed_s > timeout_s:
            reason = f"{self.state.name} timed out after {self.elapsed_s:.2f}s"
            if FactoryState.RECOVERY in ALLOWED_TRANSITIONS[self.state]:
                self.transition(FactoryState.RECOVERY, reason)
            else:
                self.transition(FactoryState.FAILED, reason)
            return False
        return True

    def begin(self, object_id: str) -> None:
        if self.state is not FactoryState.WAITING:
            raise RuntimeError(f"cannot begin {object_id}: currently {self.state.name}")
        self.object_id = object_id
        self.transition(FactoryState.OBJECT_DETECTED)

    def classification(self, status: str) -> None:
        if self.state is FactoryState.OBJECT_DETECTED:
            self.transition(FactoryState.CLASSIFYING)
        if self.state is not FactoryState.CLASSIFYING:
            raise RuntimeError("classification is only accepted in CLASSIFYING")
        self.transition(FactoryState.STOPPING_BELT if status == "ok" else FactoryState.REJECTING)

    def reset_cycle(self) -> None:
        if self.state is not FactoryState.COMPLETE:
            raise RuntimeError("only a complete cycle can return to WAITING")
        self.object_id = None
        self.transition(FactoryState.WAITING)
