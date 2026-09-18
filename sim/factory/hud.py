"""Operator-visible status panel for the streamed Isaac factory."""

from __future__ import annotations

from dataclasses import dataclass, replace


MODE_BANNERS = {
    "model": "MODEL · TRAINED PERCEPTION",
    "development": "DEVELOPMENT · PIXEL PROXY",
    "showcase": "SHOWCASE · SCRIPTED ROUTING (NOT MODEL ACCURACY)",
}


@dataclass(frozen=True)
class HudSnapshot:
    mode: str
    scenario: str
    phase: str = "STARTING"
    object_id: str = "—"
    ground_truth: str = "—"
    prediction: str = "—"
    confidence: str = "—"
    destination: str = "—"
    completed: int = 0
    total: int = 0
    successful: int = 0
    rejected: int = 0
    failures: int = 0

    @property
    def mode_banner(self) -> str:
        return MODE_BANNERS[self.mode]

    @property
    def progress_text(self) -> str:
        return f"{self.completed}/{self.total} processed  ·  {self.successful} successful  ·  {self.rejected} safe rejects  ·  {self.failures} faults"


class FactoryHud:
    """Small floating Kit window updated by the camera-to-control loop."""

    def __init__(self, mode: str, scenario: str, total: int):
        import omni.ui as ui

        self._ui = ui
        self.snapshot = HudSnapshot(mode=mode, scenario=scenario, total=total)
        self.window = ui.Window("Cheese Factory · Live", width=430, height=285)
        self.window.position_x = 105
        self.window.position_y = 125
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label("PHYSICAL AI CHEESE FACTORY", height=28)
                self._mode = ui.Label(self.snapshot.mode_banner, height=24)
                self._scenario = ui.Label(f"Scenario: {scenario}", height=20)
                ui.Separator(height=4)
                self._phase = ui.Label("State: STARTING", height=22)
                self._object = ui.Label("Item: —", height=20)
                self._decision = ui.Label("Decision: —", height=20)
                self._target = ui.Label("Destination: —", height=20)
                self._confidence = ui.Label("Confidence: —", height=20)
                ui.Separator(height=4)
                self._progress = ui.Label(self.snapshot.progress_text, height=24)

    def update(self, **changes) -> None:
        self.snapshot = replace(self.snapshot, **changes)
        snap = self.snapshot
        self._mode.text = snap.mode_banner
        self._scenario.text = f"Scenario: {snap.scenario}"
        self._phase.text = f"State: {snap.phase}"
        self._object.text = f"Item: {snap.object_id}  ·  expected {snap.ground_truth}"
        self._decision.text = f"Decision: {snap.prediction}"
        self._target.text = f"Destination: {snap.destination}"
        self._confidence.text = f"Confidence: {snap.confidence}"
        self._progress.text = snap.progress_text

    def destroy(self) -> None:
        self.window.visible = False
        self.window = None
