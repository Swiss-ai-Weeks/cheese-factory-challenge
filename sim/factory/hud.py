"""Operator-visible status panel for the streamed Isaac factory."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math


MODE_BANNERS = {
    "model": "MODEL · TRAINED PERCEPTION",
    "development": "DEVELOPMENT · PIXEL PROXY",
    "showcase": "SHOWCASE · SCRIPTED ROUTING (NOT MODEL ACCURACY)",
}

# Shared by the output legend, destination badge and probability bars. ABGR is
# the packed color convention used by omni.ui. Keep this module importable
# without Kit so display semantics can be tested independently of the renderer.
OUTPUTS = (
    ("bin_hard", "HARD", 0xFF4BAEF4),
    ("bin_semi_hard", "SEMI-HARD", 0xFF57D6ED),
    ("bin_soft", "SOFT", 0xFF91D59A),
    ("bin_fresh", "FRESH", 0xFFF0C06A),
    ("bin_blue", "BLUE", 0xFFF3A79C),
    ("reject", "REJECT / HOLD", 0xFF8288F5),
)
INK = 0xFFF3F6F8
MUTED = 0xFFAFB9C6
ACCENT = 0xFFCEED8D
PANEL = 0xF21C1712
TRACK = 0xFF38312A


def humanize(value: str) -> str:
    return value.replace("_", " ").replace("→", "/").replace("—", "-").strip()


def decision_details(mode: str, result) -> dict:
    """Never present scripted/proxy scores as learned probabilities."""
    learned = mode == "model"
    scores = tuple(
        (humanize(str(label)), float(score))
        for label, score in result.topk_types[:3]
        if math.isfinite(float(score)) and 0.0 <= float(score) <= 1.0
    ) if learned else ()
    return {
        "cheese_type": humanize(result.cheese_type),
        "route_key": result.bin or "reject",
        "topk": scores,
        "inference_ms": float(result.latency_ms) if learned else None,
        "decision_note": (
            "Type / route disagreement · arm held"
            if learned and not result.agreement
            else "Learned bin confidence · type scores below" if learned
            else "Scenario route · no classifier inference" if mode == "showcase"
            else "Pixel-color proxy · no trained inference"
        ),
    }

PRESENTATION_WINDOW_TITLES = (
    "Stage",
    "Layer",
    "Render Settings",
    "Robot Inspector",
    "Property",
    "Console",
    "Content",
)


def configure_presentation_workspace(workspace=None) -> tuple[str, ...]:
    """Hide editor-only panes so the streamed viewport is the primary surface."""
    if workspace is None:
        import omni.ui as ui

        workspace = ui.Workspace

    hidden = []
    for title in PRESENTATION_WINDOW_TITLES:
        window = workspace.get_window(title)
        if window is not None and window.visible:
            window.visible = False
            hidden.append(title)
    return tuple(hidden)


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
    timing: str = "—"
    safety: str = "ARM INHIBITED · awaiting valid decision"
    completed: int = 0
    total: int = 0
    successful: int = 0
    rejected: int = 0
    failures: int = 0
    cheese_type: str = "Waiting for inspection"
    route_key: str = ""
    topk: tuple[tuple[str, float], ...] = ()
    inference_ms: float | None = None
    decision_note: str = "Waiting for a captured observation"
    output_counts: tuple[int, ...] = (0, 0, 0, 0, 0, 0)

    @property
    def mode_banner(self) -> str:
        return MODE_BANNERS[self.mode]

    @property
    def progress_text(self) -> str:
        return f"{self.completed}/{self.total} processed  ·  {self.successful} successful  ·  {self.rejected} safe rejects  ·  {self.failures} faults"

    @property
    def progress(self) -> float:
        return min(1.0, max(0.0, self.completed / max(1, self.total)))

    @property
    def accent(self) -> int:
        return next((color for key, _, color in OUTPUTS if key == self.route_key), ACCENT)

    def with_outcome(self, destination: str | None, placed: bool, diverted: bool) -> "HudSnapshot":
        """Count completed actions, never a prediction or an attempted pick."""
        key = "reject" if diverted else destination if placed else None
        counts = list(self.output_counts)
        for index, (output, _, _) in enumerate(OUTPUTS):
            if key == output:
                counts[index] += 1
                break
        return replace(self, output_counts=tuple(counts))


class FactoryHud:
    """Viewport-attached operator dashboard; no extra camera/render product.

    The inspection image is the last captured decision crop, not a live video
    feed. It is cleared at the next item to avoid associating stale pixels with
    a new decision. Widget updates happen only on runtime events.
    """

    def __init__(self, mode: str, scenario: str, total: int):
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport_window

        self._ui = ui
        self._viewport_settings = {}
        self.snapshot = HudSnapshot(mode=mode, scenario=scenario, total=total)
        viewport = get_active_viewport_window()
        self.window = None
        if viewport is None:
            # Headless evaluation can have no active viewport; telemetry still
            # exists, but we must not make robot execution depend on a window.
            self._frame = None
            return
        self._frame = viewport.get_frame("cheese.factory.operator_hud")
        self._frame.clear()
        self._provider = ui.ByteImageProvider()
        with self._frame:
            with ui.VStack(spacing=10, opaque_for_mouse_events=False):
                ui.Spacer(height=32)  # Leave Kit's camera/renderer toolbar usable.
                with ui.HStack(height=68):
                    ui.Spacer(width=12)
                    with self._card():
                        with ui.HStack(spacing=16):
                            with ui.VStack(spacing=2):
                                self._label("CHEESE / SORTING LINE", 23, INK, height=28)
                                self._mode = self._label("", 12, ACCENT, height=18)
                            ui.Spacer()
                            self._label("ISAAC SIM / PHYSICAL AI", 13, MUTED, width=250)
                            with ui.VStack(width=215, spacing=3):
                                self._progress = self._label("", 19, INK, height=27)
                                self._progress_bar = self._bar(5, ACCENT)
                    ui.Spacer(width=12)
                with ui.HStack(spacing=12):
                    ui.Spacer(width=0)
                    # Left rail leaves the manipulation area unobstructed.
                    with ui.VStack(width=270, spacing=10):
                        with self._card(height=245):
                            with ui.VStack(spacing=5):
                                self._label("OUTPUT / COMPLETED ACTIONS", 12, MUTED, height=22)
                                self._counts = []
                                for _, title, color in OUTPUTS:
                                    with ui.HStack(height=24, spacing=10):
                                        ui.Rectangle(width=4, style={"background_color": color, "border_radius": 2})
                                        self._label(title, 13, color)
                                        self._counts.append(self._label("0", 19, INK, width=34))
                                self._label("Placements + diverted rejects", 11, MUTED, height=16)
                        ui.Spacer()
                    ui.Spacer()
                    with ui.VStack(width=270, spacing=10):
                        with self._card(height=155):
                            with ui.VStack(spacing=5):
                                self._label("FRANKA / cuMotion", 12, MUTED, height=19)
                                self._phase = self._label("", 18, INK, height=44, word_wrap=True)
                                self._safety = self._label("", 12, ACCENT, height=34, word_wrap=True)
                                self._faults = self._label("", 12, MUTED, height=19)
                        ui.Spacer()
                    ui.Spacer(width=0)
                with ui.HStack(height=220, spacing=12):
                    ui.Spacer(width=0)
                    with self._card(width=240):
                        with ui.VStack(spacing=4):
                            self._label("INSPECTION / CAPTURE", 11, MUTED, height=18)
                            with ui.ZStack(height=147):
                                ui.Rectangle(style={"background_color": 0xFF28231E, "border_radius": 5})
                                self._image = ui.ImageWithProvider(self._provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT, visible=False)
                                self._image_empty = self._label("Awaiting observation", 13, MUTED, alignment=ui.Alignment.CENTER)
                            self._capture = self._label("No image captured", 11, MUTED, height=16)
                    with self._card(width=440):
                        with ui.VStack(spacing=4):
                            with ui.HStack(height=18):
                                self._decision_heading = self._label("", 11, MUTED)
                                self._inference = self._label("", 11, MUTED, width=120, alignment=ui.Alignment.RIGHT_CENTER)
                            self._decision = self._label("", 25, INK, height=31)
                            with ui.HStack(height=29, spacing=8):
                                self._target = self._label("", 17, ACCENT)
                                self._confidence = self._label("", 19, INK, width=130, alignment=ui.Alignment.RIGHT_CENTER)
                            self._decision_note = self._label("", 11, MUTED, height=18)
                            self._score_rows = []
                            with ui.ZStack(height=53):
                                with ui.VStack(spacing=4) as self._scores:
                                    for _ in range(3):
                                        with ui.HStack(height=15, spacing=8):
                                            name = self._label("", 11, MUTED, width=175)
                                            bar = self._bar(5, ACCENT)
                                            score = self._label("", 11, INK, width=50, alignment=ui.Alignment.RIGHT_CENTER)
                                            self._score_rows.append((name, bar, score))
                                self._evidence = self._label("", 12, MUTED, word_wrap=True)
                            self._timing = self._label("", 11, MUTED, height=16)
                    ui.Spacer()
                    with ui.VStack(width=270, spacing=5):
                        ui.Spacer()
                        with self._card(height=77):
                            with ui.VStack(spacing=5):
                                self._object = self._label("", 14, INK, height=20)
                                self._expected = self._label("", 11, MUTED, height=17)
                        with self._card(height=74):
                            with ui.VStack(spacing=4):
                                self._label("SESSION", 11, MUTED, height=16)
                                self._scenario = self._label(scenario, 12, INK, height=30, word_wrap=True)
                    ui.Spacer(width=0)
                ui.Spacer(height=12)
        self._render()

    def configure_viewport(self) -> None:
        """Apply presentation settings after scene loading resets the viewport."""
        if self._frame is None:
            return
        import carb
        from omni.kit.viewport.utility import get_active_viewport_window

        viewport = get_active_viewport_window()
        if viewport is None:
            return
        settings = carb.settings.get_settings()
        overrides = {
            f"/persistent/app/viewport/{viewport.viewport_api.id}/guide/grid/visible": False,
            "/app/viewport/grid/enabled": False,
            "/app/viewport/forceHideFps": True,
        }
        for key, value in overrides.items():
            if key not in self._viewport_settings:
                self._viewport_settings[key] = settings.get(key)
            settings.set_bool(key, value)

    def _label(self, text, size, color, **kwargs):
        return self._ui.Label(text, style={"font_size": size * 1.3, "color": color}, **kwargs)

    def _bar(self, height, color):
        ui = self._ui
        with ui.ZStack(height=height):
            ui.Rectangle(style={"background_color": TRACK, "border_radius": 2})
            with ui.HStack():
                fill = ui.Rectangle(width=0, style={"background_color": color, "border_radius": 2})
                ui.Spacer()
        return fill

    @contextmanager
    def _card(self, **kwargs):
        ui = self._ui
        with ui.ZStack(**kwargs):
            ui.Rectangle(style={"background_color": PANEL, "border_radius": 9, "border_width": 1, "border_color": 0xFF484239})
            with ui.VStack():
                ui.Spacer(height=12)
                with ui.HStack():
                    ui.Spacer(width=14)
                    with ui.Frame():
                        yield
                    ui.Spacer(width=14)
                ui.Spacer(height=12)

    def update(self, **changes) -> None:
        self.snapshot = replace(self.snapshot, **changes)
        self._render()

    def record_outcome(self, destination=None, *, placed=False, diverted=False) -> None:
        self.snapshot = self.snapshot.with_outcome(destination, placed, diverted)
        self._render()

    def clear_capture(self) -> None:
        if self._frame is not None:
            self._image.visible = False
            self._image_empty.visible = True
            self._capture.text = "Awaiting current item"

    def set_capture(self, pixels, item_id: str) -> None:
        if self._frame is None:
            return
        import numpy as np
        from PIL import Image

        # Bounded, event-driven upload, never a second render product or an
        # on-disk screenshot. The crop is exactly what the classifier receives.
        image = Image.fromarray(np.asarray(pixels, dtype=np.uint8)[..., :3])
        original = image.size
        image.thumbnail((256, 256))
        rgba = np.asarray(image.convert("RGBA"))
        self._provider.set_bytes_data(rgba.ravel().tolist(), list(image.size))
        self._image.visible = True
        self._image_empty.visible = False
        self._capture.text = f"{item_id} · {original[0]} x {original[1]} px"

    def _render(self) -> None:
        if self._frame is None:
            return
        snap = self.snapshot
        self._mode.text = snap.mode_banner
        self._scenario.text = snap.scenario
        self._phase.text = humanize(snap.phase.replace("ROBOT · ", ""))
        self._object.text = snap.object_id
        self._expected.text = f"Scenario truth: {humanize(snap.ground_truth)}"
        self._decision_heading.text = {"model": "MODEL / DECISION", "showcase": "SCENARIO / SCRIPTED ROUTE", "development": "DEVELOPMENT / PIXEL PROXY"}[snap.mode]
        self._decision.text = snap.cheese_type
        self._decision.tooltip = snap.prediction
        self._target.text = (
            "SAFE HOLD" if "SAFE HOLD" in snap.destination
            else "REJECT / HOLD" if snap.route_key == "reject"
            else humanize(snap.destination).upper()
        )
        self._target.tooltip = snap.destination
        self._target.style = {"font_size": 17 * 1.3, "color": snap.accent}
        self._confidence.text = (
            f"route {humanize(snap.confidence)}"
            if snap.mode == "model" and snap.route_key else humanize(snap.confidence)
        )
        self._confidence.style = {"font_size": 15 * 1.3, "color": INK}
        self._decision_note.text = snap.decision_note
        self._inference.text = f"{snap.inference_ms:.0f} ms inference" if snap.inference_ms is not None else ""
        self._timing.text = snap.timing
        self._safety.text = snap.safety
        self._safety.style = {"font_size": 12 * 1.3, "color": OUTPUTS[-1][2] if snap.failures else ACCENT}
        self._faults.text = f"{sum(snap.output_counts[:5])} placed  /  {snap.output_counts[5]} diverted  /  {snap.failures} failed"
        self._progress.text = f"{snap.completed:02d} / {snap.total:02d} COMPLETE"
        self._progress_bar.width = self._ui.Percent(snap.progress * 100)
        self._progress_bar.visible = snap.progress > 0
        for count, label in zip(snap.output_counts, self._counts):
            label.text = str(count)
        for index, (name, bar, score) in enumerate(self._score_rows):
            available = index < len(snap.topk)
            name.visible = bar.visible = score.visible = available
            if available:
                title, probability = snap.topk[index]
                name.text = title
                score.text = f"{probability:.1%}"
                bar.width = self._ui.Percent(probability * 100)
                bar.style = {"background_color": snap.accent, "border_radius": 2}
        self._evidence.visible = not snap.topk
        self._scores.visible = bool(snap.topk)
        self._evidence.text = (
            "Rendered camera detection\nSimulated reject diversion"
            if snap.mode == "showcase" and snap.route_key == "reject"
            else "Rendered camera detection\nPhysical robot pick-and-place"
            if snap.mode == "showcase" and snap.route_key
            else "Trained probabilities appear after inference."
            if snap.mode == "model" else "Waiting for the current camera observation."
        )

    def destroy(self) -> None:
        if self._frame is not None:
            self._frame.clear()
            self._frame = None
            import carb
            settings = carb.settings.get_settings()
            for key, previous in self._viewport_settings.items():
                if previous is None:
                    settings.destroy_item(key)
                else:
                    settings.set(key, previous)
