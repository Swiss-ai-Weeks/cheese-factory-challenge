"""Pure geometry contract for the canonical factory floor plan."""

from __future__ import annotations

from dataclasses import dataclass


BELT_CENTER_XY = (0.50, -0.395)
BELT_SIZE_XY = (0.16, 1.11)
RECEIVER_SIZE_XY = (0.17, 0.17)
REJECT_SIZE_XY = (0.17, 0.17)


@dataclass(frozen=True)
class Footprint:
    name: str
    center_x: float
    center_y: float
    size_x: float
    size_y: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.center_x - self.size_x / 2.0,
            self.center_x + self.size_x / 2.0,
            self.center_y - self.size_y / 2.0,
            self.center_y + self.size_y / 2.0,
        )

    def overlaps(self, other: "Footprint", clearance: float = 0.0) -> bool:
        left, right, bottom, top = self.bounds
        other_left, other_right, other_bottom, other_top = other.bounds
        return not (
            right + clearance <= other_left
            or other_right + clearance <= left
            or top + clearance <= other_bottom
            or other_top + clearance <= bottom
        )


def canonical_footprints(config) -> list[Footprint]:
    footprints = [Footprint("conveyor", *BELT_CENTER_XY, *BELT_SIZE_XY)]
    footprints.extend(
        Footprint(name, float(position[0]), float(position[1]), *RECEIVER_SIZE_XY)
        for name, position in config.bins.items()
    )
    reject = config.raw["reject_position"]
    footprints.append(Footprint("reject", float(reject[0]), float(reject[1]), *REJECT_SIZE_XY))
    return footprints


def overlapping_pairs(footprints: list[Footprint], clearance: float = 0.0) -> list[tuple[str, str]]:
    return [
        (first.name, second.name)
        for index, first in enumerate(footprints)
        for second in footprints[index + 1 :]
        if first.overlaps(second, clearance=clearance)
    ]
