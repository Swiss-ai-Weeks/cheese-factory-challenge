from sim.factory.config import load_config
from sim.factory.layout import Footprint, canonical_footprints, overlapping_pairs


def test_overlap_detection_distinguishes_touching_and_intersection():
    first = Footprint("first", 0.0, 0.0, 1.0, 1.0)
    touching = Footprint("touching", 1.0, 0.0, 1.0, 1.0)
    intersecting = Footprint("intersecting", 0.99, 0.0, 1.0, 1.0)

    assert not first.overlaps(touching)
    assert first.overlaps(intersecting)


def test_canonical_receivers_clear_conveyor_and_each_other():
    footprints = canonical_footprints(load_config())
    assert overlapping_pairs(footprints) == []


def test_canonical_layout_preserves_visible_air_gap():
    footprints = canonical_footprints(load_config())
    assert overlapping_pairs(footprints, clearance=0.004) == []
