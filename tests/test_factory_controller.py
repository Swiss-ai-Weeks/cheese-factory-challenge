from sim.factory.controller import PHASE_TO_FACTORY_STATE


def test_controller_phases_cover_safe_pick_place_sequence():
    assert PHASE_TO_FACTORY_STATE["APPROACH_PICK"] == "APPROACHING"
    assert PHASE_TO_FACTORY_STATE["GRASP"] == "GRASPING"
    assert PHASE_TO_FACTORY_STATE["RELEASE"] == "RELEASING"
    assert PHASE_TO_FACTORY_STATE["FAILED"] == "RECOVERY"
