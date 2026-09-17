import pytest

from sim.factory.state_machine import FactoryState, FactoryStateMachine


def test_ok_is_the_only_classification_that_reaches_pick_path():
    for status in ("empty", "not_cheese", "uncertain"):
        machine = FactoryStateMachine()
        machine.begin("reject")
        machine.classification(status)
        assert machine.state is FactoryState.REJECTING

    machine = FactoryStateMachine()
    machine.begin("cheese")
    machine.classification("ok")
    assert machine.state is FactoryState.STOPPING_BELT


def test_duplicate_cycle_and_unsafe_transition_are_rejected():
    machine = FactoryStateMachine()
    machine.begin("one")
    with pytest.raises(RuntimeError):
        machine.begin("two")
    with pytest.raises(ValueError):
        machine.transition(FactoryState.RELEASING)


def test_timeout_enters_recovery():
    machine = FactoryStateMachine()
    machine.begin("one")
    machine.classification("ok")
    assert not machine.tick(2.1, timeout_s=2.0)
    assert machine.state is FactoryState.RECOVERY
    assert "timed out" in machine.failure_reason
