"""Response-driven policy checks, independent of MemriBoard state."""
from experiment_common.adaptive_programming import Feedback, TOLERANCE


def test_no_response_escalates_but_stays_bounded() -> None:
    policy = Feedback()
    for _ in range(40):
        sign, voltage = policy.command(10e-6, 40e-6)
        assert sign == 1 and .3 <= voltage <= 3.
        policy.observe(10e-6, 10e-6, 40e-6, sign, TOLERANCE)
    assert policy.ceiling_stalls >= 4


def test_overshoot_reduces_only_responsible_polarity() -> None:
    policy = Feedback()
    policy.observe(10e-6, 60e-6, 30e-6, 1, TOLERANCE)
    assert policy.levels[1] < 1.
    assert policy.levels[0] == 1.
    assert policy.command(60e-6, 30e-6)[0] == 0


def test_direction_polarities_and_recovery_from_ceiling_stall() -> None:
    policy = Feedback(ceiling_stalls=3)
    assert policy.command(80e-6, 20e-6)[0] == 0
    policy.observe(80e-6, 50e-6, 20e-6, 0, TOLERANCE)
    assert policy.ceiling_stalls == 0
