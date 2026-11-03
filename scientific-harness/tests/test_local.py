"""Check parameter-version (member) order, whole-hypothesis failures, and simulator-call counts."""

import numpy as np
import pytest

from cases.pressure import PressureCase
from controller.local import LocalExecutor, MemberFailure
from numerical.geology import Family, permeability_field, sample_prior
from numerical.pressure import PressureModel, Simulation


def test_batch_preserves_member_order() -> None:
    case = PressureCase(grid_size=8, segment_steps=2, sample_every=1)
    model = PressureModel(case)
    executor = LocalExecutor(model, 4)
    parameters = sample_prior(Family.HOMOGENEOUS, 4, np.random.default_rng(7))
    responses = executor.batch(Family.HOMOGENEOUS, parameters, case.total_steps)
    for member, values in enumerate(parameters.T):
        field = permeability_field(Family.HOMOGENEOUS, values, case)
        reference = model.simulate(field, case.total_steps).responses_pa
        np.testing.assert_array_equal(responses[:, member], reference)
    assert executor.simulations == 4


def test_failed_member_stops_batch_without_zero_imputation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never fill a failed simulation's missing pressure results with zero."""
    case = PressureCase(grid_size=8, segment_steps=2, sample_every=1)
    model = PressureModel(case)
    original = model.simulate
    calls = 0

    def simulate(field: np.ndarray, steps: int) -> Simulation:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected member failure")
        return original(field, steps)

    monkeypatch.setattr(model, "simulate", simulate)
    executor = LocalExecutor(model, 4)
    parameters = sample_prior(Family.HOMOGENEOUS, 4, np.random.default_rng(7))
    with pytest.raises(MemberFailure, match="injected member failure") as failure:
        executor.batch(Family.HOMOGENEOUS, parameters, case.total_steps)
    assert failure.value.member_id == "member_0001"
    assert executor.simulations == 2


def test_budget_is_checked_before_another_call() -> None:
    case = PressureCase(grid_size=8, segment_steps=2, sample_every=1)
    executor = LocalExecutor(PressureModel(case), 2)
    parameters = sample_prior(Family.HOMOGENEOUS, 4, np.random.default_rng(7))
    with pytest.raises(MemberFailure, match="budget exhausted"):
        executor.batch(Family.HOMOGENEOUS, parameters, case.total_steps)
    assert executor.simulations == 2
