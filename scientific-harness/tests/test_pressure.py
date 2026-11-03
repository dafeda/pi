"""Compare pressure with FiPy solves; check fluid balance, changed rates, and time steps."""

from dataclasses import replace

import numpy as np
import pytest
from fipy import CellVariable, DiffusionTerm, TransientTerm
from fipy.solvers.scipy import LinearLUSolver
from numpy.typing import NDArray

from cases.pressure import PressureCase
from numerical.geology import BandGeometry, Family, encode_parameters, permeability_field
from numerical.pressure import PressureModel


def direct_fipy_responses(
    model: PressureModel, field: NDArray[np.float64], steps: int
) -> NDArray[np.float64]:
    """Use a slower comparison that solves through FiPy again at every time step."""
    case = model.case
    pressure = CellVariable(mesh=model.mesh, value=0.0, hasOld=True)
    permeability = CellVariable(mesh=model.mesh, value=field.ravel())
    source = CellVariable(mesh=model.mesh, value=0.0)
    equation = TransientTerm(coeff=case.storage_per_pa) == (
        DiffusionTerm(coeff=permeability.harmonicFaceValue / case.viscosity_pa_s) + source
    )
    responses = []
    for step in range(steps):
        pressure.updateOld()
        source.setValue(model.source_rhs[step] / model.cell_areas_m2)
        equation.solve(var=pressure, dt=case.time_step_s, solver=LinearLUSolver(tolerance=1e-12))
        if (step + 1) % case.sample_every == 0:
            responses.extend(pressure.value[model.gauge_cells] + case.initial_pressure_pa)
    return np.array(responses)


@pytest.mark.parametrize("family", list(Family))
def test_reused_factorization_matches_direct_fipy(family: Family) -> None:
    case = PressureCase(grid_size=8, segment_steps=2, sample_every=1)
    model = PressureModel(case)
    geometry = None if family == Family.HOMOGENEOUS else BandGeometry(0.5, 0.25)
    field = permeability_field(family, encode_parameters(family, 100.0, 0.01), case, geometry)
    result = model.simulate(field, case.total_steps)
    reference = direct_fipy_responses(model, field, case.total_steps)
    np.testing.assert_allclose(result.responses_pa, reference, rtol=0, atol=1e-6)
    assert result.max_balance_error_m3 < 1e-8
    assert abs(result.final_pressure_pa.mean() - case.initial_pressure_pa) < 1e-6


def test_cell_order_source_normalization_and_rate_linearity() -> None:
    case = PressureCase(grid_size=8, segment_steps=2, sample_every=1)
    model = PressureModel(case)
    assert model.gauge_cells.tolist() == [18, 21, 42, 45]
    np.testing.assert_allclose(model.source_rhs.sum(axis=1), 0, atol=1e-20)
    assert model.source_rhs[0, model.well_cells[0]] == case.rate_m3_s / case.thickness_m
    field = permeability_field(Family.HOMOGENEOUS, np.array([np.log(100.0)]), case)
    first = model.simulate(field, case.total_steps).responses_pa - case.initial_pressure_pa
    doubled_case = replace(case, rate_m3_s=2 * case.rate_m3_s)
    doubled = PressureModel(doubled_case).simulate(field, case.total_steps).responses_pa
    np.testing.assert_allclose(doubled - case.initial_pressure_pa, 2 * first, atol=1e-6)


def test_observations_distinguish_families_and_changed_parameters() -> None:
    case = PressureCase()
    model = PressureModel(case)
    responses = [
        model.simulate(
            permeability_field(
                family,
                encode_parameters(family, 100.0, 0.01),
                case,
                None if family == Family.HOMOGENEOUS else BandGeometry(0.5, 0.1),
            ),
            4 * case.segment_steps,
        ).responses_pa
        for family in Family
    ]
    for first in range(len(responses)):
        for second in range(first):
            difference = np.max(np.abs(responses[first] - responses[second]))
            assert difference > 5 * case.observation_std_pa
    changed = model.simulate(
        permeability_field(
            Family.VERTICAL_BAND,
            encode_parameters(Family.VERTICAL_BAND, 50.0),
            case,
            BandGeometry(0.5, 0.1),
        ),
        4 * case.segment_steps,
    ).responses_pa
    assert not np.array_equal(changed, responses[1])


def test_time_refinement_reduces_difference_from_fine_reference() -> None:
    case = PressureCase(grid_size=8, segment_steps=2, sample_every=1)
    results = []
    for refinement in (1, 2, 4):
        refined = replace(
            case,
            time_step_s=case.time_step_s / refinement,
            segment_steps=case.segment_steps * refinement,
            sample_every=case.sample_every * refinement,
        )
        field = permeability_field(
            Family.VERTICAL_BAND,
            encode_parameters(Family.VERTICAL_BAND, 100.0),
            refined,
            BandGeometry(0.5, 0.25),
        )
        results.append(PressureModel(refined).simulate(field, refined.total_steps).responses_pa)
    assert np.linalg.norm(results[1] - results[2]) < np.linalg.norm(results[0] - results[2])


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_field_is_rejected(value: float) -> None:
    model = PressureModel(PressureCase())
    with pytest.raises(ValueError, match="finite and positive"):
        model.simulate(np.full((20, 20), value), 32)
