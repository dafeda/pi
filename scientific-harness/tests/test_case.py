"""Check well rates and readings split between fitting, ranking, and final testing."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from cases.pressure import Phase, PressureCase


def test_observation_order_and_disjoint_cutoffs() -> None:
    case = PressureCase()
    assert case.observation_ids[:4].tolist() == [
        "step_0002:south_west",
        "step_0002:south_east",
        "step_0002:north_west",
        "step_0002:north_east",
    ]
    masks = [case.phase_mask(phase) for phase in Phase]
    assert [int(mask.sum()) for mask in masks] == [64, 32, 32]
    np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(128))
    assert len(set(case.observation_ids)) == len(case.observation_ids)


def test_controls_are_balanced_and_change_after_conditioning() -> None:
    case = PressureCase()
    np.testing.assert_allclose(case.rates.sum(axis=1), 0, atol=1e-20)
    assert not np.array_equal(case.rates[:16], case.rates[32:48])
    assert not np.array_equal(case.rates[32:48], case.rates[48:64])
    assert "truth" not in case.specification()
    # This label is part of the saved-data format, not free-form explanatory text.
    assert (
        case.specification()["observation_error"]
        == "independent Gaussian, fixed standard deviation"
    )


@pytest.mark.parametrize("grid_size", [4, 8, 20])
def test_case_settings_do_not_depend_on_band_geometry(grid_size: int) -> None:
    case = PressureCase(grid_size=grid_size)
    settings = asdict(case)
    assert settings["grid_size"] == grid_size
    assert case.specification()["settings"] == settings
    assert "band_width_fraction" not in settings


@pytest.mark.parametrize(
    "settings",
    [
        {"grid_size": 0},
        {"grid_size": 2},
        {"grid_size": 9},
        {"storage_per_pa": -1},
        {"viscosity_pa_s": np.nan},
        {"sample_every": 3},
    ],
)
def test_invalid_case_is_rejected(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(PressureCase(), **settings)
