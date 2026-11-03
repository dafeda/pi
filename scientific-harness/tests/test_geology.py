"""Check permeability patterns, unit conversions, starting guesses, and invalid fitted numbers."""

import numpy as np
import pytest

from cases.pressure import PressureCase
from numerical.geology import (
    MILLIDARCY_M2,
    BandGeometry,
    Family,
    decode_parameters,
    encode_parameters,
    parameter_names,
    permeability_field,
    sample_prior,
)


@pytest.mark.parametrize("family", list(Family))
def test_parameter_round_trip_and_prior_reproducibility(family: Family) -> None:
    encoded = encode_parameters(family, 80.0, 0.015)
    decoded = decode_parameters(family, encoded[:, np.newaxis])[:, 0]
    np.testing.assert_allclose(decoded[0], 80.0)
    if family != Family.HOMOGENEOUS:
        np.testing.assert_allclose(decoded[1], 0.015)
    first = sample_prior(family, 50, np.random.default_rng(7))
    second = sample_prior(family, 50, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (len(parameter_names(family)), 50)
    assert (decode_parameters(family, first) > 0).all()


def test_fixed_bands_and_units() -> None:
    case = PressureCase()
    geometry = BandGeometry(0.5, 0.1)
    parameters = encode_parameters(Family.VERTICAL_BAND, 100.0, 0.01)
    vertical = permeability_field(Family.VERTICAL_BAND, parameters, case, geometry)
    horizontal = permeability_field(Family.HORIZONTAL_BAND, parameters, case, geometry)
    homogeneous = permeability_field(Family.HOMOGENEOUS, parameters[:1], case)
    np.testing.assert_allclose(vertical, horizontal.T)
    np.testing.assert_allclose(vertical[:, 9:11], MILLIDARCY_M2)
    np.testing.assert_allclose(vertical[:, :9], 100 * MILLIDARCY_M2)
    np.testing.assert_allclose(homogeneous, 100 * MILLIDARCY_M2)


@pytest.mark.parametrize("family", [Family.VERTICAL_BAND, Family.HORIZONTAL_BAND])
def test_band_families_require_explicit_geometry(family: Family) -> None:
    parameters = encode_parameters(family, 100.0, 0.01)
    with pytest.raises(ValueError, match="band families require explicit geometry"):
        permeability_field(family, parameters, PressureCase())


@pytest.mark.parametrize("ratio", [0.0, 1.0, -1.0, np.nan])
def test_invalid_physical_ratio_is_not_clipped(ratio: float) -> None:
    with pytest.raises(ValueError):
        encode_parameters(Family.VERTICAL_BAND, 100.0, ratio)


def test_invalid_updated_support_is_not_clipped() -> None:
    with pytest.raises(ValueError, match="rounded to 0 or 1"):
        decode_parameters(Family.VERTICAL_BAND, np.array([[np.log(100.0)], [1000.0]]))
    with pytest.raises(FloatingPointError):
        decode_parameters(Family.HOMOGENEOUS, np.array([[1000.0]]))
    with pytest.raises(ValueError, match="shape"):
        decode_parameters(Family.VERTICAL_BAND, np.zeros((1, 50)))
