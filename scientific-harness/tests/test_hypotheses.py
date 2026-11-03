"""Check strip positions/widths, reuse for the same grid cells, and rejected requests."""

import numpy as np
import pytest

from cases.pressure import PressureCase
from controller.hypotheses import parse_hypothesis
from numerical.geology import BandGeometry, Family, encode_parameters, permeability_field


def test_new_geometry_changes_the_realization_and_equivalent_recipes_share_identity() -> None:
    case = PressureCase()
    first = parse_hypothesis(
        {"kind": "band", "orientation": "vertical", "center_fraction": 0.5, "width_fraction": 0.1},
        case,
    )
    equivalent = parse_hypothesis(
        {
            "kind": "band",
            "orientation": "vertical",
            "center_fraction": 0.499,
            "width_fraction": 0.1,
        },
        case,
    )
    shifted = parse_hypothesis(
        {"kind": "band", "orientation": "vertical", "center_fraction": 0.6, "width_fraction": 0.2},
        case,
    )
    assert first.identifier(case) == equivalent.identifier(case)
    assert first.identifier(case) != shifted.identifier(case)
    parameters = encode_parameters(Family.VERTICAL_BAND, 100)
    original = permeability_field(first.family, parameters, case, first.geometry)
    changed = permeability_field(shifted.family, parameters, case, shifted.geometry)
    assert not np.array_equal(original, changed)
    assert parse_hypothesis({"kind": "homogeneous"}, case).geometry is None


def test_band_resolution_is_validated_separately_from_case_settings() -> None:
    case = PressureCase(grid_size=8)
    with pytest.raises(ValueError, match="band width must lie"):
        parse_hypothesis(
            {
                "kind": "band",
                "orientation": "vertical",
                "center_fraction": 0.5,
                "width_fraction": 0.1,
            },
            case,
        )
    hypothesis = parse_hypothesis(
        {"kind": "band", "orientation": "vertical", "center_fraction": 0.5, "width_fraction": 0.25},
        case,
    )
    assert hypothesis.geometry == BandGeometry(0.5, 0.25)


@pytest.mark.parametrize(
    "specification",
    [
        {"kind": "python", "code": "read protected truth"},
        {"kind": "homogeneous", "observation_std_pa": 1e9},
        {"kind": "band", "orientation": "diagonal", "center_fraction": 0.5, "width_fraction": 0.1},
        {"kind": "band", "orientation": "vertical", "center_fraction": 0.5, "width_fraction": 0.01},
        {"kind": "band", "orientation": "vertical", "center_fraction": 0.01, "width_fraction": 0.2},
        {"kind": "band", "orientation": "vertical", "center_fraction": True, "width_fraction": 0.1},
        {
            "kind": "band",
            "orientation": "vertical",
            "center_fraction": float("nan"),
            "width_fraction": 0.1,
        },
    ],
)
def test_unsupported_requests_are_rejected(specification: object) -> None:
    with pytest.raises(ValueError):
        parse_hypothesis(specification, PressureCase())
