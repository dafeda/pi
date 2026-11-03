"""Check that final comparison plots render as a PNG image without opening a window."""

import numpy as np

from cases.pressure import PressureCase
from numerical.geology import BandGeometry, Family, encode_parameters, permeability_field
from numerical.report import diagnostic_png


def test_diagnostics_render_without_a_gui() -> None:
    case = PressureCase(grid_size=8, segment_steps=2, sample_every=1)
    field = permeability_field(
        Family.VERTICAL_BAND,
        encode_parameters(Family.VERTICAL_BAND, 100.0),
        case,
        BandGeometry(0.5, 0.25),
    )
    truth = case.initial_pressure_pa + np.arange(case.observation_ids.size) * 100.0
    predictions = np.repeat(truth[:, np.newaxis], 4, axis=1)
    image = diagnostic_png(case, field, field, truth, predictions, truth, "vertical_band")
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) > 1000
