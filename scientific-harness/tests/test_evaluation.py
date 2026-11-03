"""Check the pressure-score formula, repeatable noise draws, and fitted-parameter summaries."""

import numpy as np
import pytest

from numerical.evaluation import parameter_summary, predictive_ensemble, predictive_scores
from numerical.geology import Family, decode_parameters


def test_empirical_crps_matches_pairwise_definition() -> None:
    predictions = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 6.0]]) + 2e7
    observations = np.array([2.5, 4.0]) + 2e7
    first_term = np.abs(predictions - observations[:, np.newaxis]).mean(axis=1)
    pairwise = np.abs(predictions[:, :, np.newaxis] - predictions[:, np.newaxis, :])
    expected = (first_term - 0.5 * pairwise.mean(axis=(1, 2))).mean()
    scores = predictive_scores(predictions, observations)
    np.testing.assert_allclose(scores["crps_pa"], expected, rtol=0, atol=1e-8)
    assert scores["coverage_90"] == 1
    assert scores["interval_width_90_pa"] > 0


def test_predictive_observations_include_reproducible_error() -> None:
    responses = np.full((2, 5000), 2e7)
    first = predictive_ensemble(responses, 1000.0, np.random.default_rng(7))
    second = predictive_ensemble(responses, 1000.0, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, responses)
    assert abs((first - responses).std() - 1000.0) < 30.0


def test_band_permeability_summary_uses_joint_member_parameters() -> None:
    parameters = np.array([[np.log(50.0), np.log(200.0)], [-3.0, -5.0]])
    summary = parameter_summary(Family.VERTICAL_BAND, parameters)
    physical = decode_parameters(Family.VERTICAL_BAND, parameters)
    assert summary["band_md"]["mean"] == np.mean(physical[0] * physical[1])
    assert "band_ratio" not in parameter_summary(Family.HOMOGENEOUS, parameters[:1])


def test_scoring_rejects_missing_or_misaligned_predictions() -> None:
    with pytest.raises(ValueError, match="matching observed data"):
        predictive_scores(np.ones((2, 5)), np.ones(3))
    with pytest.raises(ValueError, match="non-finite"):
        predictive_scores(np.full((2, 5), np.nan), np.ones(2))
