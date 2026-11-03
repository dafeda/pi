"""Compare ensemble fitting with an exact linear-model result with normal (bell-shaped) errors."""

import numpy as np
import pytest

from controller.assimilation import INFLATION, create_smoother, update_parameters


def test_linear_gaussian_conditional_mean_and_covariance() -> None:
    prior_mean = np.array([1.0, -1.0])
    prior_covariance = np.array([[2.0, 0.3], [0.3, 1.0]])
    operator = np.array([[1.0, 0.5], [0.2, 1.0]])
    observations = np.array([0.5, -0.4])
    variance = np.array([0.25, 0.16])
    innovation_covariance = operator @ prior_covariance @ operator.T + np.diag(variance)
    gain = np.linalg.solve(innovation_covariance, operator @ prior_covariance).T
    expected_mean = prior_mean + gain @ (observations - operator @ prior_mean)
    expected_covariance = prior_covariance - gain @ operator @ prior_covariance

    rng = np.random.default_rng(42)
    parameters = rng.multivariate_normal(prior_mean, prior_covariance, size=30_000).T
    smoother = create_smoother(observations, variance, rng)
    np.testing.assert_array_equal(smoother.alpha, INFLATION)
    assert np.sum(1 / smoother.alpha) == 1
    for _ in range(smoother.num_assimilations()):
        responses = operator @ parameters
        perturbations = rng.normal(size=responses.shape) * np.sqrt(variance)[:, np.newaxis]
        parameters = update_parameters(smoother, parameters, responses, perturbations)

    # A finite set of random parameter versions only approximates the exact fitted distribution.
    np.testing.assert_allclose(parameters.mean(axis=1), expected_mean, rtol=0, atol=0.025)
    np.testing.assert_allclose(np.cov(parameters), expected_covariance, rtol=0, atol=0.025)


def test_update_preserves_inputs_and_replays_recorded_perturbations() -> None:
    rng = np.random.default_rng(7)
    observations = np.array([0.3, -0.4])
    variance = np.array([0.1, 0.2])
    parameters = rng.normal(size=(3, 50))
    operator = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.2]])
    responses = operator @ parameters
    perturbations = rng.normal(size=responses.shape) * np.sqrt(variance)[:, np.newaxis]
    before_parameters = parameters.copy()
    before_responses = responses.copy()
    first = update_parameters(
        create_smoother(observations, variance, rng), parameters, responses, perturbations
    )
    replayed = update_parameters(
        create_smoother(observations, variance, np.random.default_rng(99)),
        parameters,
        responses,
        perturbations,
    )
    np.testing.assert_array_equal(parameters, before_parameters)
    np.testing.assert_array_equal(responses, before_responses)
    np.testing.assert_array_equal(first, replayed)


def test_invalid_array_contracts_fail_before_advancing_smoother() -> None:
    """Reject wrong row/column layouts or invalid numbers without starting an update."""
    rng = np.random.default_rng(7)
    observations = np.ones(2)
    variance = np.ones(2)
    smoother = create_smoother(observations, variance, rng)
    parameters = np.ones((3, 50))
    with pytest.raises(ValueError, match="shape"):
        update_parameters(smoother, parameters, np.ones((50, 2)), np.ones((2, 50)))
    with pytest.raises(ValueError, match="finite float64"):
        update_parameters(smoother, parameters, np.full((2, 50), np.nan), np.ones((2, 50)))
    with pytest.raises(ValueError, match="finite float64"):
        update_parameters(
            smoother, parameters, np.ones((2, 50), dtype=np.float32), np.ones((2, 50))
        )
    assert smoother.iteration == -1


@pytest.mark.parametrize("variance", [np.array([0.0]), np.array([-1.0]), np.array([np.nan])])
def test_invalid_observation_variance_is_rejected(variance: np.ndarray) -> None:
    with pytest.raises(ValueError):
        create_smoother(np.array([0.0]), variance, np.random.default_rng(7))
