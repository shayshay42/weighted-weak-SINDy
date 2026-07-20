from __future__ import annotations

import numpy as np

from lorenz63_benchmark.v2.sindy import (
    fit_sindy_coefficients, normalized_to_physical_coefficients, polynomial_exponents,
    polynomial_library, stlsq, weak_window_center_weights,
)
from lorenz63_benchmark.v2.base import Normalization
from lorenz63_benchmark.v2.config import DEFAULT_CONFIG
from lorenz63_benchmark.v2.identification import (
    lorenz_polynomial_coefficients, relative_coefficient_error,
)
from lorenz63_benchmark.v2.numerics import compact_taper


def test_weighted_sindy_equals_unweighted_for_uniform_weights() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(200, 10))
    truth = rng.normal(size=(10, 3))
    targets = features @ truth
    unweighted, _ = stlsq(
        features, targets, threshold=0.0, ridge=1e-10, max_iter=5
    )
    weighted, _ = stlsq(
        features, targets, threshold=0.0, ridge=1e-10, max_iter=5,
        sample_weights=np.ones(features.shape[0]),
    )
    np.testing.assert_allclose(weighted, unweighted, rtol=1e-12, atol=1e-12)


def test_weak_window_weights_follow_window_centers_and_modes() -> None:
    states = np.zeros((2, 11, 3), dtype=np.float64)
    weights = weak_window_center_weights(
        states, window_steps=2, stride_steps=2, modes=3
    )
    per_trajectory = np.repeat(compact_taper(5), 3)
    np.testing.assert_array_equal(weights, np.tile(per_trajectory, 2))
    assert weights.shape == (30,)


def test_weighted_weak_sindy_equals_weak_sindy_for_uniform_weights() -> None:
    rng = np.random.default_rng(14)
    states = rng.normal(size=(3, 21, 3))
    split = {
        "states": states,
        "metadata": {
            "dt": 0.01,
            "noise_level": 0.0,
            "normalization": {
                "state_mean": states.mean(axis=(0, 1)).tolist(),
                "state_std": states.std(axis=(0, 1)).tolist(),
                "time_scale": 0.9,
            },
        },
    }
    config = {
        "sindy": {
            **DEFAULT_CONFIG["sindy"],
            "degree": 2,
            "threshold": 0.0,
            "ridge": 1e-10,
            "weak_window_steps": 4,
            "weak_stride_steps": 2,
            "weak_modes": 2,
        }
    }
    weak, exponents, _ = fit_sindy_coefficients("sindy_weak", split, config)
    weighted, weighted_exponents, metadata = fit_sindy_coefficients(
        "sindy_weak_weighted", split, config, uniform_weights=True
    )
    assert weighted_exponents == exponents
    np.testing.assert_array_equal(weighted, weak)
    assert metadata["formulation"] == "weak"
    assert metadata["weighting"] == "uniform"


def test_canonical_lorenz_library_coefficients_are_exact() -> None:
    exponents = polynomial_exponents(3, 2)
    system = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
    coefficients = lorenz_polynomial_coefficients(exponents, **system)
    assert relative_coefficient_error(coefficients, exponents, system) == 0.0
    assert np.count_nonzero(coefficients) == 7


def test_normalized_sindy_coefficients_map_back_to_physical_field() -> None:
    rng = np.random.default_rng(9)
    exponents = polynomial_exponents(3, 2)
    normalized_coefficients = rng.normal(size=(len(exponents), 3))
    normalization = Normalization(
        state_mean=np.asarray([1.0, -2.0, 3.0]),
        state_std=np.asarray([2.0, 4.0, 0.5]),
        time_scale=0.9,
    )
    physical_coefficients = normalized_to_physical_coefficients(
        normalized_coefficients, exponents, normalization
    )
    physical_states = rng.normal(size=(20, 3))
    normalized_states = normalization.normalize_state(physical_states)
    expected = (
        polynomial_library(normalized_states, exponents) @ normalized_coefficients
    ) * normalization.state_std * normalization.time_scale
    actual = polynomial_library(physical_states, exponents) @ physical_coefficients
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
