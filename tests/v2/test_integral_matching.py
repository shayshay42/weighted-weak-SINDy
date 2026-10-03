from __future__ import annotations

import numpy as np

from lorenz63_benchmark.v2.numerics import LorenzParameters, integrate_rk4, lorenz_rhs
from lorenz63_benchmark.v2.parametric import estimate_integral_matching_parameters


def _lorenz_trajectories() -> tuple[np.ndarray, np.ndarray]:
    parameters = LorenzParameters(10.0, 28.0, 8.0 / 3.0)
    times = np.linspace(0.0, 2.0, 2001)
    initial = np.asarray([
        [-8.0, 7.0, 27.0],
        [4.0, -5.0, 20.0],
        [12.0, 13.0, 30.0],
    ])
    states, _ = integrate_rk4(
        lambda state: lorenz_rhs(state, parameters), initial, times, internal_step=0.001
    )
    return states, times


def test_integral_matching_recovers_lorenz_parameters_without_derivatives() -> None:
    states, times = _lorenz_trajectories()
    fitted, diagnostics = estimate_integral_matching_parameters(
        states,
        times,
        parameter_bounds={"sigma": [0.0, 50.0], "rho": [0.0, 100.0], "beta": [0.0, 20.0]},
    )
    np.testing.assert_allclose(fitted, [10.0, 28.0, 8.0 / 3.0], rtol=2e-4, atol=2e-4)
    assert diagnostics["regression_rows_per_parameter"] == 3 * 2000
    assert not any(diagnostics["projected_to_bounds"].values())
    assert all(value > 0.0 for value in diagnostics["denominators"].values())


def test_integral_matching_records_projection_to_bounds() -> None:
    states, times = _lorenz_trajectories()
    fitted, diagnostics = estimate_integral_matching_parameters(
        states,
        times,
        parameter_bounds={"sigma": [0.0, 9.0], "rho": [0.0, 27.0], "beta": [0.0, 2.0]},
    )
    np.testing.assert_array_equal(fitted, [9.0, 27.0, 2.0])
    assert all(diagnostics["projected_to_bounds"].values())


def test_integral_matching_rejects_nonmonotone_times() -> None:
    states, times = _lorenz_trajectories()
    times[10] = times[9]
    try:
        estimate_integral_matching_parameters(
            states,
            times,
            parameter_bounds={"sigma": [0.0, 50.0], "rho": [0.0, 100.0], "beta": [0.0, 20.0]},
        )
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("nonmonotone time input was accepted")
