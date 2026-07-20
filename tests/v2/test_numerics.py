from __future__ import annotations

import numpy as np
import torch

from lorenz63_benchmark.v2.losses import soft_dtw_divergence, weak_residual_loss
from lorenz63_benchmark.v2.metrics import normalized_squared_error, valid_prediction_time
from lorenz63_benchmark.v2.models import NormalizedVectorField
from lorenz63_benchmark.v2.node import TorchNodeAdapter, node_sampling_margin
from lorenz63_benchmark.v2.oracle import ORACLE_INTERNAL_STEP, load_oracle
from lorenz63_benchmark.v2.numerics import (
    finite_difference_jacobian, fourth_order_derivative, integrate_rk4,
)


def _derivative_error(dt: float) -> float:
    times = np.arange(0.0, 2.0 + 0.5 * dt, dt)
    states = np.sin(times)[None, :, None]
    _, derivative = fourth_order_derivative(states, dt)
    truth = np.cos(times[2:-2])[None, :, None]
    return float(np.sqrt(np.mean((derivative - truth) ** 2)))


def test_fourth_order_derivative_converges() -> None:
    coarse = _derivative_error(0.1)
    fine = _derivative_error(0.05)
    assert coarse / fine > 12.0


def test_soft_dtw_divergence_zero_and_has_finite_gradient() -> None:
    torch.manual_seed(3)
    values = torch.randn(2, 6, 3, dtype=torch.float64, requires_grad=True)
    divergence = soft_dtw_divergence(values, values, gamma=0.2).mean()
    assert abs(float(divergence.detach())) < 1e-12
    target = values.detach() + 0.1
    loss = soft_dtw_divergence(values, target, gamma=0.2).mean()
    loss.backward()
    assert values.grad is not None
    assert torch.all(torch.isfinite(values.grad))


def test_weak_residual_converges_under_grid_refinement() -> None:
    losses = []
    for length in (21, 41, 81):
        times = torch.linspace(0.0, 1.0, length, dtype=torch.float64)
        states = torch.exp(times)[None, :, None]
        loss = weak_residual_loss(states, states, 1.0 / (length - 1), modes=3, power=2)
        losses.append(float(loss))
    assert losses[2] < losses[1] < losses[0]


def test_vpt_uses_squared_error_and_reports_censoring_in_lyapunov_time() -> None:
    errors = np.asarray([[0.0, 0.2, 0.5, 0.8], [0.0, 0.1, 0.2, 0.3]])
    times = np.asarray([0.0, 0.5, 1.0, 1.5])
    vpt, censored = valid_prediction_time(errors, times, largest_lyapunov_exponent=0.9, threshold=0.4)
    np.testing.assert_allclose(vpt, [0.9, 1.35])
    np.testing.assert_array_equal(censored, [False, True])


def test_nonfinite_forecast_is_an_error_crossing_not_censoring() -> None:
    prediction = np.zeros((1, 3, 3))
    truth = np.zeros_like(prediction)
    prediction[0, 1] = np.nan
    error = normalized_squared_error(prediction, truth, np.ones(3))
    assert np.isinf(error[0, 1])
    vpt, censored = valid_prediction_time(error, np.asarray([0.0, 0.5, 1.0]), 0.9)
    np.testing.assert_allclose(vpt, [0.45])
    np.testing.assert_array_equal(censored, [False])


def test_batched_finite_difference_jacobian() -> None:
    matrix = np.asarray([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0], [0.0, -1.0, 2.0]])
    state = np.asarray([2.0, -3.0, 0.5])
    actual = finite_difference_jacobian(lambda points: points @ matrix.T, state)
    np.testing.assert_allclose(actual, matrix, rtol=1e-10, atol=1e-10)


def test_torch_node_rk4_matches_common_float64_evaluator() -> None:
    torch.manual_seed(12)
    model = NormalizedVectorField(hidden_dim=5, depth=1, activation_name="tanh").double()
    adapter = TorchNodeAdapter(
        method="node_strong", track="state_only_dynamics", model=model,
        device=torch.device("cpu"), state_mean=np.asarray([1.0, -2.0, 3.0]),
        state_std=np.asarray([2.0, 0.5, 4.0]), time_scale=0.9, internal_step=0.005,
    )
    initial = np.asarray([[1.2, -1.5, 2.8], [-0.2, 1.0, 4.0]])
    times = np.asarray([0.0, 0.01, 0.02])
    expected, evaluations = integrate_rk4(
        adapter.vector_field, initial, times, internal_step=0.005
    )
    actual = adapter.forecast(initial, times)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert adapter.last_vector_field_evaluations == evaluations


def test_node_sampling_margin_is_shared_and_covers_noisy_derivatives() -> None:
    config = {"sampling_margin_noisy": 10}
    assert node_sampling_margin(0.0, 2, config) == 2
    assert node_sampling_margin(0.01, 3, config) == 10
    assert node_sampling_margin(0.01, 10, config) == 10


def test_legacy_oracle_checkpoint_uses_high_accuracy_integrator(tmp_path) -> None:
    checkpoint = tmp_path / "legacy_oracle.npz"
    np.savez(
        checkpoint,
        method=np.asarray("solver_oracle"),
        parameters=np.asarray([10.0, 28.0, 8.0 / 3.0]),
    )
    adapter = load_oracle(checkpoint, internal_step=0.001)
    assert adapter.internal_step == ORACLE_INTERNAL_STEP
