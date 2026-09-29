from __future__ import annotations

from types import SimpleNamespace

import torch

from amortized_sindy.v1.model import AmortizedSINDy, AmortizedSINDyConfig
from amortized_sindy.v1.training import (
    _context_weak_form_residual,
    _weighted_birkhoff_mmd,
    source_training_loss,
)


def _one_dimensional_model(
    *,
    training_support_mode: str = "soft",
    weak_window_length: int = 33,
    weak_modes: int = 2,
) -> AmortizedSINDy:
    return AmortizedSINDy(
        AmortizedSINDyConfig(
            state_dimension=1,
            library_degree=1,
            hidden_size=4,
            min_context_steps=5,
            training_support_mode=training_support_mode,
            weak_window_length=weak_window_length,
            weak_stride_steps=max(1, weak_window_length // 2),
            weak_modes=weak_modes,
        )
    ).double()


def test_near_exact_field_has_smaller_weak_residual_than_wrong_field() -> None:
    model = _one_dimensional_model(
        weak_window_length=257,
        weak_modes=3,
    )
    times = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)
    growth_rate = 0.7
    states = torch.exp(growth_rate * times)[None, :, None]
    common = {
        "state_mean": torch.zeros((1, 1, 1), dtype=torch.float64),
        "state_scale": torch.ones((1, 1, 1), dtype=torch.float64),
        "time_scale": torch.ones((1, 1), dtype=torch.float64),
    }
    exact_dynamics = SimpleNamespace(
        effective_coefficients=torch.tensor(
            [[[0.0], [growth_rate]]], dtype=torch.float64
        ),
        **common,
    )
    wrong_dynamics = SimpleNamespace(
        effective_coefficients=torch.tensor(
            [[[0.0], [-growth_rate]]], dtype=torch.float64
        ),
        **common,
    )

    exact = _context_weak_form_residual(
        model,
        context_states=states,
        context_times=times,
        dynamics=exact_dynamics,
    )
    wrong = _context_weak_form_residual(
        model,
        context_states=states,
        context_times=times,
        dynamics=wrong_dynamics,
    )

    assert exact < 1e-8
    assert exact < wrong * 1e-4


def test_birkhoff_mmd_identity_shift_and_gradient() -> None:
    model = _one_dimensional_model()
    offsets = torch.linspace(0.0, 1.0, 129, dtype=torch.float64)
    base = (
        torch.sin(2.0 * torch.pi * offsets)
        + 0.2 * torch.cos(6.0 * torch.pi * offsets)
    )[None, :, None]
    state_scale = torch.ones((1, 1, 1), dtype=torch.float64)
    bandwidths = (0.25, 0.5, 1.0, 2.0)

    identical = _weighted_birkhoff_mmd(
        model,
        predictions=base,
        targets=base.clone(),
        forecast_offsets=offsets,
        state_scale=state_scale,
        bandwidths=bandwidths,
    )
    torch.testing.assert_close(
        identical,
        torch.zeros((), dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )

    candidate = base.clone().requires_grad_(True)
    shifted = base + 0.75
    shifted_loss = _weighted_birkhoff_mmd(
        model,
        predictions=candidate,
        targets=shifted,
        forecast_offsets=offsets,
        state_scale=state_scale,
        bandwidths=bandwidths,
    )
    assert shifted_loss > 1e-3

    shifted_loss.backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert torch.count_nonzero(candidate.grad).item() > 0


def test_straight_through_support_is_hard_forward_and_soft_backward() -> None:
    model = _one_dimensional_model(training_support_mode="straight_through")
    model.train()
    with torch.no_grad():
        assert model.encoder is not None
        for parameter in model.encoder.parameters():
            parameter.zero_()
        model.coefficient_head.weight.zero_()
        model.coefficient_head.bias.copy_(
            torch.tensor([2.0, 3.0], dtype=torch.float64)
        )
        model.support_head.weight.zero_()
        model.support_head.bias.copy_(
            torch.tensor([-0.4, 0.4], dtype=torch.float64)
        )

    times = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
    states = torch.sin(times)[None, :, None]
    dynamics = model.infer_dynamics(states, times, hard_support=False)
    expected_support = (
        dynamics.support_probabilities >= model.config.support_threshold
    ).to(dynamics.coefficients.dtype)

    torch.testing.assert_close(
        dynamics.effective_coefficients,
        dynamics.coefficients * expected_support,
        rtol=0.0,
        atol=1e-15,
    )

    dynamics.effective_coefficients.sum().backward()
    gradient = model.support_head.bias.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient.abs() > 0)


def test_pseudo_huber_has_gradient_well_beyond_its_scale() -> None:
    model = _one_dimensional_model()
    model.train()
    with torch.no_grad():
        assert model.encoder is not None
        for parameter in model.encoder.parameters():
            parameter.zero_()
        model.coefficient_head.weight.zero_()
        model.coefficient_head.bias.copy_(
            torch.tensor([50.0, 0.0], dtype=torch.float64)
        )
        model.support_head.weight.zero_()
        model.support_head.bias.fill_(20.0)

    times = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
    context = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)[None, :, None]
    offsets = torch.tensor([0.2], dtype=torch.float64)
    future = context[:, -1:, :].clone()
    huber_scale = 0.25

    with torch.no_grad():
        prediction, dynamics = model.rollout(
            context,
            times,
            offsets,
            hard_support=False,
        )
        normalized_error = (prediction - future) / dynamics.state_scale
        assert normalized_error.abs().min() > 10.0 * huber_scale

    loss = source_training_loss(
        model,
        context_states=context,
        context_times=times,
        forecast_offsets=offsets,
        future_states=future,
        rollout_error_cap=huber_scale,
        rollout_loss_kind="pseudo_huber",
        support_weight=0.0,
        coefficient_weight=0.0,
    )
    loss.total.backward()

    gradient = model.coefficient_head.bias.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient[0].abs() > 1e-6


def test_pseudo_huber_is_stable_near_float32_extremes() -> None:
    model = _one_dimensional_model().float()
    model.train()
    with torch.no_grad():
        assert model.encoder is not None
        for parameter in model.encoder.parameters():
            parameter.zero_()
        model.coefficient_head.weight.zero_()
        model.coefficient_head.bias.copy_(torch.tensor([1e30, 0.0]))
        model.support_head.weight.zero_()
        model.support_head.bias.fill_(20.0)

    times = torch.linspace(0.0, 1.0, 17)
    context = torch.linspace(-1.0, 1.0, 17)[None, :, None]
    loss = source_training_loss(
        model,
        context_states=context,
        context_times=times,
        forecast_offsets=torch.tensor([0.01]),
        future_states=context[:, -1:, :].clone(),
        rollout_error_cap=0.25,
        rollout_loss_kind="pseudo_huber",
        support_weight=0.0,
        coefficient_weight=0.0,
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    gradient = model.coefficient_head.bias.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient[0].abs() > 0
