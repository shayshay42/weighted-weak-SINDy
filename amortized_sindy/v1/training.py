from __future__ import annotations

from dataclasses import dataclass

import torch

from .library import polynomial_library
from .model import AmortizedSINDy


@dataclass
class LossBreakdown:
    total: torch.Tensor
    rollout_mse: torch.Tensor
    support_penalty: torch.Tensor
    support_binary_penalty: torch.Tensor
    coefficient_penalty: torch.Tensor
    raw_coefficient_penalty: torch.Tensor
    context_residual: torch.Tensor
    weak_form_residual: torch.Tensor
    birkhoff_mmd: torch.Tensor
    coefficient_consistency: torch.Tensor
    teacher_coefficient_mse: torch.Tensor


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _context_derivative_residual(
    model: AmortizedSINDy,
    *,
    context_states: torch.Tensor,
    context_times: torch.Tensor,
    dynamics: object,
) -> torch.Tensor:
    states, times = model._validate_context(context_states, context_times)
    state_mean = dynamics.state_mean  # type: ignore[attr-defined]
    state_scale = dynamics.state_scale  # type: ignore[attr-defined]
    time_scale = dynamics.time_scale  # type: ignore[attr-defined]
    coefficients = dynamics.effective_coefficients  # type: ignore[attr-defined]
    normalized_states = (states - state_mean) / state_scale
    normalized_times = (times - times[:, -1:]) / time_scale
    derivatives = (
        normalized_states[:, 2:, :] - normalized_states[:, :-2, :]
    ) / (normalized_times[:, 2:] - normalized_times[:, :-2]).unsqueeze(-1)
    features = polynomial_library(normalized_states[:, 1:-1, :], model.exponents)
    predicted = torch.einsum("blm,bmd->bld", features, coefficients)
    return torch.mean((predicted - derivatives) ** 2)


def _context_weak_form_residual(
    model: AmortizedSINDy,
    *,
    context_states: torch.Tensor,
    context_times: torch.Tensor,
    dynamics: object,
) -> torch.Tensor:
    """Derivative-free weak residual for the context-conditioned field.

    The calculation uses the same normalized coordinates and compact smooth
    test functions as the conditioner, but it contracts every weak equation
    with the emitted coefficient matrix.  No pointwise derivative estimate is
    formed.
    """
    states, times = model._validate_context(context_states, context_times)
    state_mean = dynamics.state_mean  # type: ignore[attr-defined]
    state_scale = dynamics.state_scale  # type: ignore[attr-defined]
    time_scale = dynamics.time_scale  # type: ignore[attr-defined]
    coefficients = dynamics.effective_coefficients  # type: ignore[attr-defined]
    normalized_states = (states - state_mean) / state_scale
    normalized_times = (times - times[:, -1:]) / time_scale

    context_length = normalized_states.shape[1]
    window_length = min(model.config.weak_window_length, context_length)
    if window_length < 5:
        raise ValueError("context is too short for a weak-form residual")
    last_start = context_length - window_length
    starts = list(range(0, last_start + 1, model.config.weak_stride_steps))
    if starts[-1] != last_start:
        starts.append(last_start)

    residuals = []
    for start in starts:
        stop = start + window_length
        local_times = normalized_times[:, start:stop]
        local_states = normalized_states[:, start:stop, :]
        duration = (local_times[:, -1] - local_times[:, 0]).clamp_min(
            model.config.min_time_scale
        )
        coordinate = (
            2.0 * (local_times - local_times[:, :1]) / duration[:, None] - 1.0
        )
        bump, bump_derivative = model._smooth_test_function(coordinate)
        polynomials, polynomial_derivatives = model._legendre_modes(
            coordinate, model.config.weak_modes
        )
        quadrature = model._trapezoid_weights(local_times)
        library = polynomial_library(local_states, model.exponents)
        for polynomial, polynomial_derivative in zip(
            polynomials, polynomial_derivatives
        ):
            phi = bump * polynomial
            phi_derivative = (
                bump_derivative * polynomial + bump * polynomial_derivative
            ) * (2.0 / duration[:, None])
            features = torch.einsum("bl,blm->bm", quadrature * phi, library)
            targets = -torch.einsum(
                "bl,bld->bd", quadrature * phi_derivative, local_states
            )
            predicted = torch.einsum("bm,bmd->bd", features, coefficients)
            equation_scale = torch.sqrt(
                (
                    torch.sum(features ** 2, dim=1)
                    + torch.sum(targets ** 2, dim=1)
                )
                / (features.shape[1] + targets.shape[1])
            ).clamp_min(model.config.min_time_scale)
            residuals.append(
                torch.mean(
                    ((predicted - targets) / equation_scale[:, None]) ** 2,
                    dim=1,
                )
            )
    return torch.stack(residuals, dim=1).mean()


def _rational_quadratic_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bandwidth: float,
) -> torch.Tensor:
    squared_distance = torch.sum(
        (left[:, :, None, :] - right[:, None, :, :]) ** 2,
        dim=-1,
    )
    return 1.0 / (1.0 + squared_distance / (2.0 * bandwidth ** 2))


def _weighted_birkhoff_mmd(
    model: AmortizedSINDy,
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    forecast_offsets: torch.Tensor,
    state_scale: torch.Tensor,
    bandwidths: tuple[float, ...],
) -> torch.Tensor:
    """Weighted empirical-measure loss for source-only forecast rollouts."""
    if not bandwidths or any(value <= 0 for value in bandwidths):
        raise ValueError("Birkhoff MMD bandwidths must be positive")
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("Birkhoff MMD trajectories must have matching [batch,time,state] shapes")
    if predictions.shape[1] < 3:
        return _zero(predictions)
    offsets = torch.as_tensor(
        forecast_offsets, dtype=predictions.dtype, device=predictions.device
    )
    if offsets.ndim == 1:
        offsets = offsets[None, :].expand(predictions.shape[0], -1)
    if offsets.shape != predictions.shape[:2]:
        raise ValueError("Birkhoff MMD time grid does not match trajectories")
    scale = torch.as_tensor(
        state_scale, dtype=predictions.dtype, device=predictions.device
    )
    if scale.ndim == 2:
        scale = scale[:, None, :]
    if scale.shape != (predictions.shape[0], 1, predictions.shape[2]):
        raise ValueError("Birkhoff MMD state scale has the wrong shape")
    normalized_predictions = predictions / scale
    normalized_targets = targets / scale
    weights = model._smooth_birkhoff_weights(offsets)
    pair_weights = weights[:, :, None] * weights[:, None, :]
    values = []
    for bandwidth in bandwidths:
        predicted_kernel = _rational_quadratic_kernel(
            normalized_predictions,
            normalized_predictions,
            bandwidth=bandwidth,
        )
        target_kernel = _rational_quadratic_kernel(
            normalized_targets,
            normalized_targets,
            bandwidth=bandwidth,
        )
        cross_kernel = _rational_quadratic_kernel(
            normalized_predictions,
            normalized_targets,
            bandwidth=bandwidth,
        )
        values.append(
            torch.sum(
                pair_weights
                * (predicted_kernel + target_kernel - 2.0 * cross_kernel),
                dim=(1, 2),
            )
        )
    # The biased empirical MMD is non-negative analytically.  Clamp only tiny
    # floating-point excursions below zero.
    return torch.clamp(torch.stack(values, dim=1), min=0.0).mean()


def _coefficient_consistency(
    physical_coefficients: torch.Tensor,
    group_ids: torch.Tensor | None,
) -> torch.Tensor:
    if group_ids is None:
        return _zero(physical_coefficients)
    group_ids = torch.as_tensor(group_ids, device=physical_coefficients.device).reshape(-1)
    if group_ids.shape[0] != physical_coefficients.shape[0]:
        raise ValueError("group_ids must have one value per context")
    losses = []
    for group_id in torch.unique(group_ids):
        selected = physical_coefficients[group_ids == group_id]
        if selected.shape[0] < 2:
            continue
        center = selected.mean(dim=0, keepdim=True)
        scale = torch.mean(center ** 2).clamp_min(1e-4)
        losses.append(torch.mean((selected - center) ** 2) / scale)
    return torch.stack(losses).mean() if losses else _zero(physical_coefficients)


def source_training_loss(
    model: AmortizedSINDy,
    *,
    context_states: torch.Tensor,
    context_times: torch.Tensor,
    forecast_offsets: torch.Tensor,
    future_states: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    teacher_physical_coefficients: torch.Tensor | None = None,
    context_embeddings: torch.Tensor | None = None,
    teacher_weight: float = 0.0,
    rollout_weight: float = 1.0,
    rollout_error_cap: float | None = None,
    rollout_loss_kind: str = "clipped_mse",
    support_weight: float = 1e-4,
    support_binary_weight: float = 0.0,
    coefficient_weight: float = 1e-6,
    raw_coefficient_weight: float = 0.0,
    context_residual_weight: float = 0.0,
    weak_form_weight: float = 0.0,
    birkhoff_mmd_weight: float = 0.0,
    birkhoff_mmd_bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    consistency_weight: float = 0.0,
) -> LossBreakdown:
    """Offline source-system objective; never call this in target inference."""
    if min(
        support_weight,
        support_binary_weight,
        coefficient_weight,
        raw_coefficient_weight,
        context_residual_weight,
        weak_form_weight,
        birkhoff_mmd_weight,
        consistency_weight,
        rollout_weight,
        teacher_weight,
    ) < 0:
        raise ValueError("regularization weights must be non-negative")
    if rollout_error_cap is not None and rollout_error_cap <= 0:
        raise ValueError("rollout_error_cap must be positive")
    if rollout_loss_kind not in {"clipped_mse", "pseudo_huber"}:
        raise ValueError("invalid rollout loss kind")
    if rollout_loss_kind == "pseudo_huber" and rollout_error_cap is None:
        raise ValueError("pseudo-Huber rollout loss requires a positive scale")
    if not birkhoff_mmd_bandwidths or any(
        value <= 0 for value in birkhoff_mmd_bandwidths
    ):
        raise ValueError("Birkhoff MMD bandwidths must be positive")
    predictions, dynamics = model.rollout(
        context_states,
        context_times,
        forecast_offsets,
        hard_support=False,
        context_embedding=context_embeddings,
    )
    if future_states.ndim == 2:
        future_states = future_states.unsqueeze(0)
    if predictions.shape != future_states.shape:
        raise ValueError("future state shape does not match model predictions")
    scale = dynamics.state_scale.squeeze(1).unsqueeze(1)
    normalized_error = (predictions - future_states) / scale
    if rollout_error_cap is None:
        rollout_mse = torch.mean(normalized_error ** 2)
    else:
        finite_error = torch.where(
            torch.isfinite(normalized_error),
            normalized_error,
            torch.full_like(normalized_error, rollout_error_cap),
        )
        if rollout_loss_kind == "clipped_mse":
            rollout_mse = torch.mean(
                torch.clamp(finite_error ** 2, max=rollout_error_cap ** 2)
            )
        else:
            delta = torch.as_tensor(
                rollout_error_cap,
                dtype=finite_error.dtype,
                device=finite_error.device,
            )
            rollout_mse = torch.mean(
                2.0
                * delta
                * (torch.hypot(finite_error, delta) - delta)
            )
    support_penalty = dynamics.support_probabilities.mean()
    support_binary_penalty = torch.mean(
        4.0
        * dynamics.support_probabilities
        * (1.0 - dynamics.support_probabilities)
    )
    coefficient_penalty = dynamics.effective_coefficients.abs().mean()
    raw_coefficient_penalty = dynamics.coefficients.abs().mean()
    context_residual = (
        _context_derivative_residual(
            model,
            context_states=context_states,
            context_times=context_times,
            dynamics=dynamics,
        )
        if context_residual_weight > 0
        else _zero(rollout_mse)
    )
    weak_form_residual = (
        _context_weak_form_residual(
            model,
            context_states=context_states,
            context_times=context_times,
            dynamics=dynamics,
        )
        if weak_form_weight > 0
        else _zero(rollout_mse)
    )
    if birkhoff_mmd_weight > 0:
        measure_predictions = predictions
        if rollout_error_cap is not None:
            cap = rollout_error_cap * scale
            measure_predictions = torch.where(
                torch.isfinite(predictions), predictions, future_states + cap
            )
        birkhoff_mmd = _weighted_birkhoff_mmd(
            model,
            predictions=measure_predictions,
            targets=future_states,
            forecast_offsets=forecast_offsets,
            state_scale=scale,
            bandwidths=birkhoff_mmd_bandwidths,
        )
    else:
        birkhoff_mmd = _zero(rollout_mse)
    coefficient_consistency = (
        _coefficient_consistency(dynamics.physical_coefficients, group_ids)
        if consistency_weight > 0
        else _zero(rollout_mse)
    )
    if teacher_weight > 0:
        if teacher_physical_coefficients is None:
            raise ValueError("teacher coefficients are required when teacher_weight is positive")
        teacher_physical_coefficients = torch.as_tensor(
            teacher_physical_coefficients,
            dtype=dynamics.physical_coefficients.dtype,
            device=dynamics.physical_coefficients.device,
        )
        if teacher_physical_coefficients.shape != dynamics.physical_coefficients.shape:
            raise ValueError("teacher coefficient shape does not match model output")
        teacher_scale = torch.sqrt(
            torch.mean(teacher_physical_coefficients ** 2, dim=(1, 2), keepdim=True)
        ).clamp_min(0.1)
        teacher_coefficient_mse = torch.mean(
            ((dynamics.physical_coefficients - teacher_physical_coefficients) / teacher_scale) ** 2
        )
    else:
        teacher_coefficient_mse = _zero(rollout_mse)
    total = (
        rollout_weight * rollout_mse
        + support_weight * support_penalty
        + support_binary_weight * support_binary_penalty
        + coefficient_weight * coefficient_penalty
        + raw_coefficient_weight * raw_coefficient_penalty
        + context_residual_weight * context_residual
        + weak_form_weight * weak_form_residual
        + birkhoff_mmd_weight * birkhoff_mmd
        + consistency_weight * coefficient_consistency
        + teacher_weight * teacher_coefficient_mse
    )
    return LossBreakdown(
        total=total,
        rollout_mse=rollout_mse,
        support_penalty=support_penalty,
        support_binary_penalty=support_binary_penalty,
        coefficient_penalty=coefficient_penalty,
        raw_coefficient_penalty=raw_coefficient_penalty,
        context_residual=context_residual,
        weak_form_residual=weak_form_residual,
        birkhoff_mmd=birkhoff_mmd,
        coefficient_consistency=coefficient_consistency,
        teacher_coefficient_mse=teacher_coefficient_mse,
    )


def source_training_step(
    model: AmortizedSINDy,
    optimizer: torch.optim.Optimizer,
    **loss_arguments: object,
) -> LossBreakdown:
    """One explicitly offline optimizer step on source-system trajectories."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = source_training_loss(model, **loss_arguments)  # type: ignore[arg-type]
    loss.total.backward()
    optimizer.step()
    return loss
