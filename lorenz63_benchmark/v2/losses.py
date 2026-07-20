from __future__ import annotations

import torch


def soft_dtw_distance(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        raise ValueError("soft-DTW gamma must be positive")
    if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError("soft-DTW inputs must be [batch,time,dimension] with matching batch/dimension")
    distances = torch.cdist(x, y, p=2).square()
    batch, n_steps, m_steps = distances.shape
    infinity = torch.finfo(x.dtype).max / 1000.0
    table = x.new_full((batch, n_steps + 1, m_steps + 1), infinity)
    table[:, 0, 0] = 0.0
    for diagonal in range(2, n_steps + m_steps + 1):
        i_start = max(1, diagonal - m_steps)
        i_stop = min(n_steps, diagonal - 1)
        if i_start > i_stop:
            continue
        i = torch.arange(i_start, i_stop + 1, device=x.device)
        j = diagonal - i
        previous = torch.stack(
            (
                table[:, i - 1, j],
                table[:, i, j - 1],
                table[:, i - 1, j - 1],
            ),
            dim=-1,
        )
        soft_minimum = -gamma * torch.logsumexp(-previous / gamma, dim=-1)
        table[:, i, j] = distances[:, i - 1, j - 1] + soft_minimum
    return table[:, n_steps, m_steps] / float(n_steps + m_steps)


def soft_dtw_divergence(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    return (
        soft_dtw_distance(x, y, gamma)
        - 0.5 * soft_dtw_distance(x, x, gamma)
        - 0.5 * soft_dtw_distance(y, y, gamma)
    )


def trapezoid_weights(length: int, step: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = torch.full((length,), float(step), device=device, dtype=dtype)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def compact_test_functions(
    length: int,
    step: float,
    modes: int,
    power: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    tapered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length < 3 or modes < 1 or power < 1:
        raise ValueError("invalid weak test-function dimensions")
    s = torch.linspace(-1.0, 1.0, length, device=device, dtype=dtype)
    support = torch.clamp(1.0 - s.square(), min=0.0)
    bump = support.pow(power)
    dbump_ds = -2.0 * power * s * support.pow(max(power - 1, 0))
    polynomials = [torch.ones_like(s)]
    derivatives = [torch.zeros_like(s)]
    if modes > 1:
        polynomials.append(s)
        derivatives.append(torch.ones_like(s))
    for order in range(2, modes):
        polynomial = ((2 * order - 1) * s * polynomials[-1] - (order - 1) * polynomials[-2]) / order
        derivative = (
            (2 * order - 1) * (polynomials[-1] + s * derivatives[-1])
            - (order - 1) * derivatives[-2]
        ) / order
        polynomials.append(polynomial)
        derivatives.append(derivative)
    polynomial = torch.stack(polynomials, dim=-1)
    derivative = torch.stack(derivatives, dim=-1)
    phi = bump[:, None] * polynomial
    dphi_ds = dbump_ds[:, None] * polynomial + bump[:, None] * derivative
    if tapered:
        u = (s + 1.0) / 2.0
        taper = torch.zeros_like(u)
        dtaper_du = torch.zeros_like(u)
        interior = (u > 0.0) & (u < 1.0)
        q = u[interior] * (1.0 - u[interior])
        taper[interior] = torch.exp(4.0 - 1.0 / q)
        dtaper_du[interior] = taper[interior] * (1.0 - 2.0 * u[interior]) / q.square()
        dtaper_ds = 0.5 * dtaper_du
        dphi_ds = dtaper_ds[:, None] * phi + taper[:, None] * dphi_ds
        phi = taper[:, None] * phi
    duration = float(step) * (length - 1)
    dphi_dt = dphi_ds * (2.0 / duration)
    phi[0] = 0.0
    phi[-1] = 0.0
    return phi, dphi_dt


def weak_residual_loss(
    states: torch.Tensor,
    rhs: torch.Tensor,
    step: float,
    *,
    modes: int = 4,
    power: int = 2,
    tapered: bool = False,
) -> torch.Tensor:
    phi, dphi = compact_test_functions(
        states.shape[1], step, modes, power, device=states.device, dtype=states.dtype, tapered=tapered
    )
    weights = trapezoid_weights(states.shape[1], step, device=states.device, dtype=states.dtype)
    residual = torch.einsum("t,btm,tk->bmk", weights, rhs, phi)
    residual = residual + torch.einsum("t,btm,tk->bmk", weights, states, dphi)
    norms = torch.sqrt(torch.einsum("t,tk->k", weights, phi.square()).clamp_min(1e-14))
    return (residual / norms[None, None, :]).square().mean()


def time_derivative(prediction: torch.Tensor, time_seconds: torch.Tensor) -> torch.Tensor:
    components = []
    for dimension in range(prediction.shape[-1]):
        components.append(torch.autograd.grad(
            prediction[..., dimension].sum(), time_seconds,
            create_graph=True, retain_graph=True,
        )[0])
    return torch.cat(components, dim=-1)
