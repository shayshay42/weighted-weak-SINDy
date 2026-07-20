from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class LorenzParameters:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


def lorenz_rhs(x: np.ndarray, parameters: LorenzParameters = LorenzParameters()) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    result[..., 0] = parameters.sigma * (x[..., 1] - x[..., 0])
    result[..., 1] = x[..., 0] * (parameters.rho - x[..., 2]) - x[..., 1]
    result[..., 2] = x[..., 0] * x[..., 1] - parameters.beta * x[..., 2]
    return result


def lorenz_jacobian(x: np.ndarray, parameters: LorenzParameters = LorenzParameters()) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    jacobian = np.zeros(x.shape[:-1] + (3, 3), dtype=np.float64)
    jacobian[..., 0, 0] = -parameters.sigma
    jacobian[..., 0, 1] = parameters.sigma
    jacobian[..., 1, 0] = parameters.rho - x[..., 2]
    jacobian[..., 1, 1] = -1.0
    jacobian[..., 1, 2] = -x[..., 0]
    jacobian[..., 2, 0] = x[..., 1]
    jacobian[..., 2, 1] = x[..., 0]
    jacobian[..., 2, 2] = -parameters.beta
    return jacobian


def integrate_scipy(
    initial_states: np.ndarray,
    times: np.ndarray,
    parameters: LorenzParameters,
    *,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
) -> np.ndarray:
    initial_states = np.asarray(initial_states, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    single = initial_states.ndim == 1
    batch = initial_states.reshape(-1, 3)

    def rhs(_time: float, flattened: np.ndarray) -> np.ndarray:
        return lorenz_rhs(flattened.reshape(-1, 3), parameters).reshape(-1)

    solution = solve_ivp(
        rhs, (float(times[0]), float(times[-1])), batch.reshape(-1), t_eval=times,
        method=method, rtol=rtol, atol=atol, max_step=max_step,
    )
    if not solution.success or solution.y.shape[1] != times.size:
        raise RuntimeError(f"{method} integration failed: {solution.message}")
    states = solution.y.T.reshape(times.size, batch.shape[0], 3).transpose(1, 0, 2)
    return states[0] if single else states


def spin_up(
    initial_states: np.ndarray,
    duration: float,
    parameters: LorenzParameters,
    solver: dict,
) -> np.ndarray:
    if duration <= 0:
        return np.asarray(initial_states, dtype=np.float64).copy()
    states = integrate_scipy(
        initial_states, np.asarray([0.0, duration]), parameters,
        method=str(solver["method"]), rtol=float(solver["rtol"]),
        atol=float(solver["atol"]), max_step=float(solver["max_step"]),
    )
    return states[:, -1]


def rk4_step(field: Callable[[np.ndarray], np.ndarray], state: np.ndarray, step: float) -> np.ndarray:
    k1 = field(state)
    k2 = field(state + 0.5 * step * k1)
    k3 = field(state + 0.5 * step * k2)
    k4 = field(state + step * k3)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_rk4(
    field: Callable[[np.ndarray], np.ndarray],
    initial_states: np.ndarray,
    times: np.ndarray,
    internal_step: float,
) -> tuple[np.ndarray, int]:
    initial_states = np.asarray(initial_states, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    result = np.empty((initial_states.shape[0], times.size, 3), dtype=np.float64)
    result[:, 0] = initial_states
    state = initial_states.copy()
    evaluations = 0
    for index in range(1, times.size):
        interval = float(times[index] - times[index - 1])
        substeps = max(1, int(np.ceil(abs(interval) / internal_step)))
        step = interval / substeps
        for _ in range(substeps):
            state = rk4_step(field, state, step)
            evaluations += 4 * state.shape[0]
        result[:, index] = state
    return result, evaluations


def fourth_order_derivative(states: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    if states.shape[1] < 5:
        raise ValueError("fourth-order differentiation needs at least five time points")
    derivative = (
        states[:, :-4] - 8.0 * states[:, 1:-3]
        + 8.0 * states[:, 3:-1] - states[:, 4:]
    ) / (12.0 * dt)
    return states[:, 2:-2], derivative


def savgol_derivative(
    states: np.ndarray, dt: float, window_length: int, polyorder: int
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    if window_length % 2 != 1 or window_length <= polyorder:
        raise ValueError("Savitzky-Golay window must be odd and greater than polyorder")
    if window_length > states.shape[1]:
        raise ValueError("Savitzky-Golay window exceeds trajectory length")
    derivative = savgol_filter(
        states, window_length=window_length, polyorder=polyorder,
        deriv=1, delta=dt, axis=1, mode="interp",
    )
    margin = window_length // 2
    return states[:, margin:-margin], derivative[:, margin:-margin]


def observed_derivatives(
    states: np.ndarray,
    dt: float,
    noise_level: float,
    *,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if noise_level == 0.0:
        points, derivatives = fourth_order_derivative(states, dt)
        return points, derivatives, 2
    points, derivatives = savgol_derivative(
        states, dt, int(savgol_window), int(savgol_polyorder)
    )
    return points, derivatives, int(savgol_window) // 2


def compact_taper(length: int) -> np.ndarray:
    if length < 3:
        return np.ones(length, dtype=np.float64)
    s = np.linspace(0.0, 1.0, length)
    weights = np.zeros_like(s)
    interior = (s > 0.0) & (s < 1.0)
    weights[interior] = np.exp(4.0 - 1.0 / (s[interior] * (1.0 - s[interior])))
    mean = weights.mean()
    return weights / mean if mean > 0 else np.ones(length, dtype=np.float64)


def _coupled_rk4_step(
    state: np.ndarray,
    tangent: np.ndarray,
    step: float,
    field: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    def tangent_rhs(x: np.ndarray, q: np.ndarray) -> np.ndarray:
        return jacobian(x) @ q

    k1x = field(state)
    k1q = tangent_rhs(state, tangent)
    k2x = field(state + 0.5 * step * k1x)
    k2q = tangent_rhs(state + 0.5 * step * k1x, tangent + 0.5 * step * k1q)
    k3x = field(state + 0.5 * step * k2x)
    k3q = tangent_rhs(state + 0.5 * step * k2x, tangent + 0.5 * step * k2q)
    k4x = field(state + step * k3x)
    k4q = tangent_rhs(state + step * k3x, tangent + step * k3q)
    return (
        state + (step / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        tangent + (step / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q),
    )


def lyapunov_spectrum(
    field: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    *,
    duration: float,
    step: float,
    qr_interval: int = 10,
    burn_in: float = 0.0,
) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float64).copy()
    tangent = np.eye(3, dtype=np.float64)
    burn_steps = int(round(burn_in / step))
    total_steps = int(round(duration / step))
    sums = np.zeros(3, dtype=np.float64)
    accumulated_time = 0.0
    for index in range(burn_steps + total_steps):
        state, tangent = _coupled_rk4_step(state, tangent, step, field, jacobian)
        if (index + 1) % qr_interval == 0:
            tangent, upper = np.linalg.qr(tangent)
            if index + 1 > burn_steps:
                sums += np.log(np.maximum(np.abs(np.diag(upper)), 1e-300))
                accumulated_time += qr_interval * step
    if accumulated_time <= 0:
        raise ValueError("Lyapunov duration is too short for the QR interval")
    return np.sort(sums / accumulated_time)[::-1]


def finite_difference_jacobian(
    field: Callable[[np.ndarray], np.ndarray], state: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    deltas = epsilon * np.maximum(1.0, np.abs(state))
    perturbations = np.eye(3, dtype=np.float64) * deltas[:, None]
    points = np.concatenate([state + perturbations, state - perturbations], axis=0)
    values = np.asarray(field(points), dtype=np.float64)
    if values.shape != points.shape:
        raise ValueError("finite-difference field must preserve the [batch,3] shape")
    derivatives_by_input = (values[:3] - values[3:]) / (2.0 * deltas[:, None])
    return derivatives_by_input.T
