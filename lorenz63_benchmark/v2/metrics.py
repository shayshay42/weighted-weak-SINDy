from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from scipy.integrate import trapezoid

from .numerics import finite_difference_jacobian, lyapunov_spectrum


def normalized_squared_error(
    prediction: np.ndarray, truth: np.ndarray, training_std: np.ndarray
) -> np.ndarray:
    denominator = float(np.sum(np.asarray(training_std, dtype=np.float64) ** 2))
    if denominator <= 0:
        raise ValueError("training variance must be positive")
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.sum((prediction - truth) ** 2, axis=-1) / denominator
    valid = np.all(np.isfinite(prediction), axis=-1) & np.all(np.isfinite(truth), axis=-1)
    return np.where(valid & np.isfinite(result), result, np.inf)


def valid_prediction_time(
    error: np.ndarray,
    times: np.ndarray,
    largest_lyapunov_exponent: float,
    threshold: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    error = np.asarray(error, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != times.size:
        raise ValueError("VPT error must have shape [trajectory,time]")
    vpt = np.full(error.shape[0], float(times[-1]) * largest_lyapunov_exponent)
    censored = np.ones(error.shape[0], dtype=bool)
    for trajectory, row in enumerate(error):
        crossing = np.flatnonzero(~np.isfinite(row) | (row > threshold))
        if crossing.size:
            vpt[trajectory] = float(times[crossing[0]]) * largest_lyapunov_exponent
            censored[trajectory] = False
    return vpt, censored


def normalized_rmse_auc(error: np.ndarray, times_lt: np.ndarray, horizon_lt: float) -> np.ndarray:
    error = np.asarray(error, dtype=np.float64)
    times_lt = np.asarray(times_lt, dtype=np.float64)
    if horizon_lt <= 0 or times_lt[-1] + 1e-12 < horizon_lt:
        return np.full(error.shape[0], np.nan)
    keep = times_lt <= horizon_lt + 1e-12
    selected_times = times_lt[keep]
    selected_values = np.sqrt(np.maximum(error[:, keep], 0.0))
    if selected_times[-1] < horizon_lt:
        right = np.searchsorted(times_lt, horizon_lt)
        left = right - 1
        fraction = (horizon_lt - times_lt[left]) / (times_lt[right] - times_lt[left])
        interpolated = selected_values[:, -1] + fraction * (
            np.sqrt(np.maximum(error[:, right], 0.0)) - selected_values[:, -1]
        )
        selected_times = np.append(selected_times, horizon_lt)
        selected_values = np.concatenate([selected_values, interpolated[:, None]], axis=1)
    return trapezoid(selected_values, selected_times, axis=1) / horizon_lt


def interval_nrmse_auc(
    error: np.ndarray, times_lt: np.ndarray, start_lt: float, end_lt: float
) -> np.ndarray:
    if end_lt <= start_lt or times_lt[-1] + 1e-12 < end_lt:
        return np.full(error.shape[0], np.nan)
    dense = np.linspace(start_lt, end_lt, 201)
    values = np.vstack([
        np.interp(dense, times_lt, np.sqrt(np.maximum(row, 0.0))) for row in error
    ])
    return trapezoid(values, dense, axis=1) / (end_lt - start_lt)


def rational_quadratic_mmd(
    first: np.ndarray, second: np.ndarray, *, maximum_samples: int = 1000, seed: int = 2026
) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    if first.shape[0] > maximum_samples:
        first = first[rng.choice(first.shape[0], maximum_samples, replace=False)]
    if second.shape[0] > maximum_samples:
        second = second[rng.choice(second.shape[0], maximum_samples, replace=False)]
    combined = np.concatenate([first, second], axis=0)
    pairwise = pdist(combined)
    positive = pairwise[pairwise > 0]
    bandwidth_squared = float(np.median(positive) ** 2) if positive.size else 1.0
    bandwidth_squared = max(bandwidth_squared, 1e-12)

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        squared_distance = cdist(left, right, metric="sqeuclidean")
        return 1.0 / (1.0 + squared_distance / (2.0 * bandwidth_squared))

    value = kernel(first, first).mean() + kernel(second, second).mean() - 2.0 * kernel(first, second).mean()
    return float(max(value, 0.0))


def distribution_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    stability_bound: float,
    maximum_samples: int,
) -> dict[str, float]:
    finite_by_trajectory = np.all(np.isfinite(prediction), axis=(1, 2))
    bounded_by_trajectory = np.nanmax(np.abs(prediction), axis=(1, 2)) < stability_bound
    stable = finite_by_trajectory & bounded_by_trajectory
    stability_fraction = float(np.mean(stable))
    if not np.any(stable):
        return {
            "stability_fraction": stability_fraction, "rq_mmd": float("nan"),
            "wasserstein_x": float("nan"), "wasserstein_y": float("nan"),
            "wasserstein_z": float("nan"), "covariance_relative_error": float("nan"),
        }
    predicted_samples = prediction[stable].reshape(-1, 3)
    truth_samples = truth[stable].reshape(-1, 3)
    covariance_truth = np.cov(truth_samples, rowvar=False)
    covariance_prediction = np.cov(predicted_samples, rowvar=False)
    result = {
        "stability_fraction": stability_fraction,
        "rq_mmd": rational_quadratic_mmd(
            predicted_samples, truth_samples, maximum_samples=maximum_samples
        ),
        "covariance_relative_error": float(
            np.linalg.norm(covariance_prediction - covariance_truth) / np.linalg.norm(covariance_truth)
        ),
    }
    for dimension, name in enumerate(("x", "y", "z")):
        result[f"wasserstein_{name}"] = float(
            wasserstein_distance(predicted_samples[:, dimension], truth_samples[:, dimension])
        )
    return result


def learned_lyapunov_metrics(
    field: Callable[[np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    reference_spectrum: np.ndarray,
    *,
    duration: float,
    step: float,
    qr_interval: int,
) -> dict[str, float | list[float]]:
    try:
        spectrum = lyapunov_spectrum(
            lambda state: np.asarray(field(np.asarray(state)[None]), dtype=np.float64)[0],
            lambda state: finite_difference_jacobian(
                lambda points: np.asarray(field(points), dtype=np.float64), state
            ),
            initial_state, duration=duration, burn_in=min(2.0, duration / 5.0),
            step=step, qr_interval=qr_interval,
        )
        reference_spectrum = np.asarray(reference_spectrum, dtype=np.float64)
        relative_error = float(np.linalg.norm(spectrum - reference_spectrum) / np.linalg.norm(reference_spectrum))
        return {"lyapunov_spectrum": spectrum.tolist(), "lyapunov_spectrum_relative_error": relative_error}
    except (FloatingPointError, ValueError, RuntimeError, OverflowError):
        return {"lyapunov_spectrum": [float("nan")] * 3, "lyapunov_spectrum_relative_error": float("nan")}
