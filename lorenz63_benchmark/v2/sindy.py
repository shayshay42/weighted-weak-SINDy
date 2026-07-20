from __future__ import annotations

import json
import math
import time
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from .artifacts import atomic_save_npz, atomic_write_csv
from .base import NumpyVectorFieldAdapter
from .base import Normalization
from .contracts import contract_for
from .numerics import compact_taper, integrate_rk4, observed_derivatives


SINDY_METHODS = {
    "sindy_strong", "sindy_weak", "sindy_weighted", "sindy_weak_weighted",
}


def polynomial_exponents(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    exponents: list[tuple[int, ...]] = [(0,) * dimension]
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(dimension), order):
            exponent = [0] * dimension
            for index in indices:
                exponent[index] += 1
            exponents.append(tuple(exponent))
    return tuple(exponents)


def exponent_name(exponent: tuple[int, ...]) -> str:
    variables = ("x", "y", "z")
    parts = []
    for variable, power in zip(variables, exponent):
        if power == 1:
            parts.append(variable)
        elif power > 1:
            parts.append(f"{variable}^{power}")
    return "1" if not parts else " ".join(parts)


def polynomial_library(states: np.ndarray, exponents: tuple[tuple[int, ...], ...]) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    columns = []
    for exponent in exponents:
        column = np.ones(states.shape[:-1], dtype=np.float64)
        for dimension, power in enumerate(exponent):
            if power:
                column *= states[..., dimension] ** power
        columns.append(column)
    return np.stack(columns, axis=-1)


def normalized_to_physical_coefficients(
    coefficients: np.ndarray,
    exponents: tuple[tuple[int, ...], ...],
    normalization: Normalization,
) -> np.ndarray:
    """Map dz/d(lambda*t)=Theta(z)C to dx/dt in the physical monomial basis."""
    lookup = {exponent: index for index, exponent in enumerate(exponents)}
    physical = np.zeros_like(coefficients, dtype=np.float64)
    for source_index, source_exponent in enumerate(exponents):
        expansions: list[tuple[tuple[int, ...], float]] = [((0, 0, 0), 1.0)]
        for dimension, power in enumerate(source_exponent):
            next_expansions = []
            for partial_exponent, partial_factor in expansions:
                for physical_power in range(power + 1):
                    exponent = list(partial_exponent)
                    exponent[dimension] = physical_power
                    factor = (
                        math.comb(power, physical_power)
                        * (-normalization.state_mean[dimension]) ** (power - physical_power)
                        / normalization.state_std[dimension] ** power
                    )
                    next_expansions.append((tuple(exponent), partial_factor * factor))
            expansions = next_expansions
        for target_exponent, basis_factor in expansions:
            target_index = lookup[target_exponent]
            physical[target_index] += (
                basis_factor
                * coefficients[source_index]
                * normalization.state_std
                * normalization.time_scale
            )
    return physical


def solve_ridge(features: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    if ridge <= 0:
        return np.linalg.lstsq(features, targets, rcond=None)[0]
    return np.linalg.solve(
        features.T @ features + ridge * np.eye(features.shape[1]), features.T @ targets
    )


def stlsq(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float,
    ridge: float,
    max_iter: int,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if sample_weights is None:
        sample_weights = np.ones(features.shape[0], dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if sample_weights.shape != (features.shape[0],) or np.any(sample_weights < 0):
        raise ValueError("invalid SINDy sample weights")
    root_weight = np.sqrt(sample_weights)[:, None]
    weighted_features = features * root_weight
    weighted_targets = targets * root_weight
    scales = np.sqrt(np.mean(weighted_features ** 2, axis=0))
    scales = np.where(scales > 1e-14, scales, 1.0)
    normalized = weighted_features / scales
    coefficients = solve_ridge(normalized, weighted_targets, ridge)
    active = np.abs(coefficients / scales[:, None]) >= threshold
    for _ in range(max_iter):
        previous = active.copy()
        for dimension in range(targets.shape[1]):
            mask = active[:, dimension]
            coefficients[~mask, dimension] = 0.0
            if np.any(mask):
                coefficients[mask, dimension] = solve_ridge(
                    normalized[:, mask], weighted_targets[:, dimension : dimension + 1], ridge
                ).ravel()
                active[:, dimension] = np.abs(coefficients[:, dimension] / scales) >= threshold
        if np.array_equal(active, previous):
            break
    physical_coefficients = coefficients / scales[:, None]
    physical_coefficients[~active] = 0.0
    return physical_coefficients, scales


def _numpy_test_functions(length: int, dt: float, modes: int, power: int = 2) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(-1.0, 1.0, length)
    support = np.maximum(1.0 - s * s, 0.0)
    bump = support ** power
    dbump = -2.0 * power * s * support ** (power - 1)
    phi = []
    derivative = []
    for mode in range(modes):
        coefficient = np.zeros(mode + 1)
        coefficient[-1] = 1.0
        polynomial = np.polynomial.legendre.legval(s, coefficient)
        derivative_coefficient = np.polynomial.legendre.legder(coefficient)
        dpolynomial = np.polynomial.legendre.legval(s, derivative_coefficient)
        phi.append(bump * polynomial)
        derivative.append(dbump * polynomial + bump * dpolynomial)
    duration = dt * (length - 1)
    return np.stack(phi, axis=-1), np.stack(derivative, axis=-1) * (2.0 / duration)


def build_weak_system(
    states: np.ndarray,
    dt: float,
    exponents: tuple[tuple[int, ...], ...],
    window_steps: int,
    stride_steps: int,
    modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    length = window_steps + 1
    phi, derivative = _numpy_test_functions(length, dt, modes)
    weights = np.full(length, dt)
    weights[[0, -1]] *= 0.5
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for trajectory in states:
        for start in range(0, trajectory.shape[0] - window_steps, stride_steps):
            window = trajectory[start : start + length]
            library = polynomial_library(window, exponents)
            for mode in range(modes):
                feature_rows.append(np.sum(weights[:, None] * phi[:, mode, None] * library, axis=0))
                target_rows.append(-np.sum(weights[:, None] * derivative[:, mode, None] * window, axis=0))
    if not feature_rows:
        raise ValueError("weak SINDy configuration produced no regression windows")
    return np.asarray(feature_rows), np.asarray(target_rows)


def weak_window_center_weights(
    states: np.ndarray,
    window_steps: int,
    stride_steps: int,
    modes: int,
    *,
    uniform: bool = False,
) -> np.ndarray:
    """Weight weak equations by their temporal window centers along each trajectory."""
    row_weights: list[np.ndarray] = []
    for trajectory in states:
        window_count = len(range(0, trajectory.shape[0] - window_steps, stride_steps))
        if window_count == 0:
            continue
        window_weights = (
            np.ones(window_count, dtype=np.float64)
            if uniform else compact_taper(window_count)
        )
        row_weights.append(np.repeat(window_weights, modes))
    if not row_weights:
        raise ValueError("weak SINDy configuration produced no regression windows")
    return np.concatenate(row_weights)


def fit_sindy_coefficients(
    method: str,
    train_split: dict[str, Any],
    config: dict[str, Any],
    trajectory_count: int | None = None,
    uniform_weights: bool = False,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...], dict[str, Any]]:
    if method not in SINDY_METHODS:
        raise ValueError(f"not a SINDy method: {method}")
    cfg = config["sindy"]
    states = np.asarray(train_split["states"], dtype=np.float64)
    if trajectory_count is not None:
        if trajectory_count < 1 or trajectory_count > states.shape[0]:
            raise ValueError("invalid SINDy trajectory count")
        states = states[:trajectory_count]
    dt = float(train_split["metadata"]["dt"])
    normalization = Normalization.from_split(train_split)
    normalized_states = normalization.normalize_state(states)
    normalized_dt = dt * normalization.time_scale
    degree = int(cfg["degree"])
    exponents = polynomial_exponents(3, degree)
    sample_weights = None
    if method in {"sindy_weak", "sindy_weak_weighted"}:
        features, targets = build_weak_system(
            normalized_states, normalized_dt, exponents, int(cfg["weak_window_steps"]),
            int(cfg["weak_stride_steps"]), int(cfg["weak_modes"]),
        )
        if method == "sindy_weak_weighted":
            sample_weights = weak_window_center_weights(
                normalized_states,
                int(cfg["weak_window_steps"]),
                int(cfg["weak_stride_steps"]),
                int(cfg["weak_modes"]),
                uniform=uniform_weights,
            )
    else:
        derivative_states, derivatives, _ = observed_derivatives(
            states, dt, float(train_split["metadata"]["noise_level"]),
            savgol_window=int(cfg["savgol_window"]),
            savgol_polyorder=int(cfg["savgol_polyorder"]),
        )
        normalized_points = normalization.normalize_state(derivative_states)
        normalized_derivatives = derivatives / (
            normalization.state_std * normalization.time_scale
        )
        features = polynomial_library(normalized_points, exponents).reshape(-1, len(exponents))
        targets = normalized_derivatives.reshape(-1, 3)
        if method == "sindy_weighted":
            per_trajectory = (
                np.ones(normalized_points.shape[1]) if uniform_weights
                else compact_taper(normalized_points.shape[1])
            )
            sample_weights = np.tile(per_trajectory, normalized_points.shape[0])
    coefficients, feature_scales = stlsq(
        features, targets, threshold=float(cfg["threshold"]), ridge=float(cfg["ridge"]),
        max_iter=int(cfg["max_iter"]), sample_weights=sample_weights,
    )
    residual = features @ coefficients - targets
    if sample_weights is None:
        rmse = np.sqrt(np.mean(residual ** 2, axis=0))
    else:
        rmse = np.sqrt(np.average(residual ** 2, axis=0, weights=sample_weights))
    weighted = method in {"sindy_weighted", "sindy_weak_weighted"} and not uniform_weights
    weighting = "uniform"
    weighting_scope = "none"
    if weighted and method == "sindy_weighted":
        weighting = "compact_endpoint_taper"
        weighting_scope = "observation_time"
    elif weighted:
        weighting = "compact_endpoint_taper"
        weighting_scope = "weak_window_center"
    metadata = {
        "degree": degree, "threshold": float(cfg["threshold"]), "ridge": float(cfg["ridge"]),
        "max_iter": int(cfg["max_iter"]), "fit_rows": int(features.shape[0]),
        "trajectory_count": int(states.shape[0]), "feature_scales": feature_scales.tolist(),
        "fit_rmse": rmse.tolist(), "uniform_weights": bool(uniform_weights),
        "formulation": "weak" if method in {"sindy_weak", "sindy_weak_weighted"} else "strong",
        "weighting": weighting, "weighting_scope": weighting_scope,
        "normalization": {
            "state_mean": normalization.state_mean.tolist(),
            "state_std": normalization.state_std.tolist(),
            "time_scale": normalization.time_scale,
        },
    }
    return coefficients, exponents, metadata


def _field(
    coefficients: np.ndarray,
    exponents: tuple[tuple[int, ...], ...],
    normalization: Normalization,
    states: np.ndarray,
) -> np.ndarray:
    normalized = normalization.normalize_state(states)
    normalized_derivative = polynomial_library(normalized, exponents) @ coefficients
    return normalization.state_std * normalization.time_scale * normalized_derivative


def train_sindy(
    method: str,
    train_split: dict[str, Any],
    validation_split: dict[str, Any],
    config: dict[str, Any],
    model_seed: int,
    output_dir: str | Path,
    trajectory_count: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    del model_seed
    started = time.perf_counter()
    coefficients, exponents, metadata = fit_sindy_coefficients(
        method, train_split, config, trajectory_count=trajectory_count
    )
    normalization = Normalization.from_split(train_split)
    physical_coefficients = normalized_to_physical_coefficients(
        coefficients, exponents, normalization
    )
    validation_states = np.asarray(validation_split["states"], dtype=np.float64)
    validation_times = np.asarray(validation_split["times"], dtype=np.float64)
    maximum = min(validation_times[-1], 2.0 / float(train_split["metadata"]["normalization"]["time_scale"]))
    keep = validation_times <= maximum + 1e-12
    prediction, evaluations = integrate_rk4(
        lambda state: _field(coefficients, exponents, normalization, state),
        validation_states[:, 0], validation_times[keep],
        internal_step=float(config["evaluation"]["internal_step"]),
    )
    scale = np.asarray(train_split["metadata"]["normalization"]["state_std"], dtype=np.float64)
    nrmse = np.sqrt(np.mean(((prediction - validation_states[:, keep]) / scale) ** 2, axis=-1))
    validation_score = float(trapezoid(np.median(nrmse, axis=0), validation_times[keep]) / maximum)
    metadata["validation_nrmse_auc_0_2LT"] = validation_score
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "model.npz"
    contract = contract_for(method)
    atomic_save_npz(
        checkpoint_path,
        schema=np.asarray("lorenz63-model-v2"), method=np.asarray(method), track=np.asarray(contract.track),
        information_contract=np.asarray(contract.information), coefficients=coefficients,
        physical_coefficients=physical_coefficients,
        exponents=np.asarray(exponents, dtype=np.int64), metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    training = {
        "optimizer_updates": 1, "wall_time_seconds": time.perf_counter() - started,
        "parameter_count": int(coefficients.size), "peak_gpu_memory_bytes": 0,
        "vector_field_evaluations_validation": evaluations,
    }
    atomic_write_csv(output_dir / "history.csv", [{"method": method, **metadata}])
    return checkpoint_path, training


def load_sindy(path: str | Path, internal_step: float) -> NumpyVectorFieldAdapter:
    with np.load(path, allow_pickle=False) as loaded:
        if str(loaded["schema"]) != "lorenz63-model-v2":
            raise ValueError(f"not a v2 checkpoint: {path}")
        method = str(loaded["method"])
        if method not in SINDY_METHODS:
            raise ValueError(f"not a SINDy checkpoint: {path}")
        track = str(loaded["track"])
        coefficients = np.asarray(loaded["coefficients"], dtype=np.float64)
        physical_coefficients = np.asarray(loaded["physical_coefficients"], dtype=np.float64)
        exponents = tuple(tuple(int(value) for value in row) for row in loaded["exponents"])
        metadata = json.loads(str(loaded["metadata"]))
    norm = metadata["normalization"]
    normalization = Normalization(
        np.asarray(norm["state_mean"], dtype=np.float64),
        np.asarray(norm["state_std"], dtype=np.float64), float(norm["time_scale"]),
    )
    return NumpyVectorFieldAdapter(
        method=method, track=track,
        field=lambda states: _field(coefficients, exponents, normalization, states),
        internal_step=internal_step,
        parameters={
            "coefficients": physical_coefficients.tolist(),
            "exponents": [list(row) for row in exponents],
        },
    )
