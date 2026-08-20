from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifacts import (
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_json,
    command_line,
    environment_snapshot,
    sha256_file,
    sha256_json,
    source_hash,
    utc_now,
)
from .base import Normalization
from .config import config_for_method, load_config
from .contracts import METHODS
from .data import load_split
from .metrics import normalized_rmse_auc, normalized_squared_error
from .numerics import integrate_rk4
from .panda import PANDA_CHECKPOINT_SCHEMA, PandaForecastAdapter
from .sindy import _field, fit_sindy_coefficients, polynomial_library


CONTEXT_MATCHED_SCHEMA = "lorenz63-context-matched-v2"
CONTEXT_DATA_SCHEMA = "lorenz63-context-prefix-data-v2"
CONTEXT_MODEL_SCHEMA = "lorenz63-context-sindy-bank-v2"
CONTEXT_METHODS = ("sindy_weak_weighted", "sindy_weak", "panda_zero_shot")
LINESTYLES = {
    "sindy_weak_weighted": "-",
    "sindy_weak": (0, (2.2, 1.4)),
    "panda_zero_shot": (0, (7, 2.5)),
}


def prefix_normalization(context_states: np.ndarray, time_scale: float) -> Normalization:
    context_states = np.asarray(context_states, dtype=np.float64)
    if context_states.ndim != 2 or context_states.shape[1] != 3:
        raise ValueError("one trajectory context must have shape [time, 3]")
    state_std = np.std(context_states, axis=0, ddof=0)
    state_std = np.maximum(state_std, 1e-12)
    return Normalization(
        state_mean=np.mean(context_states, axis=0),
        state_std=state_std,
        time_scale=float(time_scale),
    )


def _prefix_train_split(
    context_states: np.ndarray,
    *,
    dt: float,
    time_scale: float,
) -> tuple[dict[str, Any], Normalization]:
    normalization = prefix_normalization(context_states, time_scale)
    split = {
        "states": np.asarray(context_states, dtype=np.float64)[None],
        "times": np.arange(context_states.shape[0], dtype=np.float64) * dt,
        "metadata": {
            "schema": "lorenz63-context-prefix-v2",
            "role": "online_context",
            "dt": float(dt),
            "noise_level": 0.0,
            "normalization": {
                "state_mean": normalization.state_mean.tolist(),
                "state_std": normalization.state_std.tolist(),
                "time_scale": normalization.time_scale,
            },
        },
    }
    return split, normalization


def forecast_online_sindy(
    method: str,
    context_states: np.ndarray,
    forecast_times: np.ndarray,
    *,
    dt: float,
    time_scale: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one autonomous SINDy field using only one observed context prefix."""
    if method not in {"sindy_weak", "sindy_weak_weighted"}:
        raise ValueError(f"not an online SINDy method: {method}")
    train_split, normalization = _prefix_train_split(
        np.asarray(context_states, dtype=np.float64), dt=dt, time_scale=time_scale
    )
    started = time.perf_counter()
    coefficients, exponents, metadata = fit_sindy_coefficients(method, train_split, config)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction, evaluations = integrate_rk4(
        lambda states: _field(coefficients, exponents, normalization, states),
        np.asarray(context_states[-1:], dtype=np.float64),
        np.asarray(forecast_times, dtype=np.float64),
        internal_step=float(config["evaluation"]["internal_step"]),
    )
    return prediction[0], {
        "fit_seconds": fit_seconds,
        "forecast_seconds": time.perf_counter() - started,
        "vector_field_evaluations": int(evaluations),
        "fit_rows": int(metadata["fit_rows"]),
        "coefficients": coefficients,
        "exponents": np.asarray(exponents, dtype=np.int64),
    }


def extract_context_data(
    *,
    config_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> tuple[Path, Path]:
    """Split held-out trajectories into prefix-only adaptation data and future truth."""
    config = load_config(config_path)
    protocol = config["context_matched"]
    output_dir = Path(output_dir)
    context_path = output_dir / "context.npz"
    truth_path = output_dir / "forecast_truth.npz"
    extraction_manifest_path = output_dir / "manifest.json"
    identity = {
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "test_sha256": sha256_file(test_path),
    }
    if extraction_manifest_path.exists() and not force:
        existing = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and all(
            existing.get(key) == value for key, value in identity.items()
        ):
            records = existing.get("artifacts", {}).values()
            if records and all(
                Path(record["path"]).exists()
                and sha256_file(record["path"]) == record["sha256"]
                for record in records
            ):
                return context_path, truth_path

    test = load_split(test_path, "test")
    metadata = test["metadata"]
    states = np.asarray(test["states"], dtype=np.float64)
    times = np.asarray(test["times"], dtype=np.float64)
    context_steps = int(protocol["context_steps"])
    if context_steps < 2 or context_steps > times.size:
        raise ValueError("invalid context length for held-out trajectories")
    origin = context_steps - 1
    largest = float(metadata["normalization"]["time_scale"])
    relative_times = times[origin:] - times[origin]
    requested_horizon = float(protocol["forecast_lyapunov_times"])
    stop = int(np.searchsorted(relative_times * largest, requested_horizon, side="left"))
    stop = min(stop, relative_times.size - 1)
    if relative_times[stop] * largest + 1e-12 < requested_horizon:
        raise ValueError("held-out trajectory does not reach the requested forecast horizon")
    context_metadata = {
        "schema": CONTEXT_DATA_SCHEMA,
        "role": "adaptation_context",
        "data_seed": int(metadata["data_seed"]),
        "dt": float(metadata["dt"]),
        "largest_lyapunov_exponent": largest,
        "context_steps": context_steps,
        "forecast_origin_index": origin,
        "contains_future_states": False,
        "source_test_sha256": identity["test_sha256"],
    }
    truth_metadata = {
        "schema": CONTEXT_DATA_SCHEMA,
        "role": "evaluation_truth",
        "data_seed": int(metadata["data_seed"]),
        "dt": float(metadata["dt"]),
        "largest_lyapunov_exponent": largest,
        "context_steps": context_steps,
        "forecast_origin_index": origin,
        "forecast_horizon_lyapunov": float(relative_times[stop] * largest),
        "linked_context_sha256": None,
        "source_test_sha256": identity["test_sha256"],
    }
    atomic_save_npz(
        context_path,
        states=states[:, :context_steps],
        times=times[:context_steps] - times[0],
        metadata=np.asarray(json.dumps(context_metadata, sort_keys=True)),
    )
    truth_metadata["linked_context_sha256"] = sha256_file(context_path)
    atomic_save_npz(
        truth_path,
        states=states[:, origin : origin + stop + 1],
        times=relative_times[: stop + 1],
        metadata=np.asarray(json.dumps(truth_metadata, sort_keys=True)),
    )
    extraction_manifest = {
        "schema": "lorenz63-context-extraction-manifest-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "command": command_line(),
        "environment": environment_snapshot(),
        **identity,
        "artifacts": {
            "context": {"path": str(context_path.resolve()), "sha256": sha256_file(context_path)},
            "forecast_truth": {"path": str(truth_path.resolve()), "sha256": sha256_file(truth_path)},
        },
    }
    atomic_write_json(extraction_manifest_path, extraction_manifest)
    return context_path, truth_path


def _load_context_artifact(
    path: str | Path, expected_role: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as loaded:
        states = np.asarray(loaded["states"], dtype=np.float64)
        times = np.asarray(loaded["times"], dtype=np.float64)
        metadata = json.loads(str(loaded["metadata"]))
    if metadata.get("schema") != CONTEXT_DATA_SCHEMA:
        raise ValueError(f"not context-matched data: {path}")
    if metadata.get("role") != expected_role:
        raise ValueError(f"expected {expected_role}, got {metadata.get('role')}")
    return states, times, metadata


def fit_context_sindy_bank(
    *,
    config_path: str | Path,
    context_path: str | Path,
    output_dir: str | Path,
    method: str,
    force: bool = False,
) -> Path:
    """Fit one SINDy field per prefix; this interface cannot receive future truth."""
    if method not in {"sindy_weak", "sindy_weak_weighted"}:
        raise ValueError(f"not a context-matched SINDy method: {method}")
    config = load_config(config_path)
    configured = config_for_method(config, 0.0, method)
    fit_config = {
        "sindy": dict(configured["sindy"]),
        "evaluation": {"internal_step": float(configured["evaluation"]["internal_step"])},
    }
    output_dir = Path(output_dir)
    model_path = output_dir / "models.npz"
    manifest_path = output_dir / "manifest.json"
    identity = {
        "source_hash": source_hash(),
        "fit_config_hash": sha256_json(fit_config),
        "input_hashes": {"context": sha256_file(context_path)},
        "method": method,
    }
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and all(
            existing.get(key) == value for key, value in identity.items()
        ):
            record = existing.get("artifacts", {}).get("models", {})
            if (
                Path(record.get("path", "")).exists()
                and sha256_file(record["path"]) == record.get("sha256")
            ):
                return model_path

    contexts, _times, metadata = _load_context_artifact(context_path, "adaptation_context")
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients = []
    means = []
    standard_deviations = []
    fit_seconds = []
    fit_rows = []
    exponents = None
    for context in contexts:
        train_split, normalization = _prefix_train_split(
            context,
            dt=float(metadata["dt"]),
            time_scale=float(metadata["largest_lyapunov_exponent"]),
        )
        started = time.perf_counter()
        value, value_exponents, details = fit_sindy_coefficients(
            method, train_split, fit_config
        )
        fit_seconds.append(time.perf_counter() - started)
        coefficients.append(value)
        means.append(normalization.state_mean)
        standard_deviations.append(normalization.state_std)
        fit_rows.append(int(details["fit_rows"]))
        if exponents is None:
            exponents = np.asarray(value_exponents, dtype=np.int64)
        elif not np.array_equal(exponents, np.asarray(value_exponents, dtype=np.int64)):
            raise RuntimeError("online SINDy fits produced inconsistent libraries")
    atomic_save_npz(
        model_path,
        schema=np.asarray(CONTEXT_MODEL_SCHEMA),
        method=np.asarray(method),
        coefficients=np.stack(coefficients),
        exponents=exponents,
        state_mean=np.stack(means),
        state_std=np.stack(standard_deviations),
        time_scale=np.asarray(metadata["largest_lyapunov_exponent"], dtype=np.float64),
        fit_seconds=np.asarray(fit_seconds, dtype=np.float64),
        fit_rows=np.asarray(fit_rows, dtype=np.int64),
        context_sha256=np.asarray(identity["input_hashes"]["context"]),
    )
    manifest = {
        "schema": "lorenz63-context-sindy-fit-manifest-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "information_contract": [
            "observed_test_prefix_only",
            "observation_times",
            "degree_2_polynomial_library",
        ],
        "forbidden_inputs": [
            "forecast_truth",
            "separate_training_trajectories",
            "validation_trajectories",
            "exact_derivatives",
            "lorenz_parameters",
            "generator_metadata",
        ],
        "future_states_visible_during_adaptation": False,
        "trajectory_count": int(contexts.shape[0]),
        "adaptation_fits_per_trajectory": 1,
        **identity,
        "artifacts": {
            "models": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)}
        },
    }
    atomic_write_json(manifest_path, manifest)
    return model_path


def _panda_checkpoint(config: dict[str, Any], *, dt: float) -> dict[str, Any]:
    # The adapter's global normalization is deliberately neutral. Context-matched
    # inference supplies per-prefix normalized states directly.
    return {
        "schema": PANDA_CHECKPOINT_SCHEMA,
        "method": "panda_zero_shot",
        "model": dict(config["panda"]),
        "normalization": {
            "state_mean": [0.0, 0.0, 0.0],
            "state_std": [1.0, 1.0, 1.0],
            "time_scale": 1.0,
        },
        "observation_dt": float(dt),
    }


def forecast_panda_context_matched(
    adapter: PandaForecastAdapter,
    contexts: np.ndarray,
    forecast_times: np.ndarray,
    *,
    time_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    contexts = np.asarray(contexts, dtype=np.float64)
    normalizations = [prefix_normalization(context, time_scale) for context in contexts]
    normalized = np.stack([
        normalization.normalize_state(context)
        for normalization, context in zip(normalizations, contexts)
    ])
    started = time.perf_counter()
    normalized_prediction = adapter.forecast_normalized_context(normalized, forecast_times)
    prediction = np.stack([
        normalization.denormalize_state(value)
        for normalization, value in zip(normalizations, normalized_prediction)
    ])
    return prediction, {
        "fit_seconds": 0.0,
        "forecast_seconds": time.perf_counter() - started,
        "surrogate_evaluations": int(adapter.last_surrogate_evaluations),
        "inference_peak_gpu_memory_bytes": int(adapter.last_peak_gpu_memory_bytes),
        "parameter_count": int(adapter.parameter_count),
        "probabilistic": bool(adapter.probabilistic),
        "forecast_sample_count": int(adapter.forecast_sample_count),
    }


def _batched_sindy_field(
    states: np.ndarray,
    coefficients: np.ndarray,
    exponents: tuple[tuple[int, ...], ...],
    means: np.ndarray,
    standard_deviations: np.ndarray,
    time_scale: float,
) -> np.ndarray:
    normalized = (np.asarray(states, dtype=np.float64) - means) / standard_deviations
    library = polynomial_library(normalized, exponents)
    normalized_derivative = np.einsum(
        "bf,bfg->bg", library, coefficients, optimize=True
    )
    return standard_deviations * float(time_scale) * normalized_derivative


def _metric_rows(
    *,
    method: str,
    data_seed: int,
    prediction: np.ndarray,
    truth: np.ndarray,
    forecast_times: np.ndarray,
    largest: float,
    scale: np.ndarray,
    context_steps: int,
    native_prediction_steps: int,
    forecast_horizon_lt: float,
    vpt_threshold: float,
    nrmse_error_cap: float,
    stability_bound: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    error = normalized_squared_error(prediction, truth, scale)
    times_lt = forecast_times * largest
    vpt = np.full(error.shape[0], float(forecast_horizon_lt), dtype=np.float64)
    censored = np.ones(error.shape[0], dtype=bool)
    within_horizon = times_lt <= forecast_horizon_lt + 1e-12
    for trajectory_id, row in enumerate(error):
        crossing = np.flatnonzero(within_horizon & (~np.isfinite(row) | (row > vpt_threshold)))
        if crossing.size:
            vpt[trajectory_id] = float(times_lt[crossing[0]])
            censored[trajectory_id] = False
    unstable = (
        ~np.all(np.isfinite(prediction), axis=-1)
        | np.any(np.abs(prediction) >= stability_bound, axis=-1)
    )
    instability_time = np.full(error.shape[0], np.nan, dtype=np.float64)
    for trajectory_id, row in enumerate(unstable):
        crossing = np.flatnonzero(within_horizon & row)
        if crossing.size:
            instability_time[trajectory_id] = float(times_lt[crossing[0]])

    # Retain divergence in the forecast and VPT event while keeping AUC
    # aggregation defined under a fixed, declared worst-error cap.
    restricted_error = np.minimum(
        np.nan_to_num(
            error,
            nan=nrmse_error_cap,
            posinf=nrmse_error_cap,
            neginf=nrmse_error_cap,
        ),
        nrmse_error_cap,
    )
    native_horizon_lt = float(times_lt[min(native_prediction_steps, times_lt.size - 1)])
    native_auc = normalized_rmse_auc(restricted_error, times_lt, native_horizon_lt)
    auc = {
        horizon: normalized_rmse_auc(restricted_error, times_lt, horizon)
        for horizon in (1.0, 2.0, 5.0)
    }
    absolute_error_cap = float(np.sqrt(nrmse_error_cap))
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_absolute_error = np.mean(
            np.abs(prediction - truth) / np.asarray(scale, dtype=np.float64), axis=-1
        )
    restricted_absolute_error = np.minimum(
        np.nan_to_num(
            normalized_absolute_error,
            nan=absolute_error_cap,
            posinf=absolute_error_cap,
            neginf=absolute_error_cap,
        ),
        absolute_error_cap,
    )
    mean_coordinate_point_crps_native = normalized_rmse_auc(
        restricted_absolute_error ** 2, times_lt, native_horizon_lt
    )
    rows = []
    for trajectory_id in range(prediction.shape[0]):
        rows.append({
            "method": method,
            "data_seed": int(data_seed),
            "trajectory_id": trajectory_id,
            "context_steps": int(context_steps),
            "native_prediction_steps": int(native_prediction_steps),
            "native_horizon_lt": native_horizon_lt,
            "vpt_restricted_lt": float(vpt[trajectory_id]),
            "vpt_censored": bool(censored[trajectory_id]),
            "forecast_instability": bool(np.any(unstable[trajectory_id] & within_horizon)),
            "forecast_instability_lt": float(instability_time[trajectory_id]),
            "restricted_nrmse_auc_native": float(native_auc[trajectory_id]),
            "restricted_nrmse_auc_0_1LT": float(auc[1.0][trajectory_id]),
            "restricted_nrmse_auc_0_2LT": float(auc[2.0][trajectory_id]),
            "restricted_nrmse_auc_0_5LT": float(auc[5.0][trajectory_id]),
            "mean_coordinate_point_crps_native": float(
                mean_coordinate_point_crps_native[trajectory_id]
            ),
        })
    return rows, error, times_lt


def evaluate_context_matched(
    *,
    config_path: str | Path,
    context_path: str | Path,
    truth_path: str | Path,
    output_dir: str | Path,
    method: str,
    model_path: str | Path | None = None,
    device: str = "cpu",
    force: bool = False,
    panda_adapter_factory: Callable[[dict[str, Any], str], PandaForecastAdapter] | None = None,
) -> Path:
    config = load_config(config_path)
    protocol = config["context_matched"]
    if method not in protocol["methods"] or method not in CONTEXT_METHODS:
        raise ValueError(f"method {method!r} is not configured for context matching")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    input_hashes = {
        "context": sha256_file(context_path),
        "forecast_truth": sha256_file(truth_path),
    }
    if method != "panda_zero_shot":
        if model_path is None:
            raise ValueError("context-matched SINDy evaluation requires --models")
        input_hashes["models"] = sha256_file(model_path)
    elif model_path is not None:
        raise ValueError("Panda evaluation does not accept a fitted model bank")
    identity = {
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "input_hashes": input_hashes,
        "method": method,
    }
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and all(
            existing.get(key) == value for key, value in identity.items()
        ):
            artifacts = existing.get("artifacts", {}).values()
            if artifacts and all(
                Path(record["path"]).exists()
                and sha256_file(record["path"]) == record["sha256"]
                for record in artifacts
            ):
                return manifest_path

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": CONTEXT_MATCHED_SCHEMA,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "command": command_line(),
        "environment": environment_snapshot(),
        "information_contract": [
            "observed_test_prefix_only",
            "observation_times",
            *(
                ["external_pretraining_corpus"]
                if method == "panda_zero_shot"
                else ["degree_2_polynomial_library"]
            ),
        ],
        "forbidden_inputs": [
            "separate_training_trajectories",
            "validation_trajectories",
            "future_test_states_during_fit",
            "exact_derivatives",
            "lorenz_parameters",
        ],
        **identity,
    }
    atomic_write_json(manifest_path, manifest)
    try:
        contexts, _context_times, context_metadata = _load_context_artifact(
            context_path, "adaptation_context"
        )
        truth, forecast_times, truth_metadata = _load_context_artifact(
            truth_path, "evaluation_truth"
        )
        if truth_metadata["linked_context_sha256"] != input_hashes["context"]:
            raise ValueError("forecast truth is linked to a different context artifact")
        if int(context_metadata["data_seed"]) != int(truth_metadata["data_seed"]):
            raise ValueError("context and forecast truth use different data seeds")
        data_seed = int(context_metadata["data_seed"])
        context_steps = int(context_metadata["context_steps"])
        if context_steps != int(protocol["context_steps"]):
            raise ValueError("context artifact does not match the configured prefix length")
        if context_steps != int(config["panda"]["context_length"]):
            raise ValueError("context-matched prefix must equal Panda's required context length")
        if contexts.shape[0] != truth.shape[0] or contexts.shape[-1] != truth.shape[-1]:
            raise ValueError("context and forecast truth trajectory shapes do not align")
        largest = float(context_metadata["largest_lyapunov_exponent"])
        if not np.isclose(largest, float(truth_metadata["largest_lyapunov_exponent"])):
            raise ValueError("context and forecast truth use different Lyapunov conversion")
        dt = float(context_metadata["dt"])
        fit_seconds = 0.0
        forecast_seconds = 0.0
        vector_field_evaluations = 0
        surrogate_evaluations = 0
        peak_gpu_memory = 0
        parameter_count = 0
        probabilistic = False
        forecast_sample_count = 1

        if method == "panda_zero_shot":
            checkpoint = _panda_checkpoint(config, dt=dt)
            factory = panda_adapter_factory or (
                lambda value, target_device: PandaForecastAdapter(value, device=target_device)
            )
            adapter = factory(checkpoint, device)
            prediction, timing = forecast_panda_context_matched(
                adapter, contexts, forecast_times, time_scale=largest
            )
            fit_seconds = float(timing["fit_seconds"])
            forecast_seconds = float(timing["forecast_seconds"])
            surrogate_evaluations = int(timing["surrogate_evaluations"])
            peak_gpu_memory = int(timing["inference_peak_gpu_memory_bytes"])
            parameter_count = int(timing["parameter_count"])
            probabilistic = bool(timing["probabilistic"])
            forecast_sample_count = int(timing["forecast_sample_count"])
        else:
            with np.load(model_path, allow_pickle=False) as loaded:
                if str(loaded["schema"]) != CONTEXT_MODEL_SCHEMA:
                    raise ValueError(f"not a context-matched model bank: {model_path}")
                if str(loaded["method"]) != method:
                    raise ValueError("model bank method does not match requested method")
                if str(loaded["context_sha256"]) != input_hashes["context"]:
                    raise ValueError("model bank was fitted from a different context artifact")
                coefficients = np.asarray(loaded["coefficients"], dtype=np.float64)
                exponents = tuple(
                    tuple(int(value) for value in row) for row in loaded["exponents"]
                )
                means = np.asarray(loaded["state_mean"], dtype=np.float64)
                standard_deviations = np.asarray(loaded["state_std"], dtype=np.float64)
                model_time_scale = float(loaded["time_scale"])
                fit_seconds = float(np.sum(loaded["fit_seconds"]))
            if coefficients.shape[0] != contexts.shape[0]:
                raise ValueError("model bank does not contain one fit per context trajectory")
            started = time.perf_counter()
            prediction, vector_field_evaluations = integrate_rk4(
                lambda states: _batched_sindy_field(
                    states,
                    coefficients,
                    exponents,
                    means,
                    standard_deviations,
                    model_time_scale,
                ),
                contexts[:, -1],
                forecast_times,
                internal_step=float(config["evaluation"]["internal_step"]),
            )
            forecast_seconds = time.perf_counter() - started
            parameter_count = int(coefficients.shape[1] * coefficients.shape[2])

        if prediction.shape != truth.shape:
            raise RuntimeError(
                f"prediction shape {prediction.shape} does not match truth {truth.shape}"
            )
        # A prefix-only scale avoids using the separate training split while using
        # exactly the same physical normalization for all methods on each split.
        split_scale = np.std(contexts.reshape(-1, 3), axis=0, ddof=0)
        rows, error, times_lt = _metric_rows(
            method=method,
            data_seed=data_seed,
            prediction=prediction,
            truth=truth,
            forecast_times=forecast_times,
            largest=largest,
            scale=split_scale,
            context_steps=context_steps,
            native_prediction_steps=int(protocol["native_prediction_steps"]),
            forecast_horizon_lt=float(protocol["forecast_lyapunov_times"]),
            vpt_threshold=float(config["evaluation"]["vpt_threshold"]),
            nrmse_error_cap=float(protocol["nrmse_error_cap"]),
            stability_bound=float(config["evaluation"]["stability_bound"]),
        )
        metrics_path = output_dir / "trajectory_metrics.csv"
        predictions_path = output_dir / "predictions.npz"
        run_metrics_path = output_dir / "run_metrics.csv"
        atomic_write_csv(metrics_path, rows)
        atomic_save_npz(
            predictions_path,
            prediction=prediction.astype(np.float32),
            normalized_squared_error=error.astype(np.float32),
            times=forecast_times,
            times_lt=times_lt,
            context_steps=np.asarray(context_steps, dtype=np.int64),
            forecast_origin_index=np.asarray(
                context_metadata["forecast_origin_index"], dtype=np.int64
            ),
        )
        atomic_write_csv(run_metrics_path, [{
            "method": method,
            "data_seed": data_seed,
            "trajectory_count": int(contexts.shape[0]),
            "context_steps": context_steps,
            "separate_training_trajectories": 0,
            "forecast_horizon_lt": float(protocol["forecast_lyapunov_times"]),
            "stored_prediction_horizon_lt": float(times_lt[-1]),
            "native_prediction_steps": int(protocol["native_prediction_steps"]),
            "restricted_mean_vpt_lt": float(np.mean([row["vpt_restricted_lt"] for row in rows])),
            "median_vpt_lt": float(np.median([row["vpt_restricted_lt"] for row in rows])),
            "vpt_censoring_fraction": float(np.mean([row["vpt_censored"] for row in rows])),
            "forecast_instability_fraction": float(np.mean([
                row["forecast_instability"] for row in rows
            ])),
            "mean_restricted_nrmse_auc_native": float(np.mean([
                row["restricted_nrmse_auc_native"] for row in rows
            ])),
            "mean_restricted_nrmse_auc_0_1LT": float(np.mean([
                row["restricted_nrmse_auc_0_1LT"] for row in rows
            ])),
            "mean_restricted_nrmse_auc_0_2LT": float(np.mean([
                row["restricted_nrmse_auc_0_2LT"] for row in rows
            ])),
            "mean_restricted_nrmse_auc_0_5LT": float(np.mean([
                row["restricted_nrmse_auc_0_5LT"] for row in rows
            ])),
            "mean_coordinate_point_crps_native": float(np.mean([
                row["mean_coordinate_point_crps_native"] for row in rows
            ])),
            "adaptation_wall_time_seconds": fit_seconds,
            "forecast_wall_time_seconds": forecast_seconds,
            "vector_field_evaluations": vector_field_evaluations,
            "surrogate_evaluations": surrogate_evaluations,
            "parameter_count": parameter_count,
            "inference_peak_gpu_memory_bytes": peak_gpu_memory,
            "probabilistic": probabilistic,
            "forecast_sample_count": forecast_sample_count,
        }])
        artifacts = {
            "trajectory_metrics": metrics_path,
            "run_metrics": run_metrics_path,
            "predictions": predictions_path,
        }
        manifest.update({
            "status": "complete",
            "completed_at": utc_now(),
            "data_seed": data_seed,
            "context_steps": context_steps,
            "forecast_origin_index": int(context_metadata["forecast_origin_index"]),
            "future_states_visible_during_adaptation": False,
            "normalization_source": "observed_context_prefix_only",
            "metric_policy": {
                "nrmse_auc_error_cap": float(protocol["nrmse_error_cap"]),
                "point_crps_absolute_error_cap": float(
                    np.sqrt(protocol["nrmse_error_cap"])
                ),
                "raw_divergence_retained_in_predictions": True,
                "stability_bound": float(config["evaluation"]["stability_bound"]),
            },
            "probabilistic": probabilistic,
            "forecast_sample_count": forecast_sample_count,
            "artifacts": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in artifacts.items()
            },
        })
        atomic_write_json(manifest_path, manifest)
        return manifest_path
    except BaseException as error:
        manifest.update({
            "status": "failed",
            "completed_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
        })
        atomic_write_json(manifest_path, manifest)
        raise


def _paired_split_bootstrap(
    frame: Any,
    metric: str,
    *,
    resamples: int,
    seed: int,
) -> Any:
    import pandas as pd

    methods = list(CONTEXT_METHODS)
    split_ids = sorted(int(value) for value in frame["data_seed"].unique())
    trajectory_ids = sorted(int(value) for value in frame["trajectory_id"].unique())
    index = pd.MultiIndex.from_product(
        [methods, split_ids, trajectory_ids],
        names=["method", "data_seed", "trajectory_id"],
    )
    values = frame.set_index(list(index.names))[metric].reindex(index)
    if values.isna().any():
        raise ValueError(f"incomplete context-matched matrix for {metric}")
    array = values.to_numpy(dtype=np.float64).reshape(
        len(methods), len(split_ids), len(trajectory_ids)
    )
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(methods)), dtype=np.float64)
    for draw in range(resamples):
        split_draw = rng.integers(len(split_ids), size=len(split_ids))
        selected = array[:, split_draw]
        trajectory_draw = rng.integers(
            len(trajectory_ids), size=(len(split_ids), len(trajectory_ids))
        )
        sampled = np.take_along_axis(
            selected, trajectory_draw[None, :, :], axis=2
        )
        draws[draw] = sampled.mean(axis=(1, 2))
    rows = []
    point = array.mean(axis=(1, 2))
    for method_index, method in enumerate(methods):
        rows.append({
            "method": method,
            "metric": metric,
            "estimate": float(point[method_index]),
            "ci_lower": float(np.quantile(draws[:, method_index], 0.025)),
            "ci_upper": float(np.quantile(draws[:, method_index], 0.975)),
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
        })
    return pd.DataFrame(rows)


def _despine(axis: Any) -> None:
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)


def aggregate_context_matched(
    *,
    config_path: str | Path,
    runs_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    config = load_config(config_path)
    runs_root = Path(runs_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for path in sorted(runs_root.rglob("manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") == CONTEXT_MATCHED_SCHEMA and manifest.get("status") == "complete":
            manifests.append((path.parent, manifest))
    expected = {
        (method, int(seed))
        for method in CONTEXT_METHODS
        for seed in config["final"]["data_seeds"]
    }
    actual = {(manifest["method"], int(manifest["data_seed"])) for _, manifest in manifests}
    if actual != expected or len(manifests) != len(expected):
        raise ValueError(
            f"incomplete context-matched run matrix: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}, runs={len(manifests)}"
        )
    expected_source_hash = source_hash()
    expected_config_hash = sha256_json(config)
    for run_dir, manifest in manifests:
        if manifest.get("source_hash") != expected_source_hash:
            raise ValueError(f"source hash mismatch in {run_dir}")
        if manifest.get("config_hash") != expected_config_hash:
            raise ValueError(f"config hash mismatch in {run_dir}")
        if manifest.get("future_states_visible_during_adaptation") is not False:
            raise ValueError(f"adaptation leakage flag is not false in {run_dir}")
        for name, record in manifest.get("artifacts", {}).items():
            local_path = run_dir / Path(record["path"]).name
            if not local_path.exists() or sha256_file(local_path) != record["sha256"]:
                raise ValueError(f"invalid {name} artifact in {run_dir}")
    trajectory_frame = pd.concat(
        [pd.read_csv(run_dir / "trajectory_metrics.csv") for run_dir, _ in manifests],
        ignore_index=True,
    )
    run_frame = pd.concat(
        [pd.read_csv(run_dir / "run_metrics.csv") for run_dir, _ in manifests],
        ignore_index=True,
    )
    if not (run_frame["separate_training_trajectories"] == 0).all():
        raise ValueError("context-matched runs used separate target-system training data")
    panda_rows = run_frame[run_frame["method"] == "panda_zero_shot"]
    if panda_rows["probabilistic"].astype(str).str.lower().isin({"true", "1"}).any():
        raise ValueError("released Panda checkpoint unexpectedly produced probabilistic output")
    if not (panda_rows["forecast_sample_count"] == 1).all():
        raise ValueError("released Panda checkpoint must report one deterministic forecast")
    trajectory_path = output_dir / "trajectory_metrics.csv"
    run_path = output_dir / "run_metrics.csv"
    atomic_write_csv(trajectory_path, trajectory_frame.to_dict(orient="records"))
    atomic_write_csv(run_path, run_frame.to_dict(orient="records"))
    summaries = []
    resamples = int(config["aggregation"]["bootstrap_resamples"])
    seed = int(config["aggregation"]["bootstrap_seed"])
    metrics = (
        "vpt_restricted_lt",
        "forecast_instability",
        "restricted_nrmse_auc_native",
        "restricted_nrmse_auc_0_1LT",
        "restricted_nrmse_auc_0_2LT",
        "restricted_nrmse_auc_0_5LT",
        "mean_coordinate_point_crps_native",
    )
    for metric in metrics:
        summaries.append(_paired_split_bootstrap(
            trajectory_frame, metric, resamples=resamples, seed=seed
        ))
    summary = pd.concat(summaries, ignore_index=True)
    summary_path = output_dir / "bootstrap_summary.csv"
    atomic_write_csv(summary_path, summary.to_dict(orient="records"))

    split_means = trajectory_frame.groupby(
        ["method", "data_seed"], as_index=False
    )[[
        "vpt_restricted_lt",
        "forecast_instability",
        "restricted_nrmse_auc_native",
        "restricted_nrmse_auc_0_2LT",
        "restricted_nrmse_auc_0_5LT",
        "mean_coordinate_point_crps_native",
    ]].mean()
    colors = {method: METHODS[method].color for method in CONTEXT_METHODS}
    labels = {
        "sindy_weak_weighted": "Online weighted weak SINDy",
        "sindy_weak": "Online weak SINDy",
        "panda_zero_shot": "Panda zero-shot",
    }
    axis_labels = {
        "sindy_weak_weighted": "Weighted weak\nSINDy (online)",
        "sindy_weak": "Weak SINDy\n(online)",
        "panda_zero_shot": "Panda\n(zero-shot)",
    }
    figure, axes = plt.subplots(2, 3, figsize=(16, 10.4))
    plot_specs = (
        ("restricted_nrmse_auc_native", "Native 128-step restricted NRMSE AUC", True),
        ("restricted_nrmse_auc_0_2LT", "Restricted NRMSE AUC, 0-2 LT", True),
        ("vpt_restricted_lt", "Restricted mean VPT (LT)", False),
    )
    for axis, (metric, ylabel, lower_better) in zip(axes[0], plot_specs):
        paired_values: dict[str, np.ndarray] = {}
        for method_index, method in enumerate(CONTEXT_METHODS):
            selected = split_means.loc[split_means["method"] == method, metric].to_numpy()
            paired_values[method] = selected
            jitter = np.linspace(-0.1, 0.1, selected.size)
            axis.scatter(
                method_index + jitter,
                selected,
                color=colors[method],
                edgecolor="black",
                linewidth=0.4,
                s=35,
                zorder=3,
            )
            axis.plot(
                [method_index - 0.18, method_index + 0.18],
                [np.median(selected), np.median(selected)],
                color="#202020",
                linewidth=1.6,
            )
        for split_index in range(len(config["final"]["data_seeds"])):
            axis.plot(
                range(len(CONTEXT_METHODS)),
                [paired_values[method][split_index] for method in CONTEXT_METHODS],
                color="#888888",
                linewidth=0.65,
                alpha=0.45,
                zorder=1,
            )
        axis.set_xticks(
            range(len(CONTEXT_METHODS)),
            [axis_labels[method] for method in CONTEXT_METHODS],
        )
        axis.set_ylabel(ylabel)
        axis.set_title("Lower is better" if lower_better else "Higher is better")
        _despine(axis)

    horizon = float(config["context_matched"]["forecast_lyapunov_times"])
    survival_times = np.linspace(0.0, horizon, 121)
    for method in CONTEXT_METHODS:
        selected = trajectory_frame.loc[
            trajectory_frame["method"] == method, "vpt_restricted_lt"
        ].to_numpy(dtype=np.float64)
        survival = np.asarray([np.mean(selected >= value) for value in survival_times])
        axes[1, 0].plot(
            survival_times,
            survival,
            color=colors[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0,
            label=labels[method],
        )
    axes[1, 0].set_xlim(0.0, horizon)
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_xlabel("Forecast time (Lyapunov times)")
    axes[1, 0].set_ylabel("Fraction still valid")
    axes[1, 0].set_title("Valid-forecast survival, E(t) <= 0.4")
    axes[1, 0].legend(fontsize=8, frameon=False)
    _despine(axes[1, 0])

    for method in CONTEXT_METHODS:
        error_arrays = []
        times_lt = None
        for run_dir, manifest in manifests:
            if manifest["method"] != method:
                continue
            with np.load(run_dir / "predictions.npz", allow_pickle=False) as loaded:
                current_times = np.asarray(loaded["times_lt"], dtype=np.float64)
                current_error = np.asarray(
                    loaded["normalized_squared_error"], dtype=np.float64
                )
            if times_lt is None:
                times_lt = current_times
            elif not np.allclose(times_lt, current_times, rtol=0.0, atol=1e-12):
                raise ValueError("context-matched error curves use different time grids")
            error_arrays.append(current_error)
        combined = np.concatenate(error_arrays, axis=0)
        median = np.nanmedian(combined, axis=0)
        q25 = np.nanquantile(combined, 0.25, axis=0)
        q75 = np.nanquantile(combined, 0.75, axis=0)
        keep = times_lt <= horizon + 1e-12
        axes[1, 1].plot(
            times_lt[keep],
            np.clip(median[keep], 1e-10, 100.0),
            color=colors[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0,
            label=labels[method],
        )
        axes[1, 1].fill_between(
            times_lt[keep],
            np.clip(q25[keep], 1e-10, 100.0),
            np.clip(q75[keep], 1e-10, 100.0),
            color=colors[method],
            alpha=0.1,
            linewidth=0,
        )
    axes[1, 1].axhline(0.4, color="#555555", linestyle=":", linewidth=1.1)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlim(0.0, horizon)
    axes[1, 1].set_ylim(1e-8, 100.0)
    axes[1, 1].set_xlabel("Forecast time (Lyapunov times)")
    axes[1, 1].set_ylabel("Normalized squared error E(t)")
    axes[1, 1].set_title("Error growth (median and interquartile band)")
    _despine(axes[1, 1])

    crps_values: dict[str, np.ndarray] = {}
    for method_index, method in enumerate(CONTEXT_METHODS):
        selected = split_means.loc[
            split_means["method"] == method, "mean_coordinate_point_crps_native"
        ].to_numpy()
        crps_values[method] = selected
        axes[1, 2].scatter(
            method_index + np.linspace(-0.1, 0.1, selected.size),
            selected,
            color=colors[method],
            edgecolor="black",
            linewidth=0.4,
            s=35,
            zorder=3,
        )
        axes[1, 2].plot(
            [method_index - 0.18, method_index + 0.18],
            [np.median(selected), np.median(selected)],
            color="#202020",
            linewidth=1.6,
        )
    for split_index in range(len(config["final"]["data_seeds"])):
        axes[1, 2].plot(
            range(len(CONTEXT_METHODS)),
            [crps_values[method][split_index] for method in CONTEXT_METHODS],
            color="#888888",
            linewidth=0.65,
            alpha=0.45,
            zorder=1,
        )
    axes[1, 2].set_xticks(
        range(len(CONTEXT_METHODS)),
        [axis_labels[method] for method in CONTEXT_METHODS],
    )
    axes[1, 2].set_ylabel("Normalized point-mass CRPS\n(native 128 steps)")
    axes[1, 2].set_title("Lower is better; equals normalized MAE")
    _despine(axes[1, 2])

    figure.suptitle(
        "Context-matched Lorenz63 forecasting\n"
        "Same 512-point prefix; SINDy fitted online per trajectory; no separate target-system training data\n"
        "Target observations are matched; priors remain unequal; restricted AUC caps E(t) at "
        f"{float(config['context_matched']['nrmse_error_cap']):g}",
        fontsize=14,
    )
    figure.subplots_adjust(wspace=0.38, hspace=0.58, bottom=0.09, top=0.82)
    figure_paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"context_matched_comparison.{suffix}"
        figure.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        figure_paths.append(path)
    plt.close(figure)

    manifest_path = output_dir / "manifest.json"
    artifacts = {
        "trajectory_metrics": trajectory_path,
        "run_metrics": run_path,
        "bootstrap_summary": summary_path,
        "figure_png": figure_paths[0],
        "figure_svg": figure_paths[1],
    }
    aggregate_manifest = {
        "schema": "lorenz63-context-matched-aggregate-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "run_count": len(manifests),
        "data_seeds": [int(value) for value in config["final"]["data_seeds"]],
        "methods": list(CONTEXT_METHODS),
        "metric_policy": {
            "nrmse_auc_error_cap": float(
                config["context_matched"]["nrmse_error_cap"]
            ),
            "point_crps_absolute_error_cap": float(
                np.sqrt(config["context_matched"]["nrmse_error_cap"])
            ),
            "raw_divergence_retained_in_predictions": True,
            "stability_bound": float(config["evaluation"]["stability_bound"]),
        },
        "context_steps": int(config["context_matched"]["context_steps"]),
        "separate_training_trajectories": 0,
        "future_states_visible_during_adaptation": False,
        "comparison_scope": "target_observation_matched_unequal_priors",
        "panda_probabilistic": False,
        "panda_forecast_sample_count": 1,
        "crps_interpretation": (
            "mean coordinate-wise normalized point-mass CRPS; equals normalized MAE"
        ),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
    }
    atomic_write_json(manifest_path, aggregate_manifest)
    return {**artifacts, "manifest": manifest_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or aggregate the prefix-only Lorenz63 context-matched comparison."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--config", default="configs/v2/context_matched.json")
    extract.add_argument("--test", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--force", action="store_true")
    fit = subparsers.add_parser("fit-sindy")
    fit.add_argument("--config", default="configs/v2/context_matched.json")
    fit.add_argument("--context", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument(
        "--method", required=True, choices=("sindy_weak", "sindy_weak_weighted")
    )
    fit.add_argument("--force", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="configs/v2/context_matched.json")
    evaluate.add_argument("--context", required=True)
    evaluate.add_argument("--forecast-truth", required=True)
    evaluate.add_argument("--models")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--method", required=True, choices=CONTEXT_METHODS)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--force", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", default="configs/v2/context_matched.json")
    aggregate.add_argument("--runs-root", required=True)
    aggregate.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.command == "extract":
        print(*extract_context_data(
            config_path=args.config,
            test_path=args.test,
            output_dir=args.output_dir,
            force=args.force,
        ), sep="\n")
    elif args.command == "fit-sindy":
        print(fit_context_sindy_bank(
            config_path=args.config,
            context_path=args.context,
            output_dir=args.output_dir,
            method=args.method,
            force=args.force,
        ))
    elif args.command == "evaluate":
        print(evaluate_context_matched(
            config_path=args.config,
            context_path=args.context,
            truth_path=args.forecast_truth,
            output_dir=args.output_dir,
            method=args.method,
            model_path=args.models,
            device=args.device,
            force=args.force,
        ))
    else:
        outputs = aggregate_context_matched(
            config_path=args.config,
            runs_root=args.runs_root,
            output_dir=args.output_dir,
        )
        print(*outputs.values(), sep="\n")


if __name__ == "__main__":
    main()
