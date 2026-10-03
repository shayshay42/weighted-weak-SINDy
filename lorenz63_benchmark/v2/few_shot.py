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
from .config import config_for_method
from .context_matched import (
    _despine,
    _load_context_artifact,
    _metric_rows,
    _panda_checkpoint,
    forecast_panda_context_matched,
)
from .data import load_split
from .metrics import normalized_rmse_auc, normalized_squared_error
from .numerics import integrate_rk4
from .panda import PandaForecastAdapter, _PredictionPipeline
from .sindy import _field, fit_sindy_coefficients


FEW_SHOT_SCHEMA = "lorenz63-few-shot-evaluation-v2"
SHOT_DATA_SCHEMA = "lorenz63-labeled-shot-data-v2"
SHOT_MODEL_SCHEMA = "lorenz63-few-shot-model-v2"
PANDA_TUNING_SCHEMA = "lorenz63-panda-head-tuning-v2"
METHODS = ("panda", "sindy_weak", "sindy_weak_weighted")
SINDY_METHODS = ("sindy_weak", "sindy_weak_weighted")


def load_few_shot_config(path: str | Path) -> dict[str, Any]:
    """Load the self-contained protocol without materializing hidden physics defaults."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    forbidden = {"system", "parametric", "pinn"}.intersection(config)
    if forbidden:
        raise ValueError(f"few-shot fit config exposes forbidden physics: {sorted(forbidden)}")
    required = {
        "schema",
        "data",
        "panda",
        "few_shot",
        "context_matched",
        "sindy",
        "evaluation",
        "method_overrides",
        "aggregation",
        "final",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"few-shot config is not self-contained: {sorted(missing)}")
    return config


def _shot_counts(config: dict[str, Any]) -> tuple[int, ...]:
    counts = tuple(int(value) for value in config["few_shot"]["shot_counts"])
    if not counts or counts != tuple(sorted(set(counts))) or counts[0] < 1:
        raise ValueError("few-shot counts must be unique positive values in ascending order")
    return counts


def extract_shot_data(
    *,
    config_path: str | Path,
    train_path: str | Path,
    output_path: str | Path,
    force: bool = False,
) -> Path:
    """Extract one fixed labeled segment from each of distinct train trajectories."""
    config = load_few_shot_config(config_path)
    protocol = config["few_shot"]
    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    identity = {
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "train_sha256": sha256_file(train_path),
    }
    if manifest_path.exists() and output_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and all(existing.get(key) == value for key, value in identity.items())
            and existing.get("artifacts", {}).get("shots", {}).get("sha256")
            == sha256_file(output_path)
        ):
            return output_path

    split = load_split(train_path, "train")
    states = np.asarray(split["states"], dtype=np.float64)
    metadata = split["metadata"]
    if float(metadata["noise_level"]) != 0.0:
        raise ValueError("few-shot adaptation currently requires noiseless training data")
    context_steps = int(protocol["context_steps"])
    target_steps = int(protocol["target_steps"])
    segment_steps = context_steps + target_steps
    maximum_shots = max(_shot_counts(config))
    if states.shape[0] < maximum_shots or states.shape[1] < segment_steps:
        raise ValueError("training split is too small for the configured shot budget")

    data_seed = int(metadata["data_seed"])
    selection_seed = int(protocol["segment_selection_seed"]) + data_seed
    rng = np.random.default_rng(selection_seed)
    starts = rng.integers(
        0, states.shape[1] - segment_steps + 1, size=maximum_shots, endpoint=False
    )
    trajectory_ids = np.arange(maximum_shots, dtype=np.int64)
    segments = np.stack([
        states[trajectory_id, start : start + segment_steps]
        for trajectory_id, start in zip(trajectory_ids, starts)
    ])
    shot_metadata = {
        "schema": SHOT_DATA_SCHEMA,
        "role": "labeled_target_adaptation_shots",
        "data_seed": data_seed,
        "noise_level": 0.0,
        "dt": float(metadata["dt"]),
        "largest_lyapunov_exponent": float(metadata["normalization"]["time_scale"]),
        "context_steps": context_steps,
        "target_steps": target_steps,
        "segment_steps": segment_steps,
        "maximum_shots": maximum_shots,
        "selection_seed": selection_seed,
        "one_shot_definition": (
            f"one_{segment_steps}_point_segment_from_one_distinct_trajectory"
        ),
        "contains_test_states": False,
        "contains_validation_states": False,
        "source_train_sha256": identity["train_sha256"],
    }
    atomic_save_npz(
        output_path,
        states=segments,
        trajectory_ids=trajectory_ids,
        start_indices=starts.astype(np.int64),
        times=np.arange(segment_steps, dtype=np.float64) * float(metadata["dt"]),
        metadata=np.asarray(json.dumps(shot_metadata, sort_keys=True)),
    )
    manifest = {
        "schema": "lorenz63-shot-extraction-manifest-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "command": command_line(),
        "environment": environment_snapshot(),
        **identity,
        "shot_count": maximum_shots,
        "trajectory_ids": trajectory_ids.tolist(),
        "start_indices": starts.tolist(),
        "artifacts": {
            "shots": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)}
        },
    }
    atomic_write_json(manifest_path, manifest)
    return output_path


def _load_shots(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as loaded:
        states = np.asarray(loaded["states"], dtype=np.float64)
        metadata = json.loads(str(loaded["metadata"]))
    if metadata.get("schema") != SHOT_DATA_SCHEMA:
        raise ValueError(f"not a labeled shot artifact: {path}")
    if metadata.get("contains_test_states") is not False:
        raise ValueError("shot artifact does not explicitly exclude test states")
    return states, metadata


def _normalization(states: np.ndarray, time_scale: float) -> Normalization:
    flattened = np.asarray(states, dtype=np.float64).reshape(-1, 3)
    return Normalization(
        state_mean=np.mean(flattened, axis=0),
        state_std=np.maximum(np.std(flattened, axis=0, ddof=0), 1e-12),
        time_scale=float(time_scale),
    )


def _sindy_train_split(
    states: np.ndarray, metadata: dict[str, Any]
) -> tuple[dict[str, Any], Normalization]:
    normalization = _normalization(states, metadata["largest_lyapunov_exponent"])
    split = {
        "states": np.asarray(states, dtype=np.float64),
        "times": np.arange(states.shape[1], dtype=np.float64) * float(metadata["dt"]),
        "metadata": {
            "schema": "lorenz63-few-shot-train-view-v2",
            "role": "train",
            "data_seed": int(metadata["data_seed"]),
            "dt": float(metadata["dt"]),
            "noise_level": 0.0,
            "normalization": {
                "state_mean": normalization.state_mean.tolist(),
                "state_std": normalization.state_std.tolist(),
                "time_scale": normalization.time_scale,
            },
        },
    }
    return split, normalization


def fit_few_shot_sindy(
    *,
    config_path: str | Path,
    shots_path: str | Path,
    output_dir: str | Path,
    method: str,
    shot_count: int,
    force: bool = False,
) -> Path:
    """Fit SINDy from the labeled-shot artifact; no future evaluation input is accepted."""
    if method not in SINDY_METHODS:
        raise ValueError(f"not a few-shot SINDy method: {method}")
    config = load_few_shot_config(config_path)
    if shot_count not in _shot_counts(config):
        raise ValueError(f"unsupported shot count {shot_count}")
    configured = config_for_method(config, 0.0, method)
    fit_config = {
        "sindy": dict(configured["sindy"]),
        "evaluation": {"internal_step": float(configured["evaluation"]["internal_step"])},
    }
    output_dir = Path(output_dir)
    model_path = output_dir / "model.npz"
    manifest_path = output_dir / "manifest.json"
    identity = {
        "source_hash": source_hash(),
        "fit_config_hash": sha256_json(fit_config),
        "input_hashes": {"shots": sha256_file(shots_path)},
        "method": method,
        "shot_count": int(shot_count),
    }
    if manifest_path.exists() and model_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and all(existing.get(key) == value for key, value in identity.items())
            and existing.get("artifacts", {}).get("model", {}).get("sha256")
            == sha256_file(model_path)
        ):
            return model_path

    all_shots, metadata = _load_shots(shots_path)
    selected = all_shots[:shot_count]
    train_split, normalization = _sindy_train_split(selected, metadata)
    started = time.perf_counter()
    coefficients, exponents, details = fit_sindy_coefficients(
        method, train_split, fit_config
    )
    wall_time = time.perf_counter() - started
    atomic_save_npz(
        model_path,
        schema=np.asarray(SHOT_MODEL_SCHEMA),
        method=np.asarray(method),
        shot_count=np.asarray(shot_count, dtype=np.int64),
        coefficients=coefficients,
        exponents=np.asarray(exponents, dtype=np.int64),
        state_mean=normalization.state_mean,
        state_std=normalization.state_std,
        time_scale=np.asarray(normalization.time_scale, dtype=np.float64),
        shots_sha256=np.asarray(identity["input_hashes"]["shots"]),
    )
    manifest = {
        "schema": "lorenz63-few-shot-fit-manifest-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "information_contract": [
            "labeled_target_system_segments",
            "observation_times",
            "degree_2_polynomial_library",
        ],
        "forbidden_inputs": [
            "test_context",
            "forecast_truth",
            "validation_trajectories",
            "exact_derivatives",
            "lorenz_parameters",
            "generator_metadata",
        ],
        "future_test_states_visible_during_fit": False,
        "observed_state_count": int(selected.shape[0] * selected.shape[1]),
        "trajectory_segment_count": int(shot_count),
        "training": {
            "wall_time_seconds": wall_time,
            "optimizer_updates": 1,
            "fit_rows": int(details["fit_rows"]),
            "parameter_count": int(coefficients.size),
        },
        **identity,
        "artifacts": {
            "model": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)}
        },
    }
    atomic_write_json(manifest_path, manifest)
    return model_path


def _normalized_pairs(
    states: np.ndarray, context_steps: int, target_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    past = states[:, :context_steps]
    future = states[:, context_steps : context_steps + target_steps]
    means = np.mean(past, axis=1, keepdims=True)
    standard_deviations = np.maximum(np.std(past, axis=1, keepdims=True), 1e-12)
    return (
        ((past - means) / standard_deviations).astype(np.float32),
        ((future - means) / standard_deviations).astype(np.float32),
    )


def _train_panda_head_model(
    model: Any,
    past: np.ndarray,
    future: np.ndarray,
    *,
    learning_rate: float,
    updates: int,
    gradient_clip: float,
    weight_decay: float,
) -> dict[str, Any]:
    import torch

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = int(sum(parameter.numel() for parameter in trainable))
    optimizer = torch.optim.AdamW(
        trainable, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    past_tensor = torch.as_tensor(past, dtype=torch.float32, device=model.device)
    future_tensor = torch.as_tensor(future, dtype=torch.float32, device=model.device)
    losses = []
    model.train()
    started = time.perf_counter()
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(model.device)
    for _ in range(int(updates)):
        optimizer.zero_grad(set_to_none=True)
        loss = model(past_values=past_tensor, future_values=future_tensor).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Panda adaptation produced a non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(gradient_clip))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        torch.cuda.synchronize(model.device)
        peak_memory = int(torch.cuda.max_memory_allocated(model.device))
    else:
        peak_memory = 0
    model.eval()
    return {
        "wall_time_seconds": time.perf_counter() - started,
        "optimizer_updates": int(updates),
        "trainable_parameter_count": trainable_count,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "peak_gpu_memory_bytes": peak_memory,
    }


def _new_panda_adapter(config: dict[str, Any], dt: float, device: str) -> PandaForecastAdapter:
    return PandaForecastAdapter(_panda_checkpoint(config, dt=dt), device=device)


def _apply_head_weight(adapter: PandaForecastAdapter, model_path: str | Path) -> None:
    import torch

    pipeline = adapter._ensure_pipeline()
    with np.load(model_path, allow_pickle=False) as loaded:
        if str(loaded["schema"]) != SHOT_MODEL_SCHEMA or str(loaded["method"]) != "panda":
            raise ValueError(f"not a few-shot Panda head checkpoint: {model_path}")
        weight = np.asarray(loaded["head_weight"], dtype=np.float32)
    expected = tuple(pipeline.model.head.projection.weight.shape)
    if weight.shape != expected:
        raise ValueError(f"Panda head shape {weight.shape} does not match {expected}")
    with torch.no_grad():
        pipeline.model.head.projection.weight.copy_(
            torch.as_tensor(weight, device=pipeline.model.device)
        )
    pipeline.model.eval()


def _selected_tuning(
    tuning_path: str | Path, shot_count: int, config: dict[str, Any]
) -> dict[str, Any]:
    tuning = json.loads(Path(tuning_path).read_text(encoding="utf-8"))
    if tuning.get("schema") != PANDA_TUNING_SCHEMA or tuning.get("status") != "complete":
        raise ValueError(f"not a complete Panda tuning artifact: {tuning_path}")
    if tuning.get("source_hash") != source_hash():
        raise ValueError("Panda tuning artifact was produced by different benchmark source")
    if tuning.get("config_hash") != sha256_json(config):
        raise ValueError("Panda tuning artifact was produced from a different configuration")
    try:
        return dict(tuning["selected"][str(int(shot_count))])
    except KeyError as error:
        raise ValueError(f"tuning artifact has no selection for {shot_count} shots") from error


def fit_few_shot_panda(
    *,
    config_path: str | Path,
    shots_path: str | Path,
    tuning_path: str | Path,
    output_dir: str | Path,
    shot_count: int,
    device: str = "cuda",
    force: bool = False,
) -> Path:
    """Adapt only Panda's prediction head from labeled shots."""
    config = load_few_shot_config(config_path)
    if shot_count not in _shot_counts(config):
        raise ValueError(f"unsupported shot count {shot_count}")
    selected_tuning = _selected_tuning(tuning_path, shot_count, config)
    output_dir = Path(output_dir)
    model_path = output_dir / "model.npz"
    manifest_path = output_dir / "manifest.json"
    identity = {
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "input_hashes": {
            "shots": sha256_file(shots_path),
            "frozen_tuning": sha256_file(tuning_path),
        },
        "method": "panda",
        "shot_count": int(shot_count),
    }
    if manifest_path.exists() and model_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and all(existing.get(key) == value for key, value in identity.items())
            and existing.get("artifacts", {}).get("model", {}).get("sha256")
            == sha256_file(model_path)
        ):
            return model_path

    all_shots, metadata = _load_shots(shots_path)
    selected = all_shots[:shot_count]
    context_steps = int(config["few_shot"]["context_steps"])
    target_steps = int(config["few_shot"]["target_steps"])
    past, future = _normalized_pairs(selected, context_steps, target_steps)
    adapter = _new_panda_adapter(config, float(metadata["dt"]), device)
    model = adapter._ensure_pipeline().model
    training = _train_panda_head_model(
        model,
        past,
        future,
        learning_rate=float(selected_tuning["learning_rate"]),
        updates=int(selected_tuning["updates"]),
        gradient_clip=float(config["few_shot"]["gradient_clip"]),
        weight_decay=float(config["few_shot"]["weight_decay"]),
    )
    expected_trainable = int(config["few_shot"]["panda_trainable_parameters"])
    if training["trainable_parameter_count"] != expected_trainable:
        raise RuntimeError(
            f"Panda trainable parameter count changed: "
            f"{training['trainable_parameter_count']} != {expected_trainable}"
        )
    atomic_save_npz(
        model_path,
        schema=np.asarray(SHOT_MODEL_SCHEMA),
        method=np.asarray("panda"),
        shot_count=np.asarray(shot_count, dtype=np.int64),
        head_weight=model.head.projection.weight.detach().cpu().numpy().astype(np.float32),
        shots_sha256=np.asarray(identity["input_hashes"]["shots"]),
        tuning_sha256=np.asarray(identity["input_hashes"]["frozen_tuning"]),
    )
    manifest = {
        "schema": "lorenz63-few-shot-fit-manifest-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "information_contract": [
            "external_pretraining_corpus",
            "labeled_target_system_segments",
            "observed_state_context",
        ],
        "forbidden_inputs": [
            "test_context",
            "forecast_truth",
            "validation_trajectories",
            "lorenz_parameters",
            "generator_metadata",
        ],
        "future_test_states_visible_during_fit": False,
        "adaptation": "prediction_head_only",
        "observed_state_count": int(selected.shape[0] * selected.shape[1]),
        "trajectory_segment_count": int(shot_count),
        "selected_tuning": selected_tuning,
        "pretrained_model_id": config["panda"]["model_id"],
        "pretrained_model_revision": config["panda"]["model_revision"],
        "training": training,
        **identity,
        "artifacts": {
            "model": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)}
        },
    }
    atomic_write_json(manifest_path, manifest)
    return model_path


def tune_panda_head(
    *,
    config_path: str | Path,
    shots_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    force: bool = False,
    adapter_factory: Callable[[dict[str, Any], float, str], PandaForecastAdapter] | None = None,
) -> Path:
    """Select Panda head adaptation settings on development seed zero only."""
    import gc

    config = load_few_shot_config(config_path)
    output_dir = Path(output_dir)
    tuning_path = output_dir / "panda_head_tuning.json"
    table_path = output_dir / "panda_head_tuning.csv"
    identity = {
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "input_hashes": {
            "development_shots": sha256_file(shots_path),
            "development_validation": sha256_file(validation_path),
        },
    }
    if tuning_path.exists() and table_path.exists() and not force:
        existing = json.loads(tuning_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and all(existing.get(key) == value for key, value in identity.items())
            and existing.get("artifacts", {}).get("candidate_table", {}).get("sha256")
            == sha256_file(table_path)
        ):
            return tuning_path

    all_shots, shot_metadata = _load_shots(shots_path)
    if int(shot_metadata["data_seed"]) != 0:
        raise ValueError("Panda hyperparameters must be selected only on development seed 0")
    validation = load_split(validation_path, "validation")
    validation_states = np.asarray(validation["states"], dtype=np.float64)
    validation_times = np.asarray(validation["times"], dtype=np.float64)
    if int(validation["metadata"]["data_seed"]) != 0:
        raise ValueError("Panda tuning validation must come from development seed 0")
    context_steps = int(config["few_shot"]["context_steps"])
    target_steps = int(config["few_shot"]["target_steps"])
    if validation_states.shape[1] <= context_steps:
        raise ValueError("development validation trajectories have no post-context targets")
    origin = context_steps - 1
    validation_context = validation_states[:, :context_steps]
    validation_truth = validation_states[:, origin:]
    forecast_times = validation_times[origin:] - validation_times[origin]
    largest = float(validation["metadata"]["normalization"]["time_scale"])
    validation_scale = np.maximum(
        np.std(validation_context.reshape(-1, 3), axis=0), 1e-12
    )
    candidates = [(0.0, 0)]
    candidates.extend(
        (float(learning_rate), int(updates))
        for updates in config["few_shot"]["tuning_updates"]
        if int(updates) > 0
        for learning_rate in config["few_shot"]["tuning_learning_rates"]
    )
    rows = []
    selected: dict[str, dict[str, Any]] = {}
    factory = adapter_factory or _new_panda_adapter
    for shot_count in _shot_counts(config):
        selected_shots = all_shots[:shot_count]
        past, future = _normalized_pairs(selected_shots, context_steps, target_steps)
        for learning_rate, updates in candidates:
            adapter = factory(config, float(shot_metadata["dt"]), device)
            model = adapter._ensure_pipeline().model
            training = _train_panda_head_model(
                model,
                past,
                future,
                learning_rate=learning_rate,
                updates=updates,
                gradient_clip=float(config["few_shot"]["gradient_clip"]),
                weight_decay=float(config["few_shot"]["weight_decay"]),
            )
            prediction, timing = forecast_panda_context_matched(
                adapter, validation_context, forecast_times, time_scale=largest
            )
            error = normalized_squared_error(
                prediction, validation_truth, validation_scale
            )
            validation_horizon_lt = float(forecast_times[-1] * largest)
            auc = normalized_rmse_auc(
                error, forecast_times * largest, validation_horizon_lt
            )
            score = float(np.median(auc))
            rows.append({
                "shot_count": shot_count,
                "learning_rate": learning_rate,
                "updates": updates,
                "median_validation_native_nrmse_auc": score,
                "mean_validation_native_nrmse_auc": float(np.mean(auc)),
                "validation_horizon_lt": validation_horizon_lt,
                "training_wall_time_seconds": training["wall_time_seconds"],
                "forecast_wall_time_seconds": timing["forecast_seconds"],
                "initial_training_loss": training["initial_loss"],
                "final_training_loss": training["final_loss"],
            })
            del adapter, model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        baseline = next(
            row for row in rows
            if row["shot_count"] == shot_count and row["updates"] == 0
        )
        eligible = [
            row for row in rows
            if row["shot_count"] == shot_count and row["updates"] > 0
        ]
        best = min(
            eligible,
            key=lambda row: (
                row["median_validation_native_nrmse_auc"],
                row["updates"],
                row["learning_rate"],
            ),
        )
        selected[str(shot_count)] = {
            "learning_rate": float(best["learning_rate"]),
            "updates": int(best["updates"]),
            "development_score": float(best["median_validation_native_nrmse_auc"]),
            "development_validation_horizon_lt": float(best["validation_horizon_lt"]),
            "zero_shot_reference_score": float(
                baseline["median_validation_native_nrmse_auc"]
            ),
            "improves_over_zero_shot": bool(
                best["median_validation_native_nrmse_auc"]
                < baseline["median_validation_native_nrmse_auc"]
            ),
        }

    atomic_write_csv(table_path, rows)
    tuning = {
        "schema": PANDA_TUNING_SCHEMA,
        "status": "complete",
        "completed_at": utc_now(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "development_seed": 0,
        "objective": config["few_shot"]["tuning_objective"],
        "adaptation": "prediction_head_only",
        "validation_future_points": int(validation_truth.shape[1] - 1),
        "selected": selected,
        **identity,
        "artifacts": {
            "candidate_table": {
                "path": str(table_path.resolve()),
                "sha256": sha256_file(table_path),
            }
        },
    }
    atomic_write_json(tuning_path, tuning)
    return tuning_path


def _load_few_shot_model(
    model_path: str | Path, method: str, shot_count: int
) -> tuple[Any, ...]:
    with np.load(model_path, allow_pickle=False) as loaded:
        if str(loaded["schema"]) != SHOT_MODEL_SCHEMA:
            raise ValueError(f"not a few-shot model: {model_path}")
        if str(loaded["method"]) != method or int(loaded["shot_count"]) != shot_count:
            raise ValueError("few-shot model identity does not match the requested evaluation")
        if method == "panda":
            return ()
        coefficients = np.asarray(loaded["coefficients"], dtype=np.float64)
        exponents = tuple(tuple(int(value) for value in row) for row in loaded["exponents"])
        normalization = Normalization(
            state_mean=np.asarray(loaded["state_mean"], dtype=np.float64),
            state_std=np.asarray(loaded["state_std"], dtype=np.float64),
            time_scale=float(loaded["time_scale"]),
        )
    return coefficients, exponents, normalization


def evaluate_few_shot(
    *,
    config_path: str | Path,
    context_path: str | Path,
    truth_path: str | Path,
    output_dir: str | Path,
    method: str,
    shot_count: int,
    model_path: str | Path | None = None,
    device: str = "cpu",
    force: bool = False,
    panda_adapter_factory: Callable[[dict[str, Any], str], PandaForecastAdapter] | None = None,
) -> Path:
    config = load_few_shot_config(config_path)
    if method not in METHODS:
        raise ValueError(f"unsupported few-shot method {method!r}")
    if method == "panda" and shot_count == 0:
        if model_path is not None:
            raise ValueError("zero-shot Panda evaluation does not accept an adapted model")
    elif shot_count not in _shot_counts(config) or model_path is None:
        raise ValueError("adapted evaluation requires a configured shot count and model")
    if method in SINDY_METHODS and shot_count == 0:
        raise ValueError("SINDy has no zero-shot condition")

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    input_hashes = {
        "context": sha256_file(context_path),
        "forecast_truth": sha256_file(truth_path),
    }
    if model_path is not None:
        input_hashes["model"] = sha256_file(model_path)
    identity = {
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "input_hashes": input_hashes,
        "method": method,
        "shot_count": int(shot_count),
    }
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and all(
            existing.get(key) == value for key, value in identity.items()
        ):
            artifacts = existing.get("artifacts", {}).values()
            if artifacts and all(
                (output_dir / Path(record["path"]).name).exists()
                and sha256_file(output_dir / Path(record["path"]).name) == record["sha256"]
                for record in artifacts
            ):
                return manifest_path

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": FEW_SHOT_SCHEMA,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "command": command_line(),
        "environment": environment_snapshot(),
        "information_contract": [
            *(["external_pretraining_corpus"] if method == "panda" else []),
            *(["degree_2_polynomial_library"] if method in SINDY_METHODS else []),
            "labeled_target_system_segments" if shot_count else "no_target_system_fit_data",
            "observed_test_context",
        ],
        "future_test_states_visible_during_fit": False,
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
        data_seed = int(context_metadata["data_seed"])
        largest = float(context_metadata["largest_lyapunov_exponent"])
        dt = float(context_metadata["dt"])
        fit_manifest: dict[str, Any] = {}
        if model_path is not None:
            _load_few_shot_model(model_path, method, shot_count)
            fit_manifest_path = Path(model_path).parent / "manifest.json"
            fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
            if fit_manifest.get("status") != "complete":
                raise ValueError("few-shot fit manifest is not complete")
            if fit_manifest.get("source_hash") != source_hash():
                raise ValueError("few-shot model was fitted by different benchmark source")
            if (
                fit_manifest.get("method") != method
                or int(fit_manifest.get("shot_count", -1)) != shot_count
            ):
                raise ValueError("few-shot fit manifest identity does not match evaluation")
            if fit_manifest.get("future_test_states_visible_during_fit") is not False:
                raise ValueError("fit manifest does not certify test-future isolation")
            model_record = fit_manifest.get("artifacts", {}).get("model", {})
            if model_record.get("sha256") != input_hashes["model"]:
                raise ValueError("few-shot model does not match its fit manifest")
            expected_fit_inputs = (
                {"shots", "frozen_tuning"} if method == "panda" else {"shots"}
            )
            if set(fit_manifest.get("input_hashes", {})) != expected_fit_inputs:
                raise ValueError("few-shot fit manifest exposes unexpected inputs")
        started = time.perf_counter()
        vector_field_evaluations = 0
        surrogate_evaluations = 0
        peak_memory = 0
        parameter_count = 0
        trainable_parameter_count = 0
        optimizer_updates = 0
        adaptation_wall_time = 0.0
        if fit_manifest:
            training = fit_manifest.get("training", {})
            adaptation_wall_time = float(training.get("wall_time_seconds", 0.0))
            optimizer_updates = int(training.get("optimizer_updates", 0))
            trainable_parameter_count = int(
                training.get("trainable_parameter_count", training.get("parameter_count", 0))
            )

        if method == "panda":
            checkpoint = _panda_checkpoint(config, dt=dt)
            factory = panda_adapter_factory or (
                lambda value, target_device: PandaForecastAdapter(value, device=target_device)
            )
            adapter = factory(checkpoint, device)
            if model_path is not None:
                _apply_head_weight(adapter, model_path)
            prediction, timing = forecast_panda_context_matched(
                adapter, contexts, forecast_times, time_scale=largest
            )
            forecast_seconds = float(timing["forecast_seconds"])
            surrogate_evaluations = int(timing["surrogate_evaluations"])
            peak_memory = int(timing["inference_peak_gpu_memory_bytes"])
            parameter_count = int(timing["parameter_count"])
            probabilistic = bool(timing["probabilistic"])
            forecast_sample_count = int(timing["forecast_sample_count"])
        else:
            coefficients, exponents, normalization = _load_few_shot_model(
                model_path, method, shot_count
            )
            prediction, vector_field_evaluations = integrate_rk4(
                lambda state: _field(coefficients, exponents, normalization, state),
                contexts[:, -1],
                forecast_times,
                internal_step=float(config["evaluation"]["internal_step"]),
            )
            forecast_seconds = time.perf_counter() - started
            parameter_count = int(coefficients.size)
            probabilistic = False
            forecast_sample_count = 1

        if prediction.shape != truth.shape:
            raise RuntimeError(
                f"prediction shape {prediction.shape} does not match truth {truth.shape}"
            )
        metric_scale = np.maximum(np.std(contexts.reshape(-1, 3), axis=0), 1e-12)
        rows, error, times_lt = _metric_rows(
            method=method,
            data_seed=data_seed,
            prediction=prediction,
            truth=truth,
            forecast_times=forecast_times,
            largest=largest,
            scale=metric_scale,
            context_steps=int(config["few_shot"]["context_steps"]),
            native_prediction_steps=int(config["few_shot"]["target_steps"]),
            forecast_horizon_lt=float(config["few_shot"]["forecast_lyapunov_times"]),
            vpt_threshold=float(config["evaluation"]["vpt_threshold"]),
            nrmse_error_cap=float(config["few_shot"]["nrmse_error_cap"]),
            stability_bound=float(config["evaluation"]["stability_bound"]),
        )
        for row in rows:
            row["shot_count"] = int(shot_count)
            row["adaptation"] = (
                "zero_shot"
                if method == "panda" and shot_count == 0
                else "prediction_head_only"
                if method == "panda"
                else "fit_from_scratch"
            )
        trajectory_path = output_dir / "trajectory_metrics.csv"
        run_path = output_dir / "run_metrics.csv"
        predictions_path = output_dir / "predictions.npz"
        atomic_write_csv(trajectory_path, rows)
        atomic_save_npz(
            predictions_path,
            prediction=prediction.astype(np.float32),
            normalized_squared_error=error.astype(np.float32),
            times=forecast_times,
            times_lt=times_lt,
            shot_count=np.asarray(shot_count, dtype=np.int64),
        )
        atomic_write_csv(run_path, [{
            "method": method,
            "shot_count": int(shot_count),
            "adaptation": rows[0]["adaptation"],
            "data_seed": data_seed,
            "trajectory_count": int(contexts.shape[0]),
            "observed_training_state_count": int(
                shot_count
                * (int(config["few_shot"]["context_steps"]) + int(config["few_shot"]["target_steps"]))
            ),
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
            "adaptation_wall_time_seconds": adaptation_wall_time,
            "forecast_wall_time_seconds": forecast_seconds,
            "optimizer_updates": optimizer_updates,
            "vector_field_evaluations": vector_field_evaluations,
            "surrogate_evaluations": surrogate_evaluations,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "inference_peak_gpu_memory_bytes": peak_memory,
            "probabilistic": probabilistic,
            "forecast_sample_count": forecast_sample_count,
        }])
        artifacts = {
            "trajectory_metrics": trajectory_path,
            "run_metrics": run_path,
            "predictions": predictions_path,
        }
        manifest.update({
            "status": "complete",
            "completed_at": utc_now(),
            "data_seed": data_seed,
            "adaptation": rows[0]["adaptation"],
            "observed_training_state_count": int(
                shot_count
                * (int(config["few_shot"]["context_steps"]) + int(config["few_shot"]["target_steps"]))
            ),
            "normalization_source": "observed_test_context_prefix_for_panda_and_metrics",
            "probabilistic": probabilistic,
            "forecast_sample_count": forecast_sample_count,
            "metric_policy": {
                "nrmse_auc_error_cap": float(config["few_shot"]["nrmse_error_cap"]),
                "point_crps_absolute_error_cap": float(
                    np.sqrt(config["few_shot"]["nrmse_error_cap"])
                ),
                "raw_divergence_retained_in_predictions": True,
            },
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


def _expected_conditions(config: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    counts = _shot_counts(config)
    return (
        *(("panda", count) for count in (0, *counts)),
        *(("sindy_weak", count) for count in counts),
        *(("sindy_weak_weighted", count) for count in counts),
    )


def _paired_condition_bootstrap(
    frame: Any,
    metric: str,
    conditions: tuple[tuple[str, int], ...],
    *,
    resamples: int,
    seed: int,
) -> Any:
    import pandas as pd

    split_ids = sorted(int(value) for value in frame["data_seed"].unique())
    trajectory_ids = sorted(int(value) for value in frame["trajectory_id"].unique())
    arrays = []
    for method, shot_count in conditions:
        selected = frame.loc[
            (frame["method"] == method) & (frame["shot_count"] == shot_count),
            ["data_seed", "trajectory_id", metric],
        ]
        index = pd.MultiIndex.from_product(
            [split_ids, trajectory_ids], names=["data_seed", "trajectory_id"]
        )
        values = selected.set_index(["data_seed", "trajectory_id"])[metric].reindex(index)
        if values.isna().any():
            raise ValueError(f"incomplete few-shot matrix for {method}/{shot_count}/{metric}")
        arrays.append(values.to_numpy(dtype=np.float64).reshape(len(split_ids), -1))
    array = np.stack(arrays)
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(conditions)), dtype=np.float64)
    for draw in range(resamples):
        split_draw = rng.integers(len(split_ids), size=len(split_ids))
        trajectory_draw = rng.integers(
            len(trajectory_ids), size=(len(split_ids), len(trajectory_ids))
        )
        selected = array[:, split_draw]
        sampled = np.take_along_axis(selected, trajectory_draw[None], axis=2)
        draws[draw] = sampled.mean(axis=(1, 2))
    point = array.mean(axis=(1, 2))
    return pd.DataFrame([{
        "method": method,
        "shot_count": shot_count,
        "metric": metric,
        "estimate": float(point[index]),
        "ci_lower": float(np.quantile(draws[:, index], 0.025)),
        "ci_upper": float(np.quantile(draws[:, index], 0.975)),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    } for index, (method, shot_count) in enumerate(conditions)])


def _plot_learning_curve(
    axis: Any,
    split_means: Any,
    summary: Any,
    metric: str,
    ylabel: str,
    *,
    lower_better: bool,
    log_scale: bool = False,
) -> None:
    positions = {0: 0, 1: 1, 4: 2, 16: 3}
    colors = {"panda": "#117733", "sindy_weak": "#CC79A7", "sindy_weak_weighted": "#332288"}
    labels = {
        "panda": "Panda (head-adapted)",
        "sindy_weak": "Weak SINDy",
        "sindy_weak_weighted": "Weighted weak SINDy",
    }
    markers = {"panda": "o", "sindy_weak": "s", "sindy_weak_weighted": "^"}
    for method in METHODS:
        method_rows = split_means.loc[split_means["method"] == method]
        counts = sorted(int(value) for value in method_rows["shot_count"].unique())
        for data_seed in sorted(method_rows["data_seed"].unique()):
            selected = method_rows.loc[method_rows["data_seed"] == data_seed].sort_values(
                "shot_count"
            )
            axis.plot(
                [positions[int(value)] for value in selected["shot_count"]],
                selected[metric],
                color=colors[method],
                alpha=0.18,
                linewidth=0.8,
            )
        estimates = []
        lower = []
        upper = []
        for count in counts:
            row = summary.loc[
                (summary["method"] == method)
                & (summary["shot_count"] == count)
                & (summary["metric"] == metric)
            ].iloc[0]
            estimates.append(float(row["estimate"]))
            lower.append(float(row["estimate"] - row["ci_lower"]))
            upper.append(float(row["ci_upper"] - row["estimate"]))
        axis.errorbar(
            [positions[count] for count in counts],
            estimates,
            yerr=np.asarray([lower, upper]),
            color=colors[method],
            marker=markers[method],
            markersize=5,
            linewidth=2.0,
            capsize=3,
            label=labels[method],
        )
    axis.set_xticks(range(4), ["0", "1", "4", "16"])
    axis.set_xlabel("Labeled training segments (shots)")
    axis.set_ylabel(ylabel)
    axis.set_title("Lower is better" if lower_better else "Higher is better")
    if log_scale:
        axis.set_yscale("log")
    _despine(axis)


def aggregate_few_shot(
    *,
    config_path: str | Path,
    runs_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    config = load_few_shot_config(config_path)
    runs_root = Path(runs_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = _expected_conditions(config)
    expected = {
        (method, shot_count, int(data_seed))
        for method, shot_count in conditions
        for data_seed in config["final"]["data_seeds"]
    }
    manifests = []
    for path in sorted(runs_root.rglob("manifest.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") == FEW_SHOT_SCHEMA and value.get("status") == "complete":
            manifests.append((path.parent, value))
    actual = {
        (value["method"], int(value["shot_count"]), int(value["data_seed"]))
        for _, value in manifests
    }
    if actual != expected or len(manifests) != len(expected):
        raise ValueError(
            f"incomplete few-shot matrix: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}, runs={len(manifests)}"
        )
    expected_source = source_hash()
    expected_config = sha256_json(config)
    for run_dir, manifest in manifests:
        if manifest.get("source_hash") != expected_source:
            raise ValueError(f"source hash mismatch in {run_dir}")
        if manifest.get("config_hash") != expected_config:
            raise ValueError(f"config hash mismatch in {run_dir}")
        if manifest.get("future_test_states_visible_during_fit") is not False:
            raise ValueError(f"test isolation flag is not false in {run_dir}")
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
    zero = run_frame[(run_frame["method"] == "panda") & (run_frame["shot_count"] == 0)]
    if not (zero["observed_training_state_count"] == 0).all():
        raise ValueError("zero-shot Panda unexpectedly used target-system fitting observations")
    expected_states = run_frame["shot_count"] * (
        int(config["few_shot"]["context_steps"]) + int(config["few_shot"]["target_steps"])
    )
    if not (run_frame["observed_training_state_count"] == expected_states).all():
        raise ValueError("one or more runs violated the exact shot observation budget")
    if run_frame.loc[run_frame["method"] == "panda", "probabilistic"].astype(
        str
    ).str.lower().isin({"true", "1"}).any():
        raise ValueError("released Panda checkpoint unexpectedly became probabilistic")

    trajectory_path = output_dir / "trajectory_metrics.csv"
    run_path = output_dir / "run_metrics.csv"
    summary_path = output_dir / "bootstrap_summary.csv"
    atomic_write_csv(trajectory_path, trajectory_frame.to_dict(orient="records"))
    atomic_write_csv(run_path, run_frame.to_dict(orient="records"))
    metrics = (
        "vpt_restricted_lt",
        "forecast_instability",
        "restricted_nrmse_auc_native",
        "restricted_nrmse_auc_0_1LT",
        "restricted_nrmse_auc_0_2LT",
        "restricted_nrmse_auc_0_5LT",
        "mean_coordinate_point_crps_native",
    )
    summaries = [
        _paired_condition_bootstrap(
            trajectory_frame,
            metric,
            conditions,
            resamples=int(config["aggregation"]["bootstrap_resamples"]),
            seed=int(config["aggregation"]["bootstrap_seed"]),
        )
        for metric in metrics
    ]
    summary = pd.concat(summaries, ignore_index=True)
    atomic_write_csv(summary_path, summary.to_dict(orient="records"))
    split_means = trajectory_frame.groupby(
        ["method", "shot_count", "data_seed"], as_index=False
    )[[
        "vpt_restricted_lt",
        "restricted_nrmse_auc_native",
        "restricted_nrmse_auc_0_2LT",
        "restricted_nrmse_auc_0_5LT",
        "mean_coordinate_point_crps_native",
    ]].mean()

    figure, axes = plt.subplots(2, 3, figsize=(16, 10.2))
    _plot_learning_curve(
        axes[0, 0], split_means, summary, "restricted_nrmse_auc_native",
        f"Native {int(config['few_shot']['target_steps'])}-step restricted NRMSE AUC",
        lower_better=True, log_scale=True,
    )
    _plot_learning_curve(
        axes[0, 1], split_means, summary, "restricted_nrmse_auc_0_2LT",
        "Restricted NRMSE AUC, 0-2 LT", lower_better=True, log_scale=True,
    )
    _plot_learning_curve(
        axes[0, 2], split_means, summary, "vpt_restricted_lt",
        "Restricted mean VPT (LT)", lower_better=False,
    )

    colors = {"panda": "#117733", "sindy_weak": "#CC79A7", "sindy_weak_weighted": "#332288"}
    curve_conditions = (
        ("panda", 0, "Panda zero-shot", (0, (2.0, 1.5))),
        ("panda", 16, "Panda, 16-shot head adaptation", "-"),
        ("sindy_weak", 16, "Weak SINDy, 16 shots", (0, (5.0, 2.0))),
        ("sindy_weak_weighted", 16, "Weighted weak SINDy, 16 shots", "-"),
    )
    horizon = float(config["few_shot"]["forecast_lyapunov_times"])
    survival_times = np.linspace(0.0, horizon, 121)
    for method, shot_count, label, linestyle in curve_conditions:
        selected = trajectory_frame.loc[
            (trajectory_frame["method"] == method)
            & (trajectory_frame["shot_count"] == shot_count),
            "vpt_restricted_lt",
        ].to_numpy(dtype=np.float64)
        survival = np.asarray([np.mean(selected >= value) for value in survival_times])
        axes[1, 0].plot(
            survival_times,
            survival,
            color=colors[method],
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    axes[1, 0].set_xlim(0.0, horizon)
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_xlabel("Forecast time (Lyapunov times)")
    axes[1, 0].set_ylabel("Fraction still valid")
    axes[1, 0].set_title("Valid-forecast survival, E(t) <= 0.4")
    axes[1, 0].legend(fontsize=8, frameon=False)
    _despine(axes[1, 0])

    for method, shot_count, label, linestyle in curve_conditions:
        arrays = []
        times_lt = None
        for run_dir, manifest in manifests:
            if manifest["method"] != method or int(manifest["shot_count"]) != shot_count:
                continue
            with np.load(run_dir / "predictions.npz", allow_pickle=False) as loaded:
                current_times = np.asarray(loaded["times_lt"], dtype=np.float64)
                arrays.append(np.asarray(loaded["normalized_squared_error"], dtype=np.float64))
            if times_lt is None:
                times_lt = current_times
            elif not np.allclose(times_lt, current_times, rtol=0.0, atol=1e-12):
                raise ValueError("few-shot error curves use different time grids")
        combined = np.concatenate(arrays, axis=0)
        median = np.nanmedian(combined, axis=0)
        q25 = np.nanquantile(combined, 0.25, axis=0)
        q75 = np.nanquantile(combined, 0.75, axis=0)
        keep = times_lt <= horizon + 1e-12
        axes[1, 1].plot(
            times_lt[keep],
            np.clip(median[keep], 1e-10, 100.0),
            color=colors[method],
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        axes[1, 1].fill_between(
            times_lt[keep],
            np.clip(q25[keep], 1e-10, 100.0),
            np.clip(q75[keep], 1e-10, 100.0),
            color=colors[method],
            alpha=0.08,
            linewidth=0,
        )
    axes[1, 1].axhline(0.4, color="#555555", linestyle=":", linewidth=1.1)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlim(0.0, horizon)
    axes[1, 1].set_ylim(1e-8, 100.0)
    axes[1, 1].set_xlabel("Forecast time (Lyapunov times)")
    axes[1, 1].set_ylabel("Normalized squared error E(t)")
    axes[1, 1].set_title("Error growth at 16 shots (median and IQR)")
    _despine(axes[1, 1])

    _plot_learning_curve(
        axes[1, 2], split_means, summary, "mean_coordinate_point_crps_native",
        f"Restricted normalized point-mass CRPS\n"
        f"(native {int(config['few_shot']['target_steps'])} steps)",
        lower_better=True,
        log_scale=True,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.855),
        ncol=3, frameon=False,
    )
    figure.suptitle(
        "Lorenz63 target-adaptation learning curve\n"
        f"One shot = one labeled {int(config['few_shot']['context_steps'])}-context + "
        f"{int(config['few_shot']['target_steps'])}-target segment from a distinct trajectory\n"
        "Panda adapts only its prediction head; pretrained and structural priors remain unequal",
        fontsize=14,
    )
    figure.subplots_adjust(wspace=0.35, hspace=0.42, bottom=0.09, top=0.76)
    figure_paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"few_shot_learning_curve.{suffix}"
        figure.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        figure_paths.append(path)
    plt.close(figure)

    artifacts = {
        "trajectory_metrics": trajectory_path,
        "run_metrics": run_path,
        "bootstrap_summary": summary_path,
        "figure_png": figure_paths[0],
        "figure_svg": figure_paths[1],
    }
    manifest_path = output_dir / "manifest.json"
    aggregate_manifest = {
        "schema": "lorenz63-few-shot-aggregate-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "run_count": len(manifests),
        "conditions": [
            {"method": method, "shot_count": shot_count}
            for method, shot_count in conditions
        ],
        "data_seeds": [int(value) for value in config["final"]["data_seeds"]],
        "one_shot_definition": (
            f"one_{int(config['few_shot']['context_steps']) + int(config['few_shot']['target_steps'])}"
            "_point_segment_from_one_distinct_trajectory"
        ),
        "future_test_states_visible_during_fit": False,
        "comparison_scope": "matched_target_observations_unequal_priors",
        "metric_policy": {
            "nrmse_auc_error_cap": float(config["few_shot"]["nrmse_error_cap"]),
            "point_crps_absolute_error_cap": float(
                np.sqrt(config["few_shot"]["nrmse_error_cap"])
            ),
            "raw_divergence_retained_in_predictions": True,
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
    }
    atomic_write_json(manifest_path, aggregate_manifest)
    return {**artifacts, "manifest": manifest_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exact-window Lorenz63 Panda/SINDy few-shot benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract-shots")
    extract.add_argument("--config", default="configs/v2/few_shot.json")
    extract.add_argument("--train", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--force", action="store_true")
    tune = subparsers.add_parser("tune-panda")
    tune.add_argument("--config", default="configs/v2/few_shot.json")
    tune.add_argument("--shots", required=True)
    tune.add_argument("--validation", required=True)
    tune.add_argument("--output-dir", required=True)
    tune.add_argument("--device", default="cuda")
    tune.add_argument("--force", action="store_true")
    fit = subparsers.add_parser("fit")
    fit.add_argument("--config", default="configs/v2/few_shot.json")
    fit.add_argument("--shots", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--method", required=True, choices=METHODS)
    fit.add_argument("--shot-count", required=True, type=int)
    fit.add_argument("--tuning")
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--force", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="configs/v2/few_shot.json")
    evaluate.add_argument("--context", required=True)
    evaluate.add_argument("--forecast-truth", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--method", required=True, choices=METHODS)
    evaluate.add_argument("--shot-count", required=True, type=int)
    evaluate.add_argument("--model")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--force", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", default="configs/v2/few_shot.json")
    aggregate.add_argument("--runs-root", required=True)
    aggregate.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.command == "extract-shots":
        print(extract_shot_data(
            config_path=args.config,
            train_path=args.train,
            output_path=args.output,
            force=args.force,
        ))
    elif args.command == "tune-panda":
        print(tune_panda_head(
            config_path=args.config,
            shots_path=args.shots,
            validation_path=args.validation,
            output_dir=args.output_dir,
            device=args.device,
            force=args.force,
        ))
    elif args.command == "fit":
        if args.method == "panda":
            if not args.tuning:
                raise SystemExit("Panda fitting requires --tuning")
            print(fit_few_shot_panda(
                config_path=args.config,
                shots_path=args.shots,
                tuning_path=args.tuning,
                output_dir=args.output_dir,
                shot_count=args.shot_count,
                device=args.device,
                force=args.force,
            ))
        else:
            if args.tuning:
                raise SystemExit("SINDy fitting does not accept --tuning")
            print(fit_few_shot_sindy(
                config_path=args.config,
                shots_path=args.shots,
                output_dir=args.output_dir,
                method=args.method,
                shot_count=args.shot_count,
                force=args.force,
            ))
    elif args.command == "evaluate":
        print(evaluate_few_shot(
            config_path=args.config,
            context_path=args.context,
            truth_path=args.forecast_truth,
            output_dir=args.output_dir,
            method=args.method,
            shot_count=args.shot_count,
            model_path=args.model,
            device=args.device,
            force=args.force,
        ))
    else:
        outputs = aggregate_few_shot(
            config_path=args.config,
            runs_root=args.runs_root,
            output_dir=args.output_dir,
        )
        print(*outputs.values(), sep="\n")


if __name__ == "__main__":
    main()
