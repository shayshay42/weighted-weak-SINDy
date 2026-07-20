from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    atomic_save_npz, atomic_write_csv, atomic_write_json, command_line,
    environment_snapshot, sha256_file, sha256_json, source_hash, utc_now,
)
from .config import load_config
from .contracts import PINN_TRACK
from .data import load_split
from .metrics import (
    distribution_metrics, interval_nrmse_auc, learned_lyapunov_metrics,
    normalized_rmse_auc, normalized_squared_error, valid_prediction_time,
)
from .identification import relative_coefficient_error, relative_parameter_error
from .registry import load_method


def _checkpoint_path(run_dir: Path, manifest: dict[str, Any]) -> Path:
    record = manifest.get("artifacts", {}).get("checkpoint")
    if not record:
        raise ValueError(f"run has no checkpoint artifact: {run_dir}")
    path = Path(record["path"])
    return path if path.is_absolute() else run_dir / path


def _nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = values[~np.isnan(values)]
    return float(np.median(observed)) if observed.size else float("nan")


def evaluate_run(
    *,
    config_path: str | Path | None,
    run_dir: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    dataset_manifest_path: str | Path,
    device: str = "cpu",
    force: bool = False,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    run_dir = Path(run_dir)
    run_manifest_path = run_dir / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete":
        raise ValueError(f"cannot evaluate incomplete run {run_manifest.get('run_id')}")
    prediction_path = run_dir / "predictions.npz"
    run_metrics_path = run_dir / "run_metrics.csv"
    evaluation_identity = {
        "source_hash": source_hash(), "config_hash": sha256_json(config),
        "train_sha256": sha256_file(train_path), "test_sha256": sha256_file(test_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
    }
    if prediction_path.exists() and run_metrics_path.exists() and not force:
        evaluation = run_manifest.get("evaluation", {})
        records = evaluation.get("artifacts", {}).values()
        if (
            evaluation.get("status") == "complete"
            and all(evaluation.get(key) == value for key, value in evaluation_identity.items())
            and records
            and all(
                Path(record["path"]).exists()
                and sha256_file(record["path"]) == record["sha256"]
                for record in records
            )
        ):
            return prediction_path, run_metrics_path
    run_manifest["evaluation"] = {
        "status": "running", "started_at": utc_now(), **evaluation_identity,
        "command": command_line(), "environment": environment_snapshot(), "device": device,
    }
    atomic_write_json(run_manifest_path, run_manifest)

    train = load_split(train_path, "train")
    test = load_split(test_path, "test")
    dataset_manifest = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
    if int(dataset_manifest["data_seed"]) != int(run_manifest["seeds"]["data"]):
        raise ValueError("run and dataset manifest data seeds differ")
    checkpoint = _checkpoint_path(run_dir, run_manifest)
    method = str(run_manifest["method"])
    adapter = load_method(checkpoint, method, config, device=device)
    integration_internal_step = getattr(adapter, "internal_step", None)
    forecast_evaluator = {
        "kind": "fixed_step_float64_rk4" if integration_internal_step is not None else "conditional_flow_map",
        "internal_step": (
            float(integration_internal_step) if integration_internal_step is not None else None
        ),
    }
    forecast_times = np.asarray(test["times"], dtype=np.float64)
    forecast_truth = np.asarray(test["states"], dtype=np.float64)
    largest = float(dataset_manifest["largest_lyapunov_exponent"])
    if adapter.track == PINN_TRACK and method != "solver_oracle":
        maximum_lt = float(config["pinn"]["evaluation_lyapunov_times"])
        stop = int(np.searchsorted(largest * forecast_times, maximum_lt, side="left"))
        stop = min(stop, forecast_times.size - 1)
        forecast_times = forecast_times[: stop + 1]
        forecast_truth = forecast_truth[:, : stop + 1]

    started = time.perf_counter()
    forecast_prediction = adapter.forecast(forecast_truth[:, 0], forecast_times)
    inference_seconds = time.perf_counter() - started
    if forecast_prediction.shape != forecast_truth.shape:
        raise RuntimeError(
            f"prediction shape {forecast_prediction.shape} does not match truth {forecast_truth.shape}"
        )
    training_std = np.asarray(train["metadata"]["normalization"]["state_std"], dtype=np.float64)
    forecast_error = normalized_squared_error(forecast_prediction, forecast_truth, training_std)
    forecast_times_lt = forecast_times * largest
    full_vpt, full_censored = valid_prediction_time(
        forecast_error, forecast_times, largest,
        threshold=float(config["evaluation"]["vpt_threshold"]),
    )
    if adapter.track == PINN_TRACK:
        maximum_lt = min(
            float(config["pinn"]["evaluation_lyapunov_times"]), float(forecast_times_lt[-1])
        )
        metric_stop = int(np.searchsorted(forecast_times_lt, maximum_lt, side="left"))
        metric_stop = min(metric_stop, forecast_times.size - 1)
    else:
        metric_stop = forecast_times.size - 1
    metric_times = forecast_times[: metric_stop + 1]
    metric_times_lt = forecast_times_lt[: metric_stop + 1]
    metric_truth = forecast_truth[:, : metric_stop + 1]
    metric_prediction = forecast_prediction[:, : metric_stop + 1]
    error = forecast_error[:, : metric_stop + 1]
    vpt, censored = valid_prediction_time(
        error, metric_times, largest, threshold=float(config["evaluation"]["vpt_threshold"])
    )
    auc_values = {
        float(horizon): normalized_rmse_auc(error, metric_times_lt, float(horizon))
        for horizon in config["evaluation"]["auc_lyapunov_times"]
    }
    in_domain = normalized_rmse_auc(error, metric_times_lt, 2.0)
    extrapolation = interval_nrmse_auc(error, metric_times_lt, 2.0, 5.0)
    trajectory_rows = []
    for index in range(metric_truth.shape[0]):
        row = {
            "run_id": run_manifest["run_id"], "method": method, "track": adapter.track,
            "data_seed": int(run_manifest["seeds"]["data"]),
            "model_seed": int(run_manifest["seeds"]["model"]), "trajectory_id": index,
            "vpt_restricted_lt": float(vpt[index]), "vpt_censored": bool(censored[index]),
            "mean_normalized_squared_error": float(np.mean(error[index])),
            "pinn_in_domain_auc_0_2LT": float(in_domain[index]),
            "pinn_extrapolation_auc_2_5LT": float(extrapolation[index]),
        }
        for horizon, values in auc_values.items():
            row[f"nrmse_auc_0_{horizon:g}LT"] = float(values[index])
        trajectory_rows.append(row)

    burn_mask = forecast_times_lt >= min(5.0, 0.5 * forecast_times_lt[-1])
    if np.count_nonzero(burn_mask) < 2:
        burn_mask = np.arange(forecast_times.size) >= forecast_times.size // 2
    long_run: dict[str, Any] = {}
    if adapter.autonomous:
        long_run.update(distribution_metrics(
            forecast_prediction[:, burn_mask], forecast_truth[:, burn_mask],
            stability_bound=float(config["evaluation"]["stability_bound"]),
            maximum_samples=int(config["evaluation"]["long_run_subsample"]),
        ))
        long_run.update(learned_lyapunov_metrics(
            adapter.vector_field, forecast_truth[0, 0],
            np.asarray(dataset_manifest["lyapunov_spectrum"]),
            duration=float(config["evaluation"]["lyapunov_duration"]),
            step=float(config["evaluation"]["lyapunov_step"]),
            qr_interval=int(config["evaluation"]["lyapunov_qr_interval"]),
        ))

    parameters = adapter.parameters()
    system = {
        key: float(value) for key, value in dataset_manifest["generator"]["system"].items()
    }
    parameter_error = float("nan")
    coefficient_error = float("nan")
    if method.startswith("lorenz_ad"):
        parameter_error = relative_parameter_error(parameters, system)
    elif method.startswith("sindy_"):
        coefficient_error = relative_coefficient_error(
            np.asarray(parameters["coefficients"], dtype=np.float64),
            tuple(tuple(int(value) for value in row) for row in parameters["exponents"]),
            system,
        )
    run_row: dict[str, Any] = {
        "run_id": run_manifest["run_id"], "method": method, "track": adapter.track,
        "data_seed": int(run_manifest["seeds"]["data"]),
        "model_seed": int(run_manifest["seeds"]["model"]),
        "noise_level": float(train["metadata"]["noise_level"]),
        "forecast_horizon_lt": float(metric_times_lt[-1]),
        "full_forecast_horizon_lt": float(forecast_times_lt[-1]),
        "full_horizon_vpt_censoring_fraction": float(np.mean(full_censored)),
        "full_horizon_restricted_mean_vpt_lt": float(np.mean(full_vpt)),
        "restricted_mean_vpt_lt": float(np.mean(vpt)),
        "median_vpt_lt": float(np.median(vpt)),
        "vpt_censoring_fraction": float(np.mean(censored)),
        "mean_normalized_squared_error": float(np.mean(error)),
        "inference_wall_time_seconds": inference_seconds,
        "inference_seconds_per_trajectory": inference_seconds / forecast_truth.shape[0],
        "vector_field_evaluations": int(getattr(adapter, "last_vector_field_evaluations", 0)),
        "surrogate_evaluations": int(getattr(adapter, "last_surrogate_evaluations", 0)),
        "integration_internal_step": (
            float(integration_internal_step) if integration_internal_step is not None else float("nan")
        ),
        "training_vector_field_evaluations": int(
            run_manifest.get("training", {}).get("vector_field_evaluations", 0)
        ),
        "training_wall_time_seconds": float(run_manifest.get("training", {}).get("wall_time_seconds", 0.0)),
        "parameter_count": int(run_manifest.get("training", {}).get("parameter_count", 0)),
        "optimizer_updates": int(run_manifest.get("training", {}).get("optimizer_updates", 0)),
        "peak_gpu_memory_bytes": int(run_manifest.get("training", {}).get("peak_gpu_memory_bytes", 0)),
        "parameter_relative_error": parameter_error,
        "coefficient_relative_error": coefficient_error,
    }
    for horizon, values in auc_values.items():
        run_row[f"median_nrmse_auc_0_{horizon:g}LT"] = _nanmedian(values)
    run_row["median_pinn_in_domain_auc_0_2LT"] = _nanmedian(in_domain)
    run_row["median_pinn_extrapolation_auc_2_5LT"] = _nanmedian(extrapolation)
    run_row.update(long_run)

    atomic_save_npz(
        prediction_path, prediction=forecast_prediction.astype(np.float32), times=forecast_times,
        normalized_squared_error=forecast_error.astype(np.float32),
        metric_times=metric_times, metric_normalized_squared_error=error.astype(np.float32),
        vpt_restricted_lt=vpt, vpt_censored=censored,
    )
    trajectory_metrics_path = run_dir / "trajectory_metrics.csv"
    atomic_write_csv(trajectory_metrics_path, trajectory_rows)
    atomic_write_csv(run_metrics_path, [run_row])
    curve_path = run_dir / "curve.npz"
    atomic_save_npz(
        curve_path, times_lt=metric_times_lt,
        median=np.nanmedian(error, axis=0), q25=np.nanquantile(error, 0.25, axis=0),
        q75=np.nanquantile(error, 0.75, axis=0),
    )
    evaluation_record = dict(run_manifest["evaluation"])
    evaluation_record.update({
        "status": "complete", "completed_at": utc_now(),
        "forecast_evaluator": forecast_evaluator,
        "artifacts": {
            "predictions": {"path": str(prediction_path.resolve()), "sha256": sha256_file(prediction_path)},
            "trajectory_metrics": {"path": str(trajectory_metrics_path.resolve()), "sha256": sha256_file(trajectory_metrics_path)},
            "run_metrics": {"path": str(run_metrics_path.resolve()), "sha256": sha256_file(run_metrics_path)},
            "curve": {"path": str(curve_path.resolve()), "sha256": sha256_file(curve_path)},
        },
    })
    run_manifest["evaluation"] = evaluation_record
    atomic_write_json(run_manifest_path, run_manifest)
    return prediction_path, run_metrics_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one completed Lorenz63 v2 run.")
    parser.add_argument("--config", default="configs/v2/lorenz63_paper.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = evaluate_run(
        config_path=args.config, run_dir=args.run_dir, train_path=args.train,
        test_path=args.test, dataset_manifest_path=args.dataset_manifest,
        device=args.device, force=args.force,
    )
    print(*paths, sep="\n")


if __name__ == "__main__":
    main()
