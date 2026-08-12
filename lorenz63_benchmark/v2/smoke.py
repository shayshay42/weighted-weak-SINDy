from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import aggregate_runs
from .artifacts import validate_artifact_hashes
from .config import load_config
from .contracts import configured_methods
from .data import generate_split
from .evaluate import evaluate_run
from .train import run_training


def run_smoke(output_root: str | Path, config_path: str | Path) -> Path:
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    config = load_config(config_path)
    split_root = generate_split(config, output_root / "data", data_seed=1)
    train_path = split_root / "noise_0" / "train.npz"
    validation_path = split_root / "validation.npz"
    test_path = split_root / "test.npz"
    dataset_manifest = split_root / "manifest.json"
    runs_root = output_root / "runs"
    methods = sorted(configured_methods(config))
    for method in methods:
        run_dir = runs_root / method
        checkpoint = run_training(
            config_path=config_path, train_path=train_path, validation_path=validation_path,
            method=method, data_seed=1, model_seed=0, output_dir=run_dir, device="cpu",
        )
        if not checkpoint.exists() or not validate_artifact_hashes(run_dir / "manifest.json"):
            raise AssertionError(f"incomplete or invalid training artifact for {method}")
        prediction_path, _ = evaluate_run(
            config_path=config_path, run_dir=run_dir, train_path=train_path,
            test_path=test_path, dataset_manifest_path=dataset_manifest, device="cpu",
        )
        with np.load(prediction_path, allow_pickle=False) as prediction:
            if not np.all(np.isfinite(prediction["prediction"])):
                raise AssertionError(f"non-finite smoke forecast for {method}")
    outputs = aggregate_runs(
        config_path=config_path, runs_root=runs_root,
        output_dir=output_root / "results", make_figures=True,
    )
    run_metrics = pd.read_csv(outputs["run_metrics"])
    trajectory_metrics = pd.read_csv(outputs["trajectory_metrics"])
    expected_runs = len(methods)
    expected_trajectories = expected_runs * int(config["data"]["n_test"])
    if len(run_metrics) != expected_runs:
        raise AssertionError(f"expected {expected_runs} run rows, found {len(run_metrics)}")
    if len(trajectory_metrics) != expected_trajectories:
        raise AssertionError(
            f"expected {expected_trajectories} trajectory rows, found {len(trajectory_metrics)}"
        )
    oracle = run_metrics.loc[run_metrics["method"] == "solver_oracle"].iloc[0]
    if abs(float(oracle["full_horizon_vpt_censoring_fraction"]) - 1.0) > 1e-12:
        raise AssertionError("the exact numerical oracle did not reach the solver-valid horizon")
    forward = run_metrics.loc[run_metrics["track"] == "known_physics_forward_surrogate"]
    if float(forward["forecast_horizon_lt"].max() - forward["forecast_horizon_lt"].min()) > 1e-12:
        raise AssertionError("forward-surrogate VPT metrics do not share a common horizon")
    aggregate_manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    if aggregate_manifest.get("status") != "complete" or aggregate_manifest.get("run_count") != expected_runs:
        raise AssertionError("aggregate manifest is incomplete")
    return output_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run every Lorenz63 v2 method on a tiny CPU dataset.")
    parser.add_argument("--config", default="configs/v2/lorenz63_smoke.json")
    parser.add_argument("--output-root", default="runs/v2_smoke")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(run_smoke(args.output_root, args.config))


if __name__ == "__main__":
    main()
