from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    atomic_write_csv,
    atomic_write_json,
    command_line,
    environment_snapshot,
    sha256_file,
    source_hash,
    utc_now,
)
from .contracts import PRETRAINED_TRACK
from .data import noise_label
from .visualize import generate_survival_overlay


def _read_result_tables(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(directory)
    return (
        pd.read_csv(directory / "run_metrics.csv"),
        pd.read_csv(directory / "trajectory_metrics.csv"),
    )


def _select_method(
    runs: pd.DataFrame, trajectories: pd.DataFrame, method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_runs = runs[runs["method"] == method].copy()
    selected_trajectories = trajectories[trajectories["method"] == method].copy()
    if selected_runs.empty or selected_trajectories.empty:
        raise ValueError(f"extension input has no results for {method}")
    if set(selected_runs["track"]) != {PRETRAINED_TRACK}:
        raise ValueError(f"{method} does not use the pretrained information track")
    if set(selected_runs["forecast_context_steps"].astype(int)) != {512}:
        raise ValueError(f"{method} must use exactly 512 clean context samples")
    if not np.allclose(selected_runs["noise_level"], 0.0):
        raise ValueError(f"{method} source rows must be the clean zero-shot reference")
    return selected_runs, selected_trajectories


def _repeat_pretrained_reference(
    runs: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    method: str,
    panel_noises: tuple[float, ...],
    target_model_seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, trajectories = _select_method(runs, trajectories, method)
    source_model_seeds = sorted(int(value) for value in runs["model_seed"].unique())
    repeated_runs: list[pd.DataFrame] = []
    repeated_trajectories: list[pd.DataFrame] = []
    for noise in panel_noises:
        for target_seed in target_model_seeds:
            source_seed = (
                target_seed if target_seed in source_model_seeds else source_model_seeds[0]
            )
            source_runs = runs[runs["model_seed"].astype(int) == source_seed].copy()
            source_trajectories = trajectories[
                trajectories["model_seed"].astype(int) == source_seed
            ].copy()
            if source_runs.empty or source_trajectories.empty:
                raise ValueError(f"missing {method} source model seed {source_seed}")
            suffix = f"__fixed_clean_prefix__panel_n{noise_label(noise)}__m{target_seed}"
            run_id_map = {
                str(run_id): f"{run_id}{suffix}"
                for run_id in source_runs["run_id"].astype(str)
            }
            source_runs["source_run_id"] = source_runs["run_id"].astype(str)
            source_trajectories["source_run_id"] = source_trajectories["run_id"].astype(str)
            source_runs["run_id"] = source_runs["run_id"].astype(str).map(run_id_map)
            source_trajectories["run_id"] = (
                source_trajectories["run_id"].astype(str).map(run_id_map)
            )
            source_runs["model_seed"] = target_seed
            source_trajectories["model_seed"] = target_seed
            source_runs["noise_level"] = noise
            source_trajectories["noise_level"] = noise
            source_runs["fixed_clean_prefix_reference"] = True
            source_trajectories["fixed_clean_prefix_reference"] = True
            source_runs["source_noise_level"] = 0.0
            source_trajectories["source_noise_level"] = 0.0
            repeated_runs.append(source_runs)
            repeated_trajectories.append(source_trajectories)
    return (
        pd.concat(repeated_runs, ignore_index=True),
        pd.concat(repeated_trajectories, ignore_index=True),
    )


def combine_survival_extension(
    *,
    core_results: str | Path,
    integral_results: str | Path,
    panda_results: str | Path,
    chronos_results: str | Path,
    output_dir: str | Path,
    panel_noises: tuple[float, ...] = (0.001, 0.01, 0.05),
    target_model_seeds: tuple[int, ...] = (0, 1, 2),
) -> Path:
    if len(panel_noises) != 3 or any(noise <= 0.0 for noise in panel_noises):
        raise ValueError("extension requires three positive panel noise levels")
    inputs = {
        "core": Path(core_results),
        "integral": Path(integral_results),
        "panda": Path(panda_results),
        "chronos": Path(chronos_results),
    }
    core_runs, core_trajectories = _read_result_tables(inputs["core"])
    integral_runs, integral_trajectories = _read_result_tables(inputs["integral"])
    panda_runs, panda_trajectories = _read_result_tables(inputs["panda"])
    chronos_runs, chronos_trajectories = _read_result_tables(inputs["chronos"])
    keep_noise = lambda frame: frame[
        np.isclose(frame["noise_level"].to_numpy()[:, None], panel_noises).any(axis=1)
    ].copy()
    core_runs = keep_noise(core_runs)
    core_trajectories = keep_noise(core_trajectories)
    integral_runs = keep_noise(integral_runs)
    integral_trajectories = keep_noise(integral_trajectories)
    if set(integral_runs["method"]) != {"lorenz_integral"}:
        raise ValueError("integral results must contain only lorenz_integral")

    panda_reference = _repeat_pretrained_reference(
        panda_runs,
        panda_trajectories,
        method="panda_zero_shot",
        panel_noises=panel_noises,
        target_model_seeds=target_model_seeds,
    )
    chronos_reference = _repeat_pretrained_reference(
        chronos_runs,
        chronos_trajectories,
        method="chronos_zero_shot",
        panel_noises=panel_noises,
        target_model_seeds=target_model_seeds,
    )
    run_frames = [core_runs, integral_runs, panda_reference[0], chronos_reference[0]]
    trajectory_frames = [
        core_trajectories,
        integral_trajectories,
        panda_reference[1],
        chronos_reference[1],
    ]
    for frame in (*run_frames[:2], *trajectory_frames[:2]):
        frame["fixed_clean_prefix_reference"] = False
        frame["source_noise_level"] = frame["noise_level"]
        frame["source_run_id"] = frame["run_id"]
    combined_runs = pd.concat(run_frames, ignore_index=True, sort=False)
    combined_trajectories = pd.concat(trajectory_frames, ignore_index=True, sort=False)
    if combined_runs["run_id"].duplicated().any():
        raise ValueError("combined extension has duplicate run IDs")
    trajectory_key = [
        "method", "noise_level", "data_seed", "model_seed", "trajectory_id"
    ]
    if combined_trajectories.duplicated(trajectory_key).any():
        raise ValueError("combined extension has duplicate trajectory cells")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run_metrics.csv"
    trajectory_path = output_dir / "trajectory_metrics.csv"
    atomic_write_csv(run_path, combined_runs.to_dict(orient="records"))
    atomic_write_csv(trajectory_path, combined_trajectories.to_dict(orient="records"))
    manifest = {
        "schema": "lorenz63-survival-extension-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "source_hash": source_hash(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "panel_noises": list(panel_noises),
        "target_model_seeds": list(target_model_seeds),
        "pretrained_reference_protocol": {
            "context_steps": 512,
            "context_noise": 0.0,
            "repeated_across_training_noise_panels": True,
            "benchmark_fitting_performed": False,
        },
        "inputs": {
            name: {
                "run_metrics": sha256_file(path / "run_metrics.csv"),
                "trajectory_metrics": sha256_file(path / "trajectory_metrics.csv"),
            }
            for name, path in inputs.items()
        },
        "artifacts": {
            "run_metrics": {"path": str(run_path.resolve()), "sha256": sha256_file(run_path)},
            "trajectory_metrics": {
                "path": str(trajectory_path.resolve()),
                "sha256": sha256_file(trajectory_path),
            },
        },
    }
    manifest_path = output_dir / "extension_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine integral and fixed-prefix TSFM survival references.")
    parser.add_argument("--core-results", required=True)
    parser.add_argument("--integral-results", required=True)
    parser.add_argument("--panda-results", required=True)
    parser.add_argument("--chronos-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/v2/extended_survival.json")
    parser.add_argument("--figure-dir")
    parser.add_argument("--bootstrap-resamples", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = combine_survival_extension(
        core_results=args.core_results,
        integral_results=args.integral_results,
        panda_results=args.panda_results,
        chronos_results=args.chronos_results,
        output_dir=args.output_dir,
    )
    if args.figure_dir:
        generate_survival_overlay(
            config_path=args.config,
            results_dir=args.output_dir,
            output_dir=args.figure_dir,
            formats=("png", "pdf"),
            bootstrap_resamples=args.bootstrap_resamples,
        )
    print(manifest)


if __name__ == "__main__":
    main()
