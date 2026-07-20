from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lorenz63_benchmark.v2.smoke import run_smoke
from lorenz63_benchmark.v2.attractor_summary import generate_attractor_summary
from lorenz63_benchmark.v2.artifacts import sha256_file
from lorenz63_benchmark.v2.visualize import generate_visualizations


def test_all_methods_cpu_smoke(tmp_path) -> None:
    root = run_smoke(tmp_path / "smoke", Path("configs/v2/lorenz63_smoke.json"))
    assert (root / "results" / "manifest.json").exists()
    figures = list((root / "results" / "figures").glob("*.png"))
    assert len(figures) >= 3
    aggregate = json.loads((root / "results" / "manifest.json").read_text())
    assert aggregate["source_hash"]
    assert aggregate["config_hash"]
    run_manifest = json.loads((root / "runs" / "node_strong" / "manifest.json").read_text())
    assert {
        "schema", "status", "run_id", "method", "track", "information_contract",
        "source_hash", "config_hash", "input_hashes", "command", "environment",
        "seeds", "artifacts", "training", "evaluation",
    } <= run_manifest.keys()
    assert run_manifest["status"] == "complete"
    assert set(run_manifest["input_hashes"]) == {"train", "validation"}
    assert run_manifest["evaluation"]["status"] == "complete"
    assert run_manifest["evaluation"]["source_hash"]
    assert run_manifest["evaluation"]["config_hash"]
    assert set(run_manifest["evaluation"]["artifacts"]) == {
        "predictions", "trajectory_metrics", "run_metrics", "curve",
    }

    repeated = run_smoke(tmp_path / "smoke_repeated", Path("configs/v2/lorenz63_smoke.json"))
    first_trajectories = pd.read_csv(root / "results" / "trajectory_metrics.csv").sort_values(
        ["run_id", "trajectory_id"]
    ).reset_index(drop=True)
    repeated_trajectories = pd.read_csv(
        repeated / "results" / "trajectory_metrics.csv"
    ).sort_values(["run_id", "trajectory_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first_trajectories, repeated_trajectories, check_exact=True)

    timing_columns = {
        "training_wall_time_seconds",
        "inference_wall_time_seconds",
        "inference_seconds_per_trajectory",
    }
    first_runs = pd.read_csv(root / "results" / "run_metrics.csv").sort_values(
        "run_id"
    ).reset_index(drop=True)
    repeated_runs = pd.read_csv(repeated / "results" / "run_metrics.csv").sort_values(
        "run_id"
    ).reset_index(drop=True)
    scientific_columns = [
        column for column in first_runs.columns if column not in timing_columns
    ]
    pd.testing.assert_frame_equal(
        first_runs[scientific_columns], repeated_runs[scientific_columns], check_exact=True
    )

    for first_run_dir in sorted((root / "runs").iterdir()):
        repeated_prediction = repeated / "runs" / first_run_dir.name / "predictions.npz"
        with np.load(first_run_dir / "predictions.npz", allow_pickle=False) as first, np.load(
            repeated_prediction, allow_pickle=False
        ) as second:
            assert first.files == second.files
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key], strict=True)

    visualization_root = root / "visualizations"
    summary_path = generate_attractor_summary(
        runs_root=root / "runs",
        data_root=root / "data",
        run_metrics_path=root / "results" / "run_metrics.csv",
        trajectory_metrics_path=root / "results" / "trajectory_metrics.csv",
        output_path=visualization_root / "attractor_summary.npz",
        density_start_lt=0.1,
        density_stop_lt=0.4,
        return_start_lt=0.1,
        histogram_bins=16,
        return_pairs_per_run=20,
    )
    visualization_manifest_path = generate_visualizations(
        config_path=Path("configs/v2/lorenz63_smoke.json"),
        results_dir=root / "results",
        output_dir=visualization_root,
        attractor_summary_path=summary_path,
        formats=("png",),
        bootstrap_resamples=20,
        bootstrap_seed=2026,
    )
    visualization_manifest = json.loads(visualization_manifest_path.read_text())
    assert visualization_manifest["status"] == "complete"
    assert visualization_manifest["figure_count"] == 10
    for artifact in visualization_manifest["artifacts"].values():
        assert sha256_file(artifact["path"]) == artifact["sha256"]
