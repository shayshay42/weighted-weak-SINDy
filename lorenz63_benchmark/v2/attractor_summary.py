from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aggregate import discover_completed_runs
from .artifacts import (
    atomic_save_npz,
    atomic_write_json,
    command_line,
    environment_snapshot,
    sha256_file,
    sha256_json,
    source_hash,
    utc_now,
)
from .contracts import METHODS, PRIMARY_TRACK
from .data import load_split


def return_map_pairs(states: np.ndarray) -> np.ndarray:
    """Return consecutive local maxima of z for batched Lorenz trajectories."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 3 or states.shape[-1] != 3:
        raise ValueError("return-map states must have shape [trajectory,time,3]")
    pairs: list[np.ndarray] = []
    for trajectory in states:
        z = trajectory[:, 2]
        maxima = z[1:-1][(z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:])]
        if maxima.size >= 2:
            pairs.append(np.column_stack([maxima[:-1], maxima[1:]]))
    return np.concatenate(pairs, axis=0) if pairs else np.empty((0, 2), dtype=np.float64)


def normalized_xz_histogram(
    states: np.ndarray, x_edges: np.ndarray, z_edges: np.ndarray
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(states), axis=1)
    finite_states = states[finite]
    x = np.clip(
        finite_states[:, 0],
        np.nextafter(x_edges[0], x_edges[-1]),
        np.nextafter(x_edges[-1], x_edges[0]),
    )
    z = np.clip(
        finite_states[:, 2],
        np.nextafter(z_edges[0], z_edges[-1]),
        np.nextafter(z_edges[-1], z_edges[0]),
    )
    histogram, _, _ = np.histogram2d(
        x, z, bins=(x_edges, z_edges)
    )
    total = float(histogram.sum())
    return histogram / total if total > 0.0 else np.zeros_like(histogram)


def xz_in_range_fraction(
    states: np.ndarray, x_edges: np.ndarray, z_edges: np.ndarray
) -> float:
    states = np.asarray(states, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(states), axis=1)
    if not np.any(finite):
        return 0.0
    selected = states[finite]
    inside = (
        (selected[:, 0] >= x_edges[0])
        & (selected[:, 0] <= x_edges[-1])
        & (selected[:, 2] >= z_edges[0])
        & (selected[:, 2] <= z_edges[-1])
    )
    return float(np.mean(inside))


def select_representative_cell(
    trajectory_metrics: pd.DataFrame, noise_level: float
) -> dict[str, int]:
    primary = trajectory_metrics[
        (trajectory_metrics["track"] == PRIMARY_TRACK)
        & np.isclose(trajectory_metrics["noise_level"], noise_level)
    ]
    if primary.empty:
        raise ValueError("no primary-track trajectory metrics are available")
    keys = ["data_seed", "model_seed", "trajectory_id"]
    difficulty = primary.groupby(keys, as_index=False)["vpt_restricted_lt"].median()
    target = float(difficulty["vpt_restricted_lt"].median())
    difficulty["distance"] = np.abs(difficulty["vpt_restricted_lt"] - target)
    selected = difficulty.sort_values(["distance", *keys]).iloc[0]
    return {key: int(selected[key]) for key in keys}


def _balanced_pairs(
    pairs: np.ndarray, maximum: int, rng: np.random.Generator
) -> np.ndarray:
    if pairs.shape[0] <= maximum:
        return pairs
    return pairs[rng.choice(pairs.shape[0], maximum, replace=False)]


def _manifest_map(runs_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for run_dir, manifest in discover_completed_runs(runs_root):
        run_id = str(manifest["run_id"])
        if "__k" not in run_id:
            result[run_id] = (run_dir, manifest)
    return result


def _load_prediction(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = manifest["evaluation"]["artifacts"]["predictions"]
    path = Path(record["path"])
    if not path.exists():
        path = run_dir / "predictions.npz"
    with np.load(path, allow_pickle=False) as loaded:
        return (
            np.asarray(loaded["prediction"], dtype=np.float64),
            np.asarray(loaded["times"], dtype=np.float64),
            np.asarray(loaded["normalized_squared_error"], dtype=np.float64),
        )


def _window_mask(times_lt: np.ndarray, start: float, stop: float) -> np.ndarray:
    mask = (times_lt >= start - 1e-12) & (times_lt <= stop + 1e-12)
    if np.count_nonzero(mask) < 3:
        raise ValueError(f"window {start:g}-{stop:g} LT contains fewer than three samples")
    return mask


def generate_attractor_summary(
    *,
    runs_root: str | Path,
    data_root: str | Path,
    run_metrics_path: str | Path,
    trajectory_metrics_path: str | Path,
    output_path: str | Path,
    noise_level: float = 0.0,
    density_start_lt: float = 2.0,
    density_stop_lt: float = 5.0,
    return_start_lt: float = 5.0,
    histogram_bins: int = 96,
    return_pairs_per_run: int = 500,
    seed: int = 2026,
) -> Path:
    runs_root = Path(runs_root)
    data_root = Path(data_root)
    output_path = Path(output_path)
    run_metrics_path = Path(run_metrics_path)
    trajectory_metrics_path = Path(trajectory_metrics_path)
    run_metrics = pd.read_csv(run_metrics_path)
    trajectory_metrics = pd.read_csv(trajectory_metrics_path)
    selected_runs = run_metrics[np.isclose(run_metrics["noise_level"], noise_level)].copy()
    if selected_runs.empty:
        raise ValueError(f"no runs found for noise level {noise_level:g}")
    method_names = set(selected_runs["method"].astype(str))
    unknown_methods = sorted(method_names.difference(METHODS))
    if unknown_methods:
        raise ValueError(f"prediction summary contains unknown methods: {unknown_methods}")
    methods = sorted(method_names)
    manifests = _manifest_map(runs_root)
    missing_runs = sorted(set(selected_runs["run_id"]).difference(manifests))
    if missing_runs:
        raise ValueError(f"missing completed run manifests: {missing_runs[:3]}")

    data_seeds = sorted(int(value) for value in selected_runs["data_seed"].unique())
    truth_by_seed: dict[int, dict[str, Any]] = {}
    density_samples = []
    test_hashes: dict[str, str] = {}
    for data_seed in data_seeds:
        split_root = data_root / f"split_seed{data_seed}"
        test_path = split_root / "test.npz"
        test = load_split(test_path, "test")
        dataset_manifest_path = split_root / "manifest.json"
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        largest = float(dataset_manifest["largest_lyapunov_exponent"])
        times = np.asarray(test["times"], dtype=np.float64)
        states = np.asarray(test["states"], dtype=np.float64)
        times_lt = largest * times
        density_mask = _window_mask(times_lt, density_start_lt, density_stop_lt)
        density_samples.append(states[:, density_mask].reshape(-1, 3))
        truth_by_seed[data_seed] = {
            "states": states, "times": times, "times_lt": times_lt,
            "largest": largest,
        }
        test_hashes[str(data_seed)] = sha256_file(test_path)

    truth_samples = np.concatenate(density_samples, axis=0)
    finite_truth = truth_samples[np.all(np.isfinite(truth_samples), axis=1)]
    if finite_truth.size == 0:
        raise ValueError("test truth has no finite density samples")
    lower = np.quantile(finite_truth[:, [0, 2]], 0.001, axis=0)
    upper = np.quantile(finite_truth[:, [0, 2]], 0.999, axis=0)
    padding = 0.05 * np.maximum(upper - lower, 1e-12)
    x_edges = np.linspace(lower[0] - padding[0], upper[0] + padding[0], histogram_bins + 1)
    z_edges = np.linspace(lower[1] - padding[1], upper[1] + padding[1], histogram_bins + 1)
    truth_histograms = [
        normalized_xz_histogram(samples, x_edges, z_edges) for samples in density_samples
    ]

    arrays: dict[str, np.ndarray] = {
        "x_edges": x_edges,
        "z_edges": z_edges,
        "density_truth": np.mean(truth_histograms, axis=0),
        "density_in_range_fraction__truth": np.asarray(np.mean([
            xz_in_range_fraction(samples, x_edges, z_edges) for samples in density_samples
        ])),
    }
    input_manifest_hashes: dict[str, str] = {}
    return_truth_by_seed: list[np.ndarray] = []
    for data_seed in data_seeds:
        truth = truth_by_seed[data_seed]
        mask = truth["times_lt"] >= return_start_lt - 1e-12
        pairs = return_map_pairs(truth["states"][:, mask])
        rng = np.random.default_rng(np.random.SeedSequence([seed, data_seed, 991]))
        return_truth_by_seed.append(_balanced_pairs(pairs, return_pairs_per_run, rng))
    arrays["return_truth"] = (
        np.concatenate(return_truth_by_seed, axis=0)
        if return_truth_by_seed else np.empty((0, 2), dtype=np.float64)
    )

    method_index = {method: index for index, method in enumerate(methods)}
    autonomous_methods = [
        method for method in methods if METHODS[method].autonomous
    ]
    for method in methods:
        method_runs = selected_runs[selected_runs["method"] == method].sort_values(
            ["data_seed", "model_seed"]
        )
        histograms = []
        in_range_fractions = []
        method_pairs = []
        for row in method_runs.itertuples(index=False):
            run_dir, manifest = manifests[str(row.run_id)]
            input_manifest_hashes[str(row.run_id)] = sha256_file(run_dir / "manifest.json")
            prediction, times, _error = _load_prediction(run_dir, manifest)
            truth = truth_by_seed[int(row.data_seed)]
            times_lt = times * float(truth["largest"])
            density_mask = _window_mask(times_lt, density_start_lt, density_stop_lt)
            histograms.append(
                normalized_xz_histogram(prediction[:, density_mask], x_edges, z_edges)
            )
            in_range_fractions.append(
                xz_in_range_fraction(prediction[:, density_mask], x_edges, z_edges)
            )
            if method in autonomous_methods:
                return_mask = times_lt >= return_start_lt - 1e-12
                pairs = return_map_pairs(prediction[:, return_mask])
                rng = np.random.default_rng(np.random.SeedSequence([
                    seed, method_index[method], int(row.data_seed), int(row.model_seed),
                ]))
                method_pairs.append(_balanced_pairs(pairs, return_pairs_per_run, rng))
        arrays[f"density__{method}"] = np.mean(histograms, axis=0)
        arrays[f"density_in_range_fraction__{method}"] = np.asarray(
            np.mean(in_range_fractions)
        )
        if method in autonomous_methods:
            arrays[f"return__{method}"] = (
                np.concatenate(method_pairs, axis=0)
                if method_pairs else np.empty((0, 2), dtype=np.float64)
            )

    representative = select_representative_cell(trajectory_metrics, noise_level)
    representative_metadata: dict[str, Any] = dict(representative)
    representative_metadata["methods"] = {}
    representative_times: np.ndarray | None = None
    representative_truth_x: np.ndarray | None = None
    for method in methods:
        matching = selected_runs[
            (selected_runs["method"] == method)
            & (selected_runs["data_seed"] == representative["data_seed"])
            & (selected_runs["model_seed"] == representative["model_seed"])
        ]
        if len(matching) != 1:
            raise ValueError(f"expected one representative run for {method}, found {len(matching)}")
        row = matching.iloc[0]
        run_id = str(row["run_id"])
        run_dir, manifest = manifests[run_id]
        prediction, times, error = _load_prediction(run_dir, manifest)
        truth = truth_by_seed[representative["data_seed"]]
        times_lt = times * float(truth["largest"])
        stop = min(density_stop_lt, float(times_lt[-1]))
        mask = times_lt <= stop + 1e-12
        trajectory_id = representative["trajectory_id"]
        if representative_times is None:
            representative_times = times_lt[mask]
            representative_truth_x = truth["states"][trajectory_id, : times.size][mask, 0]
        elif not np.allclose(representative_times, times_lt[mask], rtol=0.0, atol=1e-12):
            raise ValueError("representative methods do not share a common forecast grid")
        arrays[f"representative_prediction_x__{method}"] = prediction[trajectory_id, mask, 0]
        arrays[f"representative_error__{method}"] = error[trajectory_id, mask]
        trajectory_row = trajectory_metrics[
            (trajectory_metrics["run_id"] == run_id)
            & (trajectory_metrics["trajectory_id"] == trajectory_id)
        ]
        if len(trajectory_row) != 1:
            raise ValueError(f"missing representative trajectory metrics for {run_id}")
        representative_metadata["methods"][method] = {
            "run_id": run_id,
            "vpt_restricted_lt": float(trajectory_row.iloc[0]["vpt_restricted_lt"]),
            "vpt_censored": bool(trajectory_row.iloc[0]["vpt_censored"]),
        }
    assert representative_times is not None and representative_truth_x is not None
    arrays["representative_times_lt"] = representative_times
    arrays["representative_truth_x"] = representative_truth_x

    metadata = {
        "schema": "lorenz63-attractor-summary-v2",
        "noise_level": float(noise_level),
        "density_window_lt": [float(density_start_lt), float(density_stop_lt)],
        "return_start_lt": float(return_start_lt),
        "histogram_bins": int(histogram_bins),
        "return_pairs_per_run": int(return_pairs_per_run),
        "seed": int(seed),
        "methods": methods,
        "autonomous_methods": autonomous_methods,
        "representative": representative_metadata,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    atomic_save_npz(output_path, **arrays)
    manifest = {
        "schema": "lorenz63-attractor-summary-manifest-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "source_hash": source_hash(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "settings": metadata,
        "inputs": {
            "run_metrics": sha256_file(run_metrics_path),
            "trajectory_metrics": sha256_file(trajectory_metrics_path),
            "run_manifest_set": sha256_json(input_manifest_hashes),
            "test_files": sha256_json(test_hashes),
        },
        "artifact": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)},
    }
    atomic_write_json(output_path.with_suffix(".manifest.json"), manifest)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Lorenz63 prediction artifacts for paper diagnostics."
    )
    parser.add_argument("--runs-root", default="runs/v2")
    parser.add_argument("--data-root", default="data/v2")
    parser.add_argument("--run-metrics", default="results/v2/run_metrics.csv")
    parser.add_argument("--trajectory-metrics", default="results/v2/trajectory_metrics.csv")
    parser.add_argument("--output", default="results/v2/visualizations/attractor_summary.npz")
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--density-start-lt", type=float, default=2.0)
    parser.add_argument("--density-stop-lt", type=float, default=5.0)
    parser.add_argument("--return-start-lt", type=float, default=5.0)
    parser.add_argument("--histogram-bins", type=int, default=96)
    parser.add_argument("--return-pairs-per-run", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    path = generate_attractor_summary(
        runs_root=args.runs_root,
        data_root=args.data_root,
        run_metrics_path=args.run_metrics,
        trajectory_metrics_path=args.trajectory_metrics,
        output_path=args.output,
        noise_level=args.noise_level,
        density_start_lt=args.density_start_lt,
        density_stop_lt=args.density_stop_lt,
        return_start_lt=args.return_start_lt,
        histogram_bins=args.histogram_bins,
        return_pairs_per_run=args.return_pairs_per_run,
        seed=args.seed,
    )
    print(path)


if __name__ == "__main__":
    main()
