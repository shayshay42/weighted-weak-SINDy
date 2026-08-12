from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    atomic_save_npz, atomic_write_csv, atomic_write_json, command_line,
    environment_snapshot, sha256_file, sha256_json, source_hash, utc_now,
)
from .config import load_config
from .plot import make_track_figures


def paired_hierarchical_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    resamples: int = 10000,
    seed: int = 2026,
) -> pd.DataFrame:
    required = {"method", "data_seed", "model_seed", "trajectory_id", value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"bootstrap frame misses columns: {sorted(missing)}")
    methods = sorted(str(value) for value in frame["method"].unique())
    data_seeds = sorted(int(value) for value in frame["data_seed"].unique())
    model_seeds = sorted(int(value) for value in frame["model_seed"].unique())
    trajectories = sorted(int(value) for value in frame["trajectory_id"].unique())
    full_index = pd.MultiIndex.from_product(
        [methods, data_seeds, model_seeds, trajectories],
        names=["method", "data_seed", "model_seed", "trajectory_id"],
    )
    indexed = frame.set_index(list(full_index.names))[value_column]
    if indexed.index.duplicated().any():
        raise ValueError("duplicate method/split/model/trajectory rows in bootstrap input")
    aligned = indexed.reindex(full_index)
    if aligned.isna().any():
        missing_count = int(aligned.isna().sum())
        raise ValueError(f"paired bootstrap requires a complete run matrix; {missing_count} cells are missing")
    values = aligned.to_numpy(dtype=np.float64).reshape(
        len(methods), len(data_seeds), len(model_seeds), len(trajectories)
    )
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(methods)), dtype=np.float64)
    chunk_size = 128
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        count = stop - start
        split_indices = rng.integers(
            len(data_seeds), size=(count, len(data_seeds))
        )
        model_indices = rng.integers(
            len(model_seeds), size=(count, len(data_seeds), len(model_seeds))
        )
        trajectory_indices = rng.integers(
            len(trajectories),
            size=(count, len(data_seeds), len(model_seeds), len(trajectories)),
        )
        selected = values[
            :,
            split_indices[:, :, None, None],
            model_indices[:, :, :, None],
            trajectory_indices,
        ]
        draws[start:stop] = selected.mean(axis=(2, 3, 4)).T
    point = values.mean(axis=(1, 2, 3))
    rows = []
    for index, method in enumerate(methods):
        rows.append({
            "method": method, "metric": value_column, "estimate": float(point[index]),
            "ci_lower": float(np.quantile(draws[:, index], 0.025)),
            "ci_upper": float(np.quantile(draws[:, index], 0.975)),
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        })
    return pd.DataFrame(rows)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def discover_completed_runs(runs_root: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    discovered = []
    seen: set[str] = set()
    for manifest_path in sorted(Path(runs_root).rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "lorenz63-run-manifest-v2":
            continue
        if manifest.get("status") != "complete" or manifest.get("evaluation", {}).get("status") != "complete":
            continue
        run_id = str(manifest["run_id"])
        if run_id in seen:
            raise ValueError(f"duplicate completed run ID {run_id}")
        seen.add(run_id)
        discovered.append((manifest_path.parent, manifest))
    return discovered


def _validate_main_matrix(run_metrics: pd.DataFrame, config: dict[str, Any]) -> None:
    expected = {
        (method, int(data_seed), int(model_seed), float(noise))
        for method in config_method_names(config)
        for data_seed in config["final"]["data_seeds"]
        for model_seed in config["final"]["model_seeds"]
        for noise in config["data"]["noise_levels"]
    }
    actual = {
        (str(row.method), int(row.data_seed), int(row.model_seed), float(row.noise_level))
        for row in run_metrics.itertuples(index=False)
    }
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"incomplete final run matrix: missing={len(missing)}, unexpected={len(extra)}"
        )


def config_method_names(config: dict[str, Any] | None = None) -> list[str]:
    from .config import DEFAULT_CONFIG
    from .contracts import configured_methods
    return sorted(configured_methods(DEFAULT_CONFIG if config is None else config))


def _validate_ablation_matrix(
    ablation_runs: list[tuple[Path, dict[str, Any]]], config: dict[str, Any]
) -> None:
    expected = {
        (method, int(data_seed), float(noise), int(count))
        for method in ("sindy_strong", "sindy_weighted")
        for data_seed in config["final"]["data_seeds"]
        for noise in config["data"]["noise_levels"]
        for count in config["sindy"]["ablation_trajectory_counts"]
    }
    actual = set()
    for _run_dir, manifest in ablation_runs:
        match = re.search(r"__k(\d+)$", str(manifest["run_id"]))
        if match is None:
            raise ValueError(f"ablation run lacks trajectory count: {manifest['run_id']}")
        metrics_path = Path(manifest["evaluation"]["artifacts"]["run_metrics"]["path"])
        row = pd.read_csv(metrics_path).iloc[0]
        actual.add((str(row["method"]), int(row["data_seed"]), float(row["noise_level"]), int(match.group(1))))
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"incomplete SINDy ablation matrix: missing={len(missing)}, unexpected={len(extra)}"
        )


def _aggregate_curves(runs: list[tuple[Path, dict[str, Any]]], output_path: Path) -> None:
    groups: dict[tuple[str, float], list[tuple[str, Path]]] = {}
    for run_dir, manifest in runs:
        metrics = pd.read_csv(run_dir / "run_metrics.csv").iloc[0]
        key = (str(manifest["track"]), float(metrics["noise_level"]))
        groups.setdefault(key, []).append((str(manifest["method"]), run_dir / "curve.npz"))
    arrays: dict[str, np.ndarray] = {}
    index_records = []
    for group_index, ((track, noise), records) in enumerate(sorted(groups.items())):
        by_method: dict[str, list[Path]] = {}
        for method, path in records:
            by_method.setdefault(method, []).append(path)
        for method_index, (method, paths) in enumerate(sorted(by_method.items())):
            loaded = [np.load(path, allow_pickle=False) for path in paths]
            maximum = min(float(item["times_lt"][-1]) for item in loaded)
            common = loaded[0]["times_lt"]
            common = common[common <= maximum + 1e-12]
            run_curves = np.vstack([
                np.interp(common, item["times_lt"], item["median"]) for item in loaded
            ])
            prefix = f"g{group_index}_m{method_index}"
            arrays[f"{prefix}_times_lt"] = common
            arrays[f"{prefix}_median"] = np.median(run_curves, axis=0)
            arrays[f"{prefix}_q25"] = np.quantile(run_curves, 0.25, axis=0)
            arrays[f"{prefix}_q75"] = np.quantile(run_curves, 0.75, axis=0)
            index_records.append({"prefix": prefix, "track": track, "noise_level": noise, "method": method})
            for item in loaded:
                item.close()
    arrays["index_json"] = np.asarray(json.dumps(index_records, sort_keys=True))
    atomic_save_npz(output_path, **arrays)


def aggregate_runs(
    *,
    config_path: str | Path | None,
    runs_root: str | Path,
    output_dir: str | Path,
    make_figures: bool = True,
) -> dict[str, Path]:
    config = load_config(config_path)
    runs = discover_completed_runs(runs_root)
    if not runs:
        raise ValueError(f"no completed evaluated v2 runs under {runs_root}")
    ablation_runs = [(run_dir, manifest) for run_dir, manifest in runs if "__k" in str(manifest["run_id"])]
    runs = [(run_dir, manifest) for run_dir, manifest in runs if "__k" not in str(manifest["run_id"])]
    if not runs:
        raise ValueError("only SINDy ablation runs were found; no main benchmark runs are complete")
    run_frames = [_read_csv(run_dir / "run_metrics.csv") for run_dir, _ in runs]
    trajectory_frames = [_read_csv(run_dir / "trajectory_metrics.csv") for run_dir, _ in runs]
    run_metrics = pd.concat(run_frames, ignore_index=True)
    trajectory_metrics = pd.concat(trajectory_frames, ignore_index=True)
    if run_metrics["run_id"].duplicated().any():
        raise ValueError("duplicate run IDs in aggregate metrics")
    if bool(config["aggregation"].get("require_complete_matrix", True)):
        _validate_main_matrix(run_metrics, config)
    if bool(config["aggregation"].get("require_complete_ablation", True)):
        _validate_ablation_matrix(ablation_runs, config)
    trajectory_metrics = trajectory_metrics.merge(
        run_metrics[["run_id", "noise_level"]], on="run_id", how="left", validate="many_to_one"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics_path = output_dir / "run_metrics.csv"
    trajectory_metrics_path = output_dir / "trajectory_metrics.csv"
    atomic_write_csv(run_metrics_path, run_metrics.to_dict(orient="records"))
    atomic_write_csv(trajectory_metrics_path, trajectory_metrics.to_dict(orient="records"))
    ablation_path = output_dir / "sindy_ablation_metrics.csv"
    ablation_frames = [_read_csv(run_dir / "run_metrics.csv") for run_dir, _ in ablation_runs]
    atomic_write_csv(
        ablation_path,
        pd.concat(ablation_frames, ignore_index=True).to_dict(orient="records") if ablation_frames else [],
    )

    bootstrap_rows = []
    bootstrap_cfg = config["aggregation"]
    candidate_metrics = [
        column for column in trajectory_metrics.columns
        if column == "vpt_restricted_lt" or column.startswith("nrmse_auc_")
    ]
    for (_track, _noise), group in trajectory_metrics.groupby(["track", "noise_level"]):
        for metric in candidate_metrics:
            observed_group = group[~group[metric].isna()]
            if observed_group.empty:
                continue
            result = paired_hierarchical_bootstrap(
                observed_group, metric,
                resamples=int(bootstrap_cfg["bootstrap_resamples"]),
                seed=int(bootstrap_cfg["bootstrap_seed"]),
            )
            result["track"] = _track
            result["noise_level"] = _noise
            bootstrap_rows.extend(result.to_dict(orient="records"))
        run_metric_candidates = [
            column for column in (
                "stability_fraction", "rq_mmd", "wasserstein_x", "wasserstein_y",
                "wasserstein_z", "covariance_relative_error",
                "lyapunov_spectrum_relative_error", "parameter_relative_error",
                "coefficient_relative_error", "training_wall_time_seconds",
                "inference_seconds_per_trajectory",
            ) if column in run_metrics.columns
        ]
        run_group = run_metrics[
            (run_metrics["track"] == _track) & (run_metrics["noise_level"] == _noise)
        ]
        for metric in run_metric_candidates:
            observed = run_group[~run_group[metric].isna()].copy()
            if observed.empty:
                continue
            observed["trajectory_id"] = 0
            try:
                result = paired_hierarchical_bootstrap(
                    observed, metric,
                    resamples=int(bootstrap_cfg["bootstrap_resamples"]),
                    seed=int(bootstrap_cfg["bootstrap_seed"]),
                )
            except ValueError:
                # Parameter metrics intentionally apply only to a subset of methods.
                continue
            result["track"] = _track
            result["noise_level"] = _noise
            bootstrap_rows.extend(result.to_dict(orient="records"))
    bootstrap_path = output_dir / "bootstrap_summary.csv"
    atomic_write_csv(bootstrap_path, bootstrap_rows)
    curves_path = output_dir / "aggregate_curves.npz"
    _aggregate_curves(runs, curves_path)
    figure_paths = (
        make_track_figures(run_metrics_path, curves_path, output_dir / "figures", config)
        if make_figures else []
    )
    comparison_summary_path: Path | None = None
    if make_figures and {
        "sindy_weak", "sindy_weak_weighted", "panda_zero_shot"
    }.issubset(set(run_metrics["method"].astype(str))):
        from .panda_comparison import make_panda_comparison_figures

        comparison_paths, comparison_summary_path = make_panda_comparison_figures(
            run_metrics=run_metrics,
            trajectory_metrics=trajectory_metrics,
            curves_path=curves_path,
            output_dir=output_dir / "figures",
            config=config,
        )
        figure_paths.extend(comparison_paths)
    manifest_path = output_dir / "manifest.json"
    input_manifest_hashes = {
        str(manifest["run_id"]): sha256_file(run_dir / "manifest.json")
        for run_dir, manifest in [*runs, *ablation_runs]
    }
    aggregate_manifest = {
        "schema": "lorenz63-aggregate-manifest-v2", "status": "complete",
        "completed_at": utc_now(), "run_count": len(runs), "ablation_run_count": len(ablation_runs),
        "run_ids": sorted(run_metrics["run_id"].astype(str).tolist()),
        "config_hash": sha256_json(config), "source_hash": source_hash(),
        "input_manifest_set_hash": sha256_json(input_manifest_hashes),
        "command": command_line(), "environment": environment_snapshot(),
        "seeds": {
            "data": list(config["final"]["data_seeds"]),
            "model": list(config["final"]["model_seeds"]),
            "bootstrap": int(config["aggregation"]["bootstrap_seed"]),
        },
        "artifacts": {},
    }
    for name, path in {
        "run_metrics": run_metrics_path, "trajectory_metrics": trajectory_metrics_path,
        "bootstrap_summary": bootstrap_path, "aggregate_curves": curves_path,
        "sindy_ablation_metrics": ablation_path,
    }.items():
        aggregate_manifest["artifacts"][name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    for index, path in enumerate(figure_paths):
        aggregate_manifest["artifacts"][f"figure_{index:02d}"] = {
            "path": str(path.resolve()), "sha256": sha256_file(path),
        }
    if comparison_summary_path is not None:
        aggregate_manifest["artifacts"]["panda_comparison_summary"] = {
            "path": str(comparison_summary_path.resolve()),
            "sha256": sha256_file(comparison_summary_path),
        }
    atomic_write_json(manifest_path, aggregate_manifest)
    outputs = {
        "run_metrics": run_metrics_path, "trajectory_metrics": trajectory_metrics_path,
        "bootstrap": bootstrap_path, "curves": curves_path,
        "sindy_ablation_metrics": ablation_path, "manifest": manifest_path,
    }
    if comparison_summary_path is not None:
        outputs["panda_comparison_summary"] = comparison_summary_path
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate completed Lorenz63 benchmark v2 runs.")
    parser.add_argument("--config", default="configs/v2/lorenz63_paper.json")
    parser.add_argument("--runs-root", default="runs/v2")
    parser.add_argument("--output-dir", default="results/v2")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    outputs = aggregate_runs(
        config_path=args.config, runs_root=args.runs_root, output_dir=args.output_dir,
        make_figures=not args.no_figures,
    )
    print(*outputs.values(), sep="\n")


if __name__ == "__main__":
    main()
