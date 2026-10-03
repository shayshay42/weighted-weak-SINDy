from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aggregate import _validate_ablation_matrix, _validate_main_matrix, discover_completed_runs
from .artifacts import (
    atomic_write_json, sha256_file, source_hash, utc_now, validate_artifact_hashes,
)
from .config import load_config
from .contracts import configured_methods


def _dataset_record_path(root: Path, record: dict[str, Any]) -> Path:
    relative = record.get("path", record.get("train_path"))
    if relative is None:
        raise ValueError("dataset file record has neither 'path' nor 'train_path'")
    return root / str(relative)


def validate_benchmark(
    *,
    config_path: str | Path | None,
    data_root: str | Path,
    runs_root: str | Path,
    results_dir: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    data_root = Path(data_root)
    for seed in config["final"]["data_seeds"]:
        manifest_path = data_root / f"split_seed{seed}" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        largest = float(manifest["largest_lyapunov_exponent"])
        lower = float(config["acceptance"]["largest_lyapunov_min"])
        upper = float(config["acceptance"]["largest_lyapunov_max"])
        record(f"dataset_seed_{seed}_complete", manifest.get("status") == "complete", manifest.get("status", "missing"))
        record(f"dataset_seed_{seed}_lyapunov", lower <= largest <= upper, f"lambda_max={largest:.8g}")
        file_records = [manifest["files"]["validation"], manifest["files"]["test"]]
        file_records.extend(manifest["files"]["train"].values())
        hashes_ok = all(
            sha256_file(_dataset_record_path(manifest_path.parent, item)) == item["sha256"]
            for item in file_records
        )
        record(f"dataset_seed_{seed}_hashes", hashes_ok, f"files={len(file_records)}")

    discovered = discover_completed_runs(runs_root)
    main_runs = [(path, manifest) for path, manifest in discovered if "__k" not in str(manifest["run_id"])]
    ablation_runs = [(path, manifest) for path, manifest in discovered if "__k" in str(manifest["run_id"])]
    run_metrics = pd.concat([pd.read_csv(path / "run_metrics.csv") for path, _ in main_runs], ignore_index=True)
    try:
        _validate_main_matrix(run_metrics, config)
        record("complete_run_matrix", True, f"runs={len(run_metrics)}")
    except ValueError as error:
        record("complete_run_matrix", False, str(error))
    if bool(config["aggregation"].get("require_complete_ablation", True)):
        try:
            _validate_ablation_matrix(ablation_runs, config)
            record("complete_sindy_ablation", True, f"runs={len(ablation_runs)}")
        except ValueError as error:
            record("complete_sindy_ablation", False, str(error))
    else:
        record("complete_sindy_ablation", True, "skipped: ablation not configured")
    record(
        "unique_run_ids", not run_metrics["run_id"].duplicated().any(),
        f"unique={run_metrics['run_id'].nunique()}, rows={len(run_metrics)}",
    )
    artifact_hashes_ok = all(validate_artifact_hashes(path / "manifest.json") for path, _ in discovered)
    record("training_artifact_hashes", artifact_hashes_ok, f"manifests={len(discovered)}")
    evaluation_hashes_ok = True
    for _path, manifest in discovered:
        for artifact in manifest["evaluation"]["artifacts"].values():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                evaluation_hashes_ok = False
                break
    record("evaluation_artifact_hashes", evaluation_hashes_ok, f"manifests={len(discovered)}")

    noiseless = run_metrics[np.isclose(run_metrics["noise_level"], 0.0)]
    selected_methods = set(configured_methods(config))
    if "solver_oracle" in selected_methods:
        oracle = noiseless[noiseless["method"] == "solver_oracle"]
        oracle_ok = (
            not oracle.empty
            and np.allclose(oracle["full_horizon_vpt_censoring_fraction"], 1.0)
            and np.all(oracle["full_forecast_horizon_lt"] > 5.0)
        )
        record("oracle_reaches_valid_horizon", oracle_ok, f"runs={len(oracle)}")
    else:
        record("oracle_reaches_valid_horizon", True, "skipped: oracle not configured")

    coefficient_methods = selected_methods.intersection({"sindy_strong", "sindy_weighted"})
    if coefficient_methods:
        sindy = noiseless[noiseless["method"].isin(coefficient_methods)]
        sindy_limit = float(config["acceptance"]["sindy_coefficient_relative_error"])
        sindy_max = (
            float(sindy["coefficient_relative_error"].max())
            if not sindy.empty else float("inf")
        )
        record(
            "clean_sindy_coefficients", sindy_max <= sindy_limit,
            f"max_relative_error={sindy_max:.8g}",
        )
    else:
        record(
            "clean_sindy_coefficients", True,
            "skipped: strong-form coefficient-recovery methods not configured",
        )

    ad_methods = selected_methods.intersection({"lorenz_ad", "lorenz_ad_tapered"})
    if ad_methods:
        ad = noiseless[noiseless["method"].isin(ad_methods)]
        ad_limit = float(config["acceptance"]["ad_parameter_relative_error"])
        ad_max = (
            float(ad["parameter_relative_error"].max())
            if not ad.empty else float("inf")
        )
        record("clean_ad_parameters", ad_max <= ad_limit, f"max_relative_error={ad_max:.8g}")
    else:
        record("clean_ad_parameters", True, "skipped: AD methods not configured")
    nonfinite_runs = 0
    oracle_predictions_finite = True
    for path, manifest in main_runs:
        with np.load(path / "predictions.npz", allow_pickle=False) as loaded:
            if not np.all(np.isfinite(loaded["prediction"])):
                nonfinite_runs += 1
                if manifest["method"] == "solver_oracle":
                    oracle_predictions_finite = False
    record(
        "oracle_predictions_finite", oracle_predictions_finite,
        f"unstable_nonoracle_runs={nonfinite_runs}",
    )

    results_manifest = Path(results_dir) / "manifest.json"
    aggregate = json.loads(results_manifest.read_text(encoding="utf-8"))
    record("aggregate_complete", aggregate.get("status") == "complete", aggregate.get("status", "missing"))
    passed = all(check["passed"] for check in checks)
    report = {
        "schema": "lorenz63-acceptance-report-v2", "status": "passed" if passed else "failed",
        "completed_at": utc_now(), "source_hash": source_hash(), "checks": checks,
        "note": "Seeded reproducibility is enforced by tests/v2/test_reproducibility.py.",
    }
    atomic_write_json(Path(results_dir) / "acceptance_report.json", report)
    if not passed:
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"benchmark acceptance failed: {failed}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate all Lorenz63 v2 acceptance gates.")
    parser.add_argument("--config", default="configs/v2/lorenz63_frozen.json")
    parser.add_argument("--data-root", default="data/v2")
    parser.add_argument("--runs-root", default="runs/v2")
    parser.add_argument("--results-dir", default="results/v2")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = validate_benchmark(
        config_path=args.config, data_root=args.data_root,
        runs_root=args.runs_root, results_dir=args.results_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
