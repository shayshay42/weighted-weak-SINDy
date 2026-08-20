from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    atomic_write_csv, atomic_write_json, command_line, environment_snapshot,
    sha256_file, sha256_json, source_hash, utc_now,
)
from .config import deep_update, load_config, save_config
from .contracts import CORE_METHODS, METHODS, TRACK_LABELS
from .data import load_split, noise_label
from .metrics import normalized_rmse_auc, normalized_squared_error
from .registry import load_method
from .train import run_training


MATCHED_SINDY_METHODS = {
    "sindy_weighted": "sindy_strong",
    "sindy_weak_weighted": "sindy_weak",
}


def _candidate_overrides(method: str, config: dict[str, Any], noise_level: float) -> list[dict[str, Any]]:
    tuning = config["tuning"]
    if method == "node_soft_dtw":
        return [{"node": {"gamma": float(value)}} for value in tuning["soft_dtw_gamma"]]
    if method == "node_strong" and noise_level > 0:
        polyorder = int(config["node"]["savgol_polyorder"])
        return [
            {"node": {"savgol_window": int(value)}}
            for value in tuning["savgol_window"] if int(value) > polyorder
        ]
    if method == "sindy_strong":
        derivative_windows = (
            tuning["savgol_window"] if noise_level > 0
            else [config["sindy"]["savgol_window"]]
        )
        return [
            {"sindy": {
                "degree": int(degree), "threshold": float(threshold),
                "ridge": float(ridge), "savgol_window": int(window),
            }}
            for degree, threshold, ridge, window in itertools.product(
                tuning["sindy_degree"], tuning["sindy_threshold"],
                tuning["sindy_ridge"], derivative_windows,
            )
        ]
    if method == "sindy_weak":
        return [
            {"sindy": {
                "degree": int(degree), "threshold": float(threshold),
                "ridge": float(ridge),
            }}
            for degree, threshold, ridge in itertools.product(
                tuning["sindy_degree"], tuning["sindy_threshold"], tuning["sindy_ridge"]
            )
        ]
    if method in MATCHED_SINDY_METHODS:
        return []
    return [{}]


def _score_validation(
    checkpoint: Path,
    method: str,
    config: dict[str, Any],
    train: dict[str, Any],
    validation: dict[str, Any],
    device: str,
) -> float:
    adapter = load_method(checkpoint, method, config, device=device)
    largest = float(train["metadata"]["normalization"]["time_scale"])
    times = np.asarray(validation["times"], dtype=np.float64)
    stop = int(np.searchsorted(largest * times, 2.0, side="left"))
    stop = min(stop, times.size - 1)
    selected_times = times[: stop + 1]
    truth = np.asarray(validation["states"], dtype=np.float64)[:, : stop + 1]
    prediction = adapter.forecast(truth[:, 0], selected_times)
    error = normalized_squared_error(
        prediction, truth,
        np.asarray(train["metadata"]["normalization"]["state_std"], dtype=np.float64),
    )
    auc = normalized_rmse_auc(error, selected_times * largest, 2.0)
    return float(np.nanmedian(auc))


def tune_methods(
    *,
    config_path: str | Path | None,
    train_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path,
    output_config_path: str | Path,
    methods: list[str],
    model_seed: int = 0,
    device: str = "auto",
) -> Path:
    config = load_config(config_path)
    train = load_split(train_path, "train")
    validation = load_split(validation_path, "validation")
    development_seed = int(config["tuning"]["development_seed"])
    if int(train["metadata"]["data_seed"]) != development_seed:
        raise ValueError(f"tuning is restricted to development data seed {development_seed}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    selected: dict[str, tuple[float, dict[str, Any]]] = {}
    noise_level = float(train["metadata"]["noise_level"])
    label = noise_label(noise_level)
    existing_method_overrides = config.get("method_overrides", {}).get(label, {})
    tuning_manifest = {
        "schema": "lorenz63-tuning-manifest-v2", "status": "running",
        "started_at": utc_now(), "development_data_seed": development_seed,
        "model_seed": model_seed, "noise_level": noise_level, "methods": methods,
        "command": command_line(), "environment": environment_snapshot(),
        "source_hash": source_hash(), "config_hash": sha256_json(config),
        "input_hashes": {
            "train": sha256_file(train_path), "validation": sha256_file(validation_path),
        },
    }
    atomic_write_json(output_dir / "manifest.json", tuning_manifest)

    def run_candidate(method: str, index: int, override: dict[str, Any]) -> float:
        candidate_method_override = deep_update(
            existing_method_overrides.get(method, {}), override
        )
        candidate_config = deep_update(config, {
            "method_overrides": {label: {method: candidate_method_override}}
        })
        candidate_epochs = int(config["tuning"]["candidate_epochs"])
        candidate_steps = int(config["tuning"]["candidate_steps_per_epoch"])
        kind = METHODS[method].kind
        if kind == "node":
            candidate_config = deep_update(candidate_config, {
                "node": {"epochs": candidate_epochs, "steps_per_epoch": candidate_steps}
            })
        elif kind == "pinn":
            candidate_config = deep_update(candidate_config, {
                "pinn": {"epochs": candidate_epochs, "steps_per_epoch": candidate_steps}
            })
        elif kind == "parametric":
            candidate_config = deep_update(candidate_config, {
                "parametric": {"epochs": candidate_epochs, "steps_per_epoch": candidate_steps}
            })
        candidate_hash = sha256_json(override)[:10]
        candidate_root = output_dir / method / f"candidate_{index:03d}_{candidate_hash}"
        candidate_config_path = candidate_root / "input_config.json"
        save_config(candidate_config, candidate_config_path)
        checkpoint = run_training(
            config_path=candidate_config_path, train_path=train_path,
            validation_path=validation_path, method=method,
            data_seed=development_seed, model_seed=model_seed,
            output_dir=candidate_root / "run", device=device,
        )
        score = _score_validation(
            checkpoint, method, candidate_config, train, validation, device="cpu"
        )
        records.append({
            "method": method, "candidate": index, "candidate_hash": candidate_hash,
            "override_json": json.dumps(override, sort_keys=True),
            "median_nrmse_auc_0_2LT": score,
        })
        return score

    for method in methods:
        if method in MATCHED_SINDY_METHODS:
            continue
        for index, override in enumerate(_candidate_overrides(method, config, noise_level)):
            score = run_candidate(method, index, override)
            if method not in selected or score < selected[method][0]:
                selected[method] = (score, override)

    for method, parent in MATCHED_SINDY_METHODS.items():
        if method not in methods:
            continue
        matched_override = selected.get(
            parent, (float("inf"), existing_method_overrides.get(parent, {}))
        )[1]
        selected[method] = (
            run_candidate(method, 0, matched_override), matched_override
        )

    selected_method_overrides = dict(existing_method_overrides)
    for method, (_score, override) in selected.items():
        selected_method_overrides[method] = deep_update(
            existing_method_overrides.get(method, {}), override
        )
    frozen = deep_update(config, {
        "method_overrides": {label: selected_method_overrides}
    })
    method_scores = {method: selected.get(method, (float("inf"), {}))[0] for method in methods}
    plot_order: dict[str, list[str]] = {}
    for track in TRACK_LABELS:
        track_methods = [method for method in methods if METHODS[method].track == track]
        plot_order[track] = sorted(track_methods, key=lambda method: (method_scores[method], method))
    frozen.setdefault("plot_order_by_noise", {})[label] = plot_order
    frozen.setdefault("tuning_result", {}).setdefault("by_noise", {})[label] = {
        "completed_at": utc_now(), "development_data_seed": development_seed,
        "noise_level": noise_level, "model_seed": model_seed,
        "objective": config["tuning"]["objective"], "selected": {
            method: {"score": score, "override": override}
            for method, (score, override) in selected.items()
        },
    }
    output_config_path = Path(output_config_path)
    save_config(frozen, output_config_path)
    atomic_write_csv(output_dir / "tuning_results.csv", records)
    tuning_manifest.update({
        "status": "complete", "completed_at": utc_now(),
        "output_config": str(output_config_path.resolve()),
        "output_config_sha256": sha256_file(output_config_path),
        "candidate_count": len(records),
    })
    atomic_write_json(output_dir / "manifest.json", tuning_manifest)
    return output_config_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune Lorenz63 v2 methods on development seed zero only.")
    parser.add_argument("--config", default="configs/v2/lorenz63_paper.json")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", default="tuning/v2")
    parser.add_argument("--output-config", default="configs/v2/lorenz63_frozen.json")
    parser.add_argument(
        "--methods", nargs="+", choices=sorted(METHODS), default=sorted(CORE_METHODS)
    )
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    path = tune_methods(
        config_path=args.config, train_path=args.train, validation_path=args.validation,
        output_dir=args.output_dir, output_config_path=args.output_config,
        methods=args.methods, model_seed=args.model_seed, device=args.device,
    )
    print(path)


if __name__ == "__main__":
    main()
