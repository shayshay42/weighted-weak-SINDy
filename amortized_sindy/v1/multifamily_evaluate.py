from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoint import sha256_file
from .io import atomic_write_json
from .multifamily_data import DATASET_SCHEMA


def _artifact_matches(
    dataset_manifest: dict[str, object],
    name: str,
    path: str | Path,
) -> None:
    record = dataset_manifest["artifacts"][name]  # type: ignore[index]
    if sha256_file(path) != record["sha256"]:  # type: ignore[index]
        raise ValueError(f"{name} does not match the dataset manifest")


def evaluate_heldout_family(
    *,
    inference_manifest_path: str | Path,
    context_path: str | Path,
    truth_path: str | Path,
    hidden_systems_path: str | Path,
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads(
        Path(dataset_manifest_path).read_text(encoding="utf-8")
    )
    if dataset_manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid multifamily dataset manifest")
    _artifact_matches(dataset_manifest, "lorenz_heldout_context", context_path)
    _artifact_matches(dataset_manifest, "lorenz_heldout_truth", truth_path)
    _artifact_matches(dataset_manifest, "hidden_systems", hidden_systems_path)
    inference_manifest_path = Path(inference_manifest_path)
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    if inference_manifest.get("schema") != "amortized-sindy-heldout-inference-v1":
        raise ValueError("invalid held-out inference manifest")
    if inference_manifest.get("target_optimizer_steps") != 0:
        raise ValueError("held-out inference used target optimization")
    if inference_manifest.get("target_sparse_regression_solves") != 0:
        raise ValueError("held-out inference used target sparse regression")
    prediction_path = inference_manifest_path.parent / inference_manifest[
        "prediction_artifact"
    ]["path"]
    if sha256_file(prediction_path) != inference_manifest["prediction_artifact"]["sha256"]:
        raise ValueError("held-out prediction artifact hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        predictions = np.asarray(loaded["predictions"], dtype=np.float64)
        predicted_coefficients = np.asarray(
            loaded["physical_coefficients"], dtype=np.float64
        )
        support_probabilities = np.asarray(
            loaded["support_probabilities"], dtype=np.float64
        )
        group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        trajectory_ids = np.asarray(loaded["trajectory_ids"], dtype=np.int64)
        forecast_offsets = np.asarray(loaded["forecast_offsets"], dtype=np.float64)
    with np.load(context_path, allow_pickle=False) as loaded:
        contexts = np.asarray(loaded["context_states"], dtype=np.float64)
        context_group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        context_trajectory_ids = np.asarray(loaded["trajectory_ids"], dtype=np.int64)
    with np.load(truth_path, allow_pickle=False) as loaded:
        truth = np.asarray(loaded["future_states"], dtype=np.float64)
        truth_offsets = np.asarray(loaded["forecast_offsets"], dtype=np.float64)
        truth_group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        truth_trajectory_ids = np.asarray(loaded["trajectory_ids"], dtype=np.int64)
    if not (
        np.array_equal(group_ids, context_group_ids)
        and np.array_equal(group_ids, truth_group_ids)
        and np.array_equal(trajectory_ids, context_trajectory_ids)
        and np.array_equal(trajectory_ids, truth_trajectory_ids)
        and np.array_equal(forecast_offsets, truth_offsets)
    ):
        raise ValueError("held-out prediction/context/truth identities do not align")
    if predictions.shape != truth.shape:
        raise ValueError("held-out prediction and truth shapes do not align")
    with np.load(hidden_systems_path, allow_pickle=False) as loaded:
        hidden_group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        hidden_coefficients = np.asarray(loaded["coefficients"], dtype=np.float64)
        hidden_families = np.asarray(loaded["family_ids"])
    true_by_group = {
        int(group_id): hidden_coefficients[index]
        for index, group_id in enumerate(hidden_group_ids)
        if hidden_families[index] == "lorenz_heldout"
    }
    trajectory_rows = []
    support_threshold = 0.5
    for index, (group_id, trajectory_id) in enumerate(zip(group_ids, trajectory_ids)):
        true_coefficients = true_by_group[int(group_id)]
        coefficient_error = np.linalg.norm(
            predicted_coefficients[index] - true_coefficients
        ) / max(np.linalg.norm(true_coefficients), 1e-12)
        true_support = np.abs(true_coefficients) > 1e-12
        predicted_support = support_probabilities[index] >= support_threshold
        true_positive = int(np.sum(true_support & predicted_support))
        false_positive = int(np.sum(~true_support & predicted_support))
        false_negative = int(np.sum(true_support & ~predicted_support))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        support_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        denominator = max(float(np.sum(np.var(contexts[index], axis=0))), 1e-12)
        squared_error = np.sum((predictions[index] - truth[index]) ** 2, axis=1)
        normalized_mse = float(np.mean(squared_error / denominator))
        constant_squared_error = np.sum(
            (contexts[index, -1] - truth[index]) ** 2, axis=1
        )
        constant_normalized_mse = float(
            np.mean(constant_squared_error / denominator)
        )
        trajectory_rows.append({
            "group_id": int(group_id),
            "trajectory_id": int(trajectory_id),
            "normalized_mse": normalized_mse,
            "normalized_rmse": float(np.sqrt(normalized_mse)),
            "constant_field_normalized_mse": constant_normalized_mse,
            "coefficient_relative_error": float(coefficient_error),
            "support_precision": float(precision),
            "support_recall": float(recall),
            "support_f1": float(support_f1),
        })
    consistency_values = []
    for group_id in np.unique(group_ids):
        selected = predicted_coefficients[group_ids == group_id]
        center = np.mean(selected, axis=0)
        consistency_values.append(
            float(np.mean((selected - center) ** 2) / max(np.mean(center ** 2), 1e-12))
        )
    metrics = {
        "trajectory_count": len(trajectory_rows),
        "group_count": int(np.unique(group_ids).size),
        "mean_normalized_mse": float(np.mean([row["normalized_mse"] for row in trajectory_rows])),
        "median_normalized_mse": float(np.median([row["normalized_mse"] for row in trajectory_rows])),
        "mean_constant_field_normalized_mse": float(np.mean([
            row["constant_field_normalized_mse"] for row in trajectory_rows
        ])),
        "relative_mse_vs_constant_field": float(
            np.mean([row["normalized_mse"] for row in trajectory_rows])
            / max(
                np.mean([
                    row["constant_field_normalized_mse"] for row in trajectory_rows
                ]),
                1e-12,
            )
        ),
        "mean_coefficient_relative_error": float(np.mean([
            row["coefficient_relative_error"] for row in trajectory_rows
        ])),
        "median_coefficient_relative_error": float(np.median([
            row["coefficient_relative_error"] for row in trajectory_rows
        ])),
        "mean_support_f1": float(np.mean([row["support_f1"] for row in trajectory_rows])),
        "mean_within_group_coefficient_variation": float(np.mean(consistency_values)),
        "all_predictions_finite": bool(np.isfinite(predictions).all()),
    }
    rows_path = atomic_write_json(output_dir / "trajectory_metrics.json", trajectory_rows)
    return atomic_write_json(output_dir / "evaluation_manifest.json", {
        "schema": "amortized-sindy-heldout-evaluation-v1",
        "status": "complete",
        "family": "lorenz_heldout",
        "selection_role": "final_heldout_family_no_tuning",
        "metrics": metrics,
        "trajectory_metrics": {
            "path": rows_path.name,
            "sha256": sha256_file(rows_path),
        },
        "inference_manifest_sha256": sha256_file(inference_manifest_path),
        "truth_sha256": sha256_file(truth_path),
        "hidden_systems_sha256": sha256_file(hidden_systems_path),
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen held-out-family predictions with isolated truth."
    )
    parser.add_argument("--inference-manifest", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--hidden-systems", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(evaluate_heldout_family(
        inference_manifest_path=args.inference_manifest,
        context_path=args.context,
        truth_path=args.truth,
        hidden_systems_path=args.hidden_systems,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
