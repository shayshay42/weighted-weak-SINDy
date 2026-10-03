from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import array_sha256, atomic_write_json, json_safe


def evaluate_ctf_predictions(
    *,
    prediction_batch_manifest: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> Path:
    """Score an immutable prediction batch in a separate truth-aware process."""
    try:
        from ctf4science.eval_module import evaluate
    except ImportError as error:
        raise RuntimeError("install the pinned CTF4Science framework first") from error
    prediction_batch_manifest = Path(prediction_batch_manifest)
    manifest = json.loads(prediction_batch_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "ctf-amortized-sindy-prediction-batch-v1":
        raise ValueError("invalid prediction batch manifest")
    dataset = manifest["dataset"]
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"evaluation directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_results: list[dict[str, Any]] = []
    failed_pairs: list[int] = []
    raw_metric_values: list[float] = []
    for record in manifest["pairs"]:
        pair_id = int(record["pair_id"])
        prediction_path = Path(record["prediction_path"])
        if not prediction_path.is_absolute():
            prediction_path = prediction_batch_manifest.parent / prediction_path
        predictions = np.load(prediction_path, allow_pickle=False)
        inference_path = Path(record["inference_manifest_path"])
        if not inference_path.is_absolute():
            inference_path = prediction_batch_manifest.parent / inference_path
        inference = json.loads(
            inference_path.read_text(encoding="utf-8")
        )
        if inference.get("pair_id") != pair_id or inference.get("zero_shot") is not True:
            raise ValueError(f"invalid zero-shot inference manifest for pair {pair_id}")
        if inference.get("target_optimizer_steps") != 0:
            raise ValueError(f"target optimization was reported for pair {pair_id}")
        if inference.get("target_sparse_regression_solves") != 0:
            raise ValueError(f"target sparse regression was reported for pair {pair_id}")
        if inference.get("prediction_sha256") != array_sha256(predictions):
            raise ValueError(f"prediction hash mismatch for pair {pair_id}")
        if not np.isfinite(predictions).all():
            results = {
                "method_failure": True,
                "failure_reason": "nonfinite_prediction",
            }
            failed_pairs.append(pair_id)
        else:
            try:
                results = json_safe(evaluate(dataset, pair_id, predictions))
            except Exception as error:  # noqa: BLE001 - isolate benchmark pairs
                results = {
                    "method_failure": True,
                    "failure_reason": (
                        f"evaluator_error:{type(error).__name__}:{error}"
                    ),
                }
                failed_pairs.append(pair_id)
            else:
                for value in results.values():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        raw_metric_values.append(float(value))
        result_path = atomic_write_json(
            output_dir / f"pair_{pair_id}" / "evaluation_results.json", results
        )
        pair_results.append({
            "pair_id": pair_id,
            "prediction_path": record["prediction_path"],
            "evaluation_results_path": result_path.relative_to(output_dir).as_posix(),
            "metrics": results,
        })
    complete = not failed_pairs and len(raw_metric_values) == 12
    return atomic_write_json(output_dir / "evaluation_batch_manifest.json", {
        "schema": "ctf-amortized-sindy-evaluation-batch-v1",
        "dataset": dataset,
        "prediction_batch_manifest": str(prediction_batch_manifest),
        "pairs": pair_results,
        "aggregate": {
            "expected_metric_count": 12,
            "completed_metric_count": len(raw_metric_values),
            "failed_pair_ids": failed_pairs,
            "raw_metric_values": raw_metric_values,
            "official_clipped_composite": (
                float(np.mean(np.clip(raw_metric_values, -100.0, 100.0)))
                if complete
                else None
            ),
            "complete": complete,
        },
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen amortized-SINDy CTF predictions in a separate process."
    )
    parser.add_argument("--prediction-batch-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(evaluate_ctf_predictions(
        prediction_batch_manifest=args.prediction_batch_manifest,
        output_dir=args.output_dir,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
