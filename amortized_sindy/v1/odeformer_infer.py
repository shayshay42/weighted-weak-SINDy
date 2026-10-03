from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import sha256_file
from .io import array_sha256, atomic_save_npz, atomic_write_json
from .library import polynomial_exponents
from .multifamily_data import DATASET_SCHEMA
from .multifamily_infer import CONTEXT_KEYS
from .odeformer_gate import (
    ODEFORMER_GATE_SCHEMA,
    _load_odeformer,
    adapt_symbolic_equations_to_sindy,
    forecast_odeformer,
)


FROZEN_CONFIG_SCHEMA = "amortized-sindy-odeformer-frozen-config-v1"


def _load_frozen_config(
    path: str | Path,
    *,
    selection_manifest_path: str | Path,
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != {"schema", "model", "inference", "selection"}:
        raise ValueError("frozen ODEFormer config contains unexpected sections")
    if config.get("schema") != FROZEN_CONFIG_SCHEMA:
        raise ValueError("invalid frozen ODEFormer config schema")
    model = config["model"]
    if model.get("pretraining_exposure") not in {
        "verified_excluded", "unknown", "known_exposed"
    }:
        raise ValueError("invalid ODEFormer pretraining exposure label")
    inference = config["inference"]
    required_inference = {
        "candidate_count": 8,
        "sampling_temperature": 0.1,
        "coefficient_adapter": (
            "analytic_degree_2_taylor_at_context_coordinate_mean"
        ),
        "library_dimension": 3,
        "library_degree": 2,
        "target_optimizer_steps": 0,
        "target_sparse_regression_solves": 0,
    }
    permitted_reranking = {
        "observed_context_snmse",
        "observed_context_taylor_sindy_nmse_with_public_grid_stability",
    }
    actual_reranking = inference.get("context_rerank")
    remaining_inference = dict(inference)
    remaining_inference.pop("context_rerank", None)
    if (
        remaining_inference != required_inference
        or actual_reranking not in permitted_reranking
    ):
        raise ValueError("ODEFormer inference settings differ from the frozen gate")
    selection = config["selection"]
    if selection.get("gate_passed") is not True:
        raise ValueError("source-family gate did not pass")
    if selection.get("target_families_used_for_selection") != []:
        raise ValueError("target families must be absent from model selection")
    if not (
        selection["conditioner_mse"] < selection["raw_gru_mse"]
        and selection["conditioner_mse"] < selection["constant_field_mse"]
        and selection.get("all_predictions_finite") is True
    ):
        raise ValueError("frozen conditioner did not beat both registered baselines")
    if sha256_file(selection_manifest_path) != selection[
        "source_gate_manifest_sha256"
    ]:
        raise ValueError("source gate manifest hash differs from frozen selection")
    gate = json.loads(Path(selection_manifest_path).read_text(encoding="utf-8"))
    if gate.get("schema") != ODEFORMER_GATE_SCHEMA:
        raise ValueError("invalid ODEFormer source gate manifest")
    if gate.get("metrics", {}).get("all_predictions_finite") is not True:
        raise ValueError("source gate contains failed rollouts")
    if not np.isclose(
        gate["metrics"]["mean_normalized_mse"], selection["conditioner_mse"]
    ):
        raise ValueError("source gate score differs from frozen selection")
    return config


def infer_odeformer_heldout_family(
    *,
    frozen_config_path: str | Path,
    selection_manifest_path: str | Path,
    checkpoint_path: str | Path,
    external_source_path: str | Path,
    context_path: str | Path,
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    seed: int = 2026,
    force: bool = False,
) -> Path:
    """Infer held-out SINDy fields without accepting a truth artifact."""
    config = _load_frozen_config(
        frozen_config_path,
        selection_manifest_path=selection_manifest_path,
    )
    if sha256_file(checkpoint_path) != config["model"]["checkpoint_sha256"]:
        raise ValueError("ODEFormer checkpoint hash differs from frozen config")
    dataset_manifest = json.loads(
        Path(dataset_manifest_path).read_text(encoding="utf-8")
    )
    if dataset_manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid multifamily dataset manifest")
    if sha256_file(context_path) != dataset_manifest["artifacts"][
        "lorenz_heldout_context"
    ]["sha256"]:
        raise ValueError("held-out context does not match the dataset manifest")
    with np.load(context_path, allow_pickle=False) as loaded:
        if set(loaded.files) != CONTEXT_KEYS:
            raise ValueError("held-out context file contains unexpected arrays")
        values = {name: np.asarray(loaded[name]) for name in loaded.files}
    contexts = np.asarray(values["context_states"], dtype=np.float64)
    times = np.asarray(values["context_times"], dtype=np.float64)
    offsets = np.asarray(values["forecast_offsets"], dtype=np.float64)
    if contexts.ndim != 3 or contexts.shape[-1] != 3:
        raise ValueError("held-out contexts must have shape [trajectory,time,3]")
    if times.shape != (contexts.shape[1],) or offsets.ndim != 1:
        raise ValueError("held-out context time grids are invalid")
    if not np.isfinite(contexts).all():
        raise ValueError("held-out contexts contain non-finite states")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"held-out output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    regressor, parameter_count = _load_odeformer(
        checkpoint_path=checkpoint_path,
        external_source_path=external_source_path,
        external_revision=config["model"]["external_source_revision"],
        device=device,
        beam_size=config["inference"]["candidate_count"],
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    symbolic_predictions, symbolic_failed, equations, _, errors = forecast_odeformer(
        contexts,
        context_times=times,
        forecast_offsets=offsets,
        regressor=regressor,
        seed=seed,
        context_rerank=True,
    )
    predictions, failed, coefficients = adapt_symbolic_equations_to_sindy(
        equations,
        contexts=contexts,
        forecast_offsets=offsets,
    )
    elapsed = time.perf_counter() - started
    support = np.where(np.isfinite(coefficients) & (np.abs(coefficients) > 1e-10), 1.0, 0.0)
    prediction_path = atomic_save_npz(
        output_dir / "predictions.npz",
        predictions=predictions,
        symbolic_predictions=symbolic_predictions,
        physical_coefficients=coefficients,
        support_probabilities=support,
        equations=equations,
        errors=errors,
        forecast_offsets=offsets,
        group_ids=np.asarray(values["group_ids"], dtype=np.int64),
        trajectory_ids=np.asarray(values["trajectory_ids"], dtype=np.int64),
    )
    return atomic_write_json(output_dir / "inference_manifest.json", {
        "schema": "amortized-sindy-heldout-inference-v1",
        "status": "complete",
        "method": "odeformer_context_conditioned_taylor_sindy_v1",
        "zero_shot": True,
        "target_time_zero_update": True,
        "strict_dataset_zero_shot": (
            config["model"]["pretraining_exposure"] == "verified_excluded"
        ),
        "target_pretraining_exposure": config["model"]["pretraining_exposure"],
        "target_optimizer_steps": 0,
        "target_sparse_regression_solves": 0,
        "future_truth_path_accepted": False,
        "context_candidate_reranking": True,
        "coefficient_adapter": config["inference"]["coefficient_adapter"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "external_source_revision": config["model"]["external_source_revision"],
        "parameter_count": parameter_count,
        "context_sha256": sha256_file(context_path),
        "context_array_sha256": array_sha256(contexts),
        "frozen_config_sha256": sha256_file(frozen_config_path),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "prediction_artifact": {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
        },
        "trajectory_count": int(contexts.shape[0]),
        "failed_sindy_rollout_count": int(np.sum(failed)),
        "failed_symbolic_rollout_count": int(np.sum(symbolic_failed)),
        "library_exponents": [list(value) for value in polynomial_exponents(3, 2)],
        "runtime": {
            "inference_seconds": elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device.startswith("cuda")
                else 0
            ),
            "seed": seed,
        },
        "training_manifest": {
            "source": "official ODEFormer released checkpoint",
            "pretraining_exposure": config["model"]["pretraining_exposure"],
            "target_evaluation_data_used_for_repository_selection": False,
            "excluded_evaluation_dataset_ids": [],
        },
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen ODEFormer-conditioned SINDy without truth access."
    )
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--external-source", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(infer_odeformer_heldout_family(
        frozen_config_path=args.frozen_config,
        selection_manifest_path=args.selection_manifest,
        checkpoint_path=args.checkpoint,
        external_source_path=args.external_source,
        context_path=args.context,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
