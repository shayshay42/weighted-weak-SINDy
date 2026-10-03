from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import sha256_file
from .io import array_sha256, atomic_save_npy, atomic_write_json
from .odeformer_gate import _load_odeformer, condition_odeformer_sindy
from .odeformer_infer import _load_frozen_config


def _one_trajectory(train_data: Any) -> np.ndarray:
    if not isinstance(train_data, (list, tuple)) or len(train_data) != 1:
        raise ValueError("CTF non-parametric pair must provide one trajectory")
    value = np.asarray(train_data[0], dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("CTF trajectory must have shape [time,state]")
    return value


def select_ctf_context(
    *,
    pair_id: int,
    train_data: Any,
    init_data: np.ndarray | None,
    context_length: int,
) -> tuple[np.ndarray, int]:
    if context_length < 2:
        raise ValueError("CTF context length must be at least two")
    if pair_id in (8, 9):
        if init_data is None:
            raise ValueError("CTF parametric pairs require warm-start context")
        context = np.asarray(init_data, dtype=np.float64)[-context_length:]
        initial_index = -1
    else:
        trajectory = _one_trajectory(train_data)
        if pair_id in (2, 4):
            context = trajectory[:context_length]
            initial_index = 0
        else:
            context = trajectory[-context_length:]
            initial_index = -1
    if context.ndim != 2 or context.shape[1] != 3 or context.shape[0] < 2:
        raise ValueError("CTF context has an invalid shape")
    if not np.isfinite(context).all():
        raise ValueError("CTF context contains non-finite values")
    return context, initial_index


def predict_odeformer_ctf_pairs(
    *,
    frozen_config_path: str | Path,
    selection_manifest_path: str | Path,
    checkpoint_path: str | Path,
    external_source_path: str | Path,
    output_dir: str | Path,
    context_length: int = 128,
    device: str = "cuda:0",
    seed: int = 2026,
    force: bool = False,
) -> Path:
    """Generate open CTF dev predictions without importing its evaluator."""
    try:
        from ctf4science.data_module import (
            get_prediction_timesteps,
            load_dataset,
        )
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("the pinned CTF4Science framework is required") from error
    config = _load_frozen_config(
        frozen_config_path,
        selection_manifest_path=selection_manifest_path,
    )
    if config["inference"]["context_rerank"] != (
        "observed_context_taylor_sindy_nmse_with_public_grid_stability"
    ):
        raise ValueError("CTF ODEFormer adapter requires post-adapter reranking")
    if sha256_file(checkpoint_path) != config["model"]["checkpoint_sha256"]:
        raise ValueError("ODEFormer checkpoint hash differs from frozen config")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"CTF prediction directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    regressor, parameter_count = _load_odeformer(
        checkpoint_path=checkpoint_path,
        external_source_path=external_source_path,
        external_revision=config["model"]["external_source_revision"],
        device=device,
        beam_size=config["inference"]["candidate_count"],
    )
    pair_records = []
    for pair_id in range(1, 10):
        train_data, init_data = load_dataset("ODE_Lorenz", pair_id)
        prediction_times = np.asarray(
            get_prediction_timesteps("ODE_Lorenz", pair_id), dtype=np.float64
        )
        context, initial_index = select_ctf_context(
            pair_id=pair_id,
            train_data=train_data,
            init_data=init_data,
            context_length=context_length,
        )
        delta_t = float(np.median(np.diff(prediction_times)))
        context_times = np.arange(context.shape[0], dtype=np.float64) * delta_t
        offsets = prediction_times - prediction_times[0]
        started = time.perf_counter()
        predictions, failed, equations, coefficients, scores, errors = (
            condition_odeformer_sindy(
                context[None],
                context_times=context_times,
                forecast_offsets=offsets,
                regressor=regressor,
                seed=seed + pair_id,
                forecast_initial_index=initial_index,
            )
        )
        prediction = predictions[0]
        pair_dir = output_dir / f"pair_{pair_id}"
        prediction_path = atomic_save_npy(pair_dir / "predictions.npy", prediction)
        inference_path = atomic_write_json(pair_dir / "inference_manifest.json", {
            "schema": "ctf-amortized-sindy-inference-v1",
            "pair_id": pair_id,
            "dataset": "ODE_Lorenz",
            "evaluation_track": "open_development",
            "method_status": "post_hoc_exploratory",
            "confirmatory_eligible": False,
            "zero_shot": True,
            "target_time_zero_update": True,
            "strict_dataset_zero_shot": False,
            "target_pretraining_exposure": config["model"]["pretraining_exposure"],
            "target_optimizer_steps": 0,
            "target_sparse_regression_solves": 0,
            "future_truth_path_accepted": False,
            "context_steps": int(context.shape[0]),
            "task_mode": "reconstruction" if pair_id in (2, 4) else "forecast",
            "rollout_initial_context_index": initial_index,
            "context_sha256": array_sha256(context),
            "prediction_sha256": array_sha256(prediction),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "frozen_config_sha256": sha256_file(frozen_config_path),
            "selection_manifest_sha256": sha256_file(selection_manifest_path),
            "failed_rollout": bool(failed[0]),
            "equation": str(equations[0]),
            "coefficient_matrix": coefficients[0].tolist(),
            "observed_context_score": float(scores[0]),
            "error": str(errors[0]),
            "runtime_seconds": time.perf_counter() - started,
            "parameter_count": parameter_count,
        })
        pair_records.append({
            "pair_id": pair_id,
            "prediction_path": prediction_path.relative_to(output_dir).as_posix(),
            "inference_manifest_path": inference_path.relative_to(output_dir).as_posix(),
        })
    return atomic_write_json(output_dir / "prediction_batch_manifest.json", {
        "schema": "ctf-amortized-sindy-prediction-batch-v1",
        "dataset": "ODE_Lorenz",
        "evaluation_track": "open_development",
        "method": "odeformer_context_conditioned_taylor_sindy_v2",
        "method_status": "post_hoc_exploratory",
        "confirmatory_eligible": False,
        "zero_shot": True,
        "pairs": pair_records,
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate exploratory ODEFormer-SINDy CTF dev predictions."
    )
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--external-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(predict_odeformer_ctf_pairs(
        frozen_config_path=args.frozen_config,
        selection_manifest_path=args.selection_manifest,
        checkpoint_path=args.checkpoint,
        external_source_path=args.external_source,
        output_dir=args.output_dir,
        context_length=args.context_length,
        device=args.device,
        seed=args.seed,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
