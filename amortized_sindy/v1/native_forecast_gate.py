from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import sha256_file
from .io import atomic_save_npz, atomic_write_json
from .multifamily_data import DATASET_SCHEMA
from .multifamily_train import _load_source
from .tabpfn_pilot import _metrics


NATIVE_GATE_SCHEMA = "amortized-sindy-native-forecast-gate-v1"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _context_normalization(
    contexts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = contexts.mean(axis=1, keepdims=True)
    scale = np.maximum(contexts.std(axis=1, keepdims=True), 1e-6)
    return (contexts - mean) / scale, mean, scale


def forecast_panda_native(
    contexts: np.ndarray,
    *,
    model: Any,
    future_steps: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Use Panda's released probabilistic forecast head with no adaptation."""
    if future_steps < 1 or batch_size < 1:
        raise ValueError("future steps and batch size must be positive")
    contexts = np.asarray(contexts, dtype=np.float64)
    normalized, mean, scale = _context_normalization(contexts)
    device = torch.device(model.device)
    batches = []
    _seed_everything(seed)
    with torch.inference_mode():
        for start in range(0, contexts.shape[0], batch_size):
            stop = min(start + batch_size, contexts.shape[0])
            tensor = torch.as_tensor(
                normalized[start:stop], dtype=torch.float32, device=device
            )
            remaining = future_steps
            generated_blocks = []
            while remaining > 0:
                generated = model.generate(tensor).sequences
                if generated.ndim != 4 or generated.shape[-1] != contexts.shape[-1]:
                    raise ValueError("Panda returned an unexpected forecast shape")
                point = generated.median(dim=1).values
                generated_blocks.append(point[:, :remaining])
                remaining -= point.shape[1]
                if remaining > 0:
                    tensor = torch.cat((tensor, point), dim=1)
                    tensor = tensor[:, -int(model.config.context_length) :]
            batches.append(torch.cat(generated_blocks, dim=1).cpu().numpy())
    normalized_future = np.concatenate(batches, axis=0)[:, :future_steps]
    future = normalized_future * scale + mean
    return np.concatenate((contexts[:, -1:, :], future), axis=1)


def forecast_chronos_t5_native(
    contexts: np.ndarray,
    *,
    pipeline: Any,
    future_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Forecast each state channel through Chronos-T5's native head."""
    if future_steps < 1 or batch_size < 1:
        raise ValueError("future steps and batch size must be positive")
    contexts = np.asarray(contexts, dtype=np.float64)
    normalized, mean, scale = _context_normalization(contexts)
    channel_series = normalized.transpose(0, 2, 1).reshape(-1, contexts.shape[1])
    forecast_blocks = []
    with torch.inference_mode():
        for start in range(0, channel_series.shape[0], batch_size):
            stop = min(start + batch_size, channel_series.shape[0])
            _, forecast_mean = pipeline.predict_quantiles(
                context=torch.as_tensor(
                    channel_series[start:stop], dtype=torch.float32
                ),
                prediction_length=future_steps,
                quantile_levels=[0.5],
            )
            values = torch.as_tensor(forecast_mean).detach().cpu().numpy()
            if values.shape != (stop - start, future_steps):
                raise ValueError("Chronos-T5 returned an unexpected forecast shape")
            forecast_blocks.append(values)
    normalized_future = np.concatenate(forecast_blocks, axis=0)
    normalized_future = normalized_future.reshape(
        contexts.shape[0], contexts.shape[2], future_steps
    ).transpose(0, 2, 1)
    future = normalized_future * scale + mean
    return np.concatenate((contexts[:, -1:, :], future), axis=1)


def _load_backend(
    *,
    backend: str,
    model_id: str,
    revision: str,
    device: str,
    seed: int,
) -> tuple[Any, int]:
    _seed_everything(seed)
    if backend == "panda":
        try:
            from panda.patchtst.patchtst import PatchTSTForPrediction
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Panda is not installed") from error
        model = PatchTSTForPrediction.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.float32,
        )
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        parameter_count = int(sum(value.numel() for value in model.parameters()))
        return model, parameter_count
    if backend == "chronos_t5":
        try:
            from chronos import ChronosPipeline
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Chronos is not installed") from error
        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            revision=revision,
            device_map=device,
            torch_dtype=(torch.bfloat16 if device.startswith("cuda") else torch.float32),
        )
        pipeline.model.eval()
        pipeline.model.requires_grad_(False)
        parameter_count = int(
            sum(value.numel() for value in pipeline.model.parameters())
        )
        return pipeline, parameter_count
    raise ValueError("unsupported native forecast backend")


def run_native_forecast_gate(
    *,
    trajectory_path: str | Path,
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    artifact_name: str = "source_validation",
    backend: str,
    model_id: str,
    revision: str,
    device: str = "cuda:0",
    batch_size: int = 32,
    seed: int = 2026,
    target_pretraining_exposure: str = "unknown",
    force: bool = False,
) -> Path:
    if not model_id or not revision:
        raise ValueError("native model ID and revision must be pinned")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"native forecast output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid multifamily dataset manifest")
    if artifact_name not in {"source_train", "source_validation"}:
        raise ValueError("native forecast gate supports source_train/source_validation")
    if manifest.get("artifacts", {}).get(artifact_name, {}).get(
        "sha256"
    ) != sha256_file(trajectory_path):
        raise ValueError(f"{artifact_name} does not match the dataset manifest")
    validation = _load_source(trajectory_path)
    contexts = np.asarray(validation["context_states"], dtype=np.float64)
    future_steps = int(validation["forecast_offsets"].size - 1)

    model, parameter_count = _load_backend(
        backend=backend,
        model_id=model_id,
        revision=revision,
        device=device,
        seed=seed,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if backend == "panda":
        predictions = forecast_panda_native(
            contexts,
            model=model,
            future_steps=future_steps,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        predictions = forecast_chronos_t5_native(
            contexts,
            pipeline=model,
            future_steps=future_steps,
            batch_size=batch_size,
        )
    inference_seconds = time.perf_counter() - started
    failed = ~np.isfinite(predictions).all(axis=(1, 2))
    metrics = _metrics(predictions, validation, failed)
    peak_memory = (
        int(torch.cuda.max_memory_allocated())
        if device.startswith("cuda")
        else 0
    )
    prediction_path = atomic_save_npz(
        output_dir / "source_validation_predictions.npz",
        predictions=predictions,
        group_ids=np.asarray(validation["group_ids"], dtype=np.int64),
        trajectory_ids=np.asarray(validation["trajectory_ids"], dtype=np.int64),
        forecast_offsets=np.asarray(validation["forecast_offsets"], dtype=np.float64),
    )
    weights_sha256 = None
    try:
        from huggingface_hub import hf_hub_download

        weights_sha256 = sha256_file(hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            revision=revision,
        ))
    except (ImportError, OSError):
        pass
    return atomic_write_json(output_dir / "manifest.json", {
        "schema": NATIVE_GATE_SCHEMA,
        "status": "complete",
        "selection_role": (
            "source_family_validation_only"
            if artifact_name == "source_validation"
            else "source_training_diagnostic"
        ),
        "artifact_role": artifact_name,
        "backend": backend,
        "model": {
            "id": model_id,
            "revision": revision,
            "weights_sha256": weights_sha256,
            "parameter_count": parameter_count,
            "target_pretraining_exposure": target_pretraining_exposure,
        },
        "information_contract": {
            "context_steps": int(contexts.shape[1]),
            "forecast_steps_including_initial": int(future_steps + 1),
            "target_optimizer_steps": 0,
            "target_sparse_regression_solves": 0,
            "native_forecast_head_used": True,
            "trajectory_normalization": "per_context_coordinate_mean_std",
            "heldout_lorenz_or_ctf_data_read": False,
            "validation_future_passed_to_forecaster": False,
        },
        "metrics": metrics,
        "runtime": {
            "inference_seconds": inference_seconds,
            "peak_gpu_memory_bytes": peak_memory,
            "batch_size": batch_size,
            "seed": seed,
        },
        "artifacts": {
            "trajectory_sha256": sha256_file(trajectory_path),
            "predictions": {
                "path": prediction_path.name,
                "sha256": sha256_file(prediction_path),
            },
        },
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a pinned foundation model's native forecast head on source validation."
    )
    parser.add_argument("--trajectory", required=True)
    parser.add_argument(
        "--artifact-name",
        choices=("source_train", "source_validation"),
        default="source_validation",
    )
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=("panda", "chronos_t5"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--target-pretraining-exposure",
        choices=("verified_excluded", "unknown", "known_exposed"),
        default="unknown",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(run_native_forecast_gate(
        trajectory_path=args.trajectory,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
        artifact_name=args.artifact_name,
        backend=args.backend,
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        target_pretraining_exposure=args.target_pretraining_exposure,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
