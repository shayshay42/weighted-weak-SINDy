from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint, sha256_file
from .foundation_embed import (
    load_foundation_embedding_manifest,
    verify_foundation_embedding_identity,
)
from .io import array_sha256, atomic_save_npz, atomic_write_json
from .multifamily_data import DATASET_SCHEMA


CONTEXT_KEYS = {
    "context_states",
    "context_times",
    "forecast_offsets",
    "group_ids",
    "trajectory_ids",
}
EMBEDDING_KEYS = {"embeddings", "group_ids", "trajectory_ids"}


def infer_heldout_family(
    *,
    checkpoint_path: str | Path,
    context_path: str | Path,
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    context_embeddings_path: str | Path | None = None,
    context_embeddings_manifest_path: str | Path | None = None,
    device: str = "cpu",
    force: bool = False,
) -> Path:
    """Infer and roll out held-out fields without accepting a truth path."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"inference output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads(
        Path(dataset_manifest_path).read_text(encoding="utf-8")
    )
    if dataset_manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid multifamily dataset manifest")
    context_record = dataset_manifest["artifacts"]["lorenz_heldout_context"]
    if sha256_file(context_path) != context_record["sha256"]:
        raise ValueError("held-out context does not match the dataset manifest")
    model, training_manifest = load_checkpoint(checkpoint_path, device=device)
    if "synthetic-multifamily-v1/lorenz_heldout" not in training_manifest[
        "excluded_evaluation_dataset_ids"
    ]:
        raise ValueError("checkpoint did not explicitly exclude the held-out Lorenz family")
    with np.load(context_path, allow_pickle=False) as loaded:
        if set(loaded.files) != CONTEXT_KEYS:
            raise ValueError("held-out context file contains unexpected arrays")
        values = {name: np.asarray(loaded[name]) for name in loaded.files}
    contexts = values["context_states"]
    times = values["context_times"]
    offsets = values["forecast_offsets"]
    if contexts.ndim != 3 or contexts.shape[-1] != model.config.state_dimension:
        raise ValueError("held-out contexts do not match the checkpoint dimension")
    if times.shape != (contexts.shape[1],) or offsets.ndim != 1:
        raise ValueError("held-out public time grids are invalid")
    if not np.isfinite(contexts).all():
        raise ValueError("held-out contexts contain non-finite values")
    count = contexts.shape[0]
    context_embeddings = None
    context_embedding_provenance = None
    foundation_encoder_provenance = training_manifest.get(
        "foundation_encoder_provenance"
    )
    if model.config.encoder_type in {
        "external", "external_mlp", "external_birkhoff_weak"
    }:
        if context_embeddings_path is None:
            raise ValueError("external checkpoint requires held-out context embeddings")
        with np.load(context_embeddings_path, allow_pickle=False) as loaded:
            if set(loaded.files) != EMBEDDING_KEYS:
                raise ValueError("held-out embedding file contains unexpected arrays")
            embedding_array = np.asarray(loaded["embeddings"], dtype=np.float32)
            embedding_group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
            embedding_trajectory_ids = np.asarray(
                loaded["trajectory_ids"], dtype=np.int64
            )
        if embedding_array.shape != (count, model.config.hidden_size):
            raise ValueError("held-out embedding shape differs from checkpoint contract")
        if not np.array_equal(embedding_group_ids, values["group_ids"]):
            raise ValueError("held-out embedding group order differs from contexts")
        if not np.array_equal(embedding_trajectory_ids, values["trajectory_ids"]):
            raise ValueError("held-out embedding trajectory order differs from contexts")
        if not np.isfinite(embedding_array).all():
            raise ValueError("held-out embeddings contain non-finite values")
        if foundation_encoder_provenance is not None:
            if context_embeddings_manifest_path is None:
                raise ValueError(
                    "checkpoint-bound foundation encoder requires a held-out "
                    "embedding manifest"
                )
        if context_embeddings_manifest_path is not None:
            context_embedding_provenance = load_foundation_embedding_manifest(
                context_embeddings_manifest_path,
                embeddings_path=context_embeddings_path,
                context_path=context_path,
                expected_artifact_role="lorenz_heldout_context",
                expected_embedding_shape=tuple(embedding_array.shape),
            )
            if foundation_encoder_provenance is not None:
                foundation_spec = verify_foundation_embedding_identity(
                    foundation_encoder_provenance,
                    context_embedding_provenance,
                )
                if (
                    foundation_spec.state_dimension != contexts.shape[-1]
                    or foundation_spec.context_length != contexts.shape[1]
                ):
                    raise ValueError(
                        "held-out context differs from the bound foundation encoder contract"
                    )
        context_embeddings = torch.as_tensor(
            embedding_array, dtype=torch.float32, device=device
        )
    elif (
        context_embeddings_path is not None
        or context_embeddings_manifest_path is not None
    ):
        raise ValueError("held-out embeddings require an external checkpoint")
    context_tensor = torch.as_tensor(contexts, dtype=torch.float32, device=device)
    time_tensor = torch.as_tensor(times, dtype=torch.float32, device=device)
    time_tensor = time_tensor[None, :].expand(count, -1).clone()
    offset_tensor = torch.as_tensor(offsets, dtype=torch.float32, device=device)
    with torch.inference_mode():
        predictions, dynamics = model.rollout(
            context_tensor,
            time_tensor,
            offset_tensor,
            hard_support=True,
            context_embedding=context_embeddings,
        )
    prediction_array = predictions.cpu().numpy().astype(np.float64)
    physical_coefficients = dynamics.physical_coefficients.cpu().numpy().astype(np.float64)
    support_probabilities = dynamics.support_probabilities.cpu().numpy().astype(np.float64)
    if not np.isfinite(prediction_array).all():
        raise RuntimeError("held-out zero-shot rollout diverged")
    prediction_path = atomic_save_npz(
        output_dir / "predictions.npz",
        predictions=prediction_array,
        physical_coefficients=physical_coefficients,
        support_probabilities=support_probabilities,
        forecast_offsets=np.asarray(offsets, dtype=np.float64),
        group_ids=np.asarray(values["group_ids"], dtype=np.int64),
        trajectory_ids=np.asarray(values["trajectory_ids"], dtype=np.int64),
    )
    return atomic_write_json(output_dir / "inference_manifest.json", {
        "schema": "amortized-sindy-heldout-inference-v1",
        "status": "complete",
        "zero_shot": True,
        "target_optimizer_steps": 0,
        "target_sparse_regression_solves": 0,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "context_sha256": sha256_file(context_path),
        "context_embeddings_sha256": (
            sha256_file(context_embeddings_path)
            if context_embeddings_path is not None
            else None
        ),
        "context_embeddings_manifest_sha256": (
            sha256_file(context_embeddings_manifest_path)
            if context_embeddings_manifest_path is not None
            else None
        ),
        "foundation_encoder_provenance": foundation_encoder_provenance,
        "context_array_sha256": array_sha256(contexts),
        "prediction_artifact": {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
        },
        "trajectory_count": count,
        "library_exponents": [list(value) for value in model.exponents],
        "training_manifest": training_manifest,
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen amortized SINDy on held-out contexts without truth access."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--context-embeddings")
    parser.add_argument("--context-embeddings-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(infer_heldout_family(
        checkpoint_path=args.checkpoint,
        context_path=args.context,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
        context_embeddings_path=args.context_embeddings,
        context_embeddings_manifest_path=args.context_embeddings_manifest,
        device=args.device,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
