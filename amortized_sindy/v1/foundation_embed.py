from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import sha256_file
from .foundation_encoder import (
    FOUNDATION_ENCODER_BINDING_SCHEMA,
    FoundationEncoderSpec,
    load_foundation_encoder,
    validate_foundation_encoder_binding,
)
from .io import atomic_save_npz, atomic_write_json
from .multifamily_data import DATASET_SCHEMA


SOURCE_KEYS = {
    "context_states",
    "context_times",
    "future_states",
    "forecast_offsets",
    "group_ids",
    "trajectory_ids",
}
CONTEXT_KEYS = SOURCE_KEYS - {"future_states"}
FOUNDATION_EMBEDDING_MANIFEST_SCHEMA = (
    "amortized-sindy-foundation-embeddings-v1"
)


def load_foundation_embedding_manifest(
    manifest_path: str | Path,
    *,
    embeddings_path: str | Path,
    context_path: str | Path,
    expected_artifact_role: str,
    expected_embedding_shape: tuple[int, int],
) -> dict[str, Any]:
    """Validate an embedding artifact and return its pinned encoder identity."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get(
        "schema"
    ) != FOUNDATION_EMBEDDING_MANIFEST_SCHEMA:
        raise ValueError("invalid foundation embedding manifest schema")
    if manifest.get("artifact_role") != expected_artifact_role:
        raise ValueError("foundation embedding manifest has the wrong artifact role")
    if manifest.get("future_states_read") is not False:
        raise ValueError("foundation embedding extraction must not read future states")
    if manifest.get("forecast_generation_called") is not False:
        raise ValueError("foundation embedding extraction must not forecast")
    if manifest.get("context_artifact_sha256") != sha256_file(context_path):
        raise ValueError("foundation embedding context hash mismatch")
    if manifest.get("embedding_shape") != list(expected_embedding_shape):
        raise ValueError("foundation embedding manifest shape mismatch")
    artifact = manifest.get("embedding_artifact")
    if not isinstance(artifact, dict) or artifact.get("sha256") != sha256_file(
        embeddings_path
    ):
        raise ValueError("foundation embedding artifact hash mismatch")
    spec = FoundationEncoderSpec.from_dict(manifest.get("encoder"))
    weights = manifest.get("model_weights")
    if not isinstance(weights, dict) or set(weights) != {"filename", "sha256"}:
        raise ValueError("foundation encoder weight identity is invalid")
    if weights["filename"] is not None and not isinstance(weights["filename"], str):
        raise ValueError("foundation encoder weight filename is invalid")
    if weights["sha256"] is not None and not (
        isinstance(weights["sha256"], str) and len(weights["sha256"]) == 64
    ):
        raise ValueError("foundation encoder weight hash is invalid")
    return {
        "encoder": spec.to_dict(),
        "model_weights": {
            "filename": weights["filename"],
            "sha256": weights["sha256"],
        },
        "manifest_sha256": sha256_file(manifest_path),
    }


def bind_foundation_embedding_manifests(
    source: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Bind matching source/validation encoder provenance into a checkpoint."""
    if source["encoder"] != validation["encoder"]:
        raise ValueError(
            "source and validation embeddings use different foundation encoders"
        )
    if source["model_weights"] != validation["model_weights"]:
        raise ValueError(
            "source and validation embeddings use different foundation weights"
        )
    binding = {
        "schema": FOUNDATION_ENCODER_BINDING_SCHEMA,
        "encoder": source["encoder"],
        "model_weights": source["model_weights"],
        "embedding_manifests": {
            "source": source["manifest_sha256"],
            "validation": validation["manifest_sha256"],
        },
    }
    validate_foundation_encoder_binding(binding)
    return binding


def verify_foundation_embedding_identity(
    binding: dict[str, Any],
    observed: dict[str, Any],
) -> FoundationEncoderSpec:
    """Require an inference embedding to come from the checkpoint-bound encoder."""
    spec = validate_foundation_encoder_binding(binding)
    if observed["encoder"] != binding["encoder"]:
        raise ValueError("foundation embedding encoder differs from checkpoint binding")
    if observed["model_weights"] != binding["model_weights"]:
        raise ValueError("foundation embedding weights differ from checkpoint binding")
    return spec


def extract_foundation_embeddings(
    *,
    context_path: str | Path,
    dataset_manifest_path: str | Path,
    artifact_name: str,
    output_dir: str | Path,
    backend: str = "panda_patchtst",
    model_id: str = "GilpinLab/panda_mlm",
    revision: str = "ad305089edeade49daf11be740ee2f4cafc839fe",
    context_length: int = 128,
    target_pretraining_exposure: str = "unknown",
    device: str = "cuda:0",
    batch_size: int = 32,
    force: bool = False,
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    context_path = Path(context_path)
    dataset_manifest = json.loads(
        Path(dataset_manifest_path).read_text(encoding="utf-8")
    )
    if dataset_manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid multifamily dataset manifest")
    record = dataset_manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(record, dict) or record.get("sha256") != sha256_file(context_path):
        raise ValueError("context artifact does not match the dataset manifest")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"embedding output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(context_path, allow_pickle=False) as loaded:
        keys = set(loaded.files)
        if keys != SOURCE_KEYS and keys != CONTEXT_KEYS:
            raise ValueError("context file contains unexpected arrays")
        contexts = np.asarray(loaded["context_states"], dtype=np.float32)
        times = np.asarray(loaded["context_times"], dtype=np.float64)
        group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        trajectory_ids = np.asarray(loaded["trajectory_ids"], dtype=np.int64)
    if contexts.ndim != 3 or contexts.shape[-1] != 3:
        raise ValueError("foundation contexts must have shape [trajectory,time,3]")
    if times.shape != (contexts.shape[1],):
        raise ValueError("foundation context time grid has the wrong shape")
    if group_ids.shape != (contexts.shape[0],) or trajectory_ids.shape != group_ids.shape:
        raise ValueError("foundation context identifiers have the wrong shape")
    if context_length != contexts.shape[1]:
        raise ValueError(
            "foundation context length must match the sanitized context exactly"
        )
    pooling = {
        "panda_patchtst": "channel_patch_mean_flatten",
        "chronos_t5": "channel_token_mean_flatten",
    }.get(backend)
    if pooling is None:
        raise ValueError("unsupported foundation encoder backend")
    spec = FoundationEncoderSpec(
        backend=backend,  # type: ignore[arg-type]
        model_id=model_id,
        revision=revision,
        state_dimension=contexts.shape[-1],
        context_length=context_length,
        pooling=pooling,  # type: ignore[arg-type]
        target_pretraining_exposure=target_pretraining_exposure,  # type: ignore[arg-type]
    )
    encoder = load_foundation_encoder(spec, device=device)
    batches = []
    for start in range(0, contexts.shape[0], batch_size):
        stop = min(start + batch_size, contexts.shape[0])
        batch_times = np.broadcast_to(times, (stop - start, times.size)).copy()
        batches.append(
            encoder.encode(
                torch.as_tensor(contexts[start:stop], dtype=torch.float32),
                torch.as_tensor(batch_times, dtype=torch.float64),
            ).cpu()
        )
    embeddings = torch.cat(batches, dim=0).numpy().astype(np.float32)
    artifact_path = atomic_save_npz(
        output_dir / "embeddings.npz",
        embeddings=embeddings,
        group_ids=group_ids,
        trajectory_ids=trajectory_ids,
    )
    weights_sha256 = None
    weights_filename = None
    try:
        from huggingface_hub import hf_hub_download

        weights_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors",
                revision=revision,
            )
        )
        weights_sha256 = sha256_file(weights_path)
        weights_filename = weights_path.name
    except (ImportError, OSError):
        # The pinned repository revision is still mandatory. A local weight
        # hash is additional evidence when the cache exposes the file.
        pass
    return atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema": FOUNDATION_EMBEDDING_MANIFEST_SCHEMA,
            "artifact_role": artifact_name,
            "context_artifact_sha256": sha256_file(context_path),
            "future_states_read": False,
            "forecast_generation_called": False,
            "encoder": spec.to_dict(),
            "embedding_shape": list(embeddings.shape),
            "embedding_artifact": {
                "path": artifact_path.name,
                "sha256": sha256_file(artifact_path),
            },
            "model_weights": {
                "filename": weights_filename,
                "sha256": weights_sha256,
            },
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frozen Panda context embeddings without forecasting."
    )
    parser.add_argument("--context", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backend", choices=("panda_patchtst", "chronos_t5"), default="panda_patchtst"
    )
    parser.add_argument("--model-id", default="GilpinLab/panda_mlm")
    parser.add_argument(
        "--revision", default="ad305089edeade49daf11be740ee2f4cafc839fe"
    )
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument(
        "--target-pretraining-exposure",
        choices=("verified_excluded", "unknown", "known_exposed"),
        default="unknown",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(
        extract_foundation_embeddings(
            context_path=args.context,
            dataset_manifest_path=args.dataset_manifest,
            artifact_name=args.artifact_name,
            output_dir=args.output_dir,
            backend=args.backend,
            model_id=args.model_id,
            revision=args.revision,
            context_length=args.context_length,
            target_pretraining_exposure=args.target_pretraining_exposure,
            device=args.device,
            batch_size=args.batch_size,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
