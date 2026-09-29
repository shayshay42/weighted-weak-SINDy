from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .foundation_encoder import validate_foundation_encoder_binding
from .model import AmortizedSINDy, AmortizedSINDyConfig


CHECKPOINT_SCHEMA = "amortized-sindy-checkpoint-v1"
TRAINING_MANIFEST_SCHEMA = "amortized-sindy-training-manifest-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != TRAINING_MANIFEST_SCHEMA:
        raise ValueError("invalid amortized SINDy training manifest schema")
    sources = manifest.get("source_dataset_ids")
    if not isinstance(sources, list) or not sources or not all(
        isinstance(value, str) and value for value in sources
    ):
        raise ValueError("training manifest must identify source datasets")
    if manifest.get("target_evaluation_data_used_for_training") is not False:
        raise ValueError("target evaluation data must not be used for training")
    excluded = manifest.get("excluded_evaluation_dataset_ids")
    if not isinstance(excluded, list) or not all(isinstance(value, str) for value in excluded):
        raise ValueError("training manifest must list excluded evaluation datasets")
    foundation_binding = manifest.get("foundation_encoder_provenance")
    if foundation_binding is not None:
        validate_foundation_encoder_binding(foundation_binding)
        explicit_fields = {
            "source_embedding_manifest_sha256": foundation_binding[
                "embedding_manifests"
            ]["source"],
            "source_validation_embedding_manifest_sha256": foundation_binding[
                "embedding_manifests"
            ]["validation"],
            "foundation_encoder_spec": foundation_binding["encoder"],
            "foundation_encoder_weights_sha256": foundation_binding[
                "model_weights"
            ]["sha256"],
        }
        supplied = {key for key in explicit_fields if key in manifest}
        if supplied and supplied != set(explicit_fields):
            raise ValueError("foundation encoder manifest fields are incomplete")
        for key in supplied:
            if manifest[key] != explicit_fields[key]:
                raise ValueError(
                    f"foundation encoder manifest field differs from binding: {key}"
                )


def save_checkpoint(
    path: str | Path,
    model: AmortizedSINDy,
    *,
    training_manifest: dict[str, Any],
) -> Path:
    validate_training_manifest(training_manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "training_manifest": training_manifest,
    }
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[AmortizedSINDy, dict[str, Any]]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument.
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("invalid amortized SINDy checkpoint schema")
    if set(payload) != {"schema", "config", "state_dict", "training_manifest"}:
        raise ValueError("checkpoint contains unexpected fields")
    manifest = payload["training_manifest"]
    validate_training_manifest(manifest)
    model = AmortizedSINDy(AmortizedSINDyConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, manifest
