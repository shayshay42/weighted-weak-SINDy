from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import save_checkpoint, sha256_file
from .foundation_embed import (
    bind_foundation_embedding_manifests,
    load_foundation_embedding_manifest,
)
from .foundation_encoder import FoundationEncoderSpec
from .io import atomic_write_json
from .model import AmortizedSINDy, AmortizedSINDyConfig
from .multifamily_data import DATASET_SCHEMA
from .training import source_training_loss


SOURCE_KEYS = {
    "context_states",
    "context_times",
    "future_states",
    "forecast_offsets",
    "group_ids",
    "trajectory_ids",
}

EMBEDDING_KEYS = {"embeddings", "group_ids", "trajectory_ids"}


def _load_source(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != SOURCE_KEYS:
            raise ValueError(f"source trajectory file has unexpected arrays: {loaded.files}")
        values = {name: np.asarray(loaded[name]) for name in loaded.files}
    contexts = values["context_states"]
    futures = values["future_states"]
    if contexts.ndim != 3 or contexts.shape[-1] != 3:
        raise ValueError("source contexts must have shape [trajectory,time,3]")
    if futures.ndim != 3 or futures.shape[0] != contexts.shape[0] or futures.shape[-1] != 3:
        raise ValueError("source futures must match context trajectory/state axes")
    if values["context_times"].shape != (contexts.shape[1],):
        raise ValueError("source context time grid has the wrong shape")
    if values["forecast_offsets"].shape != (futures.shape[1],):
        raise ValueError("source forecast grid has the wrong shape")
    if values["group_ids"].shape != (contexts.shape[0],):
        raise ValueError("source group IDs have the wrong shape")
    if not np.isfinite(contexts).all() or not np.isfinite(futures).all():
        raise ValueError("source trajectories contain non-finite values")
    return values


def _verify_source_artifact(
    manifest: dict[str, Any], name: str, path: str | Path
) -> None:
    record = manifest["artifacts"][name]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{name} does not match the dataset manifest")


def _tensors(values: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    count = values["context_states"].shape[0]
    times = torch.as_tensor(values["context_times"], dtype=torch.float32)
    return {
        "context_states": torch.as_tensor(values["context_states"], dtype=torch.float32),
        "context_times": times[None, :].expand(count, -1).clone(),
        "future_states": torch.as_tensor(values["future_states"], dtype=torch.float32),
        "forecast_offsets": torch.as_tensor(values["forecast_offsets"], dtype=torch.float32),
        "group_ids": torch.as_tensor(values["group_ids"], dtype=torch.int64),
    }


def _load_embeddings(
    path: str | Path,
    *,
    expected_group_ids: np.ndarray,
    expected_trajectory_ids: np.ndarray,
) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != EMBEDDING_KEYS:
            raise ValueError("foundation embedding file contains unexpected arrays")
        embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
        group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        trajectory_ids = np.asarray(loaded["trajectory_ids"], dtype=np.int64)
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_group_ids.shape[0]:
        raise ValueError("foundation embeddings must have shape [trajectory,feature]")
    if not np.array_equal(group_ids, expected_group_ids):
        raise ValueError("foundation embedding group order differs from source data")
    if not np.array_equal(trajectory_ids, expected_trajectory_ids):
        raise ValueError("foundation embedding trajectory order differs from source data")
    if not np.isfinite(embeddings).all():
        raise ValueError("foundation embeddings contain non-finite values")
    return torch.as_tensor(embeddings, dtype=torch.float32)


def _load_teacher_coefficients(
    *,
    labels_path: str | Path,
    teacher_manifest_path: str | Path,
    source_train_path: str | Path,
    source_group_ids: np.ndarray,
) -> torch.Tensor:
    teacher_manifest = json.loads(
        Path(teacher_manifest_path).read_text(encoding="utf-8")
    )
    if teacher_manifest.get("schema") != "amortized-sindy-source-teacher-v1":
        raise ValueError("invalid source teacher manifest")
    if teacher_manifest.get("future_source_states_used") is not False:
        raise ValueError("source teacher must not use future source states")
    if teacher_manifest.get("generator_coefficients_used") is not False:
        raise ValueError("source teacher must not use generator coefficients")
    if teacher_manifest.get("source_train_sha256") != sha256_file(source_train_path):
        raise ValueError("source teacher was fitted on a different source dataset")
    if teacher_manifest["labels"]["sha256"] != sha256_file(labels_path):
        raise ValueError("source teacher label hash mismatch")
    with np.load(labels_path, allow_pickle=False) as loaded:
        if set(loaded.files) != {
            "group_ids", "coefficients", "residual_mse", "active_terms"
        }:
            raise ValueError("source teacher labels contain unexpected arrays")
        teacher_group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        coefficients = np.asarray(loaded["coefficients"], dtype=np.float64)
    by_group = {
        int(group_id): coefficients[index]
        for index, group_id in enumerate(teacher_group_ids)
    }
    if any(int(group_id) not in by_group for group_id in source_group_ids):
        raise ValueError("source teacher labels do not cover every training group")
    return torch.as_tensor(
        np.asarray([by_group[int(group_id)] for group_id in source_group_ids]),
        dtype=torch.float32,
    )


def _validation_mse(
    model: AmortizedSINDy,
    values: dict[str, torch.Tensor],
    context_embeddings: torch.Tensor | None = None,
) -> float:
    model.eval()
    with torch.inference_mode():
        predictions, dynamics = model.rollout(
            values["context_states"],
            values["context_times"],
            values["forecast_offsets"],
            hard_support=True,
            context_embedding=context_embeddings,
        )
        scale = dynamics.state_scale.squeeze(1).unsqueeze(1)
        loss = torch.mean(((predictions - values["future_states"]) / scale) ** 2)
    value = float(loss)
    return value if np.isfinite(value) else float("inf")


def _constant_field_mse(values: dict[str, torch.Tensor]) -> float:
    contexts = values["context_states"]
    futures = values["future_states"]
    prediction = contexts[:, -1:, :].expand_as(futures)
    scale = contexts.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return float(torch.mean(((prediction - futures) / scale) ** 2))


def train_multifamily_checkpoint(
    *,
    source_train_path: str | Path,
    source_validation_path: str | Path,
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    source_teacher_labels_path: str | Path | None = None,
    source_teacher_manifest_path: str | Path | None = None,
    source_embeddings_path: str | Path | None = None,
    source_validation_embeddings_path: str | Path | None = None,
    source_embedding_manifest_path: str | Path | None = None,
    source_validation_embedding_manifest_path: str | Path | None = None,
    epochs: int = 200,
    hidden_size: int = 64,
    encoder_type: str = "weak_stats",
    hybrid_ablation: str = "full",
    learning_rate: float = 1e-3,
    eval_interval: int = 5,
    support_weight: float = 5e-4,
    support_binary_weight: float = 0.0,
    coefficient_weight: float = 1e-6,
    raw_coefficient_weight: float = 0.0,
    context_residual_weight: float = 1e-2,
    weak_form_weight: float = 0.0,
    birkhoff_mmd_weight: float = 0.0,
    birkhoff_mmd_bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    consistency_weight: float = 1e-3,
    training_support_mode: str = "soft",
    teacher_weight: float = 0.0,
    warmup_epochs: int | None = None,
    rollout_error_cap: float = 10.0,
    rollout_loss_kind: str = "clipped_mse",
    seed: int = 91,
    device: str = "cpu",
    cpu_threads: int = 1,
    force: bool = False,
) -> Path:
    if (
        epochs < 1
        or hidden_size < 1
        or learning_rate <= 0
        or eval_interval < 1
        or cpu_threads < 1
    ):
        raise ValueError("invalid multifamily training configuration")
    if min(
        support_binary_weight,
        raw_coefficient_weight,
        weak_form_weight,
        birkhoff_mmd_weight,
    ) < 0:
        raise ValueError("training regularization weights must be non-negative")
    if not birkhoff_mmd_bandwidths or any(
        value <= 0 for value in birkhoff_mmd_bandwidths
    ):
        raise ValueError("Birkhoff MMD bandwidths must be positive")
    external_encoder_types = {
        "external", "external_mlp", "external_birkhoff_weak"
    }
    if encoder_type not in {"gru", "weak_stats", *external_encoder_types}:
        raise ValueError("invalid multifamily encoder type")
    if hybrid_ablation not in {
        "full", "no_birkhoff", "no_weak", "no_birkhoff_no_weak"
    }:
        raise ValueError("invalid hybrid ablation")
    if encoder_type != "external_birkhoff_weak" and hybrid_ablation != "full":
        raise ValueError("hybrid ablations require external_birkhoff_weak")
    if warmup_epochs is None:
        warmup_epochs = max(1, epochs // 4)
    if warmup_epochs < 0 or warmup_epochs >= epochs or rollout_error_cap <= 0:
        raise ValueError("invalid rollout curriculum configuration")
    if rollout_loss_kind not in {"clipped_mse", "pseudo_huber"}:
        raise ValueError("invalid rollout loss kind")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"training output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid multifamily dataset manifest")
    if manifest.get("lorenz_used_for_training_or_selection") is not False:
        raise ValueError("Lorenz must be excluded from training and model selection")
    if "lorenz_heldout" not in manifest.get("heldout_evaluation_families", []):
        raise ValueError("dataset manifest does not reserve Lorenz for held-out evaluation")
    _verify_source_artifact(manifest, "source_train", source_train_path)
    _verify_source_artifact(manifest, "source_validation", source_validation_path)
    source = _load_source(source_train_path)
    validation = _load_source(source_validation_path)
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    source_tensors = {
        name: value.to(runtime_device) for name, value in _tensors(source).items()
    }
    validation_tensors = {
        name: value.to(runtime_device)
        for name, value in _tensors(validation).items()
    }
    embedding_paths_supplied = (
        source_embeddings_path is not None
        or source_validation_embeddings_path is not None
    )
    embedding_manifest_paths_supplied = (
        source_embedding_manifest_path is not None
        or source_validation_embedding_manifest_path is not None
    )
    if encoder_type in external_encoder_types and not (
        source_embeddings_path is not None
        and source_validation_embeddings_path is not None
    ):
        raise ValueError("external conditioning requires source and validation embeddings")
    if encoder_type not in external_encoder_types and embedding_paths_supplied:
        raise ValueError("foundation embedding paths require an external encoder type")
    if embedding_manifest_paths_supplied and not (
        source_embedding_manifest_path is not None
        and source_validation_embedding_manifest_path is not None
    ):
        raise ValueError(
            "source and validation embedding manifests must be supplied together"
        )
    if embedding_manifest_paths_supplied and encoder_type not in external_encoder_types:
        raise ValueError("foundation embedding manifests require an external encoder type")
    source_embeddings = None
    validation_embeddings = None
    foundation_encoder_provenance = None
    if encoder_type in external_encoder_types:
        source_embeddings = _load_embeddings(
            source_embeddings_path,  # type: ignore[arg-type]
            expected_group_ids=source["group_ids"],
            expected_trajectory_ids=source["trajectory_ids"],
        )
        validation_embeddings = _load_embeddings(
            source_validation_embeddings_path,  # type: ignore[arg-type]
            expected_group_ids=validation["group_ids"],
            expected_trajectory_ids=validation["trajectory_ids"],
        )
        if source_embeddings.shape[1] != validation_embeddings.shape[1]:
            raise ValueError("source and validation embedding widths differ")
        if embedding_manifest_paths_supplied:
            source_embedding_provenance = load_foundation_embedding_manifest(
                source_embedding_manifest_path,  # type: ignore[arg-type]
                embeddings_path=source_embeddings_path,  # type: ignore[arg-type]
                context_path=source_train_path,
                expected_artifact_role="source_train",
                expected_embedding_shape=tuple(source_embeddings.shape),
            )
            validation_embedding_provenance = load_foundation_embedding_manifest(
                source_validation_embedding_manifest_path,  # type: ignore[arg-type]
                embeddings_path=source_validation_embeddings_path,  # type: ignore[arg-type]
                context_path=source_validation_path,
                expected_artifact_role="source_validation",
                expected_embedding_shape=tuple(validation_embeddings.shape),
            )
            foundation_encoder_provenance = bind_foundation_embedding_manifests(
                source_embedding_provenance,
                validation_embedding_provenance,
            )
            foundation_spec = FoundationEncoderSpec.from_dict(
                foundation_encoder_provenance["encoder"]
            )
            if foundation_spec.state_dimension != source["context_states"].shape[-1]:
                raise ValueError(
                    "foundation encoder state dimension differs from source data"
                )
            if (
                source["context_states"].shape[1]
                != validation["context_states"].shape[1]
                or foundation_spec.context_length
                != source["context_states"].shape[1]
            ):
                raise ValueError(
                    "foundation encoder context length differs from source data"
                )
        source_embeddings = source_embeddings.to(runtime_device)
        validation_embeddings = validation_embeddings.to(runtime_device)
        hidden_size = int(source_embeddings.shape[1])
    validation_constant_field_mse = _constant_field_mse(validation_tensors)
    if (source_teacher_labels_path is None) != (source_teacher_manifest_path is None):
        raise ValueError("source teacher labels and manifest must be supplied together")
    if teacher_weight > 0 and source_teacher_labels_path is None:
        raise ValueError("positive teacher weight requires source teacher labels")
    teacher_coefficients = None
    if source_teacher_labels_path is not None:
        teacher_coefficients = _load_teacher_coefficients(
            labels_path=source_teacher_labels_path,
            teacher_manifest_path=source_teacher_manifest_path,
            source_train_path=source_train_path,
            source_group_ids=source["group_ids"],
        ).to(runtime_device)

    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if runtime_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = AmortizedSINDy(AmortizedSINDyConfig(
        state_dimension=3,
        library_degree=2,
        hidden_size=hidden_size,
        min_context_steps=8,
        encoder_type=encoder_type,
        hybrid_ablation=hybrid_ablation,
        training_support_mode=training_support_mode,
    )).to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stop_reason = "training_budget_exhausted"
    stopped_epoch = epochs
    total_forecast_steps = source_tensors["forecast_offsets"].shape[0]
    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:
            horizon_steps = 1
            rollout_weight = 0.0
            current_residual_weight = (
                context_residual_weight
                if weak_form_weight > 0
                else max(context_residual_weight, 0.1)
            )
            current_weak_form_weight = (
                max(weak_form_weight, 0.1) if weak_form_weight > 0 else 0.0
            )
            current_birkhoff_mmd_weight = 0.0
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            horizon_steps = min(
                total_forecast_steps,
                2 + int(progress * max(total_forecast_steps - 2, 0)),
            )
            rollout_weight = 1.0
            current_residual_weight = context_residual_weight
            current_weak_form_weight = weak_form_weight
            current_birkhoff_mmd_weight = birkhoff_mmd_weight
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = source_training_loss(
            model,
            context_states=source_tensors["context_states"],
            context_times=source_tensors["context_times"],
            forecast_offsets=source_tensors["forecast_offsets"][:horizon_steps],
            future_states=source_tensors["future_states"][:, :horizon_steps, :],
            group_ids=source_tensors["group_ids"],
            teacher_physical_coefficients=teacher_coefficients,
            context_embeddings=source_embeddings,
            teacher_weight=teacher_weight,
            rollout_weight=rollout_weight,
            rollout_error_cap=rollout_error_cap,
            rollout_loss_kind=rollout_loss_kind,
            support_weight=support_weight,
            support_binary_weight=support_binary_weight,
            coefficient_weight=coefficient_weight,
            raw_coefficient_weight=raw_coefficient_weight,
            context_residual_weight=current_residual_weight,
            weak_form_weight=current_weak_form_weight,
            birkhoff_mmd_weight=current_birkhoff_mmd_weight,
            birkhoff_mmd_bandwidths=birkhoff_mmd_bandwidths,
            consistency_weight=consistency_weight,
        )
        if not torch.isfinite(loss.total):
            stop_reason = "nonfinite_source_loss"
            stopped_epoch = epoch
            history.append({
                "epoch": epoch,
                "status": "stopped_before_optimizer_step",
                "reason": stop_reason,
                "rollout_horizon_steps": horizon_steps,
            })
            break
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            validation_mse = _validation_mse(
                model, validation_tensors, validation_embeddings
            )
            validation_is_finite = bool(np.isfinite(validation_mse))
            record = {
                "epoch": epoch,
                "source_total": float(loss.total.detach()),
                "source_rollout_mse": float(loss.rollout_mse.detach()),
                "source_context_residual": float(loss.context_residual.detach()),
                "source_weak_form_residual": float(
                    loss.weak_form_residual.detach()
                ),
                "source_birkhoff_mmd": float(loss.birkhoff_mmd.detach()),
                "source_coefficient_consistency": float(
                    loss.coefficient_consistency.detach()
                ),
                "source_support_penalty": float(loss.support_penalty.detach()),
                "source_support_binary_penalty": float(
                    loss.support_binary_penalty.detach()
                ),
                "source_coefficient_penalty": float(loss.coefficient_penalty.detach()),
                "source_raw_coefficient_penalty": float(
                    loss.raw_coefficient_penalty.detach()
                ),
                "source_teacher_coefficient_mse": float(
                    loss.teacher_coefficient_mse.detach()
                ),
                "rollout_weight": rollout_weight,
                "rollout_horizon_steps": horizon_steps,
                "context_residual_weight": current_residual_weight,
                "weak_form_weight": current_weak_form_weight,
                "birkhoff_mmd_weight": current_birkhoff_mmd_weight,
                "validation_status": (
                    "finite" if validation_is_finite else "nonfinite_rollout"
                ),
                "validation_rollout_mse": (
                    validation_mse if validation_is_finite else None
                ),
            }
            history.append(record)
            if validation_is_finite and validation_mse < best_validation:
                best_validation = validation_mse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None or not np.isfinite(best_validation):
        raise RuntimeError("no finite source-validation checkpoint was produced")
    model.load_state_dict(best_state, strict=True)
    checkpoint_path = save_checkpoint(
        output_dir / "checkpoint.pt",
        model,
        training_manifest={
            "schema": "amortized-sindy-training-manifest-v1",
            "source_dataset_ids": ["synthetic-multifamily-v1/source_train"],
            "excluded_evaluation_dataset_ids": [
                "synthetic-multifamily-v1/lorenz_heldout",
                "CTF4Science/ODE_Lorenz",
            ],
            "target_evaluation_data_used_for_training": False,
            **(
                {
                    "foundation_encoder_provenance": foundation_encoder_provenance,
                    "source_embedding_manifest_sha256": (
                        foundation_encoder_provenance["embedding_manifests"]["source"]
                    ),
                    "source_validation_embedding_manifest_sha256": (
                        foundation_encoder_provenance["embedding_manifests"][
                            "validation"
                        ]
                    ),
                    "foundation_encoder_spec": foundation_encoder_provenance[
                        "encoder"
                    ],
                    "foundation_encoder_weights_sha256": (
                        foundation_encoder_provenance["model_weights"]["sha256"]
                    ),
                }
                if foundation_encoder_provenance is not None
                else {}
            ),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "source_train_sha256": sha256_file(source_train_path),
            "source_validation_sha256": sha256_file(source_validation_path),
            "source_teacher_labels_sha256": (
                sha256_file(source_teacher_labels_path)
                if source_teacher_labels_path is not None
                else None
            ),
            "source_teacher_manifest_sha256": (
                sha256_file(source_teacher_manifest_path)
                if source_teacher_manifest_path is not None
                else None
            ),
            "source_embeddings_sha256": (
                sha256_file(source_embeddings_path)
                if source_embeddings_path is not None
                else None
            ),
            "source_validation_embeddings_sha256": (
                sha256_file(source_validation_embeddings_path)
                if source_validation_embeddings_path is not None
                else None
            ),
            "source_families": manifest["source_families"],
            "selection_families": manifest["validation_families"],
            "best_epoch": best_epoch,
            "best_validation_rollout_mse": best_validation,
            "validation_constant_field_mse": validation_constant_field_mse,
            "training_seed": seed,
            "training_device": str(runtime_device),
            "cpu_threads": cpu_threads,
            "training_epochs_budget": epochs,
            "learning_rate": learning_rate,
            "eval_interval": eval_interval,
            "stopped_epoch": stopped_epoch,
            "stop_reason": stop_reason,
            "warmup_epochs": warmup_epochs,
            "rollout_error_cap": rollout_error_cap,
            "rollout_loss_kind": rollout_loss_kind,
            "model_config": model.config.to_dict(),
            "loss_weights": {
                "support": support_weight,
                "support_binary": support_binary_weight,
                "coefficient": coefficient_weight,
                "raw_coefficient": raw_coefficient_weight,
                "context_residual": context_residual_weight,
                "weak_form": weak_form_weight,
                "birkhoff_mmd": birkhoff_mmd_weight,
                "birkhoff_mmd_bandwidths": list(birkhoff_mmd_bandwidths),
                "consistency": consistency_weight,
                "teacher": teacher_weight,
            },
        },
    )
    history_path = atomic_write_json(output_dir / "training_history.json", history)
    return atomic_write_json(output_dir / "manifest.json", {
        "schema": "amortized-sindy-multifamily-training-v1",
        "status": "complete",
        "best_epoch": best_epoch,
        "best_validation_rollout_mse": best_validation,
        "validation_constant_field_mse": validation_constant_field_mse,
        "stopped_epoch": stopped_epoch,
        "stop_reason": stop_reason,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
        },
        "training_history": {
            "path": history_path.name,
            "sha256": sha256_file(history_path),
        },
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train amortized SINDy on source trajectories with source-only selection."
    )
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-validation", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-teacher-labels")
    parser.add_argument("--source-teacher-manifest")
    parser.add_argument("--source-embeddings")
    parser.add_argument("--source-validation-embeddings")
    parser.add_argument("--source-embedding-manifest")
    parser.add_argument("--source-validation-embedding-manifest")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--encoder-type",
        choices=(
            "gru",
            "weak_stats",
            "external",
            "external_mlp",
            "external_birkhoff_weak",
        ),
        default="weak_stats",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--hybrid-ablation",
        choices=("full", "no_birkhoff", "no_weak", "no_birkhoff_no_weak"),
        default="full",
    )
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument(
        "--rollout-loss-kind",
        choices=("clipped_mse", "pseudo_huber"),
        default="clipped_mse",
    )
    parser.add_argument("--rollout-error-cap", type=float, default=10.0)
    parser.add_argument(
        "--birkhoff-mmd-bandwidths",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0, 4.0),
    )
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument(
        "--training-support-mode",
        choices=("soft", "straight_through"),
        default="soft",
    )
    parser.add_argument("--support-binary-weight", type=float, default=0.0)
    parser.add_argument("--raw-coefficient-weight", type=float, default=0.0)
    parser.add_argument("--context-residual-weight", type=float, default=1e-2)
    parser.add_argument("--weak-form-weight", type=float, default=0.0)
    parser.add_argument("--birkhoff-mmd-weight", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(train_multifamily_checkpoint(
        source_train_path=args.source_train,
        source_validation_path=args.source_validation,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
        source_teacher_labels_path=args.source_teacher_labels,
        source_teacher_manifest_path=args.source_teacher_manifest,
        source_embeddings_path=args.source_embeddings,
        source_validation_embeddings_path=args.source_validation_embeddings,
        source_embedding_manifest_path=args.source_embedding_manifest,
        source_validation_embedding_manifest_path=(
            args.source_validation_embedding_manifest
        ),
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        encoder_type=args.encoder_type,
        hybrid_ablation=args.hybrid_ablation,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        warmup_epochs=args.warmup_epochs,
        rollout_loss_kind=args.rollout_loss_kind,
        rollout_error_cap=args.rollout_error_cap,
        seed=args.seed,
        device=args.device,
        cpu_threads=args.cpu_threads,
        teacher_weight=args.teacher_weight,
        training_support_mode=args.training_support_mode,
        support_binary_weight=args.support_binary_weight,
        raw_coefficient_weight=args.raw_coefficient_weight,
        context_residual_weight=args.context_residual_weight,
        weak_form_weight=args.weak_form_weight,
        birkhoff_mmd_weight=args.birkhoff_mmd_weight,
        birkhoff_mmd_bandwidths=tuple(args.birkhoff_mmd_bandwidths),
        force=args.force,
    ))


if __name__ == "__main__":
    main()
