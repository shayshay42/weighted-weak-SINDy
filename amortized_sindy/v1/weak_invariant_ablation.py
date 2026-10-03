from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint, sha256_file
from .foundation_embed import (
    bind_foundation_embedding_manifests,
    load_foundation_embedding_manifest,
)
from .foundation_encoder import validate_foundation_encoder_binding
from .io import atomic_write_json
from .multifamily_data import DATASET_SCHEMA
from .multifamily_train import train_multifamily_checkpoint


CONFIG_SCHEMA = "amortized-sindy-weak-invariant-loss-ablation-config-v2"
RESULT_SCHEMA = "amortized-sindy-weak-invariant-loss-ablation-v2"
CHILD_SCHEMA = "amortized-sindy-multifamily-training-v1"
TRAINING_MANIFEST_SCHEMA = "amortized-sindy-training-manifest-v1"
FOUNDATION_MANIFEST_SCHEMA = "amortized-sindy-foundation-embeddings-v1"

VARIANT_ORDER = (
    "strong_control",
    "weak_form",
    "strong_birkhoff_mmd",
    "weak_birkhoff_mmd",
)
EXPECTED_RAW_GRU_MSE = 1.2122557163238525
EXPECTED_CONTEXT_RESIDUAL_WEIGHT = 0.01
EXPECTED_WEAK_FORM_WEIGHT = 0.01
EXPECTED_BIRKHOFF_MMD_WEIGHT = 0.05


def _load_json(path: str | Path, *, label: str) -> Any:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _finite_float(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _same_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_frozen_config(config: Any) -> dict[str, Any]:
    config = _require_mapping(config, label="weak-invariant config")
    _require_exact_keys(
        config,
        {
            "schema",
            "status",
            "execution_amendment",
            "information_contract",
            "expected_inputs",
            "foundation_encoder",
            "architecture",
            "training",
            "variants",
            "selection",
        },
        label="weak-invariant config",
    )
    if config["schema"] != CONFIG_SCHEMA:
        raise ValueError("invalid weak-invariant config schema")
    if config["status"] != "frozen_hyperparameters_with_runtime_guard_amendment":
        raise ValueError("weak-invariant config has an invalid amendment status")
    amendment = _require_mapping(
        config["execution_amendment"], label="execution amendment"
    )
    _require_exact_keys(
        amendment,
        {
            "recorded_on",
            "trigger_code_commit",
            "scope",
            "hyperparameters_changed",
            "target_evaluation_data_used",
        },
        label="execution amendment",
    )
    if amendment != {
        "recorded_on": "2026-08-24",
        "trigger_code_commit": "d92cb1cc73d68df5219428690a7e40e55f3a9aa2",
        "scope": (
            "encode non-finite validation rollouts as explicit standard-JSON "
            "evaluation records without changing training or checkpoint selection"
        ),
        "hyperparameters_changed": False,
        "target_evaluation_data_used": False,
    }:
        raise ValueError("weak-invariant execution amendment differs from the audit")

    contract = _require_mapping(
        config["information_contract"], label="information contract"
    )
    _require_exact_keys(
        contract,
        {
            "selection_data",
            "target_evaluation_data_used",
            "target_optimizer_steps",
            "target_sparse_regression_solves",
            "excluded_evaluation_data",
        },
        label="information contract",
    )
    if contract["selection_data"] != "synthetic-multifamily-v1/source_validation":
        raise ValueError("selection must use source validation only")
    if contract["target_evaluation_data_used"] is not False:
        raise ValueError("target evaluation data must be excluded")
    if (
        contract["target_optimizer_steps"] != 0
        or contract["target_sparse_regression_solves"] != 0
    ):
        raise ValueError("target-time fitting is forbidden")
    excluded = contract["excluded_evaluation_data"]
    if not isinstance(excluded, list) or set(excluded) != {
        "synthetic-multifamily-v1/lorenz_heldout",
        "CTF4Science/ODE_Lorenz",
    }:
        raise ValueError("the frozen config must exclude Lorenz and CTF4Science")

    expected_inputs = _require_mapping(
        config["expected_inputs"], label="expected inputs"
    )
    _require_exact_keys(
        expected_inputs,
        {
            "dataset_manifest_sha256",
            "source_train_sha256",
            "source_validation_sha256",
            "source_embeddings_sha256",
            "source_validation_embeddings_sha256",
            "source_embedding_manifest_sha256",
            "source_validation_embedding_manifest_sha256",
        },
        label="expected inputs",
    )
    if not all(_is_sha256(value) for value in expected_inputs.values()):
        raise ValueError("every frozen input hash must be SHA-256")

    foundation = _require_mapping(
        config["foundation_encoder"], label="foundation encoder"
    )
    _require_exact_keys(
        foundation,
        {
            "backend",
            "model_id",
            "revision",
            "weights_sha256",
            "pooling",
            "context_steps",
            "frozen",
            "pretraining_exposure",
        },
        label="foundation encoder",
    )
    if (
        foundation["backend"] != "panda_patchtst"
        or foundation["model_id"] != "GilpinLab/panda"
        or foundation["pooling"] != "channel_patch_mean_flatten"
        or foundation["context_steps"] != 128
        or foundation["frozen"] is not True
        or foundation["pretraining_exposure"] != "unknown"
        or not isinstance(foundation["revision"], str)
        or not foundation["revision"]
        or not _is_sha256(foundation["weights_sha256"])
    ):
        raise ValueError("invalid frozen Panda embedding contract")

    architecture = _require_mapping(config["architecture"], label="architecture")
    _require_exact_keys(
        architecture,
        {
            "encoder_type",
            "hybrid_ablation",
            "representation_branches",
            "state_dimension",
            "library_degree",
            "library_terms",
            "coefficient_count",
            "support_threshold",
            "weak_window_length",
            "weak_stride_steps",
            "weak_modes",
            "external_bottleneck_size",
            "birkhoff_scales",
            "weak_attention_heads",
            "weak_transformer_layers",
        },
        label="architecture",
    )
    branches = _require_mapping(
        architecture["representation_branches"], label="representation branches"
    )
    _require_exact_keys(
        branches, {"birkhoff", "weak_tokens"}, label="representation branches"
    )
    expected_architecture = {
        "encoder_type": "external_birkhoff_weak",
        "hybrid_ablation": "no_birkhoff_no_weak",
        "state_dimension": 3,
        "library_degree": 2,
        "library_terms": 10,
        "coefficient_count": 30,
        "support_threshold": 0.5,
        "weak_window_length": 33,
        "weak_stride_steps": 16,
        "weak_modes": 2,
        "external_bottleneck_size": 128,
        "birkhoff_scales": [1.0, 0.5],
        "weak_attention_heads": 4,
        "weak_transformer_layers": 1,
    }
    for key, expected in expected_architecture.items():
        if architecture[key] != expected:
            raise ValueError(f"frozen architecture mismatch for {key}")
    if branches != {"birkhoff": False, "weak_tokens": False}:
        raise ValueError(
            "Birkhoff and weak-token representation branches must be disabled"
        )

    training = _require_mapping(config["training"], label="training")
    _require_exact_keys(
        training,
        {
            "epochs",
            "learning_rate",
            "eval_interval",
            "warmup_epochs",
            "seed",
            "cpu_threads",
            "training_support_mode",
            "rollout_loss_kind",
            "rollout_error_cap",
            "birkhoff_mmd_bandwidths",
            "base_loss_weights",
        },
        label="training",
    )
    for key in ("epochs", "eval_interval", "seed", "cpu_threads"):
        if not isinstance(training[key], int) or isinstance(training[key], bool):
            raise ValueError(f"training {key} must be an integer")
    if training["epochs"] < 1 or training["eval_interval"] < 1:
        raise ValueError("invalid training budget")
    if training["cpu_threads"] != 32:
        raise ValueError("the frozen CPU training protocol requires 32 threads")
    if (
        not isinstance(training["warmup_epochs"], int)
        or training["warmup_epochs"] < 0
        or training["warmup_epochs"] >= training["epochs"]
    ):
        raise ValueError("invalid warmup budget")
    if (
        _finite_float(
            training["learning_rate"], label="learning rate", nonnegative=True
        )
        <= 0
    ):
        raise ValueError("learning rate must be positive")
    if training["training_support_mode"] != "straight_through":
        raise ValueError("the ablation requires straight-through support")
    if training["rollout_loss_kind"] != "pseudo_huber":
        raise ValueError("the ablation requires pseudo-Huber rollout loss")
    if (
        _finite_float(
            training["rollout_error_cap"], label="pseudo-Huber scale", nonnegative=True
        )
        <= 0
    ):
        raise ValueError("pseudo-Huber scale must be positive")
    bandwidths = training["birkhoff_mmd_bandwidths"]
    if not isinstance(bandwidths, list) or not bandwidths:
        raise ValueError("Birkhoff MMD bandwidths must be a non-empty list")
    if any(
        _finite_float(value, label="Birkhoff MMD bandwidth", nonnegative=True) <= 0
        for value in bandwidths
    ):
        raise ValueError("Birkhoff MMD bandwidths must be positive")

    base_weights = _require_mapping(
        training["base_loss_weights"], label="base loss weights"
    )
    _require_exact_keys(
        base_weights,
        {
            "support",
            "support_binary",
            "coefficient",
            "raw_coefficient",
            "consistency",
            "teacher",
        },
        label="base loss weights",
    )
    expected_base_weights = {
        "support": 5e-4,
        "support_binary": 0.0,
        "coefficient": 1e-6,
        "raw_coefficient": 0.0,
        "consistency": 1e-3,
        "teacher": 0.0,
    }
    for key, expected in expected_base_weights.items():
        if not _same_float(base_weights[key], expected):
            raise ValueError(f"frozen base loss weight mismatch for {key}")

    variants = config["variants"]
    if not isinstance(variants, list) or len(variants) != len(VARIANT_ORDER):
        raise ValueError("the config must contain the matched four-way ablation")
    by_name: dict[str, dict[str, Any]] = {}
    for record in variants:
        record = _require_mapping(record, label="variant")
        _require_exact_keys(
            record,
            {
                "name",
                "context_residual_weight",
                "weak_form_weight",
                "birkhoff_mmd_weight",
            },
            label="variant",
        )
        name = record["name"]
        if not isinstance(name, str) or name in by_name:
            raise ValueError("variant names must be unique strings")
        for key in (
            "context_residual_weight",
            "weak_form_weight",
            "birkhoff_mmd_weight",
        ):
            _finite_float(record[key], label=f"{name} {key}", nonnegative=True)
        by_name[name] = record
    if tuple(record["name"] for record in variants) != VARIANT_ORDER:
        raise ValueError("variant order or membership differs from the frozen design")
    expected_variants = {
        "strong_control": (EXPECTED_CONTEXT_RESIDUAL_WEIGHT, 0.0, 0.0),
        "weak_form": (0.0, EXPECTED_WEAK_FORM_WEIGHT, 0.0),
        "strong_birkhoff_mmd": (
            EXPECTED_CONTEXT_RESIDUAL_WEIGHT,
            0.0,
            EXPECTED_BIRKHOFF_MMD_WEIGHT,
        ),
        "weak_birkhoff_mmd": (
            0.0,
            EXPECTED_WEAK_FORM_WEIGHT,
            EXPECTED_BIRKHOFF_MMD_WEIGHT,
        ),
    }
    for name, expected in expected_variants.items():
        actual = by_name[name]
        values = (
            actual["context_residual_weight"],
            actual["weak_form_weight"],
            actual["birkhoff_mmd_weight"],
        )
        if not all(_same_float(left, right) for left, right in zip(values, expected)):
            raise ValueError(f"loss weights differ from the frozen {name} arm")

    selection = _require_mapping(config["selection"], label="selection")
    _require_exact_keys(
        selection,
        {
            "metric",
            "lower_is_better",
            "reference_raw_gru_mse",
            "must_beat_raw_gru",
            "must_beat_constant_field",
            "target_evaluation_allowed_only_if_gate_passed",
        },
        label="selection",
    )
    if (
        selection["metric"] != "source_validation_rollout_mse"
        or selection["lower_is_better"] is not True
        or not _same_float(selection["reference_raw_gru_mse"], EXPECTED_RAW_GRU_MSE)
        or selection["must_beat_raw_gru"] is not True
        or selection["must_beat_constant_field"] is not True
        or selection["target_evaluation_allowed_only_if_gate_passed"] is not True
    ):
        raise ValueError("invalid source-selection gate")
    return config


def _validate_input_hashes(
    config: dict[str, Any],
    *,
    dataset_manifest_path: Path,
    source_train_path: Path,
    source_validation_path: Path,
    source_embeddings_path: Path,
    source_validation_embeddings_path: Path,
    source_embedding_manifest_path: Path,
    source_validation_embedding_manifest_path: Path,
) -> dict[str, str]:
    paths = {
        "dataset_manifest_sha256": dataset_manifest_path,
        "source_train_sha256": source_train_path,
        "source_validation_sha256": source_validation_path,
        "source_embeddings_sha256": source_embeddings_path,
        "source_validation_embeddings_sha256": source_validation_embeddings_path,
        "source_embedding_manifest_sha256": source_embedding_manifest_path,
        "source_validation_embedding_manifest_sha256": (
            source_validation_embedding_manifest_path
        ),
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required ablation input is missing: {path}")
        digest = sha256_file(path)
        if digest != config["expected_inputs"][name]:
            raise ValueError(f"{name} differs from the frozen config")
        hashes[name] = digest
    dataset_manifest = _require_mapping(
        _load_json(dataset_manifest_path, label="dataset manifest"),
        label="dataset manifest",
    )
    if dataset_manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid multifamily dataset manifest")
    if dataset_manifest.get("lorenz_used_for_training_or_selection") is not False:
        raise ValueError("Lorenz data must not be used for training or selection")
    if "lorenz_heldout" not in dataset_manifest.get("heldout_evaluation_families", []):
        raise ValueError("dataset manifest does not reserve held-out Lorenz")
    for artifact_name, digest_name in (
        ("source_train", "source_train_sha256"),
        ("source_validation", "source_validation_sha256"),
    ):
        if (
            dataset_manifest.get("artifacts", {}).get(artifact_name, {}).get("sha256")
            != hashes[digest_name]
        ):
            raise ValueError(f"{artifact_name} is not bound to the dataset manifest")
    return hashes


def _validate_embedding_manifest(
    *,
    manifest_path: Path,
    embedding_path: Path,
    context_path: Path,
    artifact_role: str,
    foundation_config: dict[str, Any],
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"embedding manifest is missing: {manifest_path}")
    manifest = _require_mapping(
        _load_json(manifest_path, label=f"{artifact_role} embedding manifest"),
        label=f"{artifact_role} embedding manifest",
    )
    if manifest.get("schema") != FOUNDATION_MANIFEST_SCHEMA:
        raise ValueError(f"invalid {artifact_role} embedding manifest schema")
    if manifest.get("artifact_role") != artifact_role:
        raise ValueError(f"embedding manifest role differs for {artifact_role}")
    if manifest.get("context_artifact_sha256") != sha256_file(context_path):
        raise ValueError(f"embedding context hash differs for {artifact_role}")
    if manifest.get("future_states_read") is not False:
        raise ValueError("foundation embedding extraction must not read future states")
    if manifest.get("forecast_generation_called") is not False:
        raise ValueError("foundation embedding extraction must not call forecasting")
    encoder = _require_mapping(manifest.get("encoder"), label="embedding encoder")
    expected_encoder = {
        "backend": foundation_config["backend"],
        "model_id": foundation_config["model_id"],
        "revision": foundation_config["revision"],
        "state_dimension": 3,
        "context_length": foundation_config["context_steps"],
        "pooling": foundation_config["pooling"],
        "target_dataset_id": "CTF4Science/ODE_Lorenz",
        "target_pretraining_exposure": foundation_config["pretraining_exposure"],
        "adaptation_label": "target-time-zero-update",
        "strict_dataset_zero_shot": False,
    }
    if encoder != expected_encoder:
        raise ValueError(f"{artifact_role} embedding encoder differs from frozen Panda")
    embedding_artifact = _require_mapping(
        manifest.get("embedding_artifact"), label="embedding artifact"
    )
    if embedding_artifact.get("sha256") != sha256_file(embedding_path):
        raise ValueError(f"embedding artifact hash differs for {artifact_role}")
    recorded_path = Path(str(embedding_artifact.get("path", "")))
    if recorded_path.is_absolute() or recorded_path.parent != Path("."):
        raise ValueError("embedding artifact path must be a local basename")
    if (manifest_path.parent / recorded_path).resolve() != embedding_path.resolve():
        raise ValueError(f"embedding artifact path differs for {artifact_role}")
    model_weights = _require_mapping(
        manifest.get("model_weights"), label="embedding model weights"
    )
    if model_weights.get("sha256") != foundation_config["weights_sha256"]:
        raise ValueError("Panda weight hash differs from the frozen config")
    with np.load(embedding_path, allow_pickle=False) as loaded:
        if set(loaded.files) != {"embeddings", "group_ids", "trajectory_ids"}:
            raise ValueError("embedding artifact contains unexpected arrays")
        shape = list(np.asarray(loaded["embeddings"]).shape)
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("embedding artifact must have shape [trajectory, feature]")
    if manifest.get("embedding_shape") != shape:
        raise ValueError("embedding shape differs from its manifest")
    provenance = load_foundation_embedding_manifest(
        manifest_path,
        embeddings_path=embedding_path,
        context_path=context_path,
        expected_artifact_role=artifact_role,
        expected_embedding_shape=tuple(shape),
    )
    return {
        "sha256": sha256_file(manifest_path),
        "embedding_shape": shape,
        "model_weights_sha256": model_weights["sha256"],
        "provenance": provenance,
    }


def _variant_train_kwargs(
    config: dict[str, Any],
    *,
    variant: dict[str, Any],
    device: str,
    cpu_threads: int,
    force: bool,
) -> dict[str, Any]:
    training = config["training"]
    base = training["base_loss_weights"]
    return {
        "epochs": training["epochs"],
        "encoder_type": config["architecture"]["encoder_type"],
        "hybrid_ablation": config["architecture"]["hybrid_ablation"],
        "learning_rate": training["learning_rate"],
        "eval_interval": training["eval_interval"],
        "warmup_epochs": training["warmup_epochs"],
        "rollout_error_cap": training["rollout_error_cap"],
        "rollout_loss_kind": training["rollout_loss_kind"],
        "support_weight": base["support"],
        "support_binary_weight": base["support_binary"],
        "coefficient_weight": base["coefficient"],
        "raw_coefficient_weight": base["raw_coefficient"],
        "context_residual_weight": variant["context_residual_weight"],
        "weak_form_weight": variant["weak_form_weight"],
        "birkhoff_mmd_weight": variant["birkhoff_mmd_weight"],
        "birkhoff_mmd_bandwidths": tuple(training["birkhoff_mmd_bandwidths"]),
        "consistency_weight": base["consistency"],
        "training_support_mode": training["training_support_mode"],
        "teacher_weight": base["teacher"],
        "seed": training["seed"],
        "device": device,
        "cpu_threads": cpu_threads,
        "force": force,
    }


def _validate_history(
    history_path: Path,
    *,
    child_manifest: dict[str, Any],
    variant: dict[str, Any],
    training: dict[str, Any],
) -> None:
    history = _load_json(history_path, label="training history")
    if not isinstance(history, list) or not history:
        raise ValueError("training history must be a non-empty list")
    epochs = []
    evaluation_records: dict[int, dict[str, Any]] = {}
    validation_records: dict[int, dict[str, Any]] = {}
    stop_records: list[dict[str, Any]] = []
    for record in history:
        record = _require_mapping(record, label="training history record")
        epoch = record.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
            raise ValueError("training history epoch must be a positive integer")
        epochs.append(epoch)
        if record.get("status") == "stopped_before_optimizer_step":
            if (
                "validation_rollout_mse" in record
                or "validation_status" in record
            ):
                raise ValueError("explicit stop record contains validation fields")
            stop_records.append(record)
            continue
        if "validation_rollout_mse" not in record or "validation_status" not in record:
            raise ValueError("scheduled evaluation record is incomplete")
        validation_status = record["validation_status"]
        if validation_status == "finite":
            value = _finite_float(
                record["validation_rollout_mse"],
                label="validation rollout MSE",
                nonnegative=True,
            )
            if not math.isfinite(value):
                raise ValueError("validation rollout MSE must be finite")
            validation_records[epoch] = record
        elif validation_status == "nonfinite_rollout":
            if record["validation_rollout_mse"] is not None:
                raise ValueError(
                    "non-finite validation rollout must be encoded as JSON null"
                )
        else:
            raise ValueError("invalid validation rollout status")
        evaluation_records[epoch] = record
        expected_warmup = epoch <= training["warmup_epochs"]
        expected_context = (
            (
                variant["context_residual_weight"]
                if variant["weak_form_weight"] > 0
                else max(variant["context_residual_weight"], 0.1)
            )
            if expected_warmup
            else variant["context_residual_weight"]
        )
        expected_weak = (
            max(variant["weak_form_weight"], 0.1)
            if expected_warmup and variant["weak_form_weight"] > 0
            else variant["weak_form_weight"]
        )
        expected_mmd = 0.0 if expected_warmup else variant["birkhoff_mmd_weight"]
        for key, expected in (
            ("context_residual_weight", expected_context),
            ("weak_form_weight", expected_weak),
            ("birkhoff_mmd_weight", expected_mmd),
        ):
            if not _same_float(record.get(key), expected):
                raise ValueError(f"training history protocol mismatch for {key}")
    if epochs != sorted(epochs) or len(epochs) != len(set(epochs)):
        raise ValueError("training history epochs must be strictly increasing")
    stopped_epoch = child_manifest["stopped_epoch"]
    stop_reason = child_manifest["stop_reason"]
    if stop_reason == "training_budget_exhausted":
        if stopped_epoch != training["epochs"]:
            raise ValueError("exhausted training did not reach the frozen epoch budget")
        expected_epochs = [
            epoch
            for epoch in range(1, training["epochs"] + 1)
            if epoch == 1
            or epoch % training["eval_interval"] == 0
            or epoch == training["epochs"]
        ]
        if stop_records:
            raise ValueError("budget-exhausted history contains a stop record")
        if set(evaluation_records) != set(expected_epochs):
            raise ValueError("scheduled validation records are incomplete")
    elif stop_reason == "nonfinite_source_loss":
        expected_epochs = [
            epoch
            for epoch in range(1, stopped_epoch)
            if epoch == 1 or epoch % training["eval_interval"] == 0
        ] + [stopped_epoch]
        if len(stop_records) != 1 or stop_records[0] is not history[-1]:
            raise ValueError(
                "non-finite training must have exactly one final stop record"
            )
        stop_record = stop_records[0]
        if (
            stop_record.get("status") != "stopped_before_optimizer_step"
            or stop_record.get("reason") != stop_reason
            or "validation_rollout_mse" in stop_record
        ):
            raise ValueError("non-finite stop record is inconsistent")
        if set(evaluation_records) != set(expected_epochs[:-1]):
            raise ValueError(
                "scheduled validation records before the stop are incomplete"
            )
    else:
        raise ValueError("unsupported training stop reason")
    if epochs != expected_epochs:
        raise ValueError("training history does not match the frozen evaluation schedule")
    best_epoch = child_manifest["best_epoch"]
    if best_epoch not in validation_records:
        raise ValueError("best epoch is absent from training history")
    if not _same_float(
        validation_records[best_epoch]["validation_rollout_mse"],
        child_manifest["best_validation_rollout_mse"],
    ):
        raise ValueError("best validation metric differs from training history")
    minimum = min(
        float(record["validation_rollout_mse"])
        for record in validation_records.values()
    )
    if not _same_float(minimum, child_manifest["best_validation_rollout_mse"]):
        raise ValueError("child manifest does not identify the best validation epoch")
    if epochs[-1] != child_manifest["stopped_epoch"]:
        raise ValueError("training history does not end at the recorded stop epoch")


def _validate_child(
    *,
    output_dir: Path,
    variant: dict[str, Any],
    config: dict[str, Any],
    input_hashes: dict[str, str],
    dataset_manifest: dict[str, Any],
    foundation_encoder_provenance: dict[str, Any],
    embedding_width: int,
    device: str,
) -> dict[str, Any]:
    name = variant["name"]
    variant_dir = output_dir / name
    expected_files = {"checkpoint.pt", "training_history.json", "manifest.json"}
    actual_files = {path.name for path in variant_dir.iterdir()}
    if actual_files != expected_files or any(
        path.is_dir() for path in variant_dir.iterdir()
    ):
        raise ValueError(f"resume directory for {name} has unexpected contents")
    manifest_path = variant_dir / "manifest.json"
    child = _require_mapping(
        _load_json(manifest_path, label=f"{name} child manifest"),
        label=f"{name} child manifest",
    )
    _require_exact_keys(
        child,
        {
            "schema",
            "status",
            "best_epoch",
            "best_validation_rollout_mse",
            "validation_constant_field_mse",
            "stopped_epoch",
            "stop_reason",
            "checkpoint",
            "training_history",
        },
        label=f"{name} child manifest",
    )
    if child["schema"] != CHILD_SCHEMA or child["status"] != "complete":
        raise ValueError(f"incomplete child manifest for {name}")
    for key in ("best_epoch", "stopped_epoch"):
        if not isinstance(child[key], int) or isinstance(child[key], bool):
            raise ValueError(f"invalid {key} for {name}")
    if (
        not 1
        <= child["best_epoch"]
        <= child["stopped_epoch"]
        <= config["training"]["epochs"]
    ):
        raise ValueError(f"invalid epoch bounds for {name}")
    best_mse = _finite_float(
        child["best_validation_rollout_mse"],
        label=f"{name} validation MSE",
        nonnegative=True,
    )
    constant_mse = _finite_float(
        child["validation_constant_field_mse"],
        label=f"{name} constant-field MSE",
        nonnegative=True,
    )
    if child["stop_reason"] not in {
        "training_budget_exhausted",
        "nonfinite_source_loss",
    }:
        raise ValueError(f"invalid stop reason for {name}")

    artifact_paths: dict[str, Path] = {}
    for key in ("checkpoint", "training_history"):
        record = _require_mapping(child[key], label=f"{name} {key}")
        _require_exact_keys(record, {"path", "sha256"}, label=f"{name} {key}")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or relative.parent != Path("."):
            raise ValueError(f"{name} {key} path must be a local basename")
        artifact_path = variant_dir / relative
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != record["sha256"]
        ):
            raise ValueError(f"{name} {key} hash mismatch")
        artifact_paths[key] = artifact_path

    model, training_manifest = load_checkpoint(artifact_paths["checkpoint"])
    expected_training_keys = {
        "schema",
        "source_dataset_ids",
        "excluded_evaluation_dataset_ids",
        "target_evaluation_data_used_for_training",
        "dataset_manifest_sha256",
        "source_train_sha256",
        "source_validation_sha256",
        "source_teacher_labels_sha256",
        "source_teacher_manifest_sha256",
        "source_embeddings_sha256",
        "source_validation_embeddings_sha256",
        "source_families",
        "selection_families",
        "best_epoch",
        "best_validation_rollout_mse",
        "validation_constant_field_mse",
        "training_seed",
        "training_device",
        "cpu_threads",
        "training_epochs_budget",
        "learning_rate",
        "eval_interval",
        "stopped_epoch",
        "stop_reason",
        "warmup_epochs",
        "rollout_error_cap",
        "rollout_loss_kind",
        "model_config",
        "loss_weights",
    }
    provenance_training_keys = {
        "foundation_encoder_provenance",
        "source_embedding_manifest_sha256",
        "source_validation_embedding_manifest_sha256",
        "foundation_encoder_spec",
        "foundation_encoder_weights_sha256",
    }
    actual_training_keys = set(training_manifest)
    missing_training_keys = expected_training_keys - actual_training_keys
    unexpected_training_keys = actual_training_keys - (
        expected_training_keys | provenance_training_keys
    )
    if missing_training_keys or unexpected_training_keys:
        raise ValueError(
            f"{name} training manifest fields differ: "
            f"missing={sorted(missing_training_keys)}, "
            f"unexpected={sorted(unexpected_training_keys)}"
        )
    present_provenance_keys = actual_training_keys & provenance_training_keys
    if present_provenance_keys != provenance_training_keys:
        raise ValueError(
            f"{name} checkpoint must contain the complete foundation encoder "
            "provenance binding"
        )
    if training_manifest["schema"] != TRAINING_MANIFEST_SCHEMA:
        raise ValueError(f"invalid checkpoint training manifest for {name}")
    if training_manifest["target_evaluation_data_used_for_training"] is not False:
        raise ValueError(f"target data leakage reported for {name}")
    if training_manifest["source_dataset_ids"] != [
        "synthetic-multifamily-v1/source_train"
    ]:
        raise ValueError(f"source dataset identity differs for {name}")
    validate_foundation_encoder_binding(
        training_manifest["foundation_encoder_provenance"]
    )
    if (
        training_manifest["foundation_encoder_provenance"]
        != foundation_encoder_provenance
    ):
        raise ValueError(f"foundation encoder provenance differs for {name}")
    expected_provenance_fields = {
        "source_embedding_manifest_sha256": foundation_encoder_provenance[
            "embedding_manifests"
        ]["source"],
        "source_validation_embedding_manifest_sha256": (
            foundation_encoder_provenance["embedding_manifests"]["validation"]
        ),
        "foundation_encoder_spec": foundation_encoder_provenance["encoder"],
        "foundation_encoder_weights_sha256": foundation_encoder_provenance[
            "model_weights"
        ]["sha256"],
    }
    for key, expected in expected_provenance_fields.items():
        if training_manifest[key] != expected:
            raise ValueError(f"foundation encoder binding mismatch for {name}: {key}")
    if set(training_manifest["excluded_evaluation_dataset_ids"]) != {
        "synthetic-multifamily-v1/lorenz_heldout",
        "CTF4Science/ODE_Lorenz",
    }:
        raise ValueError(f"excluded target datasets differ for {name}")
    expected_hash_fields = {
        "dataset_manifest_sha256": "dataset_manifest_sha256",
        "source_train_sha256": "source_train_sha256",
        "source_validation_sha256": "source_validation_sha256",
        "source_embeddings_sha256": "source_embeddings_sha256",
        "source_validation_embeddings_sha256": "source_validation_embeddings_sha256",
    }
    for manifest_key, input_key in expected_hash_fields.items():
        if training_manifest[manifest_key] != input_hashes[input_key]:
            raise ValueError(
                f"checkpoint input hash mismatch for {name}: {manifest_key}"
            )
    if (
        training_manifest["source_teacher_labels_sha256"] is not None
        or training_manifest["source_teacher_manifest_sha256"] is not None
    ):
        raise ValueError(f"teacher artifacts are forbidden in {name}")
    if training_manifest["source_families"] != dataset_manifest["source_families"]:
        raise ValueError(f"source families differ for {name}")
    if (
        training_manifest["selection_families"]
        != dataset_manifest["validation_families"]
    ):
        raise ValueError(f"selection families differ for {name}")

    training = config["training"]
    expected_protocol = {
        "training_seed": training["seed"],
        "training_device": device,
        "cpu_threads": (
            training["cpu_threads"] if torch.device(device).type == "cpu" else 1
        ),
        "training_epochs_budget": training["epochs"],
        "learning_rate": training["learning_rate"],
        "eval_interval": training["eval_interval"],
        "warmup_epochs": training["warmup_epochs"],
        "rollout_error_cap": training["rollout_error_cap"],
        "rollout_loss_kind": training["rollout_loss_kind"],
    }
    for key, expected in expected_protocol.items():
        actual = training_manifest[key]
        equal = (
            _same_float(actual, expected)
            if isinstance(expected, float)
            else actual == expected
        )
        if not equal:
            raise ValueError(f"checkpoint protocol mismatch for {name}: {key}")
    for key in ("best_epoch", "stopped_epoch", "stop_reason"):
        if training_manifest[key] != child[key]:
            raise ValueError(f"child/checkpoint mismatch for {name}: {key}")
    for key in ("best_validation_rollout_mse", "validation_constant_field_mse"):
        if not _same_float(training_manifest[key], child[key]):
            raise ValueError(f"child/checkpoint mismatch for {name}: {key}")

    model_config = model.config.to_dict()
    if training_manifest["model_config"] != model_config:
        raise ValueError(f"serialized model config differs for {name}")
    architecture = config["architecture"]
    expected_model_fields = {
        "state_dimension": architecture["state_dimension"],
        "library_degree": architecture["library_degree"],
        "hidden_size": embedding_width,
        "support_threshold": architecture["support_threshold"],
        "training_support_mode": training["training_support_mode"],
        "encoder_type": architecture["encoder_type"],
        "weak_window_length": architecture["weak_window_length"],
        "weak_stride_steps": architecture["weak_stride_steps"],
        "weak_modes": architecture["weak_modes"],
        "external_bottleneck_size": architecture["external_bottleneck_size"],
        "birkhoff_scales": tuple(architecture["birkhoff_scales"]),
        "weak_attention_heads": architecture["weak_attention_heads"],
        "weak_transformer_layers": architecture["weak_transformer_layers"],
        "hybrid_ablation": architecture["hybrid_ablation"],
    }
    for key, expected in expected_model_fields.items():
        if model_config[key] != expected:
            raise ValueError(f"model architecture mismatch for {name}: {key}")

    expected_losses = {
        "support": training["base_loss_weights"]["support"],
        "support_binary": training["base_loss_weights"]["support_binary"],
        "coefficient": training["base_loss_weights"]["coefficient"],
        "raw_coefficient": training["base_loss_weights"]["raw_coefficient"],
        "context_residual": variant["context_residual_weight"],
        "weak_form": variant["weak_form_weight"],
        "birkhoff_mmd": variant["birkhoff_mmd_weight"],
        "birkhoff_mmd_bandwidths": training["birkhoff_mmd_bandwidths"],
        "consistency": training["base_loss_weights"]["consistency"],
        "teacher": training["base_loss_weights"]["teacher"],
    }
    actual_losses = training_manifest["loss_weights"]
    _require_exact_keys(
        actual_losses, set(expected_losses), label=f"{name} loss weights"
    )
    for key, expected in expected_losses.items():
        actual = actual_losses[key]
        if isinstance(expected, list):
            equal = actual == expected
        else:
            equal = _same_float(actual, expected)
        if not equal:
            raise ValueError(f"loss protocol mismatch for {name}: {key}")

    _validate_history(
        artifact_paths["training_history"],
        child_manifest=child,
        variant=variant,
        training=training,
    )
    return {
        "variant": name,
        "objective": {
            "context_residual_weight": variant["context_residual_weight"],
            "weak_form_weight": variant["weak_form_weight"],
            "birkhoff_mmd_weight": variant["birkhoff_mmd_weight"],
        },
        "best_epoch": child["best_epoch"],
        "source_validation_rollout_mse": best_mse,
        "constant_field_mse": constant_mse,
        "stopped_epoch": child["stopped_epoch"],
        "stop_reason": child["stop_reason"],
        "training_device": training_manifest["training_device"],
        "checkpoint": {
            "path": artifact_paths["checkpoint"].relative_to(output_dir).as_posix(),
            "sha256": sha256_file(artifact_paths["checkpoint"]),
        },
        "training_history": {
            "path": artifact_paths["training_history"]
            .relative_to(output_dir)
            .as_posix(),
            "sha256": sha256_file(artifact_paths["training_history"]),
        },
        "child_manifest": {
            "path": manifest_path.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
    }


def run_weak_invariant_ablation(
    *,
    config_path: str | Path,
    source_train_path: str | Path,
    source_validation_path: str | Path,
    dataset_manifest_path: str | Path,
    source_embeddings_path: str | Path,
    source_validation_embeddings_path: str | Path,
    source_embedding_manifest_path: str | Path,
    source_validation_embedding_manifest_path: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    cpu_threads: int = 32,
    resume: bool = False,
    force: bool = False,
) -> Path:
    """Run a frozen source-only 2x2 weak-form/invariant-loss ablation."""
    if resume and force:
        raise ValueError("resume and force are mutually exclusive")
    if not isinstance(device, str) or not device:
        raise ValueError("training device must be a non-empty string")
    try:
        runtime_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid training device: {device}") from error
    device = str(runtime_device)
    if (
        not isinstance(cpu_threads, int)
        or isinstance(cpu_threads, bool)
        or cpu_threads < 1
    ):
        raise ValueError("CPU thread count must be a positive integer")
    paths = {
        "config": Path(config_path),
        "source_train": Path(source_train_path),
        "source_validation": Path(source_validation_path),
        "dataset_manifest": Path(dataset_manifest_path),
        "source_embeddings": Path(source_embeddings_path),
        "source_validation_embeddings": Path(source_validation_embeddings_path),
        "source_embedding_manifest": Path(source_embedding_manifest_path),
        "source_validation_embedding_manifest": Path(
            source_validation_embedding_manifest_path
        ),
    }
    if not paths["config"].is_file():
        raise FileNotFoundError(f"frozen config is missing: {paths['config']}")
    config = _validate_frozen_config(
        _load_json(paths["config"], label="weak-invariant config")
    )
    if (
        runtime_device.type == "cpu"
        and cpu_threads != config["training"]["cpu_threads"]
    ):
        raise ValueError("CPU run differs from the frozen 32-thread protocol")
    if runtime_device.type == "cuda" and cpu_threads not in {
        1,
        config["training"]["cpu_threads"],
    }:
        raise ValueError("CUDA runs accept the default thread value or an explicit 1")
    trainer_cpu_threads = (
        config["training"]["cpu_threads"] if runtime_device.type == "cpu" else 1
    )
    input_hashes = _validate_input_hashes(
        config,
        dataset_manifest_path=paths["dataset_manifest"],
        source_train_path=paths["source_train"],
        source_validation_path=paths["source_validation"],
        source_embeddings_path=paths["source_embeddings"],
        source_validation_embeddings_path=paths["source_validation_embeddings"],
        source_embedding_manifest_path=paths["source_embedding_manifest"],
        source_validation_embedding_manifest_path=paths[
            "source_validation_embedding_manifest"
        ],
    )
    dataset_manifest = _require_mapping(
        _load_json(paths["dataset_manifest"], label="dataset manifest"),
        label="dataset manifest",
    )
    source_embedding_manifest = _validate_embedding_manifest(
        manifest_path=paths["source_embedding_manifest"],
        embedding_path=paths["source_embeddings"],
        context_path=paths["source_train"],
        artifact_role="source_train",
        foundation_config=config["foundation_encoder"],
    )
    validation_embedding_manifest = _validate_embedding_manifest(
        manifest_path=paths["source_validation_embedding_manifest"],
        embedding_path=paths["source_validation_embeddings"],
        context_path=paths["source_validation"],
        artifact_role="source_validation",
        foundation_config=config["foundation_encoder"],
    )
    if source_embedding_manifest["embedding_shape"][1] != (
        validation_embedding_manifest["embedding_shape"][1]
    ):
        raise ValueError("source and validation embedding widths differ")
    embedding_width = int(source_embedding_manifest["embedding_shape"][1])
    foundation_encoder_provenance = bind_foundation_embedding_manifests(
        source_embedding_manifest["provenance"],
        validation_embedding_manifest["provenance"],
    )

    output_dir = Path(output_dir)
    final_path = output_dir / "ablation_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not (resume or force):
        raise FileExistsError(f"ablation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_entries = {*VARIANT_ORDER, final_path.name}
    unexpected_entries = {path.name for path in output_dir.iterdir()} - allowed_entries
    if unexpected_entries:
        raise ValueError(
            f"ablation output contains unexpected entries: {sorted(unexpected_entries)}"
        )
    if final_path.exists() and not resume and not force:
        raise FileExistsError(f"ablation manifest already exists: {final_path}")
    if resume and final_path.is_file():
        incomplete_children = [
            name
            for name in VARIANT_ORDER
            if not (output_dir / name / "manifest.json").is_file()
        ]
        if incomplete_children:
            raise ValueError(
                "final ablation manifest exists but child runs are incomplete: "
                f"{incomplete_children}"
            )

    variant_by_name = {record["name"]: record for record in config["variants"]}
    results = []
    for name in VARIANT_ORDER:
        variant = variant_by_name[name]
        variant_dir = output_dir / name
        child_manifest_path = variant_dir / "manifest.json"
        if resume and variant_dir.exists() and any(variant_dir.iterdir()):
            if not child_manifest_path.is_file():
                raise ValueError(
                    f"cannot resume incomplete child directory: {variant_dir}"
                )
        else:
            child_manifest_path = train_multifamily_checkpoint(
                source_train_path=paths["source_train"],
                source_validation_path=paths["source_validation"],
                dataset_manifest_path=paths["dataset_manifest"],
                output_dir=variant_dir,
                source_embeddings_path=paths["source_embeddings"],
                source_validation_embeddings_path=paths["source_validation_embeddings"],
                source_embedding_manifest_path=paths["source_embedding_manifest"],
                source_validation_embedding_manifest_path=paths[
                    "source_validation_embedding_manifest"
                ],
                **_variant_train_kwargs(
                    config,
                    variant=variant,
                    device=device,
                    cpu_threads=trainer_cpu_threads,
                    force=force,
                ),
            )
        if child_manifest_path != variant_dir / "manifest.json":
            raise ValueError(f"trainer returned an unexpected manifest path for {name}")
        results.append(
            _validate_child(
                output_dir=output_dir,
                variant=variant,
                config=config,
                input_hashes=input_hashes,
                dataset_manifest=dataset_manifest,
                foundation_encoder_provenance=foundation_encoder_provenance,
                embedding_width=embedding_width,
                device=device,
            )
        )

    constant_values = [record["constant_field_mse"] for record in results]
    if not all(_same_float(value, constant_values[0]) for value in constant_values[1:]):
        raise ValueError("constant-field reference differs across matched variants")
    constant_mse = float(constant_values[0])
    winner = min(results, key=lambda record: record["source_validation_rollout_mse"])
    raw_gru_mse = float(config["selection"]["reference_raw_gru_mse"])
    for record in results:
        record["beats_raw_gru"] = record["source_validation_rollout_mse"] < raw_gru_mse
        record["beats_constant_field"] = (
            record["source_validation_rollout_mse"] < constant_mse
        )
    source_gate_passed = (
        winner["source_validation_rollout_mse"] < raw_gru_mse
        and winner["source_validation_rollout_mse"] < constant_mse
    )
    final_manifest = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "selection_scope": "source_validation_only",
        "target_evaluation_data_used": False,
        "target_optimizer_steps": 0,
        "target_sparse_regression_solves": 0,
        "frozen_config": {
            "sha256": sha256_file(paths["config"]),
            "schema": config["schema"],
            "status": config["status"],
            "execution_amendment": config["execution_amendment"],
        },
        "inputs": {
            **input_hashes,
            "source_embedding_manifest_sha256": source_embedding_manifest["sha256"],
            "source_validation_embedding_manifest_sha256": (
                validation_embedding_manifest["sha256"]
            ),
            "embedding_width": embedding_width,
            "panda_weights_sha256": source_embedding_manifest["model_weights_sha256"],
            "foundation_encoder_provenance": foundation_encoder_provenance,
        },
        "matched_protocol": {
            "encoder_type": config["architecture"]["encoder_type"],
            "hybrid_ablation": config["architecture"]["hybrid_ablation"],
            "representation_branches": config["architecture"][
                "representation_branches"
            ],
            "training_support_mode": config["training"]["training_support_mode"],
            "rollout_loss_kind": config["training"]["rollout_loss_kind"],
            "rollout_error_cap": config["training"]["rollout_error_cap"],
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["learning_rate"],
            "eval_interval": config["training"]["eval_interval"],
            "warmup_epochs": config["training"]["warmup_epochs"],
            "seed": config["training"]["seed"],
            "cpu_threads": (
                config["training"]["cpu_threads"]
                if runtime_device.type == "cpu"
                else 1
            ),
            "base_loss_weights": config["training"]["base_loss_weights"],
            "birkhoff_mmd_bandwidths": config["training"]["birkhoff_mmd_bandwidths"],
            "device": device,
        },
        "results": results,
        "selection": {
            "metric": config["selection"]["metric"],
            "selected_variant": winner["variant"],
            "selected_source_validation_rollout_mse": winner[
                "source_validation_rollout_mse"
            ],
            "reference_raw_gru_mse": raw_gru_mse,
            "constant_field_mse": constant_mse,
            "source_gate_passed": source_gate_passed,
            "reserved_lorenz_evaluation_eligible": source_gate_passed,
            "ctf4science_authorized": False,
            "ctf4science_requires": "a separately recorded passing reserved-Lorenz advancement gate",
        },
    }
    if resume and final_path.is_file():
        existing = _load_json(final_path, label="resumed ablation manifest")
        if existing != final_manifest:
            raise ValueError(
                "existing ablation manifest fails stringent resume validation"
            )
        return final_path
    return atomic_write_json(final_path, final_manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen source-only weak-form/invariant-loss ablation on "
            "external Panda embeddings."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-validation", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--source-embeddings", required=True)
    parser.add_argument("--source-validation-embeddings", required=True)
    parser.add_argument("--source-embedding-manifest", required=True)
    parser.add_argument("--source-validation-embedding-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=32,
        help="PyTorch intra-op threads (frozen at 32 on CPU; CUDA uses 1)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(
        run_weak_invariant_ablation(
            config_path=args.config,
            source_train_path=args.source_train,
            source_validation_path=args.source_validation,
            dataset_manifest_path=args.dataset_manifest,
            source_embeddings_path=args.source_embeddings,
            source_validation_embeddings_path=args.source_validation_embeddings,
            source_embedding_manifest_path=args.source_embedding_manifest,
            source_validation_embedding_manifest_path=(
                args.source_validation_embedding_manifest
            ),
            output_dir=args.output_dir,
            device=args.device,
            cpu_threads=args.cpu_threads,
            resume=args.resume,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
