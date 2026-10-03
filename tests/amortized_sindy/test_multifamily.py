from __future__ import annotations

import inspect
import json

import numpy as np

from amortized_sindy.v1.multifamily_data import generate_multifamily_data
from amortized_sindy.v1.multifamily_evaluate import evaluate_heldout_family
from amortized_sindy.v1.multifamily_infer import infer_heldout_family
from amortized_sindy.v1.multifamily_train import (
    SOURCE_KEYS,
    train_multifamily_checkpoint,
)


def test_multifamily_pipeline_preserves_the_heldout_boundary(tmp_path) -> None:
    data_root = tmp_path / "data"
    dataset_manifest_path = generate_multifamily_data(
        output_root=data_root,
        source_groups_per_family=1,
        validation_groups=1,
        lorenz_groups=1,
        trajectories_per_group=2,
        context_steps=24,
        forecast_steps=5,
    )
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    assert dataset_manifest["lorenz_used_for_training_or_selection"] is False
    assert dataset_manifest["heldout_evaluation_families"] == ["lorenz_heldout"]

    source_path = data_root / "sanitized" / "source_train.npz"
    validation_path = data_root / "sanitized" / "source_validation.npz"
    context_path = data_root / "sanitized" / "lorenz_heldout_context.npz"
    with np.load(source_path, allow_pickle=False) as loaded:
        assert set(loaded.files) == SOURCE_KEYS
        assert "coefficients" not in loaded.files
        assert "family_ids" not in loaded.files
    with np.load(context_path, allow_pickle=False) as loaded:
        assert "future_states" not in loaded.files
        assert "coefficients" not in loaded.files

    training_parameters = inspect.signature(train_multifamily_checkpoint).parameters
    inference_parameters = inspect.signature(infer_heldout_family).parameters
    assert "truth" not in training_parameters
    assert "hidden" not in training_parameters
    assert "lorenz" not in training_parameters
    assert "truth" not in inference_parameters
    assert "hidden" not in inference_parameters

    training_dir = tmp_path / "training"
    training_manifest_path = train_multifamily_checkpoint(
        source_train_path=source_path,
        source_validation_path=validation_path,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=training_dir,
        epochs=4,
        hidden_size=8,
        eval_interval=2,
    )
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    assert np.isfinite(training_manifest["best_validation_rollout_mse"])

    inference_dir = tmp_path / "inference"
    inference_manifest_path = infer_heldout_family(
        checkpoint_path=training_dir / "checkpoint.pt",
        context_path=context_path,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=inference_dir,
    )
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    assert inference_manifest["target_optimizer_steps"] == 0
    assert inference_manifest["target_sparse_regression_solves"] == 0

    evaluation_manifest_path = evaluate_heldout_family(
        inference_manifest_path=inference_manifest_path,
        context_path=context_path,
        truth_path=data_root / "evaluator_only" / "lorenz_heldout_truth.npz",
        hidden_systems_path=data_root / "evaluator_only" / "hidden_systems.npz",
        dataset_manifest_path=dataset_manifest_path,
        output_dir=tmp_path / "evaluation",
    )
    evaluation = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    assert evaluation["selection_role"] == "final_heldout_family_no_tuning"
    assert evaluation["metrics"]["all_predictions_finite"] is True
    assert np.isfinite(evaluation["metrics"]["mean_normalized_mse"])


def _embedding_file(source_path, output_path, width: int = 6):
    with np.load(source_path, allow_pickle=False) as loaded:
        contexts = np.asarray(loaded["context_states"], dtype=np.float64)
        group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        trajectory_ids = np.asarray(loaded["trajectory_ids"], dtype=np.int64)
    summary = np.concatenate(
        (contexts.mean(axis=1), contexts.std(axis=1)), axis=1
    )[:, :width]
    np.savez_compressed(
        output_path,
        embeddings=summary.astype(np.float32),
        group_ids=group_ids,
        trajectory_ids=trajectory_ids,
    )


def test_external_embeddings_train_and_infer_without_target_updates(tmp_path) -> None:
    data_root = tmp_path / "data"
    dataset_manifest_path = generate_multifamily_data(
        output_root=data_root,
        source_groups_per_family=1,
        validation_groups=1,
        lorenz_groups=1,
        trajectories_per_group=2,
        context_steps=24,
        forecast_steps=5,
    )
    source_path = data_root / "sanitized" / "source_train.npz"
    validation_path = data_root / "sanitized" / "source_validation.npz"
    context_path = data_root / "sanitized" / "lorenz_heldout_context.npz"
    source_embeddings = tmp_path / "source_embeddings.npz"
    validation_embeddings = tmp_path / "validation_embeddings.npz"
    heldout_embeddings = tmp_path / "heldout_embeddings.npz"
    _embedding_file(source_path, source_embeddings)
    _embedding_file(validation_path, validation_embeddings)
    _embedding_file(context_path, heldout_embeddings)

    training_dir = tmp_path / "training"
    training_manifest_path = train_multifamily_checkpoint(
        source_train_path=source_path,
        source_validation_path=validation_path,
        dataset_manifest_path=dataset_manifest_path,
        source_embeddings_path=source_embeddings,
        source_validation_embeddings_path=validation_embeddings,
        encoder_type="external",
        output_dir=training_dir,
        epochs=3,
        eval_interval=1,
    )
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    assert np.isfinite(training_manifest["best_validation_rollout_mse"])

    inference_manifest_path = infer_heldout_family(
        checkpoint_path=training_dir / "checkpoint.pt",
        context_path=context_path,
        context_embeddings_path=heldout_embeddings,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=tmp_path / "inference",
    )
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    assert inference_manifest["target_optimizer_steps"] == 0
    assert inference_manifest["context_embeddings_sha256"] is not None


def test_birkhoff_weak_hybrid_trains_and_replays_on_heldout_context(tmp_path) -> None:
    data_root = tmp_path / "data"
    dataset_manifest_path = generate_multifamily_data(
        output_root=data_root,
        source_groups_per_family=1,
        validation_groups=1,
        lorenz_groups=1,
        trajectories_per_group=1,
        context_steps=24,
        forecast_steps=3,
    )
    source_path = data_root / "sanitized" / "source_train.npz"
    validation_path = data_root / "sanitized" / "source_validation.npz"
    context_path = data_root / "sanitized" / "lorenz_heldout_context.npz"
    source_embeddings = tmp_path / "source_embeddings.npz"
    validation_embeddings = tmp_path / "validation_embeddings.npz"
    heldout_embeddings = tmp_path / "heldout_embeddings.npz"
    _embedding_file(source_path, source_embeddings)
    _embedding_file(validation_path, validation_embeddings)
    _embedding_file(context_path, heldout_embeddings)

    training_dir = tmp_path / "training"
    training_manifest_path = train_multifamily_checkpoint(
        source_train_path=source_path,
        source_validation_path=validation_path,
        dataset_manifest_path=dataset_manifest_path,
        source_embeddings_path=source_embeddings,
        source_validation_embeddings_path=validation_embeddings,
        encoder_type="external_birkhoff_weak",
        hybrid_ablation="full",
        output_dir=training_dir,
        epochs=2,
        eval_interval=1,
    )
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    assert np.isfinite(training_manifest["best_validation_rollout_mse"])

    inference_manifest_path = infer_heldout_family(
        checkpoint_path=training_dir / "checkpoint.pt",
        context_path=context_path,
        context_embeddings_path=heldout_embeddings,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=tmp_path / "inference",
    )
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    assert inference_manifest["target_optimizer_steps"] == 0
    assert inference_manifest["target_sparse_regression_solves"] == 0
