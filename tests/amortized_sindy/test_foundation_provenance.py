from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import torch

from amortized_sindy.v1 import ctf_adapter, ctf_predict
from amortized_sindy.v1.checkpoint import load_checkpoint, save_checkpoint, sha256_file
from amortized_sindy.v1.foundation_encoder import (
    FOUNDATION_ENCODER_BINDING_SCHEMA,
    FoundationEncoderSpec,
)
from amortized_sindy.v1.model import AmortizedSINDy, AmortizedSINDyConfig
from amortized_sindy.v1.multifamily_data import generate_multifamily_data
from amortized_sindy.v1.multifamily_infer import infer_heldout_family
from amortized_sindy.v1.multifamily_train import train_multifamily_checkpoint


def _write_embeddings_and_manifest(
    context_path,
    embedding_path,
    manifest_path,
    *,
    artifact_role: str,
    spec: FoundationEncoderSpec,
) -> None:
    with np.load(context_path, allow_pickle=False) as loaded:
        contexts = np.asarray(loaded["context_states"], dtype=np.float64)
        group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        trajectory_ids = np.asarray(loaded["trajectory_ids"], dtype=np.int64)
    embeddings = np.concatenate(
        (contexts.mean(axis=1), contexts.std(axis=1)), axis=1
    ).astype(np.float32)
    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
        group_ids=group_ids,
        trajectory_ids=trajectory_ids,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "amortized-sindy-foundation-embeddings-v1",
                "artifact_role": artifact_role,
                "context_artifact_sha256": sha256_file(context_path),
                "future_states_read": False,
                "forecast_generation_called": False,
                "encoder": spec.to_dict(),
                "embedding_shape": list(embeddings.shape),
                "embedding_artifact": {
                    "path": embedding_path.name,
                    "sha256": sha256_file(embedding_path),
                },
                "model_weights": {
                    "filename": "model.safetensors",
                    "sha256": "a" * 64,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_embedding_manifests_bind_training_and_heldout_identity(tmp_path) -> None:
    data_root = tmp_path / "data"
    dataset_manifest_path = generate_multifamily_data(
        output_root=data_root,
        source_groups_per_family=1,
        validation_groups=1,
        lorenz_groups=1,
        trajectories_per_group=1,
        context_steps=16,
        forecast_steps=2,
    )
    source_path = data_root / "sanitized" / "source_train.npz"
    validation_path = data_root / "sanitized" / "source_validation.npz"
    heldout_path = data_root / "sanitized" / "lorenz_heldout_context.npz"
    spec = FoundationEncoderSpec(
        backend="panda_patchtst",
        model_id="test/panda",
        revision="pinned-revision",
        state_dimension=3,
        context_length=16,
    )
    artifacts = {}
    for role, context_path in (
        ("source_train", source_path),
        ("source_validation", validation_path),
        ("lorenz_heldout_context", heldout_path),
    ):
        embedding_path = tmp_path / f"{role}.npz"
        manifest_path = tmp_path / f"{role}.json"
        _write_embeddings_and_manifest(
            context_path,
            embedding_path,
            manifest_path,
            artifact_role=role,
            spec=spec,
        )
        artifacts[role] = (embedding_path, manifest_path)

    mismatched_validation_manifest = tmp_path / "source-validation-mismatch.json"
    validation_manifest = json.loads(
        artifacts["source_validation"][1].read_text(encoding="utf-8")
    )
    validation_manifest["encoder"]["revision"] = "different-revision"
    mismatched_validation_manifest.write_text(
        json.dumps(validation_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different foundation encoders"):
        train_multifamily_checkpoint(
            source_train_path=source_path,
            source_validation_path=validation_path,
            dataset_manifest_path=dataset_manifest_path,
            source_embeddings_path=artifacts["source_train"][0],
            source_validation_embeddings_path=artifacts["source_validation"][0],
            source_embedding_manifest_path=artifacts["source_train"][1],
            source_validation_embedding_manifest_path=(mismatched_validation_manifest),
            encoder_type="external",
            output_dir=tmp_path / "mismatched-training",
            epochs=2,
            eval_interval=1,
        )

    training_dir = tmp_path / "training"
    train_multifamily_checkpoint(
        source_train_path=source_path,
        source_validation_path=validation_path,
        dataset_manifest_path=dataset_manifest_path,
        source_embeddings_path=artifacts["source_train"][0],
        source_validation_embeddings_path=artifacts["source_validation"][0],
        source_embedding_manifest_path=artifacts["source_train"][1],
        source_validation_embedding_manifest_path=(artifacts["source_validation"][1]),
        encoder_type="external",
        output_dir=training_dir,
        epochs=2,
        eval_interval=1,
    )
    _, training_manifest = load_checkpoint(training_dir / "checkpoint.pt")
    binding = training_manifest["foundation_encoder_provenance"]
    assert binding["schema"] == FOUNDATION_ENCODER_BINDING_SCHEMA
    assert binding["encoder"]["revision"] == "pinned-revision"
    assert training_manifest["foundation_encoder_spec"] == spec.to_dict()
    assert training_manifest["foundation_encoder_weights_sha256"] == "a" * 64
    assert training_manifest["source_embedding_manifest_sha256"] == sha256_file(
        artifacts["source_train"][1]
    )
    assert training_manifest[
        "source_validation_embedding_manifest_sha256"
    ] == sha256_file(artifacts["source_validation"][1])

    inference_manifest_path = infer_heldout_family(
        checkpoint_path=training_dir / "checkpoint.pt",
        context_path=heldout_path,
        context_embeddings_path=artifacts["lorenz_heldout_context"][0],
        context_embeddings_manifest_path=artifacts["lorenz_heldout_context"][1],
        dataset_manifest_path=dataset_manifest_path,
        output_dir=tmp_path / "inference",
    )
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    assert inference_manifest["target_optimizer_steps"] == 0
    assert inference_manifest["context_embeddings_manifest_sha256"] == sha256_file(
        artifacts["lorenz_heldout_context"][1]
    )

    mismatched_manifest = tmp_path / "heldout-mismatch.json"
    mismatch = json.loads(
        artifacts["lorenz_heldout_context"][1].read_text(encoding="utf-8")
    )
    mismatch["encoder"]["revision"] = "different-revision"
    mismatched_manifest.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from checkpoint binding"):
        infer_heldout_family(
            checkpoint_path=training_dir / "checkpoint.pt",
            context_path=heldout_path,
            context_embeddings_path=artifacts["lorenz_heldout_context"][0],
            context_embeddings_manifest_path=mismatched_manifest,
            dataset_manifest_path=dataset_manifest_path,
            output_dir=tmp_path / "mismatched-inference",
        )


class _RecordingFoundationEncoder:
    def __init__(self, spec: FoundationEncoderSpec):
        self.spec = spec
        self.output_dimension = 2
        self.contexts: list[np.ndarray] = []

    def encode(self, states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        self.contexts.append(states.detach().cpu().numpy().copy())
        return torch.tensor([[1.0, -1.0]], dtype=torch.float32)


def _external_checkpoint(path, spec: FoundationEncoderSpec):
    model = AmortizedSINDy(
        AmortizedSINDyConfig(
            state_dimension=1,
            library_degree=1,
            hidden_size=2,
            min_context_steps=4,
            encoder_type="external",
        )
    )
    with torch.no_grad():
        model.coefficient_head.weight.zero_()
        model.coefficient_head.bias.copy_(torch.tensor([0.0, -0.2]))
        model.support_head.weight.zero_()
        model.support_head.bias.fill_(20.0)
    return save_checkpoint(
        path,
        model,
        training_manifest={
            "schema": "amortized-sindy-training-manifest-v1",
            "source_dataset_ids": ["synthetic-source"],
            "excluded_evaluation_dataset_ids": ["CTF4Science/ODE_Lorenz"],
            "target_evaluation_data_used_for_training": False,
            "foundation_encoder_provenance": {
                "schema": FOUNDATION_ENCODER_BINDING_SCHEMA,
                "encoder": spec.to_dict(),
                "model_weights": {
                    "filename": "model.safetensors",
                    "sha256": "b" * 64,
                },
                "embedding_manifests": {
                    "source": "c" * 64,
                    "validation": "d" * 64,
                },
            },
        },
    )


def _write_ctf_advancement_gate(path, checkpoint, *, passed: bool) -> None:
    relative_mse = 0.9 if passed else 1.1
    path.write_text(
        json.dumps(
            {
                "schema": ctf_predict.CTF_ADVANCEMENT_GATE_SCHEMA,
                "selected_checkpoint_sha256": sha256_file(checkpoint),
                "heldout_gate": {
                    "inference_complete": True,
                    "all_predictions_finite": True,
                    "relative_mse_vs_constant_field": relative_mse,
                    "relative_mse_vs_constant_field_threshold": 1.0,
                    "passed": passed,
                },
                "ctf4science_authorized": passed,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_ctf_prediction_builds_live_bound_embedding_and_keeps_reconstruction_head(
    tmp_path, monkeypatch
) -> None:
    spec = FoundationEncoderSpec(
        backend="panda_patchtst",
        model_id="test/panda",
        revision="pinned-revision",
        state_dimension=1,
        context_length=8,
    )
    checkpoint = _external_checkpoint(tmp_path / "external.pt", spec)
    advancement_gate = tmp_path / "ctf-advancement-gate.json"
    _write_ctf_advancement_gate(advancement_gate, checkpoint, passed=True)
    recorder = _RecordingFoundationEncoder(spec)
    monkeypatch.setattr(
        ctf_adapter,
        "load_foundation_encoder",
        lambda bound_spec, device: recorder,
    )
    config = {
        "dataset": {"name": "ODE_Lorenz", "pair_id": [2]},
        "model": {
            "name": "AmortizedSINDyZeroShot",
            "zero_shot": True,
            "checkpoint": str(checkpoint),
            "context_length": -1,
            "device": "cpu",
        },
    }
    monkeypatch.setattr(ctf_predict, "_load_yaml", lambda _: config)
    context = np.linspace(1.0, 0.4, 12)[:, None]
    package = types.ModuleType("ctf4science")
    data_module = types.ModuleType("ctf4science.data_module")
    data_module.load_dataset = lambda dataset, pair_id: ([context], None)
    data_module.parse_pair_ids = lambda dataset: [2]
    data_module.get_prediction_timesteps = lambda dataset, pair_id: np.linspace(
        0.0, 0.4, 5
    )
    monkeypatch.setitem(sys.modules, "ctf4science", package)
    monkeypatch.setitem(sys.modules, "ctf4science.data_module", data_module)

    batch_path = ctf_predict.predict_ctf_pairs(
        config_path=tmp_path / "config.yaml",
        advancement_gate_path=advancement_gate,
        output_dir=tmp_path / "ctf-predictions",
    )
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    inference_path = batch_path.parent / batch["pairs"][0]["inference_manifest_path"]
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    assert batch["adaptation_label"] == "target-time-zero-update"
    assert batch["strict_dataset_zero_shot"] is False
    assert batch["pretraining_exposure"] == "unknown"
    assert batch["ctf_advancement_gate"]["selected_checkpoint_sha256"] == sha256_file(
        checkpoint
    )
    assert len(recorder.contexts) == 1
    np.testing.assert_array_equal(recorder.contexts[0], context[:8].astype(np.float32))
    assert inference["context_steps"] == 8
    assert inference["rollout_initial_context_index"] == 0
    assert inference["target_foundation_encoder_updates"] == 0
    assert inference["strict_dataset_zero_shot"] is False
    assert inference["pretraining_exposure"] == "unknown"
    assert inference["foundation_embedding_sha256"] is not None
    assert inference["ctf_advancement_gate_manifest_sha256"] == sha256_file(
        advancement_gate
    )


def test_ctf_prediction_rejects_failed_heldout_gate_before_target_access(
    tmp_path, monkeypatch
) -> None:
    spec = FoundationEncoderSpec(
        backend="panda_patchtst",
        model_id="test/panda",
        revision="pinned-revision",
        state_dimension=1,
        context_length=8,
    )
    checkpoint = _external_checkpoint(tmp_path / "external.pt", spec)
    advancement_gate = tmp_path / "ctf-advancement-gate.json"
    _write_ctf_advancement_gate(advancement_gate, checkpoint, passed=False)
    config = {
        "dataset": {"name": "ODE_Lorenz", "pair_id": [2]},
        "model": {
            "name": "AmortizedSINDyZeroShot",
            "zero_shot": True,
            "checkpoint": str(checkpoint),
            "context_length": -1,
            "device": "cpu",
        },
    }
    monkeypatch.setattr(ctf_predict, "_load_yaml", lambda _: config)
    accessed_target = False
    package = types.ModuleType("ctf4science")
    data_module = types.ModuleType("ctf4science.data_module")

    def fail_if_accessed(*args, **kwargs):
        nonlocal accessed_target
        accessed_target = True
        raise AssertionError("CTF target data must not be accessed after a failed gate")

    data_module.load_dataset = fail_if_accessed
    data_module.parse_pair_ids = fail_if_accessed
    data_module.get_prediction_timesteps = fail_if_accessed
    monkeypatch.setitem(sys.modules, "ctf4science", package)
    monkeypatch.setitem(sys.modules, "ctf4science.data_module", data_module)

    with pytest.raises(PermissionError, match="denied by the held-out gate"):
        ctf_predict.predict_ctf_pairs(
            config_path=tmp_path / "config.yaml",
            advancement_gate_path=advancement_gate,
            output_dir=tmp_path / "ctf-predictions",
        )
    assert accessed_target is False
    assert not (tmp_path / "ctf-predictions").exists()
