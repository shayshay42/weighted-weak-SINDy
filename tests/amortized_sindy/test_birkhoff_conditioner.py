from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from amortized_sindy.v1 import birkhoff_ablation
from amortized_sindy.v1.model import AmortizedSINDy, AmortizedSINDyConfig


def _hybrid_model(*, ablation: str = "full") -> AmortizedSINDy:
    return AmortizedSINDy(AmortizedSINDyConfig(
        state_dimension=3,
        library_degree=2,
        hidden_size=6,
        min_context_steps=8,
        encoder_type="external_birkhoff_weak",
        external_bottleneck_size=16,
        birkhoff_scales=(1.0, 0.5),
        weak_window_length=9,
        weak_stride_steps=4,
        weak_modes=2,
        weak_attention_heads=4,
        hybrid_ablation=ablation,
    ))


def test_smooth_birkhoff_weights_are_endpoint_zero_and_quadrature_aware() -> None:
    times = torch.tensor([[0.0, 0.1, 0.4, 0.8, 1.0]], dtype=torch.float64)
    weights = AmortizedSINDy._smooth_birkhoff_weights(times)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(weights[:, 0], torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(weights[:, -1], torch.zeros(1, dtype=torch.float64))
    assert torch.all(weights[:, 1:-1] > 0)
    constant = torch.full((1, times.shape[1]), 7.25, dtype=torch.float64)
    torch.testing.assert_close(
        torch.sum(weights * constant, dim=1),
        torch.tensor([7.25], dtype=torch.float64),
    )


def test_weak_tokens_encode_an_integration_by_parts_identity() -> None:
    config = AmortizedSINDyConfig(
        state_dimension=1,
        library_degree=1,
        hidden_size=4,
        min_context_steps=8,
        encoder_type="external_birkhoff_weak",
        external_bottleneck_size=8,
        birkhoff_scales=(1.0,),
        weak_window_length=129,
        weak_stride_steps=64,
        weak_modes=2,
        weak_attention_heads=2,
    )
    model = AmortizedSINDy(config)
    times = torch.linspace(0.0, 1.0, 129, dtype=torch.float64).unsqueeze(0)
    states = times.unsqueeze(-1)
    tokens = model._weak_equation_tokens(states, times)
    assert tokens.shape == (1, 2, model.library_size + 1 + 3)
    constant_library_integral = tokens[:, :, 0]
    derivative_target = tokens[:, :, model.library_size]
    torch.testing.assert_close(
        constant_library_integral,
        derivative_target,
        rtol=2e-3,
        atol=2e-3,
    )


def test_hybrid_conditioner_fuses_all_streams_and_is_deterministic() -> None:
    torch.manual_seed(12)
    model = _hybrid_model()
    times = torch.linspace(0.0, 0.19, 20)
    states = torch.stack((
        torch.sin(3.0 * times),
        torch.cos(2.0 * times),
        0.5 * times + times ** 2,
    ), dim=1).unsqueeze(0)
    external = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    model.eval()
    first = model.infer_dynamics(states, times, context_embedding=external)
    second = model.infer_dynamics(states, times, context_embedding=external)
    assert first.coefficients.shape == (1, 10, 3)
    torch.testing.assert_close(
        first.coefficients, second.coefficients, rtol=0.0, atol=0.0
    )
    assert torch.isfinite(first.physical_coefficients).all()

    model.train()
    loss = model.infer_dynamics(
        states, times, hard_support=False, context_embedding=external
    ).effective_coefficients.square().mean()
    loss.backward()
    assert model.coefficient_head.weight.grad is not None
    assert model.birkhoff_encoder is not None
    assert model.birkhoff_encoder[0].weight.grad is not None
    assert model.weak_token_projection is not None
    assert model.weak_token_projection[0].weight.grad is not None


def test_hybrid_ablation_is_checkpointed_in_model_config() -> None:
    model = _hybrid_model(ablation="no_weak")
    restored = AmortizedSINDyConfig.from_dict(model.config.to_dict())
    assert restored.hybrid_ablation == "no_weak"
    assert tuple(restored.birkhoff_scales) == (1.0, 0.5)


def test_ablation_runner_selects_only_on_source_validation(
    tmp_path, monkeypatch
) -> None:
    inputs = {}
    for name in (
        "source_train",
        "source_validation",
        "dataset_manifest",
        "source_embeddings",
        "source_validation_embeddings",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        inputs[name] = path
    scores = {
        "no_birkhoff_no_weak": 1.4,
        "no_weak": 1.1,
        "no_birkhoff": 1.2,
        "full": 0.9,
    }
    calls = []

    def fake_train(**kwargs):
        calls.append(kwargs["hybrid_ablation"])
        output = kwargs["output_dir"]
        output.mkdir(parents=True)
        checkpoint = output / "checkpoint.pt"
        checkpoint.write_bytes(kwargs["hybrid_ablation"].encode("utf-8"))
        path = output / "manifest.json"
        path.write_text(json.dumps({
            "schema": "amortized-sindy-multifamily-training-v1",
            "status": "complete",
            "best_epoch": 3,
            "best_validation_rollout_mse": scores[kwargs["hybrid_ablation"]],
            "validation_constant_field_mse": 1.6,
            "stopped_epoch": 3,
            "stop_reason": "training_budget_exhausted",
            "checkpoint": {
                "path": checkpoint.name,
                "sha256": birkhoff_ablation.sha256_file(checkpoint),
            },
        }), encoding="utf-8")
        return path

    monkeypatch.setattr(
        birkhoff_ablation, "train_multifamily_checkpoint", fake_train
    )

    def fake_load_checkpoint(path):
        ablation = path.read_text(encoding="utf-8")
        return SimpleNamespace(config=SimpleNamespace(
            encoder_type="external_birkhoff_weak",
            hybrid_ablation=ablation,
        )), {
            "training_seed": 91,
            "training_epochs_budget": 3,
            "training_device": "cpu",
            "source_train_sha256": birkhoff_ablation.sha256_file(
                inputs["source_train"]
            ),
            "source_validation_sha256": birkhoff_ablation.sha256_file(
                inputs["source_validation"]
            ),
        }

    monkeypatch.setattr(birkhoff_ablation, "load_checkpoint", fake_load_checkpoint)
    manifest_path = birkhoff_ablation.run_birkhoff_ablation(
        source_train_path=inputs["source_train"],
        source_validation_path=inputs["source_validation"],
        dataset_manifest_path=inputs["dataset_manifest"],
        source_embeddings_path=inputs["source_embeddings"],
        source_validation_embeddings_path=inputs[
            "source_validation_embeddings"
        ],
        output_dir=tmp_path / "runs",
        epochs=3,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_variant"] == "birkhoff_weak_tokens"
    assert manifest["selection_scope"] == "source_validation_only"
    assert manifest["target_evaluation_data_used"] is False
    assert len(manifest["results"]) == 4
    assert len(calls) == 4

    birkhoff_ablation.run_birkhoff_ablation(
        source_train_path=inputs["source_train"],
        source_validation_path=inputs["source_validation"],
        dataset_manifest_path=inputs["dataset_manifest"],
        source_embeddings_path=inputs["source_embeddings"],
        source_validation_embeddings_path=inputs[
            "source_validation_embeddings"
        ],
        output_dir=tmp_path / "runs",
        epochs=3,
        resume=True,
    )
    assert len(calls) == 4
