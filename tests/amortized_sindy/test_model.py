from __future__ import annotations

import inspect
import json
import sys
import types

import numpy as np
import pytest
import torch

from amortized_sindy.v1.checkpoint import save_checkpoint, sha256_file
from amortized_sindy.v1.ctf_adapter import CTFZeroShotSINDy
from amortized_sindy.v1 import ctf_evaluate, ctf_predict
from amortized_sindy.v1.library import (
    normalized_to_physical_coefficients,
    polynomial_exponents,
    polynomial_library,
)
from amortized_sindy.v1.model import AmortizedSINDy, AmortizedSINDyConfig
from amortized_sindy.v1.smoke import run_smoke
from amortized_sindy.v1.training import source_training_loss


def _manifest() -> dict[str, object]:
    return {
        "schema": "amortized-sindy-training-manifest-v1",
        "source_dataset_ids": ["synthetic-linear-family-v1"],
        "excluded_evaluation_dataset_ids": ["CTF4Science/ODE_Lorenz"],
        "target_evaluation_data_used_for_training": False,
    }


def _constant_model() -> AmortizedSINDy:
    config = AmortizedSINDyConfig(
        state_dimension=1,
        library_degree=1,
        hidden_size=4,
        min_context_steps=4,
    )
    model = AmortizedSINDy(config)
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.zero_()
        model.coefficient_head.weight.zero_()
        model.coefficient_head.bias.copy_(torch.tensor([0.0, -0.2]))
        model.support_head.weight.zero_()
        model.support_head.bias.fill_(20.0)
    return model


def test_polynomial_library_is_dimension_generic() -> None:
    exponents = polynomial_exponents(2, 2)
    assert exponents == ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    states = torch.tensor([[2.0, 3.0]])
    values = polynomial_library(states, exponents)
    torch.testing.assert_close(values, torch.tensor([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]]))


def test_normalized_coefficients_convert_to_physical_coordinates() -> None:
    coefficients = torch.tensor([[3.0], [7.0], [11.0]])
    physical = normalized_to_physical_coefficients(
        coefficients,
        polynomial_exponents(1, 2),
        state_mean=torch.tensor([2.0]),
        state_scale=torch.tensor([4.0]),
        time_scale=torch.tensor([5.0]),
    )
    torch.testing.assert_close(physical, torch.tensor([[1.8], [-0.8], [0.55]]))


def test_rollout_is_differentiable_during_source_training() -> None:
    torch.manual_seed(4)
    model = AmortizedSINDy(
        AmortizedSINDyConfig(
            state_dimension=1,
            library_degree=1,
            hidden_size=8,
            min_context_steps=4,
        )
    )
    times = torch.linspace(0.0, 0.7, 8)
    context = torch.exp(-0.2 * times).reshape(1, -1, 1)
    offsets = torch.tensor([0.0, 0.1, 0.2])
    future = torch.exp(-0.2 * (times[-1] + offsets)).reshape(1, -1, 1)
    loss = source_training_loss(
        model,
        context_states=context,
        context_times=times,
        forecast_offsets=offsets,
        future_states=future,
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.coefficient_head.weight.grad is not None
    assert torch.isfinite(model.coefficient_head.weight.grad).all()


def test_checkpoint_rejects_target_training_data(tmp_path) -> None:
    manifest = _manifest()
    manifest["target_evaluation_data_used_for_training"] = True
    with pytest.raises(ValueError, match="target evaluation data"):
        save_checkpoint(
            tmp_path / "bad.pt", _constant_model(), training_manifest=manifest
        )


def test_ctf_adapter_requires_explicit_target_dataset_exclusion(tmp_path) -> None:
    manifest = _manifest()
    manifest["excluded_evaluation_dataset_ids"] = []
    checkpoint = save_checkpoint(
        tmp_path / "not_excluded.pt", _constant_model(), training_manifest=manifest
    )
    with pytest.raises(ValueError, match="explicitly exclude"):
        CTFZeroShotSINDy(checkpoint)


def test_ctf_adapter_is_frozen_deterministic_and_has_no_truth_route(tmp_path) -> None:
    checkpoint = save_checkpoint(
        tmp_path / "model.pt", _constant_model(), training_manifest=_manifest()
    )
    adapter = CTFZeroShotSINDy(checkpoint, context_length=-1)
    assert adapter.frozen
    assert "future" not in inspect.signature(adapter.predict).parameters
    assert "truth" not in inspect.signature(adapter.predict).parameters

    times = np.linspace(0.0, 0.7, 8)
    context = np.exp(-0.2 * times)[:, None]
    prediction_times = np.linspace(0.0, 0.5, 6)
    before = {
        name: value.detach().clone()
        for name, value in adapter.model.state_dict().items()
    }
    first = adapter.predict(
        pair_id=1,
        train_data=[context],
        init_data=None,
        prediction_timesteps=prediction_times,
    )
    second = adapter.predict(
        pair_id=1,
        train_data=[context],
        init_data=None,
        prediction_timesteps=prediction_times,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[0], context[-1], rtol=1e-6, atol=1e-6)
    assert adapter.last_inference is not None
    assert adapter.last_inference["target_optimizer_steps"] == 0
    assert adapter.last_inference["target_sparse_regression_solves"] == 0
    for name, value in adapter.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


def test_ctf_prediction_process_does_not_import_truth_evaluator() -> None:
    source = inspect.getsource(ctf_predict)
    assert "ctf4science.eval_module" not in source
    parameters = inspect.signature(ctf_predict.predict_ctf_pairs).parameters
    assert "truth" not in parameters
    assert "test" not in parameters


def test_ctf_prediction_and_separate_evaluation_processes(
    tmp_path, monkeypatch
) -> None:
    checkpoint = save_checkpoint(
        tmp_path / "model.pt", _constant_model(), training_manifest=_manifest()
    )
    advancement_gate = tmp_path / "ctf-advancement-gate.json"
    advancement_gate.write_text(
        json.dumps(
            {
                "schema": ctf_predict.CTF_ADVANCEMENT_GATE_SCHEMA,
                "selected_checkpoint_sha256": sha256_file(checkpoint),
                "heldout_gate": {
                    "inference_complete": True,
                    "all_predictions_finite": True,
                    "relative_mse_vs_constant_field": 0.9,
                    "relative_mse_vs_constant_field_threshold": 1.0,
                    "passed": True,
                },
                "ctf4science_authorized": True,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "dataset": {"name": "ODE_Lorenz", "pair_id": [1]},
        "model": {
            "name": "AmortizedSINDyZeroShot",
            "zero_shot": True,
            "checkpoint": str(checkpoint),
            "context_length": -1,
            "device": "cpu",
        },
    }
    monkeypatch.setattr(ctf_predict, "_load_yaml", lambda _: config)
    context = np.linspace(1.0, 0.8, 8)[:, None]
    package = types.ModuleType("ctf4science")
    data_module = types.ModuleType("ctf4science.data_module")
    data_module.load_dataset = lambda dataset, pair_id: ([context], None)
    data_module.parse_pair_ids = lambda dataset: [1]
    data_module.get_prediction_timesteps = lambda dataset, pair_id: np.linspace(
        0.0, 0.4, 5
    )
    monkeypatch.setitem(sys.modules, "ctf4science", package)
    monkeypatch.setitem(sys.modules, "ctf4science.data_module", data_module)
    batch_path = ctf_predict.predict_ctf_pairs(
        config_path=tmp_path / "config.yaml",
        advancement_gate_path=advancement_gate,
        output_dir=tmp_path / "predictions",
    )
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    assert batch["zero_shot"] is True

    eval_module = types.ModuleType("ctf4science.eval_module")
    eval_module.evaluate = lambda dataset, pair_id, predictions: {
        "score": float(np.mean(predictions))
    }
    monkeypatch.setitem(sys.modules, "ctf4science.eval_module", eval_module)
    evaluation_path = ctf_evaluate.evaluate_ctf_predictions(
        prediction_batch_manifest=batch_path,
        output_dir=tmp_path / "evaluation",
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["pairs"][0]["pair_id"] == 1
    assert np.isfinite(evaluation["pairs"][0]["metrics"]["score"])


def test_ctf_parametric_pairs_use_only_warm_start_context(tmp_path) -> None:
    checkpoint = save_checkpoint(
        tmp_path / "model.pt", _constant_model(), training_manifest=_manifest()
    )
    adapter = CTFZeroShotSINDy(checkpoint)
    warm_start = np.linspace(1.0, 0.8, 8)[:, None]
    output = adapter.predict(
        pair_id=8,
        train_data=[np.zeros((8, 7)), np.ones((8, 7))],
        init_data=warm_start,
        prediction_timesteps=np.linspace(0.0, 0.4, 5),
    )
    np.testing.assert_allclose(output[0], warm_start[-1], rtol=1e-6, atol=1e-6)
    assert adapter.last_inference is not None
    assert adapter.last_inference["context_steps"] == warm_start.shape[0]


def test_ctf_reconstruction_pairs_roll_out_from_first_observation(tmp_path) -> None:
    checkpoint = save_checkpoint(
        tmp_path / "model.pt", _constant_model(), training_manifest=_manifest()
    )
    adapter = CTFZeroShotSINDy(checkpoint)
    context = np.linspace(1.0, 0.8, 8)[:, None]
    output = adapter.predict(
        pair_id=2,
        train_data=[context],
        init_data=None,
        prediction_timesteps=np.linspace(0.0, 0.4, 5),
    )
    np.testing.assert_allclose(output[0], context[0], rtol=1e-6, atol=1e-6)
    assert adapter.last_inference is not None
    assert adapter.last_inference["task_mode"] == "reconstruction"
    assert adapter.last_inference["rollout_initial_context_index"] == 0


def test_mechanism_smoke_generalizes_to_held_out_parameters() -> None:
    _, metrics = run_smoke(epochs=35, seed=17)
    assert metrics["final_train_normalized_mse"] < (
        metrics["initial_train_normalized_mse"] / 10.0
    )
    assert metrics["held_out_parameter_normalized_mse"] < 0.2
