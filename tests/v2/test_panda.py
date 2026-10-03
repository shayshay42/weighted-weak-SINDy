from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from lorenz63_benchmark.v2.evaluate import contextual_forecast_view
from lorenz63_benchmark.v2.panda import PandaForecastAdapter, create_panda_checkpoint


class _FakeConfig:
    context_length = 4
    prediction_length = 2
    num_parallel_samples = 100


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(7))
        self.config = _FakeConfig()

    @property
    def device(self) -> torch.device:
        return self.weight.device


class _FakePipeline:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.grad_enabled: list[bool] = []

    @property
    def device(self) -> torch.device:
        return self.model.device

    def predict(self, context: torch.Tensor, prediction_length: int, **_kwargs) -> torch.Tensor:
        self.grad_enabled.append(torch.is_grad_enabled())
        steps = torch.arange(1, prediction_length + 1, dtype=context.dtype)
        future = context[:, -1:, :] + 0.5 * steps[None, :, None]
        return future[:, None]


def _checkpoint() -> dict:
    return {
        "schema": "lorenz63-panda-checkpoint-v2",
        "method": "panda_zero_shot",
        "model": {
            "model_id": "fake/panda",
            "model_revision": "abc",
            "source_repository": "https://example.test/panda",
            "source_revision": "def",
            "context_length": 4,
            "prediction_length": 2,
            "batch_size": 2,
            "dtype": "float32",
            "sliding_context": True,
            "inference_seed": 99,
        },
        "normalization": {
            "state_mean": [10.0, 20.0, 30.0],
            "state_std": [2.0, 4.0, 6.0],
            "time_scale": 0.9,
        },
        "observation_dt": 0.01,
    }


def test_panda_forecast_uses_context_and_restores_physical_units() -> None:
    pipeline = _FakePipeline()
    adapter = PandaForecastAdapter(_checkpoint(), pipeline=pipeline)
    normalized = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3) / 10.0
    mean = np.asarray([10.0, 20.0, 30.0])
    std = np.asarray([2.0, 4.0, 6.0])
    context = mean + std * normalized
    times = np.arange(4) * 0.01
    prediction = adapter.forecast_from_context(context, times)
    expected_normalized = np.concatenate([
        normalized[:, -1:, :],
        normalized[:, -1:, :] + 0.5 * np.arange(1, 4)[None, :, None],
    ], axis=1)
    np.testing.assert_allclose(prediction, mean + std * expected_normalized)
    assert pipeline.grad_enabled and not any(pipeline.grad_enabled)
    assert adapter.parameter_count == 7
    assert adapter.last_surrogate_evaluations == 2
    assert adapter.inference_seed == 99
    assert adapter.probabilistic is False
    assert adapter.forecast_sample_count == 1
    with pytest.raises(ValueError, match="needs 4 context points"):
        adapter.forecast_from_context(context[:, :3], times)


def test_contextual_forecast_view_has_shared_origin() -> None:
    states = np.arange(2 * 8 * 3).reshape(2, 8, 3)
    times = np.arange(8) * 0.01
    context, forecast_times, truth, origin = contextual_forecast_view(states, times, 4)
    assert origin == 3
    np.testing.assert_array_equal(context, states[:, :4])
    np.testing.assert_array_equal(truth, states[:, 3:])
    np.testing.assert_allclose(forecast_times, np.arange(5) * 0.01)


def test_panda_checkpoint_records_external_model_without_fitting(tmp_path) -> None:
    config = {"panda": _checkpoint()["model"]}
    train = {
        "metadata": {
            "dt": 0.01,
            "normalization": {
                "state_mean": [1.0, 2.0, 3.0],
                "state_std": [4.0, 5.0, 6.0],
                "time_scale": 0.9,
            },
        }
    }
    path, training = create_panda_checkpoint(train, config, tmp_path)
    checkpoint = json.loads(path.read_text())
    assert checkpoint["benchmark_fitting_performed"] is False
    assert checkpoint["information_contract"] == [
        "external_pretraining_corpus", "observed_state_context"
    ]
    assert training["optimizer_updates"] == 0


def test_panda_can_forecast_prefix_normalized_per_trajectory() -> None:
    pipeline = _FakePipeline()
    adapter = PandaForecastAdapter(_checkpoint(), pipeline=pipeline)
    context = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) / 10.0
    prediction = adapter.forecast_normalized_context(context, np.arange(4) * 0.01)
    expected = np.concatenate([
        context[:, -1:, :],
        context[:, -1:, :] + 0.5 * np.arange(1, 4)[None, :, None],
    ], axis=1)
    np.testing.assert_allclose(prediction, expected)
