from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from amortized_sindy.v1.foundation_encoder import (
    ChronosT5ContextEncoder,
    FoundationEncoderSpec,
    PandaPatchTSTContextEncoder,
)
from amortized_sindy.v1.model import AmortizedSINDy, AmortizedSINDyConfig
from amortized_sindy.v1.tabpfn_conditioner import (
    TabPFNCoefficientConditioner,
    TabPFNCoefficientSpec,
)
from amortized_sindy.v1.library import normalized_to_physical_coefficients
from amortized_sindy.v1.multifamily_data import EXPONENTS
from amortized_sindy.v1.tabpfn_pilot import (
    _fit_source_pca,
    _physical_to_normalized_coefficients,
    _project,
    _rollout,
)


class _FakePandaBackbone:
    def __call__(self, *, past_values, **_):
        # Two deterministic patches with d_model=2 for each channel.
        channel_mean = past_values.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        offsets = torch.tensor(
            [[0.0, 1.0], [2.0, 3.0]], dtype=past_values.dtype
        ).reshape(1, 1, 2, 2)
        return SimpleNamespace(last_hidden_state=channel_mean + offsets)


class _FakePandaPredictionModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(d_model=2)
        self.model = _FakePandaBackbone()
        self.training = True
        self.requires_grad = True

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


def test_panda_encoder_is_frozen_multichannel_and_requires_uniform_time() -> None:
    prediction_model = _FakePandaPredictionModel()
    spec = FoundationEncoderSpec(
        backend="panda_patchtst",
        model_id="GilpinLab/panda_mlm",
        revision="pinned-test-revision",
        state_dimension=3,
        target_pretraining_exposure="unknown",
    )
    encoder = PandaPatchTSTContextEncoder(
        pipeline=SimpleNamespace(model=prediction_model), spec=spec
    )
    states = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)
    times = torch.linspace(0.0, 0.9, 10)
    embedding = encoder.encode(states, times)
    assert embedding.shape == (1, 6)
    assert encoder.output_dimension == 6
    assert prediction_model.training is False
    assert prediction_model.requires_grad is False
    assert spec.adaptation_label == "target-time-zero-update"
    assert spec.strict_dataset_zero_shot is False

    irregular = times.clone()
    irregular[-1] += 0.01
    with pytest.raises(ValueError, match="uniform time"):
        encoder.encode(states, irregular)


class _FakeChronosModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(config=SimpleNamespace(d_model=2))
        self.training = True
        self.requires_grad = True

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


class _FakeChronosPipeline:
    def __init__(self):
        self.model = _FakeChronosModel()
        self.forecast_called = False

    def embed(self, channel_series):
        channel_mean = channel_series.mean(dim=1).reshape(-1, 1, 1)
        offsets = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).reshape(1, 2, 2)
        return channel_mean + offsets, torch.ones(channel_series.shape[0])

    def predict(self, *_args, **_kwargs):
        self.forecast_called = True
        raise AssertionError("the embedding adapter must not forecast")


def test_chronos_encoder_pools_each_channel_without_forecasting() -> None:
    pipeline = _FakeChronosPipeline()
    spec = FoundationEncoderSpec(
        backend="chronos_t5",
        model_id="amazon/chronos-t5-small",
        revision="pinned-test-revision",
        state_dimension=3,
        pooling="channel_token_mean_flatten",
        target_pretraining_exposure="unknown",
    )
    encoder = ChronosT5ContextEncoder(pipeline=pipeline, spec=spec)
    states = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)
    embedding = encoder.encode(states, torch.linspace(0.0, 0.9, 10))
    channel_means = states.mean(dim=1).reshape(-1, 1)
    expected = (channel_means + torch.tensor([[1.0, 2.0]])).reshape(1, -1)
    torch.testing.assert_close(embedding, expected)
    assert embedding.shape == (1, 6)
    assert pipeline.forecast_called is False
    assert pipeline.model.training is False
    assert pipeline.model.requires_grad is False


def test_external_embedding_drives_fixed_sindy_head() -> None:
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
    times = torch.linspace(0.0, 0.7, 8)
    context = torch.exp(-0.2 * times).reshape(1, -1, 1)
    with pytest.raises(ValueError, match="requires a frozen context embedding"):
        model.rollout(context, times, torch.tensor([0.0, 0.1]))
    prediction, _ = model.rollout(
        context,
        times,
        torch.tensor([0.0, 0.1]),
        context_embedding=torch.tensor([[2.0, 3.0]]),
    )
    assert prediction.shape == (1, 2, 1)
    assert torch.isfinite(prediction).all()


class _PerCoefficientMeanRegressor:
    def __init__(self, categorical_index: int):
        self.categorical_index = categorical_index
        self.means = None

    def fit(self, features, targets):
        indices = features[:, self.categorical_index].astype(int)
        self.means = {
            index: float(np.mean(targets[indices == index]))
            for index in np.unique(indices)
        }
        return self

    def predict(self, features):
        indices = features[:, self.categorical_index].astype(int)
        return np.asarray([self.means[index] for index in indices])


def test_tabpfn_conditioner_fits_source_once_and_never_updates_on_target() -> None:
    conditioner = TabPFNCoefficientConditioner(
        TabPFNCoefficientSpec(),
        regressor_factory=lambda **kwargs: _PerCoefficientMeanRegressor(**kwargs),
    )
    source_embeddings = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    source_coefficients = np.arange(18, dtype=np.float64).reshape(3, 3, 2)
    conditioner.fit_source(source_embeddings, source_coefficients)
    predicted = conditioner.predict_coefficients(np.asarray([[0.25, 0.75]]))
    np.testing.assert_allclose(predicted[0], source_coefficients.mean(axis=0))
    with pytest.raises(RuntimeError, match="already frozen"):
        conditioner.fit_source(source_embeddings, source_coefficients)
    manifest = conditioner.protocol_manifest(
        source_dataset_ids=["synthetic-source"],
        excluded_evaluation_dataset_ids=["CTF4Science/ODE_Lorenz"],
    )
    assert manifest["target_conditioner_updates"] == 0
    assert manifest["target_sparse_regression_solves"] == 0


def test_source_pca_and_zero_field_rollout_are_deterministic() -> None:
    source = np.arange(48, dtype=np.float64).reshape(8, 6)
    mean, components, scale, singular_values = _fit_source_pca(
        source, component_count=3
    )
    first = _project(source, mean=mean, components=components, scale=scale)
    second = _project(source, mean=mean, components=components, scale=scale)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (8, 3)
    assert singular_values.shape == (3,)

    initial = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    coefficients = np.zeros((2, 10, 3), dtype=np.float64)
    predictions, failed = _rollout(
        initial, np.asarray([0.0, 0.01, 0.02]), coefficients
    )
    np.testing.assert_allclose(
        predictions, np.broadcast_to(initial[:, None, :], predictions.shape)
    )
    assert not np.any(failed)


def test_physical_and_context_normalized_coefficient_conversion_round_trip() -> None:
    rng = np.random.default_rng(4)
    physical = rng.normal(size=(2, len(EXPONENTS), 3))
    contexts = rng.normal(size=(2, 20, 3)) * np.asarray([2.0, 0.5, 3.0])
    times = np.linspace(2.0, 3.9, 20)
    normalized, mean, scale, duration = _physical_to_normalized_coefficients(
        physical, contexts, times
    )
    recovered = normalized_to_physical_coefficients(
        torch.as_tensor(normalized, dtype=torch.float64),
        EXPONENTS,
        state_mean=torch.as_tensor(mean, dtype=torch.float64),
        state_scale=torch.as_tensor(scale, dtype=torch.float64),
        time_scale=torch.full((2,), duration, dtype=torch.float64),
    ).numpy()
    np.testing.assert_allclose(recovered, physical, rtol=1e-10, atol=1e-10)
