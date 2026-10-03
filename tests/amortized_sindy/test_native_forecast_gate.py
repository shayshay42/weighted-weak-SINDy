from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from amortized_sindy.v1.native_forecast_gate import (
    forecast_chronos_t5_native,
    forecast_panda_native,
)


class _ConstantPanda:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(context_length=8)

    def generate(self, values):
        future = values[:, -1:, :].expand(-1, 4, -1)
        return SimpleNamespace(sequences=future[:, None, :, :])


class _ConstantChronos:
    def predict_quantiles(
        self, *, context, prediction_length, quantile_levels
    ):
        assert quantile_levels == [0.5]
        mean = context[:, -1:].expand(-1, prediction_length)
        return mean[:, :, None], mean


def _contexts() -> np.ndarray:
    first = np.arange(24, dtype=np.float64).reshape(8, 3)
    second = 2.0 * first - 3.0
    return np.stack((first, second))


def test_native_panda_forecast_receives_only_context_and_restores_scale() -> None:
    contexts = _contexts()
    prediction = forecast_panda_native(
        contexts,
        model=_ConstantPanda(),
        future_steps=3,
        batch_size=1,
        seed=7,
    )
    expected = np.broadcast_to(contexts[:, -1:, :], (2, 4, 3))
    np.testing.assert_allclose(prediction, expected)


def test_native_chronos_forecast_preserves_channel_order_and_scale() -> None:
    contexts = _contexts()
    prediction = forecast_chronos_t5_native(
        contexts,
        pipeline=_ConstantChronos(),
        future_steps=3,
        batch_size=2,
    )
    expected = np.broadcast_to(contexts[:, -1:, :], (2, 4, 3))
    np.testing.assert_allclose(prediction, expected)
