from __future__ import annotations

import numpy as np
import torch

from lorenz63_benchmark.v2.chronos import ChronosForecastAdapter


class _FakePipeline:
    model = None

    def __init__(self) -> None:
        self.calls = 0

    def predict_quantiles(self, *, context, prediction_length, quantile_levels):
        assert quantile_levels == [0.5]
        self.calls += 1
        last = torch.as_tensor(context)[:, -1:]
        mean = last.repeat(1, prediction_length)
        quantiles = mean[:, :, None]
        return quantiles, mean


def _checkpoint() -> dict:
    return {
        "model": {
            "model_id": "test/chronos",
            "model_revision": "0" * 40,
            "context_length": 4,
            "prediction_length": 2,
            "batch_size": 4,
            "dtype": "float32",
            "inference_seed": 1,
        },
        "observation_dt": 0.1,
    }


def test_chronos_adapter_uses_context_and_recurses_in_blocks() -> None:
    pipeline = _FakePipeline()
    adapter = ChronosForecastAdapter(_checkpoint(), pipeline=pipeline)
    context = np.asarray([
        [[0.0, 10.0, -3.0], [1.0, 11.0, -2.0], [2.0, 12.0, -1.0], [3.0, 13.0, 0.0]],
        [[4.0, 2.0, 9.0], [5.0, 3.0, 8.0], [6.0, 4.0, 7.0], [7.0, 5.0, 6.0]],
    ])
    prediction = adapter.forecast_from_context(context, np.arange(6) * 0.1)
    assert prediction.shape == (2, 6, 3)
    np.testing.assert_allclose(prediction[:, 0], context[:, -1])
    np.testing.assert_allclose(
        prediction[:, 1:], np.repeat(context[:, -1:, :], 5, axis=1), atol=1e-7
    )
    # Six univariate series, batch size four, three recursive forecast blocks.
    assert pipeline.calls == 6
    assert adapter.last_surrogate_evaluations == 6
