from __future__ import annotations

import numpy as np

from amortized_sindy.v1.odeformer_ctf_predict import select_ctf_context


def test_ctf_context_policy_separates_forecast_reconstruction_and_warm_start() -> None:
    trajectory = np.arange(300 * 3, dtype=np.float64).reshape(300, 3)
    warm_start = -trajectory[:100]
    forecast, forecast_index = select_ctf_context(
        pair_id=1,
        train_data=[trajectory],
        init_data=None,
        context_length=128,
    )
    reconstruction, reconstruction_index = select_ctf_context(
        pair_id=2,
        train_data=[trajectory],
        init_data=None,
        context_length=128,
    )
    parametric, parametric_index = select_ctf_context(
        pair_id=8,
        train_data=[trajectory, trajectory, trajectory],
        init_data=warm_start,
        context_length=128,
    )
    np.testing.assert_array_equal(forecast, trajectory[-128:])
    np.testing.assert_array_equal(reconstruction, trajectory[:128])
    np.testing.assert_array_equal(parametric, warm_start)
    assert forecast_index == parametric_index == -1
    assert reconstruction_index == 0
