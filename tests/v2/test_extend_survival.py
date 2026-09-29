from __future__ import annotations

import pandas as pd

from lorenz63_benchmark.v2.contracts import PRETRAINED_TRACK
from lorenz63_benchmark.v2.extend_survival import _repeat_pretrained_reference


def test_pretrained_reference_is_repeated_with_explicit_source_provenance() -> None:
    runs = pd.DataFrame([
        {
            "run_id": "chronos__d1__m0__n0",
            "method": "chronos_zero_shot",
            "track": PRETRAINED_TRACK,
            "data_seed": 1,
            "model_seed": 0,
            "noise_level": 0.0,
            "forecast_context_steps": 512,
        }
    ])
    trajectories = pd.DataFrame([
        {
            "run_id": "chronos__d1__m0__n0",
            "method": "chronos_zero_shot",
            "track": PRETRAINED_TRACK,
            "data_seed": 1,
            "model_seed": 0,
            "trajectory_id": 7,
            "noise_level": 0.0,
        }
    ])
    repeated_runs, repeated_trajectories = _repeat_pretrained_reference(
        runs,
        trajectories,
        method="chronos_zero_shot",
        panel_noises=(0.001, 0.01, 0.05),
        target_model_seeds=(0, 1, 2),
    )
    assert len(repeated_runs) == 9
    assert len(repeated_trajectories) == 9
    assert set(repeated_runs["noise_level"]) == {0.001, 0.01, 0.05}
    assert set(repeated_runs["model_seed"]) == {0, 1, 2}
    assert set(repeated_runs["source_run_id"]) == {"chronos__d1__m0__n0"}
    assert set(repeated_runs["source_noise_level"]) == {0.0}
    assert repeated_runs["fixed_clean_prefix_reference"].all()
    assert repeated_runs["run_id"].is_unique
    assert repeated_trajectories["run_id"].is_unique
