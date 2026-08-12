from __future__ import annotations

import numpy as np
import pandas as pd

from lorenz63_benchmark.v2.artifacts import atomic_save_npz
from lorenz63_benchmark.v2.attractor_summary import (
    normalized_xz_histogram,
    return_map_pairs,
    select_representative_cell,
    xz_in_range_fraction,
)
from lorenz63_benchmark.v2.panda_comparison import make_panda_comparison_figures
from lorenz63_benchmark.v2.visualize import (
    _survival_overlay_noises,
    hierarchical_survival_curves,
)


def test_return_map_pairs_extract_consecutive_z_maxima() -> None:
    z = np.asarray([0.0, 2.0, 0.0, 3.0, 0.0, 5.0, 0.0])
    states = np.zeros((1, z.size, 3))
    states[0, :, 2] = z
    np.testing.assert_array_equal(
        return_map_pairs(states),
        np.asarray([[2.0, 3.0], [3.0, 5.0]]),
    )


def test_normalized_histogram_has_unit_mass_and_ignores_nonfinite() -> None:
    states = np.asarray([
        [[-0.5, 0.0, 0.5], [0.5, 0.0, 1.5], [np.nan, 0.0, 1.0]]
    ])
    histogram = normalized_xz_histogram(
        states, np.linspace(-1.0, 1.0, 5), np.linspace(0.0, 2.0, 5)
    )
    assert histogram.shape == (4, 4)
    assert np.isclose(histogram.sum(), 1.0)
    assert np.count_nonzero(histogram) == 2
    assert np.isclose(xz_in_range_fraction(states, np.linspace(-1.0, 1.0, 5), np.linspace(0.0, 2.0, 5)), 1.0)


def test_histogram_retains_out_of_range_mass_at_boundaries() -> None:
    states = np.asarray([[[100.0, 0.0, -100.0], [200.0, 0.0, 200.0]]])
    x_edges = np.linspace(-1.0, 1.0, 5)
    z_edges = np.linspace(0.0, 2.0, 5)
    histogram = normalized_xz_histogram(states, x_edges, z_edges)
    assert np.isclose(histogram.sum(), 1.0)
    assert xz_in_range_fraction(states, x_edges, z_edges) == 0.0


def test_representative_cell_selection_is_shared_and_deterministic() -> None:
    rows = []
    for method, offset in (("node_strong", 0.0), ("node_weak", 0.2)):
        for data_seed in (1, 2):
            for model_seed in (0, 1):
                for trajectory_id in (0, 1):
                    rows.append({
                        "method": method,
                        "track": "state_only_dynamics",
                        "noise_level": 0.0,
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "trajectory_id": trajectory_id,
                        "vpt_restricted_lt": data_seed + model_seed + trajectory_id + offset,
                    })
    frame = pd.DataFrame(rows)
    first = select_representative_cell(frame, 0.0)
    second = select_representative_cell(frame.sample(frac=1.0, random_state=9), 0.0)
    assert first == second
    assert set(first) == {"data_seed", "model_seed", "trajectory_id"}


def test_survival_curves_handle_censoring_and_are_deterministic() -> None:
    rows = []
    for method in ("a", "b"):
        for data_seed in (1, 2):
            for model_seed in (0, 1):
                for trajectory_id, (vpt, censored) in enumerate(((1.0, False), (2.0, True))):
                    rows.append({
                        "method": method,
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "trajectory_id": trajectory_id,
                        "vpt_restricted_lt": vpt + (0.5 if method == "b" else 0.0),
                        "vpt_censored": censored,
                    })
    frame = pd.DataFrame(rows)
    times = np.asarray([0.0, 1.0, 2.0])
    first = hierarchical_survival_curves(frame, times, resamples=50, seed=2026)
    second = hierarchical_survival_curves(frame, times, resamples=50, seed=2026)
    np.testing.assert_array_equal(first["a"]["estimate"], np.asarray([1.0, 0.5, 0.5]))
    for method in first:
        for key in first[method]:
            np.testing.assert_array_equal(first[method][key], second[method][key])


def test_survival_overlay_selects_the_three_nonzero_noise_levels() -> None:
    frame = pd.DataFrame({
        "noise_level": [0.05, 0.0, 0.001, 0.01, 0.001],
    })
    assert _survival_overlay_noises(frame) == [0.001, 0.01, 0.05]


def test_panda_comparison_figure_and_summary_are_generated(tmp_path) -> None:
    methods = ("sindy_weak_weighted", "sindy_weak", "panda_zero_shot")
    tracks = {
        "sindy_weak_weighted": "state_only_dynamics",
        "sindy_weak": "state_only_dynamics",
        "panda_zero_shot": "pretrained_sequence_forecasting",
    }
    run_rows = []
    trajectory_rows = []
    for method_index, method in enumerate(methods):
        for data_seed in (1, 2):
            for model_seed in (0, 1):
                run_id = f"{method}__d{data_seed}__m{model_seed}__n0"
                run_rows.append({
                    "run_id": run_id,
                    "method": method,
                    "track": tracks[method],
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "noise_level": 0.0,
                    "forecast_context_steps": 512,
                    "forecast_horizon_lt": 5.0,
                    "vpt_censoring_fraction": 0.0,
                    "rq_mmd": 0.03 + 0.01 * method_index,
                    "wasserstein_x": 0.1 + method_index,
                    "wasserstein_y": 0.2 + method_index,
                    "wasserstein_z": 0.3 + method_index,
                    "covariance_relative_error": 0.05 + method_index,
                })
                for trajectory_id in range(4):
                    vpt = 0.5 + 0.1 * method_index + 0.01 * trajectory_id
                    trajectory_rows.append({
                        "run_id": run_id,
                        "method": method,
                        "track": tracks[method],
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "trajectory_id": trajectory_id,
                        "noise_level": 0.0,
                        "vpt_restricted_lt": vpt,
                        "vpt_censored": False,
                        "nrmse_auc_0_1LT": 0.2 + method_index,
                        "nrmse_auc_0_2LT": 0.3 + method_index,
                        "nrmse_auc_0_5LT": 0.4 + method_index,
                    })
    arrays = {}
    curve_index = []
    for index, method in enumerate(methods):
        prefix = f"m{index}"
        curve_index.append({
            "prefix": prefix,
            "track": tracks[method],
            "noise_level": 0.0,
            "method": method,
        })
        arrays[f"{prefix}_times_lt"] = np.linspace(0.0, 5.0, 21)
        arrays[f"{prefix}_median"] = np.linspace(1e-6, 1.0 + index, 21)
        arrays[f"{prefix}_q25"] = np.linspace(1e-6, 0.8 + index, 21)
        arrays[f"{prefix}_q75"] = np.linspace(1e-5, 1.2 + index, 21)
    arrays["index_json"] = np.asarray(__import__("json").dumps(curve_index))
    curves_path = tmp_path / "curves.npz"
    atomic_save_npz(curves_path, **arrays)
    outputs, summary = make_panda_comparison_figures(
        run_metrics=pd.DataFrame(run_rows),
        trajectory_metrics=pd.DataFrame(trajectory_rows),
        curves_path=curves_path,
        output_dir=tmp_path / "figures",
        config={
            "aggregation": {"bootstrap_resamples": 20, "bootstrap_seed": 2026},
            "evaluation": {"error_plot_max": 100.0},
        },
    )
    assert {path.suffix for path in outputs} == {".png", ".svg"}
    assert all(path.exists() for path in outputs)
    summary_frame = pd.read_csv(summary)
    assert list(summary_frame["method"]) == list(methods)
    assert set(summary_frame["forecast_context_steps"]) == {512}
