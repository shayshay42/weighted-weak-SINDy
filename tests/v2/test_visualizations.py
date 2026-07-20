from __future__ import annotations

import numpy as np
import pandas as pd

from lorenz63_benchmark.v2.attractor_summary import (
    normalized_xz_histogram,
    return_map_pairs,
    select_representative_cell,
    xz_in_range_fraction,
)
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
