from __future__ import annotations

import copy

import numpy as np
import torch

from lorenz63_benchmark.v2.config import DEFAULT_CONFIG
from lorenz63_benchmark.v2.metrics import (
    normalized_rmse_auc,
    normalized_squared_error,
    valid_prediction_time,
)
from lorenz63_benchmark.v2.node import _window_indices, load_node, train_node


def _split(role: str, states: np.ndarray) -> dict:
    return {
        "states": states,
        "initial_states": states[:, 0],
        "times": np.arange(states.shape[1]) * 0.01,
        "metadata": {
            "role": role, "data_seed": 1, "dt": 0.01, "noise_level": 0.0,
            "normalization": {
                "state_mean": states.reshape(-1, 3).mean(axis=0).tolist(),
                "state_std": (states.reshape(-1, 3).std(axis=0) + 1.0).tolist(),
                "time_scale": 0.9,
            },
        },
    }


def test_repeated_seeded_node_fit_has_identical_checkpoint(tmp_path) -> None:
    rng = np.random.default_rng(8)
    states = rng.normal(size=(2, 10, 3)).cumsum(axis=1)
    train = _split("train", states)
    validation = _split("validation", states[:, :6])
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["node"].update({
        "hidden_dim": 8, "depth": 1, "epochs": 1, "steps_per_epoch": 2,
        "windows_per_update": 2, "window_steps": 3,
    })
    first, _ = train_node("node_strong", train, validation, config, 4, tmp_path / "a", "cpu")
    second, _ = train_node("node_strong", train, validation, config, 4, tmp_path / "b", "cpu")
    first_state = torch.load(first, map_location="cpu", weights_only=False)["model_state"]
    second_state = torch.load(second, map_location="cpu", weights_only=False)["model_state"]
    assert first_state.keys() == second_state.keys()
    for key in first_state:
        torch.testing.assert_close(first_state[key], second_state[key], rtol=0.0, atol=0.0)

    times = validation["times"]
    truth = validation["states"]
    first_prediction = load_node(first, internal_step=0.01).forecast(truth[:, 0], times)
    second_prediction = load_node(second, internal_step=0.01).forecast(truth[:, 0], times)
    np.testing.assert_array_equal(first_prediction, second_prediction)

    training_std = np.asarray(train["metadata"]["normalization"]["state_std"])
    first_error = normalized_squared_error(first_prediction, truth, training_std)
    second_error = normalized_squared_error(second_prediction, truth, training_std)
    np.testing.assert_array_equal(first_error, second_error)
    first_vpt = valid_prediction_time(first_error, times, 0.9)
    second_vpt = valid_prediction_time(second_error, times, 0.9)
    np.testing.assert_array_equal(first_vpt[0], second_vpt[0])
    np.testing.assert_array_equal(first_vpt[1], second_vpt[1])
    np.testing.assert_array_equal(
        normalized_rmse_auc(first_error, times * 0.9, 0.04),
        normalized_rmse_auc(second_error, times * 0.9, 0.04),
    )


def test_node_variants_have_identical_initial_weights_and_window_streams(tmp_path) -> None:
    rng = np.random.default_rng(19)
    states = rng.normal(size=(3, 14, 3)).cumsum(axis=1)
    train = _split("train", states)
    validation = _split("validation", states[:, :8])
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["node"].update({
        "hidden_dim": 8, "depth": 1, "epochs": 0, "steps_per_epoch": 1,
        "windows_per_update": 3, "window_steps": 3,
    })

    model_states = []
    for method in ("node_strong", "node_soft_dtw", "node_weak"):
        checkpoint, _ = train_node(
            method, train, validation, config, 7, tmp_path / method, "cpu"
        )
        model_states.append(torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )["model_state"])
    for key in model_states[0]:
        torch.testing.assert_close(model_states[0][key], model_states[1][key], rtol=0.0, atol=0.0)
        torch.testing.assert_close(model_states[0][key], model_states[2][key], rtol=0.0, atol=0.0)

    streams = []
    for _method in ("node_strong", "node_soft_dtw", "node_weak"):
        stream_rng = np.random.default_rng(np.random.SeedSequence([7, 7321]))
        streams.append(_window_indices(stream_rng, 64, 2001, 40, 10, 32))
    for trajectories, starts in streams[1:]:
        np.testing.assert_array_equal(trajectories, streams[0][0])
        np.testing.assert_array_equal(starts, streams[0][1])
