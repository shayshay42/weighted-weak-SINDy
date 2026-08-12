from __future__ import annotations

import json
import copy

import numpy as np
import pytest

from lorenz63_benchmark.v2.artifacts import atomic_save_npz
from lorenz63_benchmark.v2.config import DEFAULT_CONFIG, config_for_method
from lorenz63_benchmark.v2.contracts import (
    METHODS,
    PARAMETRIC_TRACK,
    PINN_TRACK,
    PRETRAINED_TRACK,
    PRIMARY_TRACK,
    configured_methods,
)
from lorenz63_benchmark.v2.data import load_split
from lorenz63_benchmark.v2.train import _method_config, build_arg_parser
from lorenz63_benchmark.v2.validate import _dataset_record_path


def _split_metadata(role: str) -> str:
    return json.dumps({
        "schema": "lorenz63-split-v2", "role": role, "data_seed": 1,
        "dt": 0.01, "noise_level": 0.0,
        "normalization": {"state_mean": [0.0] * 3, "state_std": [1.0] * 3, "time_scale": 0.9},
    })


def test_training_split_is_isolated(tmp_path) -> None:
    path = tmp_path / "train.npz"
    atomic_save_npz(
        path, states=np.zeros((2, 6, 3)), initial_states=np.zeros((2, 3)),
        times=np.arange(6) * 0.01, metadata=np.asarray(_split_metadata("train")),
    )
    split = load_split(path, "train")
    assert "derivatives" not in split
    assert "params" not in split["metadata"]
    assert "generator" not in split["metadata"]
    with pytest.raises(ValueError):
        load_split(path, "test")


def test_train_cli_has_no_test_argument() -> None:
    destinations = {action.dest for action in build_arg_parser()._actions}
    assert "train" in destinations
    assert "validation" in destinations
    assert "test" not in destinations


def test_hidden_physics_is_removed_from_non_pinn_trainers() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["panda"] = {}
    for method, contract in METHODS.items():
        sanitized = _method_config(config, method)
        if contract.track in {PRIMARY_TRACK, PARAMETRIC_TRACK}:
            assert "system" not in sanitized
        elif contract.track == PINN_TRACK:
            assert "system" in sanitized
        elif contract.track == PRETRAINED_TRACK:
            assert set(sanitized) == {"schema", "evaluation", "normalization", "panda"}


def test_external_pretrained_methods_require_explicit_config_enablement() -> None:
    assert "panda_zero_shot" not in configured_methods(DEFAULT_CONFIG)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["final"]["methods"] = ["sindy_weak", "panda_zero_shot"]
    assert configured_methods(config) == ["sindy_weak", "panda_zero_shot"]


def test_focused_config_does_not_inherit_optional_acceptance_methods() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["final"]["methods"] = [
        "sindy_weak", "sindy_weak_weighted", "panda_zero_shot",
    ]
    selected = set(configured_methods(config))
    assert "solver_oracle" not in selected
    assert not selected.intersection({"sindy_strong", "sindy_weighted"})
    assert not selected.intersection({"lorenz_ad", "lorenz_ad_tapered"})


def test_method_specific_hyperparameters_do_not_cross_methods() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["method_overrides"] = {
        "0": {
            "sindy_strong": {"sindy": {"threshold": 0.01}},
            "sindy_weak": {"sindy": {"threshold": 0.1}},
        }
    }
    strong = config_for_method(config, 0.0, "sindy_strong")
    weak = config_for_method(config, 0.0, "sindy_weak")
    weighted_strong = config_for_method(config, 0.0, "sindy_weighted")
    weighted_weak = config_for_method(config, 0.0, "sindy_weak_weighted")
    assert strong["sindy"]["threshold"] == 0.01
    assert weak["sindy"]["threshold"] == 0.1
    assert weighted_strong["sindy"]["threshold"] == 0.01
    assert weighted_weak["sindy"]["threshold"] == 0.1


def test_dataset_validator_accepts_public_and_training_manifest_paths(tmp_path) -> None:
    assert _dataset_record_path(tmp_path, {"path": "test.npz"}) == tmp_path / "test.npz"
    assert _dataset_record_path(
        tmp_path, {"train_path": "noise_0/train.npz"}
    ) == tmp_path / "noise_0/train.npz"
    with pytest.raises(ValueError, match="neither 'path' nor 'train_path'"):
        _dataset_record_path(tmp_path, {"sha256": "missing-path"})
