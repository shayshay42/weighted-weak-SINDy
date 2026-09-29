from __future__ import annotations

import json
from pathlib import Path

import pytest

from amortized_sindy.v1.checkpoint import sha256_file
from amortized_sindy.v1.odeformer_gate import ODEFORMER_GATE_SCHEMA
from amortized_sindy.v1.odeformer_infer import _load_frozen_config


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_frozen_config_requires_passing_source_gate(tmp_path: Path) -> None:
    gate_path = tmp_path / "gate.json"
    _write_json(gate_path, {
        "schema": ODEFORMER_GATE_SCHEMA,
        "metrics": {"all_predictions_finite": True, "mean_normalized_mse": 1.0},
    })
    config_path = tmp_path / "config.json"
    config = {
        "schema": "amortized-sindy-odeformer-frozen-config-v1",
        "model": {"pretraining_exposure": "unknown"},
        "inference": {
            "candidate_count": 8,
            "sampling_temperature": 0.1,
            "context_rerank": "observed_context_snmse",
            "coefficient_adapter": "analytic_degree_2_taylor_at_context_coordinate_mean",
            "library_dimension": 3,
            "library_degree": 2,
            "target_optimizer_steps": 0,
            "target_sparse_regression_solves": 0,
        },
        "selection": {
            "source_gate_manifest_sha256": sha256_file(gate_path),
            "conditioner_mse": 1.0,
            "raw_gru_mse": 1.2,
            "constant_field_mse": 1.6,
            "all_predictions_finite": True,
            "gate_passed": True,
            "target_families_used_for_selection": [],
        },
    }
    _write_json(config_path, config)
    assert _load_frozen_config(
        config_path, selection_manifest_path=gate_path
    ) == config

    config["selection"]["target_families_used_for_selection"] = ["lorenz"]
    _write_json(config_path, config)
    with pytest.raises(ValueError, match="target families"):
        _load_frozen_config(config_path, selection_manifest_path=gate_path)
