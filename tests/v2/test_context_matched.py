from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from lorenz63_benchmark.v2.artifacts import atomic_save_npz, sha256_file
from lorenz63_benchmark.v2.context_matched import (
    CONTEXT_DATA_SCHEMA,
    _metric_rows,
    build_arg_parser,
    evaluate_context_matched,
    extract_context_data,
    fit_context_sindy_bank,
)
from lorenz63_benchmark.v2.prepare_context_matched import prepare_context_matched_queues


def _config(path: Path, *, context_steps: int = 20, data_seeds: list[int] | None = None) -> Path:
    config = {
        "data": {"noise_levels": [0.0]},
        "panda": {
            "model_id": "fake/panda",
            "model_revision": "abc",
            "source_repository": "https://example.test/panda",
            "source_revision": "def",
            "context_length": context_steps,
            "prediction_length": 4,
            "batch_size": 2,
            "dtype": "float32",
            "sliding_context": True,
            "inference_seed": 99,
        },
        "context_matched": {
            "methods": ["sindy_weak_weighted", "sindy_weak", "panda_zero_shot"],
            "context_steps": context_steps,
            "forecast_lyapunov_times": 0.1,
            "native_prediction_steps": 4,
            "nrmse_error_cap": 100.0,
        },
        "method_overrides": {
            "0": {
                "sindy_weak": {
                    "sindy": {
                        "degree": 2,
                        "ridge": 1e-8,
                        "threshold": 0.0,
                        "weak_window_steps": 4,
                        "weak_stride_steps": 2,
                        "weak_modes": 2,
                    }
                }
            }
        },
        "evaluation": {"internal_step": 0.01, "vpt_threshold": 0.4},
        "aggregation": {"bootstrap_resamples": 20, "bootstrap_seed": 2026},
        "final": {
            "data_seeds": data_seeds or [1],
            "model_seeds": [0],
            "methods": ["sindy_weak_weighted", "sindy_weak", "panda_zero_shot"],
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _test_split(path: Path, *, future_offset: float = 0.0, context_steps: int = 20) -> Path:
    times = np.arange(context_steps + 15, dtype=np.float64) * 0.01
    base = np.stack([
        np.sin(times),
        np.cos(1.3 * times),
        1.0 + np.sin(0.7 * times),
    ], axis=-1)
    states = np.stack([base, base + np.asarray([0.2, -0.1, 0.3])])
    states[:, context_steps:] += future_offset
    metadata = {
        "schema": "lorenz63-split-v2",
        "role": "test",
        "data_seed": 1,
        "dt": 0.01,
        "noise_level": 0.0,
        "normalization": {
            "state_mean": [0.0, 0.0, 0.0],
            "state_std": [1.0, 1.0, 1.0],
            "time_scale": 1.0,
        },
    }
    atomic_save_npz(path, states=states, times=times, metadata=np.asarray(json.dumps(metadata)))
    return path


class _FakePanda:
    parameter_count = 7
    probabilistic = False
    forecast_sample_count = 1
    last_surrogate_evaluations = 0
    last_peak_gpu_memory_bytes = 0

    def forecast_normalized_context(
        self, context: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        future = np.repeat(context[:, -1:, :], len(times) - 1, axis=1)
        self.last_surrogate_evaluations = 1
        return np.concatenate([context[:, -1:, :], future], axis=1)


def test_online_fit_interface_cannot_accept_truth() -> None:
    parameters = set(inspect.signature(fit_context_sindy_bank).parameters)
    assert "context_path" in parameters
    assert "truth_path" not in parameters
    assert "test_path" not in parameters
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args([
            "fit-sindy",
            "--context", "context.npz",
            "--forecast-truth", "future.npz",
            "--output-dir", "out",
            "--method", "sindy_weak",
        ])


def test_future_mutation_does_not_change_context_or_sindy_fit(tmp_path: Path) -> None:
    config = _config(tmp_path / "config.json")
    first_test = _test_split(tmp_path / "first_test.npz", future_offset=0.0)
    second_test = _test_split(tmp_path / "second_test.npz", future_offset=1000.0)
    first_context, first_truth = extract_context_data(
        config_path=config, test_path=first_test, output_dir=tmp_path / "first_data"
    )
    second_context, second_truth = extract_context_data(
        config_path=config, test_path=second_test, output_dir=tmp_path / "second_data"
    )
    with np.load(first_context, allow_pickle=False) as first, np.load(
        second_context, allow_pickle=False
    ) as second:
        np.testing.assert_array_equal(first["states"], second["states"])
        assert json.loads(str(first["metadata"]))["contains_future_states"] is False
    with np.load(first_truth, allow_pickle=False) as first, np.load(
        second_truth, allow_pickle=False
    ) as second:
        assert not np.array_equal(first["states"], second["states"])

    first_models = fit_context_sindy_bank(
        config_path=config,
        context_path=first_context,
        output_dir=tmp_path / "first_models",
        method="sindy_weak",
    )
    second_models = fit_context_sindy_bank(
        config_path=config,
        context_path=second_context,
        output_dir=tmp_path / "second_models",
        method="sindy_weak",
    )
    with np.load(first_models, allow_pickle=False) as first, np.load(
        second_models, allow_pickle=False
    ) as second:
        np.testing.assert_array_equal(first["coefficients"], second["coefficients"])
    manifest = json.loads((first_models.parent / "manifest.json").read_text())
    assert set(manifest["input_hashes"]) == {"context"}
    assert manifest["future_states_visible_during_adaptation"] is False
    assert "forecast_truth" in manifest["forbidden_inputs"]
    assert "system" not in json.dumps(manifest)


def test_deterministic_panda_reports_point_mass_crps(tmp_path: Path) -> None:
    config = _config(tmp_path / "config.json")
    test = _test_split(tmp_path / "test.npz")
    context, truth = extract_context_data(
        config_path=config, test_path=test, output_dir=tmp_path / "data"
    )
    manifest_path = evaluate_context_matched(
        config_path=config,
        context_path=context,
        truth_path=truth,
        output_dir=tmp_path / "panda",
        method="panda_zero_shot",
        panda_adapter_factory=lambda _checkpoint, _device: _FakePanda(),
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["probabilistic"] is False
    assert manifest["forecast_sample_count"] == 1
    with (manifest_path.parent / "trajectory_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(
        np.isfinite(float(row["mean_coordinate_point_crps_native"])) for row in rows
    )


def test_context_queue_routes_truth_away_from_fit_commands(tmp_path: Path) -> None:
    config = _config(tmp_path / "config.json", data_seeds=[1, 2])
    paths = prepare_context_matched_queues(
        config_path=config,
        output_dir=tmp_path / "queues",
        project_root=tmp_path / "project",
        cpu_python="/cpu/python",
        gpu_python="/gpu/python",
    )
    cpu_tasks = [json.loads(line) for line in paths["cpu"].read_text().splitlines()]
    gpu_tasks = [json.loads(line) for line in paths["gpu"].read_text().splitlines()]
    fit_tasks = [task for task in cpu_tasks if task["id"].startswith("fit_context_")]
    assert len(cpu_tasks) == 10
    assert len(gpu_tasks) == 2
    assert len(fit_tasks) == 4
    for task in fit_tasks:
        command = task["command"]
        assert "--context" in command
        assert "--forecast-truth" not in command
        assert "--test" not in command
        assert "--train" not in command
        assert "--validation" not in command
    assert all("--forecast-truth" in task["command"] for task in gpu_tasks)


def test_divergent_forecast_has_finite_restricted_auc_and_explicit_flag() -> None:
    prediction = np.zeros((1, 4, 3), dtype=np.float64)
    prediction[:, 2:] = np.inf
    rows, error, _ = _metric_rows(
        method="sindy_weak",
        data_seed=1,
        prediction=prediction,
        truth=np.zeros_like(prediction),
        forecast_times=np.arange(4, dtype=np.float64),
        largest=1.0,
        scale=np.ones(3),
        context_steps=20,
        native_prediction_steps=3,
        forecast_horizon_lt=3.0,
        vpt_threshold=0.4,
        nrmse_error_cap=100.0,
        stability_bound=1000.0,
    )
    assert np.isinf(error[0, 2])
    assert rows[0]["forecast_instability"] is True
    assert rows[0]["forecast_instability_lt"] == 2.0
    assert np.isfinite(rows[0]["restricted_nrmse_auc_0_2LT"])
