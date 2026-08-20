from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lorenz63_benchmark.v2.artifacts import (
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_json,
    source_hash,
)
from lorenz63_benchmark.v2.context_matched import extract_context_data
from lorenz63_benchmark.v2.few_shot import (
    _load_shots,
    _train_panda_head_model,
    aggregate_few_shot,
    build_arg_parser,
    evaluate_few_shot,
    extract_shot_data,
    fit_few_shot_panda,
    fit_few_shot_sindy,
    load_few_shot_config,
)
from lorenz63_benchmark.v2.prepare_few_shot import prepare_few_shot_queues


def _config(
    path: Path,
    *,
    data_seeds: list[int] | None = None,
    shot_counts: list[int] | None = None,
) -> Path:
    config = {
        "schema": "lorenz63-benchmark-v2",
        "data": {"noise_levels": [0.0]},
        "panda": {
            "model_id": "fake/panda",
            "model_revision": "abc",
            "source_repository": "https://example.test/panda",
            "source_revision": "def",
            "context_length": 4,
            "prediction_length": 2,
            "batch_size": 2,
            "dtype": "float32",
            "sliding_context": True,
            "inference_seed": 99,
        },
        "few_shot": {
            "shot_counts": shot_counts or [1, 2],
            "context_steps": 4,
            "target_steps": 2,
            "segment_selection_seed": 2027,
            "forecast_lyapunov_times": 0.04,
            "nrmse_error_cap": 100.0,
            "panda_adaptation": "prediction_head_only",
            "panda_trainable_parameters": 6,
            "gradient_clip": 1.0,
            "weight_decay": 0.0,
            "tuning_learning_rates": [1e-3],
            "tuning_updates": [1],
            "tuning_objective": "median_validation_native_nrmse_auc",
        },
        "context_matched": {
            "context_steps": 4,
            "forecast_lyapunov_times": 0.04,
        },
        "sindy": {
            "degree": 2,
            "threshold": 0.0,
            "ridge": 1e-8,
            "max_iter": 20,
            "weak_window_steps": 4,
            "weak_stride_steps": 2,
            "weak_modes": 2,
            "savgol_window": 5,
            "savgol_polyorder": 2,
            "ablation_trajectory_counts": [1, 2],
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
        "evaluation": {
            "internal_step": 0.01,
            "vpt_threshold": 0.4,
            "stability_bound": 1000.0,
        },
        "aggregation": {"bootstrap_resamples": 20, "bootstrap_seed": 2026},
        "final": {"data_seeds": data_seeds or [1], "model_seeds": [0]},
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _split(
    path: Path,
    role: str,
    *,
    trajectories: int,
    steps: int,
    data_seed: int = 1,
) -> Path:
    times = np.arange(steps, dtype=np.float64) * 0.01
    base = np.stack([
        np.sin(times),
        np.cos(1.3 * times),
        1.0 + np.sin(0.7 * times),
    ], axis=-1)
    states = np.stack([base + 0.1 * index for index in range(trajectories)])
    metadata = {
        "schema": "lorenz63-split-v2",
        "role": role,
        "data_seed": data_seed,
        "dt": 0.01,
        "noise_level": 0.0,
        "normalization": {
            "state_mean": [0.0, 0.0, 0.0],
            "state_std": [1.0, 1.0, 1.0],
            "time_scale": 1.0,
        },
    }
    atomic_save_npz(
        path,
        states=states,
        times=times,
        metadata=np.asarray(json.dumps(metadata)),
    )
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
        self.last_surrogate_evaluations = 1
        return np.concatenate([
            context[:, -1:],
            np.repeat(context[:, -1:], len(times) - 1, axis=1),
        ], axis=1)


class _TinyPanda(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3, bias=False)
        self.head = torch.nn.Linear(3, 2, bias=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, *, past_values: torch.Tensor, future_values: torch.Tensor) -> SimpleNamespace:
        hidden = self.backbone(past_values.mean(dim=1))
        prediction = self.head(hidden).unsqueeze(-1).repeat(1, 1, 3)
        return SimpleNamespace(loss=torch.mean((prediction - future_values) ** 2))


def test_shots_are_nested_distinct_and_exactly_budgeted(tmp_path: Path) -> None:
    config = _config(tmp_path / "config.json")
    train = _split(tmp_path / "train.npz", "train", trajectories=4, steps=12)
    output = extract_shot_data(
        config_path=config, train_path=train, output_path=tmp_path / "shots.npz"
    )
    shots, metadata = _load_shots(output)
    assert shots.shape == (2, 6, 3)
    assert metadata["segment_steps"] == 6
    with np.load(output, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["trajectory_ids"], [0, 1])
        assert len(set(loaded["start_indices"].tolist())) <= 2
    repeated = extract_shot_data(
        config_path=config, train_path=train, output_path=tmp_path / "shots.npz"
    )
    with np.load(output, allow_pickle=False) as first, np.load(
        repeated, allow_pickle=False
    ) as second:
        np.testing.assert_array_equal(first["states"], second["states"])


def test_fit_interfaces_cannot_receive_evaluation_data() -> None:
    for function in (fit_few_shot_sindy, fit_few_shot_panda):
        parameters = set(inspect.signature(function).parameters)
        assert "shots_path" in parameters
        assert "truth_path" not in parameters
        assert "test_path" not in parameters
        assert "validation_path" not in parameters
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args([
            "fit",
            "--shots", "shots.npz",
            "--forecast-truth", "future.npz",
            "--output-dir", "out",
            "--method", "panda",
            "--shot-count", "1",
        ])


def test_fit_config_rejects_hidden_physics(tmp_path: Path) -> None:
    config_path = _config(tmp_path / "config.json")
    config = json.loads(config_path.read_text())
    config["system"] = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden physics"):
        load_few_shot_config(config_path)


def test_head_adaptation_leaves_backbone_frozen() -> None:
    torch.manual_seed(0)
    model = _TinyPanda()
    before_backbone = model.backbone.weight.detach().clone()
    before_head = model.head.weight.detach().clone()
    training = _train_panda_head_model(
        model,
        np.ones((1, 4, 3), dtype=np.float32),
        np.zeros((1, 2, 3), dtype=np.float32),
        learning_rate=0.1,
        updates=2,
        gradient_clip=1.0,
        weight_decay=0.0,
    )
    torch.testing.assert_close(model.backbone.weight, before_backbone)
    assert not torch.equal(model.head.weight, before_head)
    assert training["trainable_parameter_count"] == 6
    assert training["optimizer_updates"] == 2


def test_zero_shot_evaluation_records_zero_training_observations(tmp_path: Path) -> None:
    config = _config(tmp_path / "config.json")
    test = _split(tmp_path / "test.npz", "test", trajectories=2, steps=10)
    context, truth = extract_context_data(
        config_path=config, test_path=test, output_dir=tmp_path / "protocol"
    )
    manifest_path = evaluate_few_shot(
        config_path=config,
        context_path=context,
        truth_path=truth,
        output_dir=tmp_path / "evaluation",
        method="panda",
        shot_count=0,
        panda_adapter_factory=lambda _checkpoint, _device: _FakePanda(),
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["observed_training_state_count"] == 0
    with (manifest_path.parent / "run_metrics.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert int(row["observed_training_state_count"]) == 0
    assert row["adaptation"] == "zero_shot"


def test_queue_routes_fit_data_without_truth(tmp_path: Path) -> None:
    config = _config(tmp_path / "config.json", data_seeds=[1, 2])
    paths = prepare_few_shot_queues(
        config_path=config,
        output_dir=tmp_path / "queues",
        project_root=tmp_path / "project",
        cpu_python="/cpu/python",
        gpu_python="/gpu/python",
    )
    queues = {
        name: [json.loads(line) for line in path.read_text().splitlines()]
        for name, path in paths.items()
    }
    assert len(queues["dev_cpu"]) == 1
    assert len(queues["dev_gpu"]) == 1
    assert len(queues["cpu"]) == 20
    assert len(queues["gpu"]) == 10
    assert len(queues["aggregate"]) == 1
    fit_tasks = [
        task
        for queue in (queues["cpu"], queues["gpu"])
        for task in queue
        if task["id"].startswith("fit_few_shot_")
    ]
    assert len(fit_tasks) == 12
    for task in fit_tasks:
        command = task["command"]
        assert "--shots" in command
        assert "--forecast-truth" not in command
        assert "--context" not in command
        assert "--test" not in command
        assert "--validation" not in command


def test_complete_matrix_aggregation_generates_figures(tmp_path: Path) -> None:
    config_path = _config(
        tmp_path / "config.json", data_seeds=[1], shot_counts=[1, 4, 16]
    )
    config = json.loads(config_path.read_text())
    conditions = (
        *(("panda", count) for count in (0, 1, 4, 16)),
        *(("sindy_weak", count) for count in (1, 4, 16)),
        *(("sindy_weak_weighted", count) for count in (1, 4, 16)),
    )
    times_lt = np.linspace(0.0, 0.04, 6)
    for condition_index, (method, shot_count) in enumerate(conditions):
        run_dir = tmp_path / "runs" / f"{method}_k{shot_count}"
        value = 0.01 * (condition_index + 1)
        trajectory_rows = [{
            "method": method,
            "data_seed": 1,
            "trajectory_id": trajectory_id,
            "shot_count": shot_count,
            "vpt_restricted_lt": 0.04 - value / 10.0,
            "vpt_censored": False,
            "forecast_instability": False,
            "forecast_instability_lt": np.nan,
            "restricted_nrmse_auc_native": value,
            "restricted_nrmse_auc_0_1LT": value,
            "restricted_nrmse_auc_0_2LT": value,
            "restricted_nrmse_auc_0_5LT": value,
            "mean_coordinate_point_crps_native": value,
        } for trajectory_id in range(2)]
        trajectory_path = run_dir / "trajectory_metrics.csv"
        run_path = run_dir / "run_metrics.csv"
        predictions_path = run_dir / "predictions.npz"
        atomic_write_csv(trajectory_path, trajectory_rows)
        atomic_write_csv(run_path, [{
            "method": method,
            "shot_count": shot_count,
            "data_seed": 1,
            "observed_training_state_count": shot_count * 6,
            "probabilistic": False,
        }])
        atomic_save_npz(
            predictions_path,
            normalized_squared_error=np.full((2, 6), value, dtype=np.float32),
            times_lt=times_lt,
        )
        artifacts = {
            "trajectory_metrics": trajectory_path,
            "run_metrics": run_path,
            "predictions": predictions_path,
        }
        atomic_write_json(run_dir / "manifest.json", {
            "schema": "lorenz63-few-shot-evaluation-v2",
            "status": "complete",
            "method": method,
            "shot_count": shot_count,
            "data_seed": 1,
            "source_hash": source_hash(),
            "config_hash": sha256_json(config),
            "future_test_states_visible_during_fit": False,
            "artifacts": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in artifacts.items()
            },
        })
    outputs = aggregate_few_shot(
        config_path=config_path,
        runs_root=tmp_path / "runs",
        output_dir=tmp_path / "results",
    )
    assert outputs["figure_png"].stat().st_size > 0
    assert outputs["figure_svg"].stat().st_size > 0
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["status"] == "complete"
    assert manifest["run_count"] == 10
