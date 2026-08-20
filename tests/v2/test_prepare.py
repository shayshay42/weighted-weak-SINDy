from __future__ import annotations

import json

from lorenz63_benchmark.v2.artifacts import atomic_write_json
from lorenz63_benchmark.v2.prepare import prepare_queues
from lorenz63_benchmark.v2.worker import load_tasks


def test_final_data_seeds_are_independent_resumable_tasks(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    atomic_write_json(config_path, {
        "data": {"noise_levels": [0.0]},
        "final": {"data_seeds": [1, 2], "model_seeds": [0]},
        "sindy": {"ablation_trajectory_counts": [1]},
    })
    paths = prepare_queues(
        config_path=config_path,
        output_dir=tmp_path / "queues",
        project_root=tmp_path / "project",
        cpu_python="/cpu/python",
        gpu_python="/gpu/python",
    )

    data_tasks = load_tasks(paths["data"])
    assert [task["id"] for task in data_tasks] == [
        "generate_split_seed1", "generate_split_seed2",
    ]
    assert [task["command"][-1] for task in data_tasks] == ["1", "2"]
    aggregate_tasks = load_tasks(paths["aggregate"])
    assert aggregate_tasks[1]["depends_on"] == ["aggregate_final_runs"]
    evaluation_tasks = load_tasks(paths["evaluate"])
    assert all(task["command"][-1] == "cuda" for task in evaluation_tasks)
    cpu_tasks = load_tasks(paths["cpu_train"])
    assert any("sindy_weak_weighted" in task["id"] for task in cpu_tasks)
    queue_manifest = json.loads((tmp_path / "queues" / "manifest.json").read_text())
    assert queue_manifest["status"] == "complete"
    assert queue_manifest["config_hash"]
    assert queue_manifest["source_hash"]
    assert queue_manifest["seeds"] == {"data": [1, 2], "model": [0]}


def test_method_only_cpu_amendment_queue_skips_existing_work(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    atomic_write_json(config_path, {
        "data": {"noise_levels": [0.0, 0.01]},
        "final": {"data_seeds": [1, 2], "model_seeds": [0, 1]},
        "sindy": {"ablation_trajectory_counts": [1]},
    })
    paths = prepare_queues(
        config_path=config_path,
        output_dir=tmp_path / "queues",
        project_root=tmp_path / "project",
        cpu_python="/cpu/python",
        gpu_python="/gpu/python",
        methods=["sindy_weak_weighted"],
        include_data=False,
        include_ablation=False,
        evaluation_python="/cpu/python",
        evaluation_device="cpu",
    )
    assert load_tasks(paths["data"]) == []
    train_tasks = load_tasks(paths["cpu_train"])
    assert len(train_tasks) == 8
    assert load_tasks(paths["gpu_train"]) == []
    evaluation_tasks = load_tasks(paths["evaluate"])
    assert len(evaluation_tasks) == 8
    assert all(task["command"][0] == "/cpu/python" for task in evaluation_tasks)
    assert all(task["command"][-1] == "cpu" for task in evaluation_tasks)
    assert all("ablation" not in task["id"] for task in train_tasks)


def test_panda_comparison_uses_explicit_methods_and_runs_root(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    atomic_write_json(config_path, {
        "data": {"noise_levels": [0.0]},
        "panda": {"model_id": "fake"},
        "final": {
            "data_seeds": [1],
            "model_seeds": [0],
            "methods": ["sindy_weak", "sindy_weak_weighted", "panda_zero_shot"],
        },
        "aggregation": {"require_complete_ablation": False},
    })
    paths = prepare_queues(
        config_path=config_path,
        output_dir=tmp_path / "queues",
        project_root=tmp_path / "project",
        cpu_python="/cpu/python",
        gpu_python="/gpu/python",
        include_data=False,
        include_ablation=False,
        runs_root="runs/v2_panda_comparison",
    )
    train_tasks = load_tasks(paths["cpu_train"])
    assert len(train_tasks) == 3
    assert load_tasks(paths["gpu_train"]) == []
    assert all("v2_panda_comparison" in " ".join(task["command"]) for task in train_tasks)
    aggregate = load_tasks(paths["aggregate"])[0]
    assert "v2_panda_comparison" in " ".join(aggregate["command"])
