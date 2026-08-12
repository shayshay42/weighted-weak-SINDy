from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .config import load_config
from .contracts import METHODS, configured_methods
from .data import noise_label
from .artifacts import (
    atomic_write_json, command_line, environment_snapshot, sha256_json,
    source_hash, utc_now,
)


GPU_METHODS = {name for name, contract in METHODS.items() if contract.kind in {"node", "pinn", "parametric"}}
CPU_METHODS = set(METHODS) - GPU_METHODS


def _atomic_jsonl(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run_dir(runs_root: Path, noise: str, data_seed: int, method: str, model_seed: int) -> Path:
    return runs_root / f"noise_{noise}" / f"split_seed{data_seed}" / method / f"model_seed{model_seed}"


def prepare_queues(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    cpu_python: str,
    gpu_python: str,
    methods: list[str] | None = None,
    include_data: bool = True,
    include_ablation: bool = True,
    evaluation_python: str | None = None,
    evaluation_device: str = "cuda",
    results_dir: str | Path | None = None,
    runs_root: str | Path | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    project = Path(project_root)
    selected_methods = sorted(configured_methods(config) if methods is None else set(methods))
    unknown = set(selected_methods).difference(METHODS)
    if unknown:
        raise ValueError(f"unknown methods in queue request: {sorted(unknown)}")
    if evaluation_device not in {"cpu", "cuda"}:
        raise ValueError("evaluation device must be 'cpu' or 'cuda'")
    evaluation_python = evaluation_python or gpu_python
    results_root = Path(results_dir) if results_dir is not None else project / "results" / "v2"
    if runs_root is None:
        run_root = project / "runs" / "v2"
    else:
        run_root = Path(runs_root)
        if not run_root.is_absolute():
            run_root = project / run_root
    config_absolute = project / config_path if not Path(config_path).is_absolute() else Path(config_path)
    data_root = project / "data" / "v2"
    data_seeds = [int(value) for value in config["final"]["data_seeds"]]
    model_seeds = [int(value) for value in config["final"]["model_seeds"]]
    levels = [float(value) for value in config["data"]["noise_levels"]]
    queues: dict[str, list[dict[str, Any]]] = {
        "data": [], "cpu_train": [], "gpu_train": [], "evaluate": [], "aggregate": [],
    }
    if include_data:
        for data_seed in data_seeds:
            queues["data"].append({
                "id": f"generate_split_seed{data_seed}",
                "cwd": str(project),
                "command": [
                    cpu_python, "-m", "lorenz63_benchmark.v2.data",
                    "--config", str(config_absolute), "--output-root", str(data_root),
                    "--seeds", str(data_seed),
                ],
            })
    for data_seed in data_seeds:
        split_root = data_root / f"split_seed{data_seed}"
        validation = split_root / "validation.npz"
        test = split_root / "test.npz"
        dataset_manifest = split_root / "manifest.json"
        for noise_level in levels:
            label = noise_label(noise_level)
            train = split_root / f"noise_{label}" / "train.npz"
            for method in selected_methods:
                target_queue = "gpu_train" if method in GPU_METHODS else "cpu_train"
                python_executable = gpu_python if method in GPU_METHODS else cpu_python
                for model_seed in model_seeds:
                    run_dir = _run_dir(run_root, label, data_seed, method, model_seed)
                    task_id = f"train_{method}_d{data_seed}_m{model_seed}_n{label}"
                    queues[target_queue].append({
                        "id": task_id, "cwd": str(project),
                        "command": [
                            python_executable, "-m", "lorenz63_benchmark.v2.train",
                            "--config", str(config_absolute), "--train", str(train),
                            "--validation", str(validation), "--method", method,
                            "--data-seed", str(data_seed), "--model-seed", str(model_seed),
                            "--output-dir", str(run_dir),
                            "--device", "cuda" if method in GPU_METHODS else "cpu",
                        ],
                    })
                    queues["evaluate"].append({
                        "id": f"evaluate_{method}_d{data_seed}_m{model_seed}_n{label}",
                        "cwd": str(project),
                        "command": [
                            evaluation_python, "-m", "lorenz63_benchmark.v2.evaluate",
                            "--config", str(config_absolute), "--run-dir", str(run_dir),
                            "--train", str(train), "--test", str(test),
                            "--dataset-manifest", str(dataset_manifest),
                            "--device", evaluation_device,
                        ],
                    })
            ablation_methods = [
                method for method in ("sindy_strong", "sindy_weighted")
                if method in selected_methods
            ]
            for count in config["sindy"]["ablation_trajectory_counts"] if include_ablation else []:
                for method in ablation_methods:
                    model_seed = 0
                    run_dir = (
                        run_root / "ablation" / f"noise_{label}"
                        / f"split_seed{data_seed}" / method / f"trajectories_{count}"
                    )
                    queues["cpu_train"].append({
                        "id": f"ablation_{method}_k{count}_d{data_seed}_n{label}",
                        "cwd": str(project),
                        "command": [
                            cpu_python, "-m", "lorenz63_benchmark.v2.train",
                            "--config", str(config_absolute), "--train", str(train),
                            "--validation", str(validation), "--method", method,
                            "--data-seed", str(data_seed), "--model-seed", str(model_seed),
                            "--trajectory-count", str(count), "--output-dir", str(run_dir),
                            "--device", "cpu",
                        ],
                    })
                    queues["evaluate"].append({
                        "id": f"evaluate_ablation_{method}_k{count}_d{data_seed}_n{label}",
                        "cwd": str(project),
                        "command": [
                            evaluation_python, "-m", "lorenz63_benchmark.v2.evaluate",
                            "--config", str(config_absolute), "--run-dir", str(run_dir),
                            "--train", str(train), "--test", str(test),
                            "--dataset-manifest", str(dataset_manifest),
                            "--device", evaluation_device,
                        ],
                    })
    queues["aggregate"].append({
        "id": "aggregate_final_runs", "cwd": str(project),
        "command": [
            cpu_python, "-m", "lorenz63_benchmark.v2.aggregate",
            "--config", str(config_absolute), "--runs-root", str(run_root),
            "--output-dir", str(results_root),
        ],
    })
    queues["aggregate"].append({
        "id": "validate_final_runs", "cwd": str(project),
        "depends_on": ["aggregate_final_runs"],
        "command": [
            cpu_python, "-m", "lorenz63_benchmark.v2.validate",
            "--config", str(config_absolute), "--data-root", str(data_root),
            "--runs-root", str(run_root),
            "--results-dir", str(results_root),
        ],
    })
    output_dir = Path(output_dir)
    paths = {}
    for name, tasks in queues.items():
        path = output_dir / f"{name}.jsonl"
        _atomic_jsonl(path, tasks)
        paths[name] = path
    summary = {
        "schema": "lorenz63-queue-set-v2", "project_root": str(project),
        "cpu_python": cpu_python, "gpu_python": gpu_python,
        "status": "complete", "completed_at": utc_now(),
        "config_hash": sha256_json(config), "source_hash": source_hash(),
        "command": command_line(), "environment": environment_snapshot(),
        "seeds": {"data": data_seeds, "model": model_seeds},
        "methods": selected_methods,
        "include_data": bool(include_data), "include_ablation": bool(include_ablation),
        "evaluation": {"python": evaluation_python, "device": evaluation_device},
        "results_dir": str(results_root),
        "runs_root": str(run_root),
        "queues": {name: {"path": str(path), "task_count": len(queues[name])} for name, path in paths.items()},
    }
    atomic_write_json(output_dir / "manifest.json", summary)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare no-Slurm Lorenz63 v2 task queues.")
    parser.add_argument("--config", default="configs/v2/lorenz63_frozen.json")
    parser.add_argument("--output-dir", default="queues/v2")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--cpu-python", default=sys.executable)
    parser.add_argument("--gpu-python", default=sys.executable)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS))
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--evaluation-python")
    parser.add_argument("--evaluation-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--results-dir")
    parser.add_argument("--runs-root")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = prepare_queues(
        config_path=args.config, output_dir=args.output_dir,
        project_root=args.project_root, cpu_python=args.cpu_python, gpu_python=args.gpu_python,
        methods=args.methods, include_data=not args.skip_data,
        include_ablation=not args.skip_ablation,
        evaluation_python=args.evaluation_python,
        evaluation_device=args.evaluation_device,
        results_dir=args.results_dir,
        runs_root=args.runs_root,
    )
    print(*paths.values(), sep="\n")


if __name__ == "__main__":
    main()
