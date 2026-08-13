from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .artifacts import (
    atomic_write_json,
    command_line,
    environment_snapshot,
    sha256_json,
    source_hash,
    utc_now,
)
from .few_shot import load_few_shot_config
from .prepare import _atomic_jsonl


def prepare_few_shot_queues(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    cpu_python: str,
    gpu_python: str,
    data_root: str | Path = "data/v2",
    protocol_data_root: str | Path = "data/v2_few_shot",
    runs_root: str | Path = "runs/v2_few_shot",
    tuning_dir: str | Path = "tuning/v2_few_shot",
    results_dir: str | Path = "results/v2/few_shot",
) -> dict[str, Path]:
    config = load_few_shot_config(config_path)
    project = Path(project_root)

    def absolute(path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else project / value

    config_absolute = absolute(config_path)
    source_data = absolute(data_root)
    protocol_data = absolute(protocol_data_root)
    runs = absolute(runs_root)
    tuning = absolute(tuning_dir)
    results = absolute(results_dir)
    tuning_file = tuning / "panda_head_tuning.json"
    queues: dict[str, list[dict[str, Any]]] = {
        "dev_cpu": [],
        "dev_gpu": [],
        "cpu": [],
        "gpu": [],
        "aggregate": [],
    }
    dev_shots = protocol_data / "split_seed0" / "shots.npz"
    queues["dev_cpu"].append({
        "id": "extract_few_shot_development",
        "cwd": str(project),
        "command": [
            cpu_python,
            "-m",
            "lorenz63_benchmark.v2.few_shot",
            "extract-shots",
            "--config",
            str(config_absolute),
            "--train",
            str(source_data / "split_seed0" / "noise_0" / "train.npz"),
            "--output",
            str(dev_shots),
        ],
    })
    queues["dev_gpu"].append({
        "id": "tune_panda_head_development",
        "cwd": str(project),
        "command": [
            gpu_python,
            "-m",
            "lorenz63_benchmark.v2.few_shot",
            "tune-panda",
            "--config",
            str(config_absolute),
            "--shots",
            str(dev_shots),
            "--validation",
            str(source_data / "split_seed0" / "validation.npz"),
            "--output-dir",
            str(tuning),
            "--device",
            "cuda",
        ],
    })

    shot_counts = [int(value) for value in config["few_shot"]["shot_counts"]]
    data_seeds = [int(value) for value in config["final"]["data_seeds"]]
    for data_seed in data_seeds:
        split = f"split_seed{data_seed}"
        split_data = protocol_data / split
        shots = split_data / "shots.npz"
        context = split_data / "context.npz"
        truth = split_data / "forecast_truth.npz"
        extract_shots_id = f"extract_few_shot_d{data_seed}"
        extract_context_id = f"extract_few_shot_context_d{data_seed}"
        queues["cpu"].append({
            "id": extract_shots_id,
            "cwd": str(project),
            "command": [
                cpu_python,
                "-m",
                "lorenz63_benchmark.v2.few_shot",
                "extract-shots",
                "--config",
                str(config_absolute),
                "--train",
                str(source_data / split / "noise_0" / "train.npz"),
                "--output",
                str(shots),
            ],
        })
        queues["cpu"].append({
            "id": extract_context_id,
            "cwd": str(project),
            "command": [
                cpu_python,
                "-m",
                "lorenz63_benchmark.v2.context_matched",
                "extract",
                "--config",
                str(config_absolute),
                "--test",
                str(source_data / split / "test.npz"),
                "--output-dir",
                str(split_data),
            ],
        })
        for method in ("sindy_weak", "sindy_weak_weighted"):
            for shot_count in shot_counts:
                condition = f"{method}_k{shot_count}"
                fit_dir = runs / "models" / split / condition
                fit_id = f"fit_few_shot_{condition}_d{data_seed}"
                queues["cpu"].append({
                    "id": fit_id,
                    "depends_on": [extract_shots_id],
                    "cwd": str(project),
                    "command": [
                        cpu_python,
                        "-m",
                        "lorenz63_benchmark.v2.few_shot",
                        "fit",
                        "--config",
                        str(config_absolute),
                        "--shots",
                        str(shots),
                        "--output-dir",
                        str(fit_dir),
                        "--method",
                        method,
                        "--shot-count",
                        str(shot_count),
                    ],
                })
                queues["cpu"].append({
                    "id": f"evaluate_few_shot_{condition}_d{data_seed}",
                    "depends_on": [fit_id, extract_context_id],
                    "cwd": str(project),
                    "command": [
                        cpu_python,
                        "-m",
                        "lorenz63_benchmark.v2.few_shot",
                        "evaluate",
                        "--config",
                        str(config_absolute),
                        "--context",
                        str(context),
                        "--forecast-truth",
                        str(truth),
                        "--output-dir",
                        str(runs / "evaluations" / split / condition),
                        "--method",
                        method,
                        "--shot-count",
                        str(shot_count),
                        "--model",
                        str(fit_dir / "model.npz"),
                        "--device",
                        "cpu",
                    ],
                })

        queues["gpu"].append({
            "id": f"evaluate_few_shot_panda_k0_d{data_seed}",
            "cwd": str(project),
            "command": [
                gpu_python,
                "-m",
                "lorenz63_benchmark.v2.few_shot",
                "evaluate",
                "--config",
                str(config_absolute),
                "--context",
                str(context),
                "--forecast-truth",
                str(truth),
                "--output-dir",
                str(runs / "evaluations" / split / "panda_k0"),
                "--method",
                "panda",
                "--shot-count",
                "0",
                "--device",
                "cuda",
            ],
        })
        for shot_count in shot_counts:
            condition = f"panda_k{shot_count}"
            fit_dir = runs / "models" / split / condition
            fit_id = f"fit_few_shot_{condition}_d{data_seed}"
            queues["gpu"].append({
                "id": fit_id,
                "cwd": str(project),
                "command": [
                    gpu_python,
                    "-m",
                    "lorenz63_benchmark.v2.few_shot",
                    "fit",
                    "--config",
                    str(config_absolute),
                    "--shots",
                    str(shots),
                    "--tuning",
                    str(tuning_file),
                    "--output-dir",
                    str(fit_dir),
                    "--method",
                    "panda",
                    "--shot-count",
                    str(shot_count),
                    "--device",
                    "cuda",
                ],
            })
            queues["gpu"].append({
                "id": f"evaluate_few_shot_{condition}_d{data_seed}",
                "depends_on": [fit_id],
                "cwd": str(project),
                "command": [
                    gpu_python,
                    "-m",
                    "lorenz63_benchmark.v2.few_shot",
                    "evaluate",
                    "--config",
                    str(config_absolute),
                    "--context",
                    str(context),
                    "--forecast-truth",
                    str(truth),
                    "--output-dir",
                    str(runs / "evaluations" / split / condition),
                    "--method",
                    "panda",
                    "--shot-count",
                    str(shot_count),
                    "--model",
                    str(fit_dir / "model.npz"),
                    "--device",
                    "cuda",
                ],
            })
    queues["aggregate"].append({
        "id": "aggregate_few_shot",
        "cwd": str(project),
        "command": [
            cpu_python,
            "-m",
            "lorenz63_benchmark.v2.few_shot",
            "aggregate",
            "--config",
            str(config_absolute),
            "--runs-root",
            str(runs / "evaluations"),
            "--output-dir",
            str(results),
        ],
    })

    output_dir = Path(output_dir)
    paths = {}
    for name, tasks in queues.items():
        path = output_dir / f"{name}.jsonl"
        _atomic_jsonl(path, tasks)
        paths[name] = path
    manifest = {
        "schema": "lorenz63-few-shot-queue-set-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "project_root": str(project),
        "cpu_python": cpu_python,
        "gpu_python": gpu_python,
        "shot_counts": shot_counts,
        "data_seeds": data_seeds,
        "tuning_file": str(tuning_file),
        "routes": {
            "shot_and_context_extraction": "cpu",
            "sindy_fit_and_evaluate": "cpu",
            "panda_tune_fit_and_evaluate": "gpu",
            "aggregate": "cpu",
        },
        "queues": {
            name: {"path": str(path), "task_count": len(queues[name])}
            for name, path in paths.items()
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare exact-window few-shot queues.")
    parser.add_argument("--config", default="configs/v2/few_shot.json")
    parser.add_argument("--output-dir", default="queues/v2_few_shot")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--cpu-python", default=sys.executable)
    parser.add_argument("--gpu-python", default=sys.executable)
    parser.add_argument("--data-root", default="data/v2")
    parser.add_argument("--protocol-data-root", default="data/v2_few_shot")
    parser.add_argument("--runs-root", default="runs/v2_few_shot")
    parser.add_argument("--tuning-dir", default="tuning/v2_few_shot")
    parser.add_argument("--results-dir", default="results/v2/few_shot")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = prepare_few_shot_queues(
        config_path=args.config,
        output_dir=args.output_dir,
        project_root=args.project_root,
        cpu_python=args.cpu_python,
        gpu_python=args.gpu_python,
        data_root=args.data_root,
        protocol_data_root=args.protocol_data_root,
        runs_root=args.runs_root,
        tuning_dir=args.tuning_dir,
        results_dir=args.results_dir,
    )
    print(*paths.values(), sep="\n")


if __name__ == "__main__":
    main()
