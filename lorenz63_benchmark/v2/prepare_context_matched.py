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
from .config import load_config
from .context_matched import CONTEXT_METHODS
from .prepare import _atomic_jsonl


def prepare_context_matched_queues(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    cpu_python: str,
    gpu_python: str,
    data_root: str | Path = "data/v2",
    protocol_data_root: str | Path = "data/v2_context_matched",
    runs_root: str | Path = "runs/v2_context_matched",
    results_dir: str | Path = "results/v2/context_matched",
) -> dict[str, Path]:
    config = load_config(config_path)
    project = Path(project_root)

    def absolute(path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else project / value

    config_absolute = absolute(config_path)
    source_data = absolute(data_root)
    protocol_data = absolute(protocol_data_root)
    runs = absolute(runs_root)
    results = absolute(results_dir)
    queues: dict[str, list[dict[str, Any]]] = {"cpu": [], "gpu": [], "aggregate": []}
    data_seeds = [int(value) for value in config["final"]["data_seeds"]]
    for data_seed in data_seeds:
        split = f"split_seed{data_seed}"
        test = source_data / split / "test.npz"
        extracted = protocol_data / split
        context = extracted / "context.npz"
        truth = extracted / "forecast_truth.npz"
        extract_id = f"extract_context_d{data_seed}"
        queues["cpu"].append({
            "id": extract_id,
            "cwd": str(project),
            "command": [
                cpu_python,
                "-m",
                "lorenz63_benchmark.v2.context_matched",
                "extract",
                "--config",
                str(config_absolute),
                "--test",
                str(test),
                "--output-dir",
                str(extracted),
            ],
        })
        for method in ("sindy_weak_weighted", "sindy_weak"):
            model_dir = runs / "models" / split / method
            models = model_dir / "models.npz"
            fit_id = f"fit_context_{method}_d{data_seed}"
            queues["cpu"].append({
                "id": fit_id,
                "depends_on": [extract_id],
                "cwd": str(project),
                "command": [
                    cpu_python,
                    "-m",
                    "lorenz63_benchmark.v2.context_matched",
                    "fit-sindy",
                    "--config",
                    str(config_absolute),
                    "--context",
                    str(context),
                    "--output-dir",
                    str(model_dir),
                    "--method",
                    method,
                ],
            })
            evaluation_dir = runs / "evaluations" / split / method
            queues["cpu"].append({
                "id": f"evaluate_context_{method}_d{data_seed}",
                "depends_on": [fit_id],
                "cwd": str(project),
                "command": [
                    cpu_python,
                    "-m",
                    "lorenz63_benchmark.v2.context_matched",
                    "evaluate",
                    "--config",
                    str(config_absolute),
                    "--context",
                    str(context),
                    "--forecast-truth",
                    str(truth),
                    "--models",
                    str(models),
                    "--output-dir",
                    str(evaluation_dir),
                    "--method",
                    method,
                    "--device",
                    "cpu",
                ],
            })
        panda_dir = runs / "evaluations" / split / "panda_zero_shot"
        queues["gpu"].append({
            "id": f"evaluate_context_panda_zero_shot_d{data_seed}",
            "cwd": str(project),
            "command": [
                gpu_python,
                "-m",
                "lorenz63_benchmark.v2.context_matched",
                "evaluate",
                "--config",
                str(config_absolute),
                "--context",
                str(context),
                "--forecast-truth",
                str(truth),
                "--output-dir",
                str(panda_dir),
                "--method",
                "panda_zero_shot",
                "--device",
                "cuda",
            ],
        })
    queues["aggregate"].append({
        "id": "aggregate_context_matched",
        "cwd": str(project),
        "command": [
            cpu_python,
            "-m",
            "lorenz63_benchmark.v2.context_matched",
            "aggregate",
            "--config",
            str(config_absolute),
            "--runs-root",
            str(runs),
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
        "schema": "lorenz63-context-matched-queue-set-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "project_root": str(project),
        "cpu_python": cpu_python,
        "gpu_python": gpu_python,
        "data_seeds": data_seeds,
        "methods": list(CONTEXT_METHODS),
        "routes": {
            "extract_fit_sindy_evaluate": "cpu",
            "panda_evaluate": "gpu",
            "aggregate": "cpu",
        },
        "paths": {
            "source_data": str(source_data),
            "protocol_data": str(protocol_data),
            "runs": str(runs),
            "results": str(results),
        },
        "queues": {
            name: {"path": str(path), "task_count": len(queues[name])}
            for name, path in paths.items()
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare context-matched Lorenz63 queues.")
    parser.add_argument("--config", default="configs/v2/context_matched.json")
    parser.add_argument("--output-dir", default="queues/v2_context_matched")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--cpu-python", default=sys.executable)
    parser.add_argument("--gpu-python", default=sys.executable)
    parser.add_argument("--data-root", default="data/v2")
    parser.add_argument("--protocol-data-root", default="data/v2_context_matched")
    parser.add_argument("--runs-root", default="runs/v2_context_matched")
    parser.add_argument("--results-dir", default="results/v2/context_matched")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = prepare_context_matched_queues(
        config_path=args.config,
        output_dir=args.output_dir,
        project_root=args.project_root,
        cpu_python=args.cpu_python,
        gpu_python=args.gpu_python,
        data_root=args.data_root,
        protocol_data_root=args.protocol_data_root,
        runs_root=args.runs_root,
        results_dir=args.results_dir,
    )
    print(*paths.values(), sep="\n")


if __name__ == "__main__":
    main()
