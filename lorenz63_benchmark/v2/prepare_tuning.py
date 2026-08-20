from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .contracts import CORE_METHODS
from .data import noise_label
from .prepare import _atomic_jsonl


def prepare_tuning_queues(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    cpu_python: str,
    gpu_python: str,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    project = Path(project_root)
    config_path = Path(config_path)
    remote_config = config_path if config_path.is_absolute() else project / config_path
    split_root = project / "data" / "v2" / "split_seed0"
    tune_tasks = []
    tuned_paths = []
    for level in config["data"]["noise_levels"]:
        label = noise_label(float(level))
        tuned_path = project / "configs" / "v2" / f"tuned_noise_{label}.json"
        tuned_paths.append(tuned_path)
        tune_tasks.append({
            "id": f"tune_noise_{label}", "cwd": str(project),
            "command": [
                gpu_python, "-m", "lorenz63_benchmark.v2.tune",
                "--config", str(remote_config),
                "--train", str(split_root / f"noise_{label}" / "train.npz"),
                "--validation", str(split_root / "validation.npz"),
                "--output-dir", str(project / "tuning" / "v2" / f"noise_{label}"),
                "--output-config", str(tuned_path),
                "--methods", *sorted(CORE_METHODS), "--model-seed", "0", "--device", "cuda",
            ],
        })
    frozen = project / "configs" / "v2" / "lorenz63_frozen.json"
    merge_tasks = [{
        "id": "merge_tuned_noise_configs", "cwd": str(project),
        "command": [
            gpu_python, "-m", "lorenz63_benchmark.v2.merge_tuning",
            "--base-config", str(remote_config),
            "--tuned-configs", *[str(path) for path in tuned_paths],
            "--output", str(frozen),
        ],
    }, {
        "id": "prepare_frozen_final_queues", "cwd": str(project),
        "depends_on": ["merge_tuned_noise_configs"],
        "command": [
            gpu_python, "-m", "lorenz63_benchmark.v2.prepare",
            "--config", str(frozen), "--output-dir", str(project / "queues" / "v2"),
            "--project-root", str(project), "--cpu-python", cpu_python,
            "--gpu-python", gpu_python,
        ],
    }]
    output_dir = Path(output_dir)
    tune_path = output_dir / "tune.jsonl"
    merge_path = output_dir / "tune_merge.jsonl"
    _atomic_jsonl(tune_path, tune_tasks)
    _atomic_jsonl(merge_path, merge_tasks)
    return tune_path, merge_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare parallel development tuning queues.")
    parser.add_argument("--config", default="configs/v2/lorenz63_paper.json")
    parser.add_argument("--output-dir", default="queues/v2")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--cpu-python", default=sys.executable)
    parser.add_argument("--gpu-python", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(*prepare_tuning_queues(
        config_path=args.config, output_dir=args.output_dir,
        project_root=args.project_root, cpu_python=args.cpu_python, gpu_python=args.gpu_python,
    ), sep="\n")


if __name__ == "__main__":
    main()
