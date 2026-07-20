from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, utc_now


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            if "id" not in task or "command" not in task:
                raise ValueError(f"invalid task on line {line_number}")
            tasks.append(task)
    task_ids = [str(task["id"]) for task in tasks]
    if len(set(task_ids)) != len(tasks):
        raise ValueError("task IDs must be unique")
    positions = {task_id: index for index, task_id in enumerate(task_ids)}
    for index, task in enumerate(tasks):
        dependencies = task.get("depends_on", [])
        if not isinstance(dependencies, list) or not all(
            isinstance(value, str) for value in dependencies
        ):
            raise ValueError(f"task {task_ids[index]} has invalid depends_on")
        for dependency in dependencies:
            if dependency not in positions:
                raise ValueError(f"task {task_ids[index]} has unknown dependency {dependency}")
            if positions[dependency] >= index:
                raise ValueError(
                    f"task {task_ids[index]} dependency {dependency} must appear earlier"
                )
    return tasks


def run_worker(
    queue_path: str | Path,
    state_dir: str | Path,
    *,
    worker_id: str,
    gpu_index: int | None = None,
    poll_seconds: float = 2.0,
    stale_claim_seconds: float = 21600.0,
) -> int:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(queue_path)
    environment = os.environ.copy()
    if gpu_index is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    completed = 0
    for task in tasks:
        task_id = str(task["id"])
        done_path = state_dir / f"{task_id}.done.json"
        failed_path = state_dir / f"{task_id}.failed.json"
        claim_path = state_dir / f"{task_id}.claim"
        if done_path.exists():
            continue
        dependencies = [str(value) for value in task.get("depends_on", [])]
        if any(not (state_dir / f"{dependency}.done.json").exists() for dependency in dependencies):
            continue
        try:
            claim_path.mkdir()
        except FileExistsError:
            age = time.time() - claim_path.stat().st_mtime
            if age <= stale_claim_seconds:
                continue
            stale_path = state_dir / f"{task_id}.stale.{os.getpid()}"
            try:
                os.replace(claim_path, stale_path)
                for child in stale_path.iterdir():
                    child.unlink()
                stale_path.rmdir()
                claim_path.mkdir()
            except (FileExistsError, FileNotFoundError, OSError):
                continue
        record = {
            "schema": "lorenz63-worker-task-v2", "task_id": task_id,
            "worker_id": worker_id, "host": socket.gethostname(),
            "gpu_index": gpu_index, "command": task["command"], "started_at": utc_now(),
        }
        atomic_write_json(claim_path / "status.json", record)
        log_path = state_dir / f"{task_id}.log"
        try:
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.run(
                    [str(value) for value in task["command"]],
                    cwd=task.get("cwd"), env=environment, stdout=log, stderr=subprocess.STDOUT,
                    check=False,
                )
            record["completed_at"] = utc_now()
            record["return_code"] = process.returncode
            record["log"] = str(log_path.resolve())
            if process.returncode == 0:
                atomic_write_json(done_path, record)
                failed_path.unlink(missing_ok=True)
                completed += 1
            else:
                atomic_write_json(failed_path, record)
                done_path.unlink(missing_ok=True)
        finally:
            for child in claim_path.iterdir():
                child.unlink()
            claim_path.rmdir()
        if poll_seconds > 0:
            time.sleep(poll_seconds)
    return completed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a resumable no-Slurm Lorenz63 v2 task queue.")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}-{os.getpid()}")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--poll-seconds", type=float, default=0.0)
    parser.add_argument("--stale-claim-seconds", type=float, default=21600.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    completed = run_worker(
        args.queue, args.state_dir, worker_id=args.worker_id,
        gpu_index=args.gpu_index, poll_seconds=args.poll_seconds,
        stale_claim_seconds=args.stale_claim_seconds,
    )
    print(json.dumps({"worker_id": args.worker_id, "tasks_completed": completed}, sort_keys=True))


if __name__ == "__main__":
    main()
