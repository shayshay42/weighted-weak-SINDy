from __future__ import annotations

import json
import sys

import pytest

from lorenz63_benchmark.v2.prepare import _atomic_jsonl
from lorenz63_benchmark.v2.worker import load_tasks, run_worker


def test_worker_runs_explicit_dependencies_in_order(tmp_path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    queue = tmp_path / "queue.jsonl"
    _atomic_jsonl(queue, [
        {
            "id": "first",
            "command": [sys.executable, "-c", f"from pathlib import Path; Path({str(first)!r}).write_text('ok')"],
        },
        {
            "id": "second",
            "depends_on": ["first"],
            "command": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"assert Path({str(first)!r}).read_text() == 'ok'; "
                    f"Path({str(second)!r}).write_text('ok')"
                ),
            ],
        },
    ])

    completed = run_worker(queue, tmp_path / "state", worker_id="test", poll_seconds=0.0)

    assert completed == 2
    assert second.read_text() == "ok"
    assert (tmp_path / "state" / "first.done.json").exists()
    assert (tmp_path / "state" / "second.done.json").exists()


def test_worker_rejects_forward_dependency(tmp_path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join([
            json.dumps({"id": "first", "depends_on": ["second"], "command": ["true"]}),
            json.dumps({"id": "second", "command": ["true"]}),
        ]) + "\n"
    )
    with pytest.raises(ValueError, match="must appear earlier"):
        load_tasks(queue)


def test_worker_successful_retry_clears_stale_failure_marker(tmp_path) -> None:
    attempt = tmp_path / "attempt.txt"
    queue = tmp_path / "queue.jsonl"
    code = (
        "from pathlib import Path; import sys; "
        f"p=Path({str(attempt)!r}); "
        "first=not p.exists(); p.write_text('attempted'); sys.exit(1 if first else 0)"
    )
    queue.write_text(json.dumps({"id": "retry", "command": [sys.executable, "-c", code]}) + "\n")
    state = tmp_path / "state"

    assert run_worker(queue, state, worker_id="first", poll_seconds=0.0) == 0
    assert (state / "retry.failed.json").exists()
    assert run_worker(queue, state, worker_id="second", poll_seconds=0.0) == 1
    assert (state / "retry.done.json").exists()
    assert not (state / "retry.failed.json").exists()
