from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_save_npz(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            if fieldnames:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_hash(root: str | Path | None = None) -> str:
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "torch", "pandas", "matplotlib"):
        try:
            module = __import__(name)
            packages[name] = str(module.__version__)
        except Exception:
            packages[name] = "unavailable"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "packages": packages,
    }


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def git_state(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def begin_manifest(
    output_dir: str | Path,
    *,
    run_id: str,
    method: str,
    track: str,
    config: dict[str, Any],
    inputs: dict[str, str | Path],
    seeds: dict[str, int],
    information_contract: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "lorenz63-run-manifest-v2",
        "run_id": run_id,
        "method": method,
        "track": track,
        "information_contract": list(information_contract),
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "command": command_line(),
        "environment": environment_snapshot(),
        "source_hash": source_hash(),
        "config_hash": sha256_json(config),
        "input_hashes": {name: sha256_file(path) for name, path in inputs.items()},
        "seeds": seeds,
        "artifacts": {},
        "training": {},
        "error": None,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def finish_manifest(
    output_dir: str | Path,
    manifest: dict[str, Any],
    *,
    artifacts: dict[str, str | Path],
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest = dict(manifest)
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["training"] = training or {}
    manifest["artifacts"] = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        for name, path in artifacts.items()
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def fail_manifest(output_dir: str | Path, manifest: dict[str, Any], error: BaseException) -> None:
    failed = dict(manifest)
    failed["status"] = "failed"
    failed["completed_at"] = utc_now()
    failed["error"] = f"{type(error).__name__}: {error}"
    atomic_write_json(Path(output_dir) / "manifest.json", failed)


def validate_artifact_hashes(manifest_path: str | Path) -> bool:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("artifacts", {}).values():
        path = Path(record["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.exists() or sha256_file(path) != record["sha256"]:
            return False
    return True
