from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lorenz63_benchmark.v2.aggregate import paired_hierarchical_bootstrap
from lorenz63_benchmark.v2.artifacts import (
    atomic_save_npz, atomic_write_json, sha256_file, validate_artifact_hashes,
)


def _bootstrap_frame() -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(("a", "b")):
        for data_seed in (1, 2):
            for model_seed in (0, 1):
                for trajectory in range(4):
                    rows.append({
                        "method": method, "data_seed": data_seed, "model_seed": model_seed,
                        "trajectory_id": trajectory,
                        "value": method_index + 0.1 * data_seed + 0.01 * model_seed + trajectory,
                    })
    return pd.DataFrame(rows)


def test_hierarchical_bootstrap_is_deterministic_and_paired() -> None:
    first = paired_hierarchical_bootstrap(_bootstrap_frame(), "value", resamples=100, seed=2026)
    second = paired_hierarchical_bootstrap(_bootstrap_frame(), "value", resamples=100, seed=2026)
    pd.testing.assert_frame_equal(first, second)
    estimates = first.set_index("method")["estimate"]
    assert abs((estimates["b"] - estimates["a"]) - 1.0) < 1e-12


def test_artifact_hash_validation_detects_mutation(tmp_path) -> None:
    artifact = tmp_path / "artifact.npz"
    atomic_save_npz(artifact, values=np.arange(5))
    manifest = tmp_path / "manifest.json"
    atomic_write_json(manifest, {
        "artifacts": {"artifact": {"path": str(artifact), "sha256": sha256_file(artifact)}}
    })
    assert validate_artifact_hashes(manifest)
    artifact.write_bytes(b"changed")
    assert not validate_artifact_hashes(manifest)
