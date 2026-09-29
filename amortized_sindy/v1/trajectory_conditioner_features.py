from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import sha256_file
from .io import atomic_save_npz, atomic_write_json
from .multifamily_data import DATASET_SCHEMA, EXPONENTS, _library_numpy
from .multifamily_train import EMBEDDING_KEYS, SOURCE_KEYS
from .native_forecast_gate import NATIVE_GATE_SCHEMA


FEATURE_SCHEMA = "amortized-sindy-trajectory-conditioner-features-v1"


def _load_context_only(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != SOURCE_KEYS:
            raise ValueError("trajectory artifact contains unexpected arrays")
        values = {
            "context_states": np.asarray(loaded["context_states"], dtype=np.float64),
            "context_times": np.asarray(loaded["context_times"], dtype=np.float64),
            "forecast_offsets": np.asarray(
                loaded["forecast_offsets"], dtype=np.float64
            ),
            "group_ids": np.asarray(loaded["group_ids"], dtype=np.int64),
            "trajectory_ids": np.asarray(loaded["trajectory_ids"], dtype=np.int64),
        }
    if not np.isfinite(values["context_states"]).all():
        raise ValueError("trajectory contexts contain non-finite values")
    return values


def _load_foundation_embeddings(
    path: str | Path,
    *,
    group_ids: np.ndarray,
    trajectory_ids: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != EMBEDDING_KEYS:
            raise ValueError("foundation embeddings contain unexpected arrays")
        features = np.asarray(loaded["embeddings"], dtype=np.float64)
        loaded_groups = np.asarray(loaded["group_ids"], dtype=np.int64)
        loaded_trajectories = np.asarray(loaded["trajectory_ids"], dtype=np.int64)
    if not np.array_equal(group_ids, loaded_groups) or not np.array_equal(
        trajectory_ids, loaded_trajectories
    ):
        raise ValueError("foundation embedding identities do not match contexts")
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("foundation embedding matrix is invalid")
    return features


def _load_native_predictions(
    manifest_path: str | Path,
    *,
    artifact_role: str,
    group_ids: np.ndarray,
    trajectory_ids: np.ndarray,
    forecast_offsets: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != NATIVE_GATE_SCHEMA:
        raise ValueError("invalid native forecast manifest")
    if manifest.get("artifact_role") != artifact_role:
        raise ValueError("native forecast artifact role does not match contexts")
    contract = manifest.get("information_contract", {})
    if contract.get("validation_future_passed_to_forecaster") is not False:
        raise ValueError("native forecaster must not receive validation future")
    if contract.get("target_optimizer_steps") != 0:
        raise ValueError("native forecaster used target optimizer steps")
    record = manifest.get("artifacts", {}).get("predictions", {})
    prediction_path = manifest_path.parent / record.get("path", "")
    if record.get("sha256") != sha256_file(prediction_path):
        raise ValueError("native prediction hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        if set(loaded.files) != {
            "predictions", "group_ids", "trajectory_ids", "forecast_offsets"
        }:
            raise ValueError("native prediction artifact contains unexpected arrays")
        predictions = np.asarray(loaded["predictions"], dtype=np.float64)
        loaded_groups = np.asarray(loaded["group_ids"], dtype=np.int64)
        loaded_trajectories = np.asarray(loaded["trajectory_ids"], dtype=np.int64)
        loaded_offsets = np.asarray(loaded["forecast_offsets"], dtype=np.float64)
    if not (
        np.array_equal(group_ids, loaded_groups)
        and np.array_equal(trajectory_ids, loaded_trajectories)
        and np.array_equal(forecast_offsets, loaded_offsets)
    ):
        raise ValueError("native prediction identities or time grid do not match")
    if not np.isfinite(predictions).all():
        raise ValueError("native predictions contain non-finite values")
    return predictions, manifest


def _weak_moments(
    normalized_states: np.ndarray,
    normalized_times: np.ndarray,
    *,
    window_length: int = 33,
    stride: int = 16,
) -> np.ndarray:
    context_length = normalized_states.shape[0]
    window_length = min(window_length, context_length)
    last_start = context_length - window_length
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    feature_rows = []
    target_rows = []
    for start in starts:
        stop = start + window_length
        times = normalized_times[start:stop]
        states = normalized_states[start:stop]
        duration = max(float(times[-1] - times[0]), 1e-8)
        coordinate = 2.0 * (times - times[0]) / duration - 1.0
        support = np.maximum(1.0 - coordinate ** 2, 0.0)
        bump = support ** 2
        bump_derivative = -4.0 * coordinate * support
        trap = np.zeros_like(times)
        trap[0] = 0.5 * (times[1] - times[0])
        trap[-1] = 0.5 * (times[-1] - times[-2])
        trap[1:-1] = 0.5 * (times[2:] - times[:-2])
        library = _library_numpy(states)
        for polynomial, polynomial_derivative in (
            (np.ones_like(coordinate), np.zeros_like(coordinate)),
            (coordinate, np.ones_like(coordinate)),
        ):
            phi = bump * polynomial
            derivative = (
                bump_derivative * polynomial + bump * polynomial_derivative
            ) * (2.0 / duration)
            feature_rows.append(np.einsum("l,lm->m", trap * phi, library))
            target_rows.append(-np.einsum("l,ld->d", trap * derivative, states))
    features = np.asarray(feature_rows)
    targets = np.asarray(target_rows)
    row_count = features.shape[0]
    gram = features.T @ features / row_count
    cross = features.T @ targets / row_count
    statistics = np.concatenate((gram.reshape(-1), cross.reshape(-1)))
    return np.sign(statistics) * np.log1p(np.abs(statistics))


def _raw_features(
    *,
    contexts: np.ndarray,
    times: np.ndarray,
    native_predictions: np.ndarray,
    foundation_embeddings: np.ndarray,
    context_tail_steps: int,
) -> tuple[np.ndarray, dict[str, int]]:
    if context_tail_steps < 8 or context_tail_steps > contexts.shape[1]:
        raise ValueError("invalid ordered context tail length")
    if native_predictions.shape[0] != contexts.shape[0] or native_predictions.shape[2] != 3:
        raise ValueError("native predictions do not match context batch/state axes")
    if not np.allclose(native_predictions[:, 0], contexts[:, -1]):
        raise ValueError("native prediction must start at the context endpoint")
    mean = contexts.mean(axis=1, keepdims=True)
    scale = np.maximum(contexts.std(axis=1, keepdims=True), 1e-6)
    duration = max(float(times[-1] - times[0]), 1e-8)
    normalized_context = (contexts - mean) / scale
    normalized_native = (native_predictions - mean) / scale
    normalized_times = (times - times[-1]) / duration
    ordered_context = normalized_context[:, -context_tail_steps:].reshape(
        contexts.shape[0], -1
    )
    ordered_native = normalized_native[:, 1:].reshape(contexts.shape[0], -1)
    native_increments = np.diff(normalized_native, axis=1).reshape(
        contexts.shape[0], -1
    )
    weak = np.asarray([
        _weak_moments(states, normalized_times) for states in normalized_context
    ])
    normalization = np.concatenate((
        (mean[:, 0] / scale[:, 0]),
        np.log(scale[:, 0]),
        np.full((contexts.shape[0], 1), np.log(duration)),
    ), axis=1)
    blocks = {
        "foundation": foundation_embeddings,
        "ordered_context": ordered_context,
        "ordered_native_forecast": ordered_native,
        "native_forecast_increments": native_increments,
        "weak_moments": weak,
        "normalization": normalization,
    }
    widths = {name: int(value.shape[1]) for name, value in blocks.items()}
    return np.concatenate(tuple(blocks.values()), axis=1), widths


def build_trajectory_conditioner_features(
    *,
    source_train_path: str | Path,
    source_validation_path: str | Path,
    dataset_manifest_path: str | Path,
    source_foundation_embeddings_path: str | Path,
    validation_foundation_embeddings_path: str | Path,
    source_native_manifest_path: str | Path,
    validation_native_manifest_path: str | Path,
    output_dir: str | Path,
    context_tail_steps: int = 64,
    force: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"trajectory feature output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads(
        Path(dataset_manifest_path).read_text(encoding="utf-8")
    )
    if dataset_manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid multifamily dataset manifest")
    for name, path in (
        ("source_train", source_train_path),
        ("source_validation", source_validation_path),
    ):
        if dataset_manifest.get("artifacts", {}).get(name, {}).get(
            "sha256"
        ) != sha256_file(path):
            raise ValueError(f"{name} does not match the dataset manifest")
    source = _load_context_only(source_train_path)
    validation = _load_context_only(source_validation_path)
    source_foundation = _load_foundation_embeddings(
        source_foundation_embeddings_path,
        group_ids=source["group_ids"],
        trajectory_ids=source["trajectory_ids"],
    )
    validation_foundation = _load_foundation_embeddings(
        validation_foundation_embeddings_path,
        group_ids=validation["group_ids"],
        trajectory_ids=validation["trajectory_ids"],
    )
    source_native, source_native_manifest = _load_native_predictions(
        source_native_manifest_path,
        artifact_role="source_train",
        group_ids=source["group_ids"],
        trajectory_ids=source["trajectory_ids"],
        forecast_offsets=source["forecast_offsets"],
    )
    validation_native, validation_native_manifest = _load_native_predictions(
        validation_native_manifest_path,
        artifact_role="source_validation",
        group_ids=validation["group_ids"],
        trajectory_ids=validation["trajectory_ids"],
        forecast_offsets=validation["forecast_offsets"],
    )
    if source_native_manifest.get("model") != validation_native_manifest.get("model"):
        raise ValueError("source and validation native forecasts use different models")
    raw_source, widths = _raw_features(
        contexts=source["context_states"],
        times=source["context_times"],
        native_predictions=source_native,
        foundation_embeddings=source_foundation,
        context_tail_steps=context_tail_steps,
    )
    raw_validation, validation_widths = _raw_features(
        contexts=validation["context_states"],
        times=validation["context_times"],
        native_predictions=validation_native,
        foundation_embeddings=validation_foundation,
        context_tail_steps=context_tail_steps,
    )
    if widths != validation_widths:
        raise ValueError("source and validation feature layouts differ")
    mean = raw_source.mean(axis=0)
    scale = np.maximum(raw_source.std(axis=0), 1e-6)
    source_features = ((raw_source - mean) / scale).astype(np.float32)
    validation_features = ((raw_validation - mean) / scale).astype(np.float32)
    source_path = atomic_save_npz(
        output_dir / "source_features.npz",
        embeddings=source_features,
        group_ids=source["group_ids"],
        trajectory_ids=source["trajectory_ids"],
    )
    validation_path = atomic_save_npz(
        output_dir / "validation_features.npz",
        embeddings=validation_features,
        group_ids=validation["group_ids"],
        trajectory_ids=validation["trajectory_ids"],
    )
    normalization_path = atomic_save_npz(
        output_dir / "source_feature_normalization.npz",
        mean=mean,
        scale=scale,
    )
    return atomic_write_json(output_dir / "manifest.json", {
        "schema": FEATURE_SCHEMA,
        "future_states_read": False,
        "source_only_feature_normalization": True,
        "target_optimizer_steps": 0,
        "target_sparse_regression_solves": 0,
        "context_tail_steps": context_tail_steps,
        "feature_width": int(source_features.shape[1]),
        "feature_blocks": widths,
        "library_exponents": [list(value) for value in EXPONENTS],
        "native_model": source_native_manifest.get("model"),
        "artifacts": {
            "source_features": {
                "path": source_path.name,
                "sha256": sha256_file(source_path),
            },
            "validation_features": {
                "path": validation_path.name,
                "sha256": sha256_file(validation_path),
            },
            "normalization": {
                "path": normalization_path.name,
                "sha256": sha256_file(normalization_path),
            },
            "source_native_manifest_sha256": sha256_file(
                source_native_manifest_path
            ),
            "validation_native_manifest_sha256": sha256_file(
                validation_native_manifest_path
            ),
            "source_foundation_embeddings_sha256": sha256_file(
                source_foundation_embeddings_path
            ),
            "validation_foundation_embeddings_sha256": sha256_file(
                validation_foundation_embeddings_path
            ),
        },
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ordered native-forecast and weak-moment conditioner features."
    )
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-validation", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--source-foundation-embeddings", required=True)
    parser.add_argument("--validation-foundation-embeddings", required=True)
    parser.add_argument("--source-native-manifest", required=True)
    parser.add_argument("--validation-native-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-tail-steps", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(build_trajectory_conditioner_features(
        source_train_path=args.source_train,
        source_validation_path=args.source_validation,
        dataset_manifest_path=args.dataset_manifest,
        source_foundation_embeddings_path=args.source_foundation_embeddings,
        validation_foundation_embeddings_path=args.validation_foundation_embeddings,
        source_native_manifest_path=args.source_native_manifest,
        validation_native_manifest_path=args.validation_native_manifest,
        output_dir=args.output_dir,
        context_tail_steps=args.context_tail_steps,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
