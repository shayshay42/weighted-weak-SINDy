from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import sha256_file
from .io import atomic_save_npz, atomic_write_json
from .multifamily_data import DATASET_SCHEMA, EXPONENTS, _library_numpy
from .multifamily_train import EMBEDDING_KEYS, SOURCE_KEYS, _load_source
from .library import normalized_to_physical_coefficients
from .tabpfn_conditioner import (
    TabPFNCoefficientConditioner,
    TabPFNCoefficientSpec,
)


PILOT_SCHEMA = "amortized-sindy-tabpfn-source-validation-v1"
TEACHER_KEYS = {"group_ids", "coefficients", "residual_mse", "active_terms"}


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_embeddings(
    path: str | Path,
    *,
    expected_group_ids: np.ndarray,
    expected_trajectory_ids: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != EMBEDDING_KEYS:
            raise ValueError("foundation embedding file contains unexpected arrays")
        embeddings = np.asarray(loaded["embeddings"], dtype=np.float64)
        group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        trajectory_ids = np.asarray(loaded["trajectory_ids"], dtype=np.int64)
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_group_ids.size:
        raise ValueError("foundation embeddings must have shape [trajectory,feature]")
    if not np.array_equal(group_ids, expected_group_ids):
        raise ValueError("foundation embedding group order differs from trajectory data")
    if not np.array_equal(trajectory_ids, expected_trajectory_ids):
        raise ValueError(
            "foundation embedding trajectory order differs from trajectory data"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("foundation embeddings contain non-finite values")
    return embeddings


def _load_teacher_by_source_trajectory(
    *,
    labels_path: str | Path,
    teacher_manifest_path: str | Path,
    source_train_path: str | Path,
    source_group_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = _load_json(teacher_manifest_path)
    if manifest.get("schema") != "amortized-sindy-source-teacher-v1":
        raise ValueError("invalid source teacher manifest")
    if manifest.get("future_source_states_used") is not False:
        raise ValueError("source teacher must not read source futures")
    if manifest.get("generator_coefficients_used") is not False:
        raise ValueError("source teacher must not read generator coefficients")
    if manifest.get("source_train_sha256") != sha256_file(source_train_path):
        raise ValueError("source teacher was fitted on a different source dataset")
    if manifest.get("labels", {}).get("sha256") != sha256_file(labels_path):
        raise ValueError("source teacher label hash mismatch")
    with np.load(labels_path, allow_pickle=False) as loaded:
        if set(loaded.files) != TEACHER_KEYS:
            raise ValueError("source teacher labels contain unexpected arrays")
        teacher_group_ids = np.asarray(loaded["group_ids"], dtype=np.int64)
        coefficients = np.asarray(loaded["coefficients"], dtype=np.float64)
    if coefficients.shape != (teacher_group_ids.size, len(EXPONENTS), 3):
        raise ValueError("source teacher coefficients have an unexpected shape")
    by_group = {
        int(group_id): coefficients[index]
        for index, group_id in enumerate(teacher_group_ids)
    }
    if any(int(group_id) not in by_group for group_id in source_group_ids):
        raise ValueError("source teacher does not cover every source trajectory")
    return (
        np.asarray([by_group[int(group_id)] for group_id in source_group_ids]),
        manifest,
    )


def _fit_source_pca(
    source: np.ndarray,
    *,
    component_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if component_count < 1:
        raise ValueError("PCA component count must be positive")
    mean = source.mean(axis=0)
    centered = source - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    retained = min(component_count, source.shape[0] - 1, right_vectors.shape[0])
    if retained < 1:
        raise ValueError("at least two source embeddings are required for PCA")
    components = right_vectors[:retained]
    projected = centered @ components.T
    scale = np.maximum(projected.std(axis=0), 1e-8)
    return mean, components, scale, singular_values[:retained]


def _project(
    embeddings: np.ndarray,
    *,
    mean: np.ndarray,
    components: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    if embeddings.shape[1] != mean.size:
        raise ValueError("embedding width differs from the source PCA width")
    projected = ((embeddings - mean) @ components.T) / scale
    if not np.isfinite(projected).all():
        raise ValueError("PCA projection contains non-finite values")
    return projected


def _rk4_step(
    states: np.ndarray,
    step: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    def field(values: np.ndarray) -> np.ndarray:
        return np.einsum(
            "bi,bij->bj", _library_numpy(values), coefficients, optimize=True
        )

    k1 = field(states)
    k2 = field(states + 0.5 * step * k1)
    k3 = field(states + 0.5 * step * k2)
    k4 = field(states + step * k3)
    return states + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _rollout(
    initial_states: np.ndarray,
    offsets: np.ndarray,
    coefficients: np.ndarray,
    *,
    max_step: float = 0.005,
    state_bound: float = 1e6,
) -> tuple[np.ndarray, np.ndarray]:
    if offsets.ndim != 1 or offsets.size < 2 or offsets[0] != 0.0:
        raise ValueError("forecast offsets must be one-dimensional and start at zero")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("forecast offsets must be strictly increasing")
    states = np.asarray(initial_states, dtype=np.float64).copy()
    predictions = np.empty((states.shape[0], offsets.size, states.shape[1]))
    predictions[:, 0] = states
    failed = np.zeros(states.shape[0], dtype=bool)
    for index, interval in enumerate(np.diff(offsets), start=1):
        substeps = max(1, int(np.ceil(float(interval) / max_step)))
        step = float(interval) / substeps
        for _ in range(substeps):
            active = ~failed
            if np.any(active):
                with np.errstate(over="ignore", invalid="ignore"):
                    states[active] = _rk4_step(
                        states[active], step, coefficients[active]
                    )
                failed |= ~np.isfinite(states).all(axis=1)
                failed |= np.max(np.abs(states), axis=1) > state_bound
                states[failed] = np.nan
        predictions[:, index] = states
    return predictions, failed


def _metrics(
    predictions: np.ndarray,
    validation: dict[str, np.ndarray],
    failed: np.ndarray,
) -> dict[str, Any]:
    contexts = np.asarray(validation["context_states"], dtype=np.float64)
    future = np.asarray(validation["future_states"], dtype=np.float64)
    scale = np.maximum(contexts.std(axis=1, keepdims=True), 1e-6)
    constant = np.broadcast_to(contexts[:, -1:, :], future.shape)
    constant_mse = float(np.mean(((constant - future) / scale) ** 2))
    finite_rows = ~failed & np.isfinite(predictions).all(axis=(1, 2))
    if np.all(finite_rows):
        mse = float(np.mean(((predictions - future) / scale) ** 2))
    else:
        mse = float("inf")
    return {
        "trajectory_count": int(predictions.shape[0]),
        "failed_rollout_count": int(np.sum(~finite_rows)),
        "all_predictions_finite": bool(np.all(finite_rows)),
        "mean_normalized_mse": mse,
        "mean_constant_field_normalized_mse": constant_mse,
        "relative_mse_vs_constant_field": mse / max(constant_mse, 1e-12),
    }


def _physical_to_normalized_coefficients(
    physical_coefficients: np.ndarray,
    contexts: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Change polynomial coordinates without fitting any observed derivatives."""
    contexts = np.asarray(contexts, dtype=np.float64)
    physical_coefficients = np.asarray(physical_coefficients, dtype=np.float64)
    mean = contexts.mean(axis=1)
    scale = np.maximum(contexts.std(axis=1), 1e-6)
    duration = max(float(times[-1] - times[0]), 1e-8)
    rng = np.random.default_rng(20260820)
    normalized_probe = rng.uniform(-2.0, 2.0, size=(64, contexts.shape[-1]))
    normalized_library = _library_numpy(normalized_probe)
    inverse_library = np.linalg.pinv(normalized_library)
    physical_probe = (
        mean[:, None, :] + scale[:, None, :] * normalized_probe[None, :, :]
    )
    physical_derivative = np.einsum(
        "bql,bls->bqs",
        _library_numpy(physical_probe),
        physical_coefficients,
        optimize=True,
    )
    normalized_derivative = (
        duration * physical_derivative / scale[:, None, :]
    )
    normalized_coefficients = np.einsum(
        "lq,bqs->bls", inverse_library, normalized_derivative, optimize=True
    )
    return normalized_coefficients, mean, scale, duration


def _normalized_rollout(
    *,
    contexts: np.ndarray,
    context_times: np.ndarray,
    offsets: np.ndarray,
    normalized_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = contexts.mean(axis=1)
    scale = np.maximum(contexts.std(axis=1), 1e-6)
    duration = max(float(context_times[-1] - context_times[0]), 1e-8)
    initial = (contexts[:, -1] - mean) / scale
    normalized_prediction, failed = _rollout(
        initial,
        offsets / duration,
        normalized_coefficients,
    )
    prediction = normalized_prediction * scale[:, None, :] + mean[:, None, :]
    return prediction, failed


def run_tabpfn_source_validation(
    *,
    source_train_path: str | Path,
    source_validation_path: str | Path,
    dataset_manifest_path: str | Path,
    source_embeddings_path: str | Path,
    validation_embeddings_path: str | Path,
    teacher_labels_path: str | Path,
    teacher_manifest_path: str | Path,
    output_dir: str | Path,
    pca_components: int = 32,
    coefficient_threshold: float = 1e-2,
    coefficient_coordinate: str = "physical",
    model_path: str = "auto",
    n_estimators: int = 1,
    device: str = "cuda:0",
    random_state: int = 2026,
    force: bool = False,
) -> Path:
    if coefficient_threshold < 0:
        raise ValueError("coefficient threshold must be non-negative")
    if coefficient_coordinate not in {"physical", "normalized"}:
        raise ValueError("coefficient coordinate must be physical or normalized")
    model_file = Path(model_path)
    model_weights = {
        "path": model_path,
        "sha256": sha256_file(model_file) if model_file.is_file() else None,
    }
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"TabPFN output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = _load_json(dataset_manifest_path)
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

    source = _load_source(source_train_path)
    validation = _load_source(source_validation_path)
    if set(source) != SOURCE_KEYS or set(validation) != SOURCE_KEYS:
        raise ValueError("invalid source trajectory contract")
    source_embeddings = _load_embeddings(
        source_embeddings_path,
        expected_group_ids=source["group_ids"],
        expected_trajectory_ids=source["trajectory_ids"],
    )
    validation_embeddings = _load_embeddings(
        validation_embeddings_path,
        expected_group_ids=validation["group_ids"],
        expected_trajectory_ids=validation["trajectory_ids"],
    )
    source_coefficients, teacher_manifest = _load_teacher_by_source_trajectory(
        labels_path=teacher_labels_path,
        teacher_manifest_path=teacher_manifest_path,
        source_train_path=source_train_path,
        source_group_ids=source["group_ids"],
    )
    if coefficient_coordinate == "normalized":
        source_coefficients, _, _, _ = _physical_to_normalized_coefficients(
            source_coefficients,
            source["context_states"],
            source["context_times"],
        )

    pca_mean, pca_vectors, pca_scale, singular_values = _fit_source_pca(
        source_embeddings, component_count=pca_components
    )
    projected_source = _project(
        source_embeddings,
        mean=pca_mean,
        components=pca_vectors,
        scale=pca_scale,
    )
    projected_validation = _project(
        validation_embeddings,
        mean=pca_mean,
        components=pca_vectors,
        scale=pca_scale,
    )
    conditioner = TabPFNCoefficientConditioner(
        TabPFNCoefficientSpec(
            model_path=model_path,
            n_estimators=n_estimators,
            device=device,
            random_state=random_state,
        )
    )
    started = time.perf_counter()
    try:
        conditioner.fit_source(projected_source, source_coefficients)
    except Exception as error:
        error_type = type(error).__name__
        status = "blocked" if error_type == "TabPFNLicenseError" else "failed"
        atomic_write_json(output_dir / "manifest.json", {
            "schema": PILOT_SCHEMA,
            "status": status,
            "selection_role": "source_family_validation_only",
            "heldout_evaluation_data_read": False,
            "target_optimizer_steps": 0,
            "target_sparse_regression_solves": 0,
            "blocker": {
                "type": error_type,
                "message": str(error),
            },
            "source_pca": {
                "fit_scope": "source_embeddings_only",
                "requested_components": pca_components,
                "retained_components": int(pca_vectors.shape[0]),
            },
            "coefficient_coordinate": coefficient_coordinate,
            "model_weights": model_weights,
            "artifacts": {
                "source_train": sha256_file(source_train_path),
                "source_validation": sha256_file(source_validation_path),
                "source_embeddings": sha256_file(source_embeddings_path),
                "source_validation_embeddings": sha256_file(
                    validation_embeddings_path
                ),
                "teacher_labels": sha256_file(teacher_labels_path),
                "teacher_manifest": sha256_file(teacher_manifest_path),
            },
        })
        raise
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predicted_coefficients = conditioner.predict_coefficients(projected_validation)
    inference_seconds = time.perf_counter() - started
    predicted_coefficients[
        np.abs(predicted_coefficients) < coefficient_threshold
    ] = 0.0
    if coefficient_coordinate == "normalized":
        predictions, failed = _normalized_rollout(
            contexts=validation["context_states"],
            context_times=validation["context_times"],
            offsets=validation["forecast_offsets"],
            normalized_coefficients=predicted_coefficients,
        )
        validation_mean = validation["context_states"].mean(axis=1)
        validation_scale = np.maximum(
            validation["context_states"].std(axis=1), 1e-6
        )
        validation_duration = max(
            float(validation["context_times"][-1] - validation["context_times"][0]),
            1e-8,
        )
        physical_coefficients = normalized_to_physical_coefficients(
            torch.as_tensor(predicted_coefficients, dtype=torch.float64),
            EXPONENTS,
            state_mean=torch.as_tensor(validation_mean, dtype=torch.float64),
            state_scale=torch.as_tensor(validation_scale, dtype=torch.float64),
            time_scale=torch.full(
                (predicted_coefficients.shape[0],),
                validation_duration,
                dtype=torch.float64,
            ),
        ).numpy()
    else:
        predictions, failed = _rollout(
            validation["context_states"][:, -1],
            validation["forecast_offsets"],
            predicted_coefficients,
        )
        physical_coefficients = predicted_coefficients
    metrics = _metrics(predictions, validation, failed)

    arrays_path = atomic_save_npz(
        output_dir / "source_validation_predictions.npz",
        predictions=predictions,
        physical_coefficients=physical_coefficients,
        conditioner_coefficients=predicted_coefficients,
        support=(np.abs(predicted_coefficients) > 0),
        group_ids=np.asarray(validation["group_ids"], dtype=np.int64),
        trajectory_ids=np.asarray(validation["trajectory_ids"], dtype=np.int64),
        forecast_offsets=np.asarray(validation["forecast_offsets"], dtype=np.float64),
    )
    projection_path = atomic_save_npz(
        output_dir / "source_pca.npz",
        mean=pca_mean,
        components=pca_vectors,
        projected_scale=pca_scale,
        singular_values=singular_values,
    )
    protocol = conditioner.protocol_manifest(
        source_dataset_ids=list(dataset_manifest.get("source_families", [])),
        excluded_evaluation_dataset_ids=[
            "synthetic/lorenz_heldout",
            "CTF4Science/ODE_Lorenz",
        ],
    )
    manifest = {
        "schema": PILOT_SCHEMA,
        "selection_role": "source_family_validation_only",
        "source_validation_family": dataset_manifest.get("validation_families"),
        "heldout_evaluation_data_read": False,
        "target_optimizer_steps": 0,
        "target_sparse_regression_solves": 0,
        "protocol": protocol,
        "source_pca": {
            "fit_scope": "source_embeddings_only",
            "requested_components": pca_components,
            "retained_components": int(pca_vectors.shape[0]),
            "artifact": {
                "path": projection_path.name,
                "sha256": sha256_file(projection_path),
            },
        },
        "coefficient_threshold": coefficient_threshold,
        "coefficient_coordinate": coefficient_coordinate,
        "model_weights": model_weights,
        "library_exponents": [list(value) for value in EXPONENTS],
        "metrics": metrics,
        "runtime_seconds": {
            "fit": fit_seconds,
            "inference": inference_seconds,
        },
        "artifacts": {
            "source_train": sha256_file(source_train_path),
            "source_validation": sha256_file(source_validation_path),
            "source_embeddings": sha256_file(source_embeddings_path),
            "source_validation_embeddings": sha256_file(
                validation_embeddings_path
            ),
            "teacher_labels": sha256_file(teacher_labels_path),
            "teacher_manifest": sha256_file(teacher_manifest_path),
            "predictions": {
                "path": arrays_path.name,
                "sha256": sha256_file(arrays_path),
            },
        },
        "teacher": {
            "ridge": teacher_manifest.get("ridge"),
            "threshold": teacher_manifest.get("threshold"),
        },
    }
    return atomic_write_json(output_dir / "manifest.json", manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a source-only PCA+TabPFN SINDy coefficient map and score it "
            "on a held-out source family."
        )
    )
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-validation", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--source-embeddings", required=True)
    parser.add_argument("--validation-embeddings", required=True)
    parser.add_argument("--teacher-labels", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--coefficient-threshold", type=float, default=1e-2)
    parser.add_argument(
        "--coefficient-coordinate",
        choices=("physical", "normalized"),
        default="physical",
    )
    parser.add_argument("--model-path", default="auto")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(run_tabpfn_source_validation(
        source_train_path=args.source_train,
        source_validation_path=args.source_validation,
        dataset_manifest_path=args.dataset_manifest,
        source_embeddings_path=args.source_embeddings,
        validation_embeddings_path=args.validation_embeddings,
        teacher_labels_path=args.teacher_labels,
        teacher_manifest_path=args.teacher_manifest,
        output_dir=args.output_dir,
        pca_components=args.pca_components,
        coefficient_threshold=args.coefficient_threshold,
        coefficient_coordinate=args.coefficient_coordinate,
        model_path=args.model_path,
        n_estimators=args.n_estimators,
        device=args.device,
        random_state=args.random_state,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
