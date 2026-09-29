from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .checkpoint import sha256_file
from .io import atomic_save_npz, atomic_write_json
from .multifamily_data import EXPONENTS, _library_numpy
from .multifamily_train import SOURCE_KEYS, _load_source


def _ridge(features: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    return np.linalg.solve(
        features.T @ features + ridge * np.eye(features.shape[1]),
        features.T @ targets,
    )


def _stlsq(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
    threshold: float,
    iterations: int = 20,
) -> np.ndarray:
    scales = np.sqrt(np.mean(features ** 2, axis=0))
    scales = np.where(scales > 1e-12, scales, 1.0)
    normalized = features / scales
    coefficients = _ridge(normalized, targets, ridge)
    active = np.abs(coefficients / scales[:, None]) >= threshold
    for _ in range(iterations):
        previous = active.copy()
        for output in range(targets.shape[1]):
            mask = active[:, output]
            coefficients[~mask, output] = 0.0
            if np.any(mask):
                coefficients[mask, output] = _ridge(
                    normalized[:, mask], targets[:, output : output + 1], ridge
                ).ravel()
                active[:, output] = (
                    np.abs(coefficients[:, output] / scales) >= threshold
                )
        if np.array_equal(previous, active):
            break
    result = coefficients / scales[:, None]
    result[~active] = 0.0
    return result


def build_source_teacher_labels(
    *,
    source_train_path: str | Path,
    output_dir: str | Path,
    ridge: float = 1e-6,
    threshold: float = 1e-2,
    force: bool = False,
) -> Path:
    if ridge < 0 or threshold < 0:
        raise ValueError("teacher ridge and threshold must be non-negative")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"teacher output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    values = _load_source(source_train_path)
    if set(values) != SOURCE_KEYS:
        raise ValueError("teacher received an invalid source trajectory contract")
    contexts = values["context_states"]
    times = values["context_times"]
    group_ids = values["group_ids"]
    coefficients = []
    residuals = []
    active_terms = []
    unique_groups = np.unique(group_ids)
    for group_id in unique_groups:
        feature_blocks = []
        target_blocks = []
        for states in contexts[group_ids == group_id]:
            derivatives = (states[2:] - states[:-2]) / (
                times[2:] - times[:-2]
            )[:, None]
            feature_blocks.append(_library_numpy(states[1:-1]))
            target_blocks.append(derivatives)
        features = np.concatenate(feature_blocks, axis=0)
        targets = np.concatenate(target_blocks, axis=0)
        fitted = _stlsq(
            features,
            targets,
            ridge=ridge,
            threshold=threshold,
        )
        prediction = features @ fitted
        coefficients.append(fitted)
        residuals.append(float(np.mean((prediction - targets) ** 2)))
        active_terms.append(int(np.sum(np.abs(fitted) > 0)))
    labels_path = atomic_save_npz(
        output_dir / "source_teacher_labels.npz",
        group_ids=np.asarray(unique_groups, dtype=np.int64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        residual_mse=np.asarray(residuals, dtype=np.float64),
        active_terms=np.asarray(active_terms, dtype=np.int64),
    )
    return atomic_write_json(output_dir / "manifest.json", {
        "schema": "amortized-sindy-source-teacher-v1",
        "method": "context_only_central_difference_stlsq",
        "future_source_states_used": False,
        "generator_coefficients_used": False,
        "library_exponents": [list(value) for value in EXPONENTS],
        "ridge": ridge,
        "threshold": threshold,
        "source_train_sha256": sha256_file(source_train_path),
        "labels": {
            "path": labels_path.name,
            "sha256": sha256_file(labels_path),
        },
        "mean_residual_mse": float(np.mean(residuals)),
        "mean_active_terms": float(np.mean(active_terms)),
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit source-only SINDy teachers for amortized coefficient distillation."
    )
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(build_source_teacher_labels(
        source_train_path=args.source_train,
        output_dir=args.output_dir,
        ridge=args.ridge,
        threshold=args.threshold,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
