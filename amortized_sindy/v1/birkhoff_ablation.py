from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint, sha256_file
from .io import atomic_write_json
from .multifamily_train import train_multifamily_checkpoint


ABLATION_SCHEMA = "amortized-sindy-birkhoff-weak-ablation-v1"
VARIANTS = {
    "ordered_foundation": "no_birkhoff_no_weak",
    "birkhoff": "no_weak",
    "weak_tokens": "no_birkhoff",
    "birkhoff_weak_tokens": "full",
}


def run_birkhoff_ablation(
    *,
    source_train_path: str | Path,
    source_validation_path: str | Path,
    dataset_manifest_path: str | Path,
    source_embeddings_path: str | Path,
    source_validation_embeddings_path: str | Path,
    output_dir: str | Path,
    source_teacher_labels_path: str | Path | None = None,
    source_teacher_manifest_path: str | Path | None = None,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    eval_interval: int = 5,
    warmup_epochs: int | None = None,
    teacher_weight: float = 0.0,
    seed: int = 91,
    device: str = "cpu",
    resume: bool = False,
    force: bool = False,
) -> Path:
    """Train a source-selected, parameter-matched four-way feature ablation."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not (force or resume):
        raise FileExistsError(f"ablation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for variant, ablation in VARIANTS.items():
        variant_dir = output_dir / variant
        manifest_path = variant_dir / "manifest.json"
        if not (resume and manifest_path.is_file()):
            manifest_path = train_multifamily_checkpoint(
                source_train_path=source_train_path,
                source_validation_path=source_validation_path,
                dataset_manifest_path=dataset_manifest_path,
                output_dir=variant_dir,
                source_teacher_labels_path=source_teacher_labels_path,
                source_teacher_manifest_path=source_teacher_manifest_path,
                source_embeddings_path=source_embeddings_path,
                source_validation_embeddings_path=source_validation_embeddings_path,
                epochs=epochs,
                encoder_type="external_birkhoff_weak",
                hybrid_ablation=ablation,
                learning_rate=learning_rate,
                eval_interval=eval_interval,
                warmup_epochs=warmup_epochs,
                teacher_weight=teacher_weight,
                seed=seed,
                device=device,
                force=force,
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "amortized-sindy-multifamily-training-v1"
            or manifest.get("status") != "complete"
        ):
            raise ValueError(f"incomplete child manifest for {variant}")
        checkpoint = variant_dir / manifest.get("checkpoint", {}).get("path", "")
        if manifest.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint):
            raise ValueError(f"checkpoint hash mismatch for {variant}")
        model, training_manifest = load_checkpoint(checkpoint)
        if (
            model.config.encoder_type != "external_birkhoff_weak"
            or model.config.hybrid_ablation != ablation
            or training_manifest.get("training_seed") != seed
            or training_manifest.get("training_epochs_budget") != epochs
            or training_manifest.get("source_train_sha256")
            != sha256_file(source_train_path)
            or training_manifest.get("source_validation_sha256")
            != sha256_file(source_validation_path)
        ):
            raise ValueError(f"checkpoint protocol mismatch for {variant}")
        results.append({
            "variant": variant,
            "hybrid_ablation": ablation,
            "best_epoch": manifest["best_epoch"],
            "source_validation_rollout_mse": manifest[
                "best_validation_rollout_mse"
            ],
            "constant_field_mse": manifest["validation_constant_field_mse"],
            "stopped_epoch": manifest["stopped_epoch"],
            "stop_reason": manifest["stop_reason"],
            "training_device": training_manifest.get("training_device", "unrecorded"),
            "manifest": {
                "path": str(manifest_path.relative_to(output_dir)),
                "sha256": sha256_file(manifest_path),
            },
        })
    winner = min(results, key=lambda row: row["source_validation_rollout_mse"])
    return atomic_write_json(output_dir / "ablation_manifest.json", {
        "schema": ABLATION_SCHEMA,
        "status": "complete",
        "selection_scope": "source_validation_only",
        "target_evaluation_data_used": False,
        "target_optimizer_steps": 0,
        "target_sparse_regression_solves": 0,
        "matched_training_budget": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "eval_interval": eval_interval,
            "warmup_epochs": warmup_epochs,
            "teacher_weight": teacher_weight,
            "seed": seed,
            "device_for_missing_variants": device,
        },
        "inputs": {
            "source_train_sha256": sha256_file(source_train_path),
            "source_validation_sha256": sha256_file(source_validation_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "source_embeddings_sha256": sha256_file(source_embeddings_path),
            "source_validation_embeddings_sha256": sha256_file(
                source_validation_embeddings_path
            ),
        },
        "results": results,
        "selected_variant": winner["variant"],
        "selected_source_validation_rollout_mse": winner[
            "source_validation_rollout_mse"
        ],
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the matched Birkhoff/weak-token conditioner ablation."
    )
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-validation", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--source-embeddings", required=True)
    parser.add_argument("--source-validation-embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-teacher-labels")
    parser.add_argument("--source-teacher-manifest")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(run_birkhoff_ablation(
        source_train_path=args.source_train,
        source_validation_path=args.source_validation,
        dataset_manifest_path=args.dataset_manifest,
        source_embeddings_path=args.source_embeddings,
        source_validation_embeddings_path=args.source_validation_embeddings,
        output_dir=args.output_dir,
        source_teacher_labels_path=args.source_teacher_labels,
        source_teacher_manifest_path=args.source_teacher_manifest,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        warmup_epochs=args.warmup_epochs,
        teacher_weight=args.teacher_weight,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
