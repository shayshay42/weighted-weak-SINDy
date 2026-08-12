from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import (
    begin_manifest, fail_manifest, finish_manifest, sha256_file, sha256_json,
    validate_artifact_hashes,
)
from .config import config_for_method, load_config, save_config
from .contracts import METHODS, PARAMETRIC_TRACK, PINN_TRACK, contract_for
from .data import load_split
from .registry import fit_method


def _method_config(config: dict[str, Any], method: str) -> dict[str, Any]:
    """Remove hidden physics before handing configuration to a trainer."""
    common = {
        "schema": config["schema"], "evaluation": config["evaluation"],
        "normalization": config["normalization"],
    }
    if method.startswith("node_"):
        common["node"] = config["node"]
    elif method.startswith("sindy_"):
        common["sindy"] = config["sindy"]
    elif method.startswith("lorenz_ad"):
        common["parametric"] = config["parametric"]
    elif method.startswith("pinn_") or method == "solver_oracle":
        common["pinn"] = config["pinn"]
        common["system"] = config["system"]
    elif method == "panda_zero_shot":
        common["panda"] = config["panda"]
    return common


def _checkpoint_from_manifest(manifest: dict[str, Any], output_dir: Path) -> Path | None:
    record = manifest.get("artifacts", {}).get("checkpoint")
    if not record:
        return None
    path = Path(record["path"])
    return path if path.is_absolute() else output_dir / path


def run_training(
    *,
    config_path: str | Path | None,
    train_path: str | Path,
    validation_path: str | Path,
    method: str,
    data_seed: int,
    model_seed: int,
    output_dir: str | Path,
    device: str = "auto",
    trajectory_count: int | None = None,
    force: bool = False,
) -> Path:
    contract = contract_for(method)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    config = load_config(config_path)
    # PINNs can read only initial conditions; other trainers receive observed states.
    train_split = load_split(train_path, "train")
    validation_split = load_split(validation_path, "validation")
    config = config_for_method(
        config, float(train_split["metadata"]["noise_level"]), method
    )
    trainer_config = _method_config(config, method)
    if contract.track == PINN_TRACK:
        train_split = {
            "initial_states": train_split["initial_states"],
            "metadata": train_split["metadata"],
        }
        validation_split = {"metadata": validation_split["metadata"]}
    elif contract.kind == "pretrained":
        train_split = {"metadata": train_split["metadata"]}
        validation_split = {"metadata": validation_split["metadata"]}
    run_id = f"{method}__d{data_seed}__m{model_seed}__n{train_split['metadata']['noise_level']:g}"
    if trajectory_count is not None:
        run_id += f"__k{trajectory_count}"
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = _checkpoint_from_manifest(existing, output_dir)
        expected_inputs = {
            "train": sha256_file(train_path), "validation": sha256_file(validation_path),
        }
        if (
            existing.get("status") == "complete"
            and existing.get("run_id") == run_id
            and existing.get("config_hash") == sha256_json(trainer_config)
            and existing.get("input_hashes") == expected_inputs
            and checkpoint is not None
            and validate_artifact_hashes(manifest_path)
        ):
            return checkpoint
    manifest = begin_manifest(
        output_dir, run_id=run_id, method=method, track=contract.track,
        config=trainer_config, inputs={"train": train_path, "validation": validation_path},
        seeds={"data": data_seed, "model": model_seed},
        information_contract=contract.information,
    )
    try:
        save_config(trainer_config, output_dir / "config.json")
        checkpoint, training = fit_method(
            method, train_split, validation_split, trainer_config, model_seed,
            output_dir, device, trajectory_count=trajectory_count,
        )
        artifacts = {"checkpoint": checkpoint, "config": output_dir / "config.json"}
        history_path = output_dir / "history.csv"
        if history_path.exists():
            artifacts["history"] = history_path
        finish_manifest(output_dir, manifest, artifacts=artifacts, training=training)
        return checkpoint
    except BaseException as error:
        fail_manifest(output_dir, manifest, error)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one Lorenz63 benchmark v2 run.")
    parser.add_argument("--config", default="configs/v2/lorenz63_paper.json")
    parser.add_argument("--train", required=True, help="Explicit training split; no test argument is accepted.")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--data-seed", required=True, type=int)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trajectory-count", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    checkpoint = run_training(
        config_path=args.config, train_path=args.train, validation_path=args.validation,
        method=args.method, data_seed=args.data_seed, model_seed=args.model_seed,
        output_dir=args.output_dir, device=args.device,
        trajectory_count=args.trajectory_count, force=args.force,
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
