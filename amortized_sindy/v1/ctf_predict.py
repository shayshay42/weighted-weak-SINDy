from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .ctf_adapter import CTFZeroShotSINDy
from .io import atomic_save_npy, atomic_write_json


CTF_ADVANCEMENT_GATE_SCHEMA = "amortized-sindy-ctf-advancement-gate-v1"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "install the 'ctf' optional dependency to read CTF configs"
        ) from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CTF configuration must contain a mapping")
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_advancement_gate(path: Path, *, checkpoint: Path) -> dict[str, str]:
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CTF advancement gate: {path}") from error
    if not isinstance(gate, dict) or gate.get("schema") != CTF_ADVANCEMENT_GATE_SCHEMA:
        raise ValueError("unsupported CTF advancement-gate schema")
    expected_checkpoint = gate.get("selected_checkpoint_sha256")
    if not isinstance(expected_checkpoint, str) or len(expected_checkpoint) != 64:
        raise ValueError("advancement gate lacks a selected checkpoint SHA-256")
    actual_checkpoint = _sha256_file(checkpoint)
    if actual_checkpoint != expected_checkpoint:
        raise ValueError("CTF checkpoint does not match the advancement gate")

    heldout = gate.get("heldout_gate")
    if not isinstance(heldout, dict):
        raise ValueError("advancement gate lacks its held-out decision")
    threshold = heldout.get("relative_mse_vs_constant_field_threshold")
    observed = heldout.get("relative_mse_vs_constant_field")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(float(observed))
    ):
        raise ValueError("advancement gate has invalid held-out MSE values")
    recomputed_pass = (
        heldout.get("inference_complete") is True
        and heldout.get("all_predictions_finite") is True
        and float(observed) < float(threshold)
    )
    if heldout.get("passed") is not recomputed_pass:
        raise ValueError("advancement-gate decision is internally inconsistent")
    if gate.get("ctf4science_authorized") is not recomputed_pass:
        raise ValueError("CTF authorization is inconsistent with the held-out gate")
    if not recomputed_pass:
        raise PermissionError("CTF4Science advancement was denied by the held-out gate")
    return {
        "schema": CTF_ADVANCEMENT_GATE_SCHEMA,
        "manifest_sha256": _sha256_file(path),
        "selected_checkpoint_sha256": actual_checkpoint,
    }


def predict_ctf_pairs(
    *,
    config_path: str | Path,
    advancement_gate_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> Path:
    """Generate CTF predictions without importing the truth-aware evaluator."""
    config_path = Path(config_path)
    config = _load_yaml(config_path)
    if set(config) != {"dataset", "model"}:
        raise ValueError("CTF zero-shot config permits only dataset and model sections")
    dataset = config["dataset"]
    model_config = config["model"]
    if dataset.get("name") != "ODE_Lorenz":
        raise ValueError("version 1 CTF zero-shot evaluation is scoped to ODE_Lorenz")
    if model_config.get("zero_shot") is not True:
        raise ValueError("amortized SINDy CTF inference requires zero_shot: true")
    permitted_model_keys = {
        "name",
        "zero_shot",
        "checkpoint",
        "context_length",
        "device",
    }
    unexpected = set(model_config) - permitted_model_keys
    if unexpected:
        raise ValueError(
            f"unexpected target-inference config fields: {sorted(unexpected)}"
        )
    checkpoint = Path(model_config["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (config_path.parent / checkpoint).resolve()
    gate_binding = _load_advancement_gate(
        Path(advancement_gate_path), checkpoint=checkpoint
    )

    # The CTF package and target data remain untouched unless an independently
    # recorded held-out decision authorizes advancement for this checkpoint.
    try:
        from ctf4science.data_module import (
            get_prediction_timesteps,
            load_dataset,
            parse_pair_ids,
        )
    except ImportError as error:
        raise RuntimeError("install the pinned CTF4Science framework first") from error

    adapter = CTFZeroShotSINDy(
        checkpoint,
        context_length=int(model_config.get("context_length", -1)),
        device=str(model_config.get("device", "cpu")),
    )
    pair_ids = [int(value) for value in parse_pair_ids(dataset)]
    if not pair_ids:
        raise ValueError("CTF config selected no pairs")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"prediction directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_records = []
    for pair_id in pair_ids:
        train_data, init_data = load_dataset(dataset["name"], pair_id)
        prediction_timesteps = get_prediction_timesteps(dataset["name"], pair_id)
        predictions = adapter.predict(
            pair_id=pair_id,
            train_data=train_data,
            init_data=init_data,
            prediction_timesteps=prediction_timesteps,
        )
        pair_dir = output_dir / f"pair_{pair_id}"
        prediction_path = atomic_save_npy(pair_dir / "predictions.npy", predictions)
        inference = dict(adapter.last_inference or {})
        inference.update(
            {
                "dataset": dataset["name"],
                "prediction_path": "predictions.npy",
                "config_hash": _sha256_json(config),
                "ctf_advancement_gate_manifest_sha256": gate_binding["manifest_sha256"],
            }
        )
        inference_path = atomic_write_json(
            pair_dir / "inference_manifest.json", inference
        )
        pair_records.append(
            {
                "pair_id": pair_id,
                "prediction_path": prediction_path.relative_to(output_dir).as_posix(),
                "inference_manifest_path": inference_path.relative_to(
                    output_dir
                ).as_posix(),
            }
        )
    return atomic_write_json(
        output_dir / "prediction_batch_manifest.json",
        {
            "schema": "ctf-amortized-sindy-prediction-batch-v1",
            "dataset": dataset["name"],
            "zero_shot": True,
            "adaptation_label": "target-time-zero-update",
            "strict_dataset_zero_shot": adapter.strict_dataset_zero_shot,
            "pretraining_exposure": adapter.pretraining_exposure,
            "config_hash": _sha256_json(config),
            "ctf_advancement_gate": gate_binding,
            "pairs": pair_records,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create frozen amortized-SINDy CTF predictions in a truth-blind process."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--advancement-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(
        predict_ctf_pairs(
            config_path=args.config,
            advancement_gate_path=args.advancement_gate,
            output_dir=args.output_dir,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
