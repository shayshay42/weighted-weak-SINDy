from __future__ import annotations

import json
from pathlib import Path

import pytest

from amortized_sindy.v1.io import atomic_write_json
from amortized_sindy.v1.weak_invariant_ablation import _validate_history


TRAINING = {
    "epochs": 10,
    "eval_interval": 4,
    "warmup_epochs": 3,
}

VARIANT = {
    "context_residual_weight": 0.01,
    "weak_form_weight": 0.02,
    "birkhoff_mmd_weight": 0.05,
}


def _validation_record(epoch: int, mse: float) -> dict[str, object]:
    warmup = epoch <= TRAINING["warmup_epochs"]
    return {
        "epoch": epoch,
        "validation_status": "finite",
        "validation_rollout_mse": mse,
        "context_residual_weight": VARIANT["context_residual_weight"],
        "weak_form_weight": 0.1 if warmup else VARIANT["weak_form_weight"],
        "birkhoff_mmd_weight": 0.0 if warmup else VARIANT["birkhoff_mmd_weight"],
    }


def _nonfinite_validation_record(epoch: int) -> dict[str, object]:
    record = _validation_record(epoch, 0.0)
    record["validation_status"] = "nonfinite_rollout"
    record["validation_rollout_mse"] = None
    return record


def _stop_record(epoch: int) -> dict[str, object]:
    return {
        "epoch": epoch,
        "status": "stopped_before_optimizer_step",
        "reason": "nonfinite_source_loss",
    }


def _write_history(tmp_path: Path, history: list[dict[str, object]]) -> Path:
    path = tmp_path / "training_history.json"
    path.write_text(json.dumps(history), encoding="utf-8")
    return path


def _exhausted_manifest() -> dict[str, object]:
    return {
        "best_epoch": 8,
        "best_validation_rollout_mse": 0.2,
        "stopped_epoch": 10,
        "stop_reason": "training_budget_exhausted",
    }


def _nonfinite_manifest() -> dict[str, object]:
    return {
        "best_epoch": 8,
        "best_validation_rollout_mse": 0.2,
        "stopped_epoch": 9,
        "stop_reason": "nonfinite_source_loss",
    }


def _exhausted_history() -> list[dict[str, object]]:
    return [
        _validation_record(1, 0.4),
        _validation_record(4, 0.3),
        _validation_record(8, 0.2),
        _validation_record(10, 0.25),
    ]


def _nonfinite_history() -> list[dict[str, object]]:
    return [
        _validation_record(1, 0.4),
        _validation_record(4, 0.3),
        _validation_record(8, 0.2),
        _stop_record(9),
    ]


def _validate(path: Path, child_manifest: dict[str, object]) -> None:
    _validate_history(
        path,
        child_manifest=child_manifest,
        variant=VARIANT,
        training=TRAINING,
    )


def test_atomic_json_rejects_nonstandard_numeric_constants(tmp_path: Path) -> None:
    output = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        atomic_write_json(output, {"invalid": float("inf")})

    assert not output.exists()


def test_validate_history_accepts_complete_exhausted_schedule(tmp_path: Path) -> None:
    _validate(_write_history(tmp_path, _exhausted_history()), _exhausted_manifest())


def test_validate_history_accepts_complete_nonfinite_schedule(tmp_path: Path) -> None:
    _validate(_write_history(tmp_path, _nonfinite_history()), _nonfinite_manifest())


def test_validate_history_accepts_nonfinite_evaluation_that_later_recovers(
    tmp_path: Path,
) -> None:
    history = _exhausted_history()
    history[1] = _nonfinite_validation_record(4)

    _validate(_write_history(tmp_path, history), _exhausted_manifest())


def test_validate_history_rejects_nonfinite_evaluation_as_best_epoch(
    tmp_path: Path,
) -> None:
    history = _exhausted_history()
    history[1] = _nonfinite_validation_record(4)
    manifest = _exhausted_manifest()
    manifest["best_epoch"] = 4

    with pytest.raises(ValueError):
        _validate(_write_history(tmp_path, history), manifest)


def test_validate_history_rejects_legacy_infinite_validation_metric(
    tmp_path: Path,
) -> None:
    history = _exhausted_history()
    history[1]["validation_rollout_mse"] = float("inf")

    with pytest.raises(ValueError):
        _validate(_write_history(tmp_path, history), _exhausted_manifest())


def test_validate_history_rejects_missing_scheduled_validation(tmp_path: Path) -> None:
    history = _exhausted_history()
    del history[2]

    with pytest.raises(ValueError, match="scheduled validation records are incomplete"):
        _validate(_write_history(tmp_path, history), _exhausted_manifest())


def test_validate_history_rejects_replaced_scheduled_validation(tmp_path: Path) -> None:
    history = _exhausted_history()
    history[1] = _validation_record(5, 0.3)

    with pytest.raises(ValueError, match="scheduled validation records are incomplete"):
        _validate(_write_history(tmp_path, history), _exhausted_manifest())


def test_validate_history_rejects_stop_record_after_exhausted_run(tmp_path: Path) -> None:
    history = _exhausted_history()
    history.insert(-1, _stop_record(9))

    with pytest.raises(ValueError, match="budget-exhausted history contains a stop record"):
        _validate(_write_history(tmp_path, history), _exhausted_manifest())


def test_validate_history_rejects_extra_nonfinite_stop_record(tmp_path: Path) -> None:
    history = _nonfinite_history()
    history.insert(-2, _stop_record(7))

    with pytest.raises(
        ValueError,
        match="non-finite training must have exactly one final stop record",
    ):
        _validate(_write_history(tmp_path, history), _nonfinite_manifest())
