from __future__ import annotations

import json
import sys
import types

import numpy as np

from amortized_sindy.v1.ctf_evaluate import evaluate_ctf_predictions
from amortized_sindy.v1.io import array_sha256


def _prediction_batch(tmp_path, predictions: np.ndarray):
    prediction_path = tmp_path / "prediction.npy"
    np.save(prediction_path, predictions, allow_pickle=False)
    inference_path = tmp_path / "inference.json"
    inference_path.write_text(json.dumps({
        "pair_id": 2,
        "zero_shot": True,
        "target_optimizer_steps": 0,
        "target_sparse_regression_solves": 0,
        "prediction_sha256": array_sha256(predictions),
    }), encoding="utf-8")
    batch_path = tmp_path / "prediction_batch.json"
    batch_path.write_text(json.dumps({
        "schema": "ctf-amortized-sindy-prediction-batch-v1",
        "dataset": "ODE_Lorenz",
        "pairs": [{
            "pair_id": 2,
            "prediction_path": str(prediction_path),
            "inference_manifest_path": str(inference_path),
        }],
    }), encoding="utf-8")
    return batch_path


def test_nonfinite_prediction_is_an_explicit_failure_without_fallback(
    tmp_path, monkeypatch
) -> None:
    eval_module = types.ModuleType("ctf4science.eval_module")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("the evaluator must not receive a failed prediction")

    eval_module.evaluate = fail_if_called
    monkeypatch.setitem(sys.modules, "ctf4science.eval_module", eval_module)
    predictions = np.array([[0.0, np.nan, 1.0]], dtype=np.float64)
    output = evaluate_ctf_predictions(
        prediction_batch_manifest=_prediction_batch(tmp_path, predictions),
        output_dir=tmp_path / "evaluation",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["pairs"][0]["metrics"] == {
        "method_failure": True,
        "failure_reason": "nonfinite_prediction",
    }
    assert result["aggregate"]["failed_pair_ids"] == [2]
    assert result["aggregate"]["official_clipped_composite"] is None
    assert result["aggregate"]["complete"] is False


def test_official_composite_requires_all_twelve_metrics_and_clips_components(
    tmp_path, monkeypatch
) -> None:
    eval_module = types.ModuleType("ctf4science.eval_module")
    values = [-150.0, 150.0, *range(10)]
    eval_module.evaluate = lambda *_args, **_kwargs: {
        f"E{index + 1}": value for index, value in enumerate(values)
    }
    monkeypatch.setitem(sys.modules, "ctf4science.eval_module", eval_module)
    predictions = np.zeros((2, 3), dtype=np.float64)
    output = evaluate_ctf_predictions(
        prediction_batch_manifest=_prediction_batch(tmp_path, predictions),
        output_dir=tmp_path / "evaluation",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    expected = float(np.mean(np.clip(values, -100.0, 100.0)))
    assert result["aggregate"]["completed_metric_count"] == 12
    assert result["aggregate"]["official_clipped_composite"] == expected
    assert result["aggregate"]["complete"] is True
