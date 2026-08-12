from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json
from .base import ForecastMethod, Normalization
from .contracts import PRETRAINED_TRACK, contract_for


PANDA_CHECKPOINT_SCHEMA = "lorenz63-panda-checkpoint-v2"


class _PredictionPipeline:
    """Minimal official-Panda rollout wrapper without its dataset dependencies."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @property
    def device(self) -> Any:
        return self.model.device

    def predict(
        self,
        context: Any,
        prediction_length: int,
        *,
        limit_prediction_length: bool,
        sliding_context: bool,
        verbose: bool,
    ) -> Any:
        del limit_prediction_length, verbose
        import torch

        context = context.to(self.device)
        remaining = int(prediction_length)
        predictions = []
        while remaining > 0:
            prediction = self.model.generate(context).sequences
            predictions.append(prediction)
            remaining -= int(prediction.shape[2])
            if remaining <= 0:
                break
            context = torch.cat([context, prediction.median(dim=1).values], dim=1)
            if sliding_context:
                context = context[:, -int(self.model.config.context_length) :]
        return torch.cat(predictions, dim=2)[:, :, :prediction_length]


def create_panda_checkpoint(
    train_split: dict[str, Any], config: dict[str, Any], output_dir: str | Path
) -> tuple[Path, dict[str, Any]]:
    """Create a pinned model reference; no benchmark fitting is performed."""
    panda_config = dict(config["panda"])
    normalization = Normalization.from_split(train_split)
    metadata = train_split["metadata"]
    checkpoint = {
        "schema": PANDA_CHECKPOINT_SCHEMA,
        "method": "panda_zero_shot",
        "track": PRETRAINED_TRACK,
        "information_contract": list(contract_for("panda_zero_shot").information),
        "model": panda_config,
        "normalization": {
            "state_mean": normalization.state_mean.tolist(),
            "state_std": normalization.state_std.tolist(),
            "time_scale": normalization.time_scale,
        },
        "observation_dt": float(metadata["dt"]),
        "benchmark_fitting_performed": False,
    }
    path = Path(output_dir) / "model.json"
    atomic_write_json(path, checkpoint)
    return path, {
        "optimizer_updates": 0,
        "wall_time_seconds": 0.0,
        "parameter_count": 0,
        "peak_gpu_memory_bytes": 0,
        "vector_field_evaluations": 0,
        "benchmark_fitting_performed": False,
        "pretrained_model_id": panda_config["model_id"],
        "pretrained_model_revision": panda_config["model_revision"],
    }


class PandaForecastAdapter(ForecastMethod):
    method = "panda_zero_shot"
    track = PRETRAINED_TRACK
    autonomous = False
    supports_distribution_metrics = True

    def __init__(
        self,
        checkpoint: dict[str, Any],
        device: str = "cpu",
        *,
        pipeline: Any | None = None,
    ) -> None:
        model_config = checkpoint["model"]
        normalization = checkpoint["normalization"]
        self.normalization = Normalization(
            state_mean=np.asarray(normalization["state_mean"], dtype=np.float64),
            state_std=np.asarray(normalization["state_std"], dtype=np.float64),
            time_scale=float(normalization["time_scale"]),
        )
        self.model_id = str(model_config["model_id"])
        self.model_revision = str(model_config["model_revision"])
        self.source_repository = str(model_config["source_repository"])
        self.source_revision = str(model_config["source_revision"])
        self.required_context_steps = int(model_config["context_length"])
        self.prediction_length = int(model_config["prediction_length"])
        self.batch_size = int(model_config["batch_size"])
        self.dtype = str(model_config["dtype"])
        self.sliding_context = bool(model_config["sliding_context"])
        self.observation_dt = float(checkpoint["observation_dt"])
        self.device = str(device)
        self._pipeline = pipeline
        self.parameter_count = 0
        self.last_surrogate_evaluations = 0
        self.last_peak_gpu_memory_bytes = 0
        if self.required_context_steps < 1 or self.prediction_length < 1 or self.batch_size < 1:
            raise ValueError("Panda context length, prediction length, and batch size must be positive")
        if pipeline is not None:
            self._validate_pipeline(pipeline)

    def _validate_pipeline(self, pipeline: Any) -> None:
        model = pipeline.model
        context_length = int(model.config.context_length)
        prediction_length = int(model.config.prediction_length)
        if context_length != self.required_context_steps:
            raise ValueError(
                f"Panda checkpoint context length is {context_length}, expected {self.required_context_steps}"
            )
        if prediction_length != self.prediction_length:
            raise ValueError(
                f"Panda checkpoint prediction length is {prediction_length}, expected {self.prediction_length}"
            )
        self.parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from panda.patchtst.patchtst import PatchTSTForPrediction
        except ImportError as exc:
            raise RuntimeError(
                "Panda inference dependencies are missing. Install the pinned official "
                "Panda source with the benchmark's 'panda' optional dependency."
            ) from exc
        if self.dtype != "float32":
            raise ValueError(f"unsupported Panda inference dtype {self.dtype!r}")
        requested_device = self.device
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Panda was requested on CUDA, but CUDA is unavailable")
        model = PatchTSTForPrediction.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            torch_dtype=torch.float32,
        )
        model.to(requested_device)
        model.eval()
        self._pipeline = _PredictionPipeline(model)
        self._validate_pipeline(self._pipeline)
        return self._pipeline

    def forecast(self, initial_states: np.ndarray, times: np.ndarray) -> np.ndarray:
        del initial_states, times
        raise ValueError("Panda requires an observed trajectory context, not only an initial state")

    def forecast_from_context(
        self, context_states: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        import torch

        context_states = np.asarray(context_states, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if context_states.ndim != 3 or context_states.shape[-1] != 3:
            raise ValueError("Panda context must have shape [trajectory, time, 3]")
        if context_states.shape[1] < self.required_context_steps:
            raise ValueError(
                f"Panda needs {self.required_context_steps} context points, got {context_states.shape[1]}"
            )
        if times.ndim != 1 or times.size < 1 or abs(float(times[0])) > 1e-12:
            raise ValueError("forecast times must be one-dimensional and start at zero")
        if times.size > 1 and not np.allclose(
            np.diff(times), self.observation_dt, rtol=0.0, atol=1e-10
        ):
            raise ValueError("Panda forecasts must use the observation sampling interval")

        pipeline = self._ensure_pipeline()
        context = context_states[:, -self.required_context_steps :]
        normalized_context = self.normalization.normalize_state(context).astype(np.float32)
        future_steps = times.size - 1
        if future_steps == 0:
            return context[:, -1:, :].copy()

        if torch.cuda.is_available() and str(pipeline.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(pipeline.device)
        batches = []
        model_calls = 0
        with torch.inference_mode():
            for start in range(0, normalized_context.shape[0], self.batch_size):
                stop = min(start + self.batch_size, normalized_context.shape[0])
                tensor = torch.as_tensor(normalized_context[start:stop], dtype=torch.float32)
                generated = pipeline.predict(
                    tensor,
                    future_steps,
                    limit_prediction_length=False,
                    sliding_context=self.sliding_context,
                    verbose=False,
                )
                if generated.ndim != 4 or generated.shape[0] != stop - start:
                    raise RuntimeError(f"unexpected Panda output shape {tuple(generated.shape)}")
                reduced = generated.median(dim=1).values[:, :future_steps]
                batches.append(reduced.detach().cpu().numpy())
                model_calls += math.ceil(future_steps / self.prediction_length)
        normalized_future = np.concatenate(batches, axis=0).astype(np.float64)
        future = self.normalization.denormalize_state(normalized_future)
        prediction = np.concatenate([context[:, -1:, :], future], axis=1)
        self.last_surrogate_evaluations = model_calls
        if torch.cuda.is_available() and str(pipeline.device).startswith("cuda"):
            self.last_peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(pipeline.device))
        return prediction

    def parameters(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "benchmark_fitting_performed": False,
        }


def load_panda(path: str | Path, device: str = "cpu") -> PandaForecastAdapter:
    checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
    if checkpoint.get("schema") != PANDA_CHECKPOINT_SCHEMA:
        raise ValueError(f"not a Panda benchmark checkpoint: {path}")
    if checkpoint.get("method") != "panda_zero_shot":
        raise ValueError(f"unexpected Panda method in checkpoint: {checkpoint.get('method')}")
    return PandaForecastAdapter(checkpoint, device=device)
