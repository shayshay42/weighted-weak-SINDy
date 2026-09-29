from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json
from .base import ForecastMethod
from .contracts import PRETRAINED_TRACK, contract_for


CHRONOS_CHECKPOINT_SCHEMA = "lorenz63-chronos-checkpoint-v2"


def create_chronos_checkpoint(
    train_split: dict[str, Any], config: dict[str, Any], output_dir: str | Path
) -> tuple[Path, dict[str, Any]]:
    """Create a pinned Chronos model reference; no benchmark fitting occurs."""
    model_config = dict(config["chronos"])
    checkpoint = {
        "schema": CHRONOS_CHECKPOINT_SCHEMA,
        "method": "chronos_zero_shot",
        "track": PRETRAINED_TRACK,
        "information_contract": list(contract_for("chronos_zero_shot").information),
        "model": model_config,
        "observation_dt": float(train_split["metadata"]["dt"]),
        "benchmark_fitting_performed": False,
        "normalization": "per_channel_context_mean_and_std",
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
        "pretrained_model_id": model_config["model_id"],
        "pretrained_model_revision": model_config["model_revision"],
    }


class ChronosForecastAdapter(ForecastMethod):
    """Channel-wise recursive Chronos-T5 forecast from an observed prefix."""

    method = "chronos_zero_shot"
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
        cfg = checkpoint["model"]
        self.model_id = str(cfg["model_id"])
        self.model_revision = str(cfg["model_revision"])
        self.required_context_steps = int(cfg["context_length"])
        self.prediction_length = int(cfg["prediction_length"])
        self.batch_size = int(cfg["batch_size"])
        self.dtype = str(cfg["dtype"])
        self.inference_seed = int(cfg.get("inference_seed", 2026))
        self.observation_dt = float(checkpoint["observation_dt"])
        self.device = str(device)
        self._pipeline = pipeline
        self.parameter_count = 0
        self.last_surrogate_evaluations = 0
        self.last_peak_gpu_memory_bytes = 0
        if self.required_context_steps < 2:
            raise ValueError("Chronos context length must be at least two")
        if self.prediction_length < 1 or self.batch_size < 1:
            raise ValueError("Chronos prediction length and batch size must be positive")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"unsupported Chronos dtype {self.dtype!r}")
        if pipeline is not None:
            self._record_parameter_count(pipeline)

    def _record_parameter_count(self, pipeline: Any) -> None:
        model = getattr(pipeline, "model", None)
        if model is not None and hasattr(model, "parameters"):
            self.parameter_count = int(sum(value.numel() for value in model.parameters()))

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from chronos import ChronosPipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Chronos inference dependencies are missing. Install the benchmark's "
                "'chronos' optional dependency."
            ) from exc
        requested_device = self.device
        if requested_device == "auto":
            requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Chronos was requested on CUDA, but CUDA is unavailable")
        random.seed(self.inference_seed)
        np.random.seed(self.inference_seed)
        torch.manual_seed(self.inference_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.inference_seed)
        torch_dtype = (
            torch.bfloat16
            if self.dtype == "bfloat16" and requested_device.startswith("cuda")
            else torch.float32
        )
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            device_map=requested_device,
            torch_dtype=torch_dtype,
        )
        model = getattr(self._pipeline, "model", None)
        if model is not None:
            model.eval()
            model.requires_grad_(False)
        self._record_parameter_count(self._pipeline)
        return self._pipeline

    def forecast(self, initial_states: np.ndarray, times: np.ndarray) -> np.ndarray:
        del initial_states, times
        raise ValueError("Chronos requires an observed trajectory context, not only an initial state")

    def forecast_from_context(
        self, context_states: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        import torch

        contexts = np.asarray(context_states, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if contexts.ndim != 3 or contexts.shape[-1] != 3:
            raise ValueError("Chronos context must have shape [trajectory, time, 3]")
        if contexts.shape[1] < self.required_context_steps:
            raise ValueError(
                f"Chronos needs {self.required_context_steps} context points, got {contexts.shape[1]}"
            )
        if times.ndim != 1 or times.size < 1 or abs(float(times[0])) > 1e-12:
            raise ValueError("forecast times must be one-dimensional and start at zero")
        if times.size > 1 and not np.allclose(
            np.diff(times), self.observation_dt, rtol=0.0, atol=1e-10
        ):
            raise ValueError("Chronos forecasts must use the observation sampling interval")

        contexts = contexts[:, -self.required_context_steps :]
        # Chronos is univariate. Treat the three state channels as independent
        # series, using only statistics of each supplied context window.
        mean = contexts.mean(axis=1, keepdims=True)
        scale = contexts.std(axis=1, keepdims=True)
        scale = np.where(scale > 1e-12, scale, 1.0)
        normalized = (contexts - mean) / scale
        channel_context = normalized.transpose(0, 2, 1).reshape(
            -1, self.required_context_steps
        )
        future_steps = times.size - 1
        if future_steps == 0:
            return contexts[:, -1:, :].copy()

        pipeline = self._ensure_pipeline()
        requested_device = self.device
        if requested_device == "auto":
            requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        remaining = future_steps
        generated_blocks: list[np.ndarray] = []
        model_calls = 0
        rolling = channel_context
        with torch.inference_mode():
            while remaining > 0:
                block_steps = min(self.prediction_length, remaining)
                batch_blocks: list[np.ndarray] = []
                for start in range(0, rolling.shape[0], self.batch_size):
                    stop = min(start + self.batch_size, rolling.shape[0])
                    _, forecast_mean = pipeline.predict_quantiles(
                        context=torch.as_tensor(rolling[start:stop], dtype=torch.float32),
                        prediction_length=block_steps,
                        quantile_levels=[0.5],
                    )
                    values = torch.as_tensor(forecast_mean).detach().cpu().numpy()
                    if values.shape != (stop - start, block_steps):
                        raise RuntimeError(
                            f"unexpected Chronos output shape {tuple(values.shape)}"
                        )
                    batch_blocks.append(values)
                    model_calls += 1
                block = np.concatenate(batch_blocks, axis=0).astype(np.float64)
                generated_blocks.append(block)
                rolling = np.concatenate((rolling, block), axis=1)[
                    :, -self.required_context_steps :
                ]
                remaining -= block_steps

        normalized_future = np.concatenate(generated_blocks, axis=1)[:, :future_steps]
        normalized_future = normalized_future.reshape(
            contexts.shape[0], contexts.shape[2], future_steps
        ).transpose(0, 2, 1)
        future = normalized_future * scale + mean
        self.last_surrogate_evaluations = model_calls
        if requested_device.startswith("cuda"):
            self.last_peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
        return np.concatenate((contexts[:, -1:, :], future), axis=1)

    def parameters(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "benchmark_fitting_performed": False,
            "normalization": "per_channel_context_mean_and_std",
            "channels_conditioned_independently": True,
            "inference_seed": self.inference_seed,
        }


def load_chronos(path: str | Path, device: str = "cpu") -> ChronosForecastAdapter:
    checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
    if checkpoint.get("schema") != CHRONOS_CHECKPOINT_SCHEMA:
        raise ValueError(f"not a Chronos benchmark checkpoint: {path}")
    if checkpoint.get("method") != "chronos_zero_shot":
        raise ValueError(f"unexpected Chronos method in checkpoint: {checkpoint.get('method')}")
    return ChronosForecastAdapter(checkpoint, device=device)
