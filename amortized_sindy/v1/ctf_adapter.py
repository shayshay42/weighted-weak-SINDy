from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .checkpoint import load_checkpoint, sha256_file
from .foundation_encoder import (
    FoundationEncoderSpec,
    load_foundation_encoder,
    validate_foundation_encoder_binding,
)
from .io import array_sha256
from .model import AmortizedSINDy, InferredDynamics


def _one_training_trajectory(train_data: Any) -> np.ndarray:
    if isinstance(train_data, (list, tuple)):
        if len(train_data) != 1:
            raise ValueError(
                "non-parametric zero-shot inference requires one contiguous context trajectory"
            )
        train_data = train_data[0]
    value = np.asarray(train_data, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("CTF context must have shape [time,state]")
    return value


class CTFZeroShotSINDy:
    """CTF4Science-compatible frozen inference wrapper.

    The wrapper intentionally exposes no target-fitting method. Loading a
    checkpoint freezes every parameter; ``predict`` consumes only the observed
    context and public forecast grid.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        context_length: int = -1,
        device: str = "cpu",
    ):
        if context_length == 0 or context_length < -1:
            raise ValueError("context_length must be -1 or a positive integer")
        self.checkpoint_path = Path(checkpoint_path)
        self.context_length = context_length
        self.device = torch.device(device)
        self.model, self.training_manifest = load_checkpoint(
            self.checkpoint_path, device=self.device
        )
        if "CTF4Science/ODE_Lorenz" not in self.training_manifest[
            "excluded_evaluation_dataset_ids"
        ]:
            raise ValueError(
                "CTF checkpoint manifest must explicitly exclude CTF4Science/ODE_Lorenz"
            )
        self.foundation_encoder = None
        self.foundation_spec: FoundationEncoderSpec | None = None
        self.foundation_encoder_provenance = self.training_manifest.get(
            "foundation_encoder_provenance"
        )
        if self.model.config.encoder_type in {
            "external", "external_mlp", "external_birkhoff_weak"
        }:
            if self.foundation_encoder_provenance is None:
                raise ValueError(
                    "external CTF checkpoint requires bound foundation encoder provenance"
                )
            self.foundation_spec = validate_foundation_encoder_binding(
                self.foundation_encoder_provenance
            )
            if self.foundation_spec.target_dataset_id != "CTF4Science/ODE_Lorenz":
                raise ValueError("foundation encoder binding targets a different dataset")
            if self.foundation_spec.state_dimension != self.model.config.state_dimension:
                raise ValueError(
                    "foundation encoder state dimension differs from the SINDy checkpoint"
                )
            if (
                self.context_length > 0
                and self.context_length > self.foundation_spec.context_length
            ):
                raise ValueError(
                    "CTF context length exceeds the bound foundation encoder context"
                )
            self.foundation_encoder = load_foundation_encoder(
                self.foundation_spec,
                device=self.device,
            )
            if self.foundation_encoder.output_dimension != self.model.config.hidden_size:
                raise ValueError(
                    "foundation encoder output width differs from the SINDy checkpoint"
                )
        elif self.foundation_encoder_provenance is not None:
            raise ValueError(
                "foundation encoder provenance requires an external SINDy checkpoint"
            )
        self.last_inference: dict[str, Any] | None = None

    @property
    def strict_dataset_zero_shot(self) -> bool:
        if self.foundation_spec is None:
            return True
        return self.foundation_spec.strict_dataset_zero_shot

    @property
    def pretraining_exposure(self) -> str:
        if self.foundation_spec is None:
            return "verified_excluded"
        return self.foundation_spec.target_pretraining_exposure

    @property
    def frozen(self) -> bool:
        return not self.model.training and not any(
            parameter.requires_grad for parameter in self.model.parameters()
        )

    def _select_context(
        self,
        *,
        pair_id: int,
        train_data: Any,
        init_data: np.ndarray | None,
    ) -> np.ndarray:
        if pair_id in (8, 9):
            if init_data is None:
                raise ValueError("CTF pairs 8 and 9 require warm-start context")
            context = np.asarray(init_data, dtype=np.float64)
            if context.ndim != 2:
                raise ValueError("CTF warm-start context must have shape [time,state]")
        else:
            context = _one_training_trajectory(train_data)
        context_limit = self.context_length
        if context_limit == -1 and self.foundation_spec is not None:
            context_limit = self.foundation_spec.context_length
        if context_limit > 0:
            if pair_id in (2, 4):
                context = context[:context_limit]
            else:
                context = context[-context_limit:]
        if context.shape[0] < self.model.config.min_context_steps:
            raise ValueError("CTF context is shorter than the checkpoint contract")
        if context.shape[1] != self.model.config.state_dimension:
            raise ValueError("CTF state dimension does not match the checkpoint")
        if not np.isfinite(context).all():
            raise ValueError("CTF context contains non-finite values")
        return context

    def predict(
        self,
        *,
        pair_id: int,
        train_data: Any,
        init_data: np.ndarray | None,
        prediction_timesteps: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        if not self.frozen:
            raise RuntimeError("target inference requires a frozen checkpoint")
        context = self._select_context(
            pair_id=pair_id, train_data=train_data, init_data=init_data
        )
        prediction_times = np.asarray(prediction_timesteps, dtype=np.float64)
        if prediction_times.ndim != 1 or prediction_times.size < 2:
            raise ValueError("prediction_timesteps must be a vector with at least two points")
        if not np.isfinite(prediction_times).all() or not np.all(
            np.diff(prediction_times) > 0
        ):
            raise ValueError("prediction_timesteps must be finite and increasing")
        delta_t = float(np.median(np.diff(prediction_times)))
        context_times = np.arange(context.shape[0], dtype=np.float64) * delta_t
        forecast_offsets = prediction_times - prediction_times[0]
        states_tensor = torch.as_tensor(
            context, dtype=torch.float32, device=self.device
        )
        times_tensor = torch.as_tensor(
            context_times, dtype=torch.float32, device=self.device
        )
        offsets_tensor = torch.as_tensor(
            forecast_offsets, dtype=torch.float32, device=self.device
        )
        context_embedding = None
        if self.foundation_encoder is not None:
            context_embedding = self.foundation_encoder.encode(
                states_tensor,
                times_tensor,
            ).to(device=self.device, dtype=torch.float32)
            if context_embedding.shape != (1, self.model.config.hidden_size):
                raise RuntimeError(
                    "foundation encoder produced an invalid CTF embedding shape"
                )
            if not torch.isfinite(context_embedding).all():
                raise RuntimeError("foundation encoder produced a non-finite embedding")
        reconstruction = pair_id in (2, 4)
        initial_tensor = states_tensor[0] if reconstruction else states_tensor[-1]
        with torch.inference_mode():
            predictions, dynamics = self.model.rollout(
                states_tensor,
                times_tensor,
                offsets_tensor,
                initial_states=initial_tensor,
                hard_support=True,
                context_embedding=context_embedding,
            )
        output = predictions.squeeze(0).cpu().numpy().astype(np.float64)
        if output.shape != (prediction_times.size, context.shape[1]):
            raise RuntimeError("zero-shot adapter produced an invalid prediction shape")
        if not np.isfinite(output).all():
            raise RuntimeError("zero-shot SINDy rollout diverged")
        self.last_inference = self._inference_record(
            pair_id=pair_id,
            context=context,
            predictions=output,
            dynamics=dynamics,
            context_embedding=context_embedding,
        )
        return output

    def _inference_record(
        self,
        *,
        pair_id: int,
        context: np.ndarray,
        predictions: np.ndarray,
        dynamics: InferredDynamics,
        context_embedding: torch.Tensor | None,
    ) -> dict[str, Any]:
        return {
            "schema": "ctf-amortized-sindy-inference-v1",
            "pair_id": int(pair_id),
            "zero_shot": True,
            "adaptation_label": "target-time-zero-update",
            "strict_dataset_zero_shot": self.strict_dataset_zero_shot,
            "pretraining_exposure": self.pretraining_exposure,
            "target_optimizer_steps": 0,
            "target_sparse_regression_solves": 0,
            "target_foundation_encoder_updates": 0,
            "context_sha256": array_sha256(context),
            "prediction_sha256": array_sha256(predictions),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "context_steps": int(context.shape[0]),
            "foundation_embedding_sha256": (
                array_sha256(context_embedding.detach().cpu().numpy())
                if context_embedding is not None
                else None
            ),
            "foundation_encoder_provenance": self.foundation_encoder_provenance,
            "task_mode": "reconstruction" if pair_id in (2, 4) else "forecast",
            "rollout_initial_context_index": 0 if pair_id in (2, 4) else -1,
            "library_exponents": [list(value) for value in self.model.exponents],
            "coefficients_normalized": dynamics.effective_coefficients.squeeze(0)
            .detach()
            .cpu()
            .tolist(),
            "coefficients_physical": dynamics.physical_coefficients.squeeze(0)
            .detach()
            .cpu()
            .tolist(),
            "support_probabilities": dynamics.support_probabilities.squeeze(0)
            .detach()
            .cpu()
            .tolist(),
            "training_manifest": self.training_manifest,
        }
