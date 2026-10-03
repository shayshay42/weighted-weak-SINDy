from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from types import SimpleNamespace
from typing import Any, Literal

import torch


ExposureStatus = Literal["verified_excluded", "unknown", "known_exposed"]
FOUNDATION_ENCODER_BINDING_SCHEMA = (
    "amortized-sindy-foundation-encoder-binding-v1"
)


@dataclass(frozen=True)
class FoundationEncoderSpec:
    """Pinned identity and protocol metadata for a frozen context encoder."""

    backend: Literal["panda_patchtst", "chronos_t5"]
    model_id: str
    revision: str
    state_dimension: int
    context_length: int = 128
    pooling: Literal[
        "channel_patch_mean_flatten", "channel_token_mean_flatten"
    ] = "channel_patch_mean_flatten"
    target_dataset_id: str = "CTF4Science/ODE_Lorenz"
    target_pretraining_exposure: ExposureStatus = "unknown"

    def __post_init__(self) -> None:
        if self.backend not in {"panda_patchtst", "chronos_t5"}:
            raise ValueError("unsupported foundation encoder backend")
        if not self.model_id or not self.revision:
            raise ValueError("foundation encoder model ID and revision must be pinned")
        if self.state_dimension < 1 or self.context_length < 8:
            raise ValueError("invalid foundation encoder dimensions")
        if not self.target_dataset_id:
            raise ValueError("foundation encoder target dataset ID must be non-empty")
        expected_pooling = {
            "panda_patchtst": "channel_patch_mean_flatten",
            "chronos_t5": "channel_token_mean_flatten",
        }[self.backend]
        if self.pooling != expected_pooling:
            raise ValueError("unsupported foundation encoder pooling")
        if self.target_pretraining_exposure not in {
            "verified_excluded",
            "unknown",
            "known_exposed",
        }:
            raise ValueError("invalid target pretraining exposure label")

    @property
    def adaptation_label(self) -> str:
        return "target-time-zero-update"

    @property
    def strict_dataset_zero_shot(self) -> bool:
        return self.target_pretraining_exposure == "verified_excluded"

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["adaptation_label"] = self.adaptation_label
        values["strict_dataset_zero_shot"] = self.strict_dataset_zero_shot
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FoundationEncoderSpec":
        if not isinstance(values, Mapping):
            raise ValueError("foundation encoder spec must be a mapping")
        field_names = {field.name for field in fields(cls)}
        derived_names = {"adaptation_label", "strict_dataset_zero_shot"}
        unexpected = set(values) - field_names - derived_names
        if unexpected:
            raise ValueError(
                f"foundation encoder spec has unexpected fields: {sorted(unexpected)}"
            )
        try:
            spec = cls(**{name: values[name] for name in field_names if name in values})
        except (KeyError, TypeError) as error:
            raise ValueError("foundation encoder spec is incomplete") from error
        if (
            "adaptation_label" in values
            and values["adaptation_label"] != spec.adaptation_label
        ):
            raise ValueError("foundation encoder adaptation label is inconsistent")
        if (
            "strict_dataset_zero_shot" in values
            and values["strict_dataset_zero_shot"]
            is not spec.strict_dataset_zero_shot
        ):
            raise ValueError(
                "foundation encoder zero-shot exposure label is inconsistent"
            )
        return spec


def validate_foundation_encoder_binding(
    value: Any,
) -> FoundationEncoderSpec:
    """Validate the artifact-independent encoder identity bound to a checkpoint."""
    if not isinstance(value, dict):
        raise ValueError("foundation encoder binding must be a mapping")
    if value.get("schema") != FOUNDATION_ENCODER_BINDING_SCHEMA:
        raise ValueError("invalid foundation encoder binding schema")
    if set(value) != {
        "schema",
        "encoder",
        "model_weights",
        "embedding_manifests",
    }:
        raise ValueError("foundation encoder binding contains unexpected fields")
    spec = FoundationEncoderSpec.from_dict(value.get("encoder"))
    weights = value.get("model_weights")
    if not isinstance(weights, dict) or set(weights) != {"filename", "sha256"}:
        raise ValueError("foundation encoder weight identity is invalid")
    if weights["filename"] is not None and not isinstance(weights["filename"], str):
        raise ValueError("foundation encoder weight filename is invalid")
    if weights["sha256"] is not None and not (
        isinstance(weights["sha256"], str) and len(weights["sha256"]) == 64
    ):
        raise ValueError("foundation encoder weight hash is invalid")
    manifests = value.get("embedding_manifests")
    if not isinstance(manifests, dict) or set(manifests) != {
        "source",
        "validation",
    }:
        raise ValueError("foundation embedding manifest binding is invalid")
    if not all(
        isinstance(digest, str) and len(digest) == 64
        for digest in manifests.values()
    ):
        raise ValueError("foundation embedding manifest hash is invalid")
    return spec


def load_foundation_encoder(
    spec: FoundationEncoderSpec,
    *,
    device: str | torch.device = "cpu",
) -> "PandaPatchTSTContextEncoder | ChronosT5ContextEncoder":
    """Instantiate the pinned frozen encoder described by a checkpoint binding."""
    device_name = str(torch.device(device))
    if spec.backend == "panda_patchtst":
        return PandaPatchTSTContextEncoder.from_pretrained(
            spec,
            device=device_name,
            dtype=torch.float32,
        )
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    return ChronosT5ContextEncoder.from_pretrained(
        spec,
        device=device_name,
        dtype=dtype,
    )


class PandaPatchTSTContextEncoder:
    """Expose Panda's frozen multichannel PatchTST hidden states.

    The adapter calls only the forecasting model's encoder backbone. It never
    asks Panda to generate target future values. Patch embeddings are averaged
    over time and flattened in channel order, preserving which state variable
    produced each channel embedding.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        spec: FoundationEncoderSpec,
        uniform_time_rtol: float = 1e-5,
        uniform_time_atol: float = 1e-8,
    ) -> None:
        if uniform_time_rtol < 0 or uniform_time_atol < 0:
            raise ValueError("uniform-time tolerances must be non-negative")
        self.pipeline = pipeline
        self.spec = spec
        self.uniform_time_rtol = uniform_time_rtol
        self.uniform_time_atol = uniform_time_atol
        prediction_model = self.pipeline.model
        prediction_model.eval()
        prediction_model.requires_grad_(False)
        self._device = torch.device(prediction_model.device)
        d_model = int(prediction_model.config.d_model)
        if d_model < 1:
            raise ValueError("Panda checkpoint has an invalid embedding dimension")
        self.output_dimension = spec.state_dimension * d_model

    @classmethod
    def from_pretrained(
        cls,
        spec: FoundationEncoderSpec,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "PandaPatchTSTContextEncoder":
        try:
            from panda.patchtst.patchtst import PatchTSTForPrediction
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Panda is not installed; install the project's 'panda' extra"
            ) from error
        prediction_model = PatchTSTForPrediction.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            torch_dtype=dtype,
        )
        prediction_model.to(device)
        return cls(pipeline=SimpleNamespace(model=prediction_model), spec=spec)

    def _validate(
        self,
        context_states: torch.Tensor,
        context_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = torch.as_tensor(context_states, dtype=torch.float32)
        times = torch.as_tensor(context_times, dtype=torch.float64)
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if times.ndim == 1:
            times = times.unsqueeze(0)
        if states.ndim != 3 or times.ndim != 2:
            raise ValueError("context states/times must be [batch,time,state] and [batch,time]")
        if states.shape[:2] != times.shape:
            raise ValueError("context state and time shapes do not agree")
        if states.shape[-1] != self.spec.state_dimension:
            raise ValueError("context state dimension does not match encoder spec")
        if states.shape[1] < 8:
            raise ValueError("Panda context must contain at least eight steps")
        if not torch.isfinite(states).all() or not torch.isfinite(times).all():
            raise ValueError("foundation context contains non-finite values")
        differences = times[:, 1:] - times[:, :-1]
        if not torch.all(differences > 0):
            raise ValueError("foundation context times must be strictly increasing")
        reference = differences[:, :1].expand_as(differences)
        if not torch.allclose(
            differences,
            reference,
            rtol=self.uniform_time_rtol,
            atol=self.uniform_time_atol,
        ):
            raise ValueError("Panda foundation conditioning requires uniform time steps")
        if states.shape[1] > self.spec.context_length:
            states = states[:, -self.spec.context_length :, :]
            times = times[:, -self.spec.context_length :]
        return states, times

    @torch.inference_mode()
    def encode(
        self,
        context_states: torch.Tensor,
        context_times: torch.Tensor,
    ) -> torch.Tensor:
        states, _ = self._validate(context_states, context_times)
        states = states.to(self._device)
        observed = torch.ones_like(states, dtype=torch.bool)
        output = self.pipeline.model.model(
            past_values=states,
            past_observed_mask=observed,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        if hidden.ndim != 4:
            raise ValueError(
                "Panda hidden state must have shape [batch,channel,patch,d_model]"
            )
        if hidden.shape[0] != states.shape[0] or hidden.shape[1] != states.shape[2]:
            raise ValueError("Panda hidden state does not match the context batch/channels")
        embedding = hidden.mean(dim=2).reshape(hidden.shape[0], -1)
        if embedding.shape[1] != self.output_dimension:
            raise ValueError("Panda pooled embedding has an unexpected width")
        if not torch.isfinite(embedding).all():
            raise ValueError("Panda pooled embedding contains non-finite values")
        return embedding.to(dtype=torch.float32)


class ChronosT5ContextEncoder:
    """Pool frozen Chronos-T5 encoder tokens independently per state channel.

    Chronos-T5 is a univariate TSFM. Each state coordinate is therefore encoded
    as a separate series and the pooled channel embeddings are concatenated in
    state order. No forecast is generated and no target-time update occurs.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        spec: FoundationEncoderSpec,
        uniform_time_rtol: float = 1e-5,
        uniform_time_atol: float = 1e-8,
    ) -> None:
        if spec.backend != "chronos_t5":
            raise ValueError("Chronos encoder requires backend='chronos_t5'")
        if uniform_time_rtol < 0 or uniform_time_atol < 0:
            raise ValueError("uniform-time tolerances must be non-negative")
        self.pipeline = pipeline
        self.spec = spec
        self.uniform_time_rtol = uniform_time_rtol
        self.uniform_time_atol = uniform_time_atol
        chronos_model = self.pipeline.model
        chronos_model.eval()
        chronos_model.requires_grad_(False)
        self._device = torch.device(chronos_model.device)
        d_model = int(chronos_model.model.config.d_model)
        if d_model < 1:
            raise ValueError("Chronos checkpoint has an invalid embedding dimension")
        self.output_dimension = spec.state_dimension * d_model

    @classmethod
    def from_pretrained(
        cls,
        spec: FoundationEncoderSpec,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "ChronosT5ContextEncoder":
        try:
            from chronos import ChronosPipeline
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Chronos is not installed; install chronos-forecasting"
            ) from error
        pipeline = ChronosPipeline.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            device_map=device,
            torch_dtype=dtype,
        )
        return cls(pipeline=pipeline, spec=spec)

    def _validate(
        self,
        context_states: torch.Tensor,
        context_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = torch.as_tensor(context_states, dtype=torch.float32)
        times = torch.as_tensor(context_times, dtype=torch.float64)
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if times.ndim == 1:
            times = times.unsqueeze(0)
        if states.ndim != 3 or times.ndim != 2:
            raise ValueError(
                "context states/times must be [batch,time,state] and [batch,time]"
            )
        if states.shape[:2] != times.shape:
            raise ValueError("context state and time shapes do not agree")
        if states.shape[-1] != self.spec.state_dimension:
            raise ValueError("context state dimension does not match encoder spec")
        if states.shape[1] < 8:
            raise ValueError("Chronos context must contain at least eight steps")
        if not torch.isfinite(states).all() or not torch.isfinite(times).all():
            raise ValueError("foundation context contains non-finite values")
        differences = times[:, 1:] - times[:, :-1]
        if not torch.all(differences > 0):
            raise ValueError("foundation context times must be strictly increasing")
        if not torch.allclose(
            differences,
            differences[:, :1].expand_as(differences),
            rtol=self.uniform_time_rtol,
            atol=self.uniform_time_atol,
        ):
            raise ValueError("Chronos foundation conditioning requires uniform time steps")
        if states.shape[1] > self.spec.context_length:
            states = states[:, -self.spec.context_length :, :]
            times = times[:, -self.spec.context_length :]
        return states, times

    @torch.inference_mode()
    def encode(
        self,
        context_states: torch.Tensor,
        context_times: torch.Tensor,
    ) -> torch.Tensor:
        states, _ = self._validate(context_states, context_times)
        batch, length, channels = states.shape
        channel_series = states.permute(0, 2, 1).reshape(batch * channels, length)
        # Chronos tokenization owns the CPU-to-model-device transfer. Passing
        # CUDA values here would conflict with its CPU quantization boundaries.
        hidden, _ = self.pipeline.embed(channel_series.cpu())
        hidden = torch.as_tensor(hidden)
        if hidden.ndim != 3 or hidden.shape[0] != batch * channels:
            raise ValueError(
                "Chronos hidden state must have shape [batch*channel,token,d_model]"
            )
        embedding = hidden.mean(dim=1).reshape(batch, -1)
        if embedding.shape[1] != self.output_dimension:
            raise ValueError("Chronos pooled embedding has an unexpected width")
        if not torch.isfinite(embedding).all():
            raise ValueError("Chronos pooled embedding contains non-finite values")
        return embedding.to(dtype=torch.float32)
