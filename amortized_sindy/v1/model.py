from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .library import (
    normalized_to_physical_coefficients,
    polynomial_exponents,
    polynomial_library,
)


@dataclass(frozen=True)
class AmortizedSINDyConfig:
    state_dimension: int
    library_degree: int = 2
    hidden_size: int = 64
    encoder_layers: int = 1
    min_context_steps: int = 8
    min_state_scale: float = 1e-6
    min_time_scale: float = 1e-8
    support_temperature: float = 1.0
    support_threshold: float = 0.5
    max_normalized_step: float = 0.05
    encoder_type: str = "gru"
    weak_window_length: int = 33
    weak_stride_steps: int = 16
    weak_modes: int = 2
    external_bottleneck_size: int = 128
    birkhoff_scales: tuple[float, ...] = (1.0, 0.5)
    weak_attention_heads: int = 4
    weak_transformer_layers: int = 1
    hybrid_ablation: str = "full"
    training_support_mode: str = "soft"

    def __post_init__(self) -> None:
        if self.state_dimension < 1:
            raise ValueError("state_dimension must be positive")
        if self.library_degree < 1:
            raise ValueError("library_degree must be positive")
        if self.hidden_size < 1 or self.encoder_layers < 1:
            raise ValueError("encoder sizes must be positive")
        if self.min_context_steps < 2:
            raise ValueError("min_context_steps must be at least two")
        if self.min_state_scale <= 0 or self.min_time_scale <= 0:
            raise ValueError("normalization floors must be positive")
        if self.support_temperature <= 0:
            raise ValueError("support_temperature must be positive")
        if not 0 < self.support_threshold < 1:
            raise ValueError("support_threshold must be between zero and one")
        if self.training_support_mode not in {"soft", "straight_through"}:
            raise ValueError("invalid training support mode")
        if self.max_normalized_step <= 0:
            raise ValueError("max_normalized_step must be positive")
        if self.encoder_type not in {
            "gru",
            "weak_stats",
            "external",
            "external_mlp",
            "external_birkhoff_weak",
        }:
            raise ValueError(
                "invalid encoder_type"
            )
        if self.external_bottleneck_size < 1:
            raise ValueError("external bottleneck size must be positive")
        if self.weak_window_length < 5 or self.weak_stride_steps < 1 or self.weak_modes < 1:
            raise ValueError("invalid weak-statistic encoder configuration")
        if not self.birkhoff_scales or any(
            not 0.0 < scale <= 1.0 for scale in self.birkhoff_scales
        ):
            raise ValueError("Birkhoff scales must lie in (0, 1]")
        if len(set(self.birkhoff_scales)) != len(self.birkhoff_scales):
            raise ValueError("Birkhoff scales must be unique")
        if self.weak_attention_heads < 1 or self.weak_transformer_layers < 1:
            raise ValueError("invalid weak-token transformer configuration")
        if (
            self.encoder_type == "external_birkhoff_weak"
            and self.external_bottleneck_size % self.weak_attention_heads != 0
        ):
            raise ValueError(
                "external bottleneck size must be divisible by weak attention heads"
            )
        if self.hybrid_ablation not in {
            "full", "no_birkhoff", "no_weak", "no_birkhoff_no_weak"
        }:
            raise ValueError("invalid hybrid ablation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AmortizedSINDyConfig":
        return cls(**values)


@dataclass
class InferredDynamics:
    """One fixed sparse vector field inferred from each context prefix."""

    coefficients: torch.Tensor
    support_probabilities: torch.Tensor
    effective_coefficients: torch.Tensor
    physical_coefficients: torch.Tensor
    state_mean: torch.Tensor
    state_scale: torch.Tensor
    time_scale: torch.Tensor


class AmortizedSINDy(nn.Module):
    """Map an observed trajectory prefix directly to a sparse vector field.

    The encoder and coefficient heads are trained offline across source systems.
    At target inference this module performs one forward pass and holds the
    resulting coefficient matrix fixed throughout integration.
    """

    def __init__(self, config: AmortizedSINDyConfig):
        super().__init__()
        self.config = config
        self.exponents = polynomial_exponents(
            config.state_dimension, config.library_degree
        )
        self.library_size = len(self.exponents)
        if config.encoder_type == "gru":
            self.encoder: nn.Module | None = nn.GRU(
                input_size=config.state_dimension + 1,
                hidden_size=config.hidden_size,
                num_layers=config.encoder_layers,
                batch_first=True,
            )
            self.stats_encoder: nn.Module | None = None
        elif config.encoder_type == "weak_stats":
            self.encoder = None
            statistic_size = (
                self.library_size * self.library_size
                + self.library_size * config.state_dimension
            )
            self.stats_encoder = nn.Sequential(
                nn.Linear(statistic_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
            )
        else:
            # A frozen foundation-model adapter supplies a precomputed context
            # embedding. Its weights and provenance are intentionally kept out
            # of this small, head-only SINDy checkpoint.
            self.encoder = None
            self.stats_encoder = None
        self.external_normalization: nn.Module = (
            nn.LayerNorm(config.hidden_size)
            if config.encoder_type in {
                "external", "external_mlp", "external_birkhoff_weak"
            }
            else nn.Identity()
        )
        if config.encoder_type == "external_mlp":
            self.external_projector: nn.Module = nn.Sequential(
                nn.Linear(config.hidden_size, config.external_bottleneck_size),
                nn.GELU(),
                nn.Linear(
                    config.external_bottleneck_size,
                    config.external_bottleneck_size,
                ),
                nn.GELU(),
            )
            head_size = config.external_bottleneck_size
        elif config.encoder_type == "external_birkhoff_weak":
            bottleneck = config.external_bottleneck_size
            self.external_projector = nn.Sequential(
                nn.Linear(config.hidden_size, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, bottleneck),
                nn.GELU(),
            )
            self.ordered_encoder: nn.Module | None = nn.GRU(
                input_size=config.state_dimension + 1,
                hidden_size=bottleneck,
                batch_first=True,
            )
            birkhoff_size = self.library_size * (
                1 + 2 * len(config.birkhoff_scales)
            )
            self.birkhoff_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(birkhoff_size, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, bottleneck),
                nn.GELU(),
            )
            weak_token_size = (
                self.library_size + config.state_dimension + 3
            )
            self.weak_token_projection: nn.Module | None = nn.Sequential(
                nn.Linear(weak_token_size, bottleneck),
                nn.LayerNorm(bottleneck),
                nn.GELU(),
            )
            weak_layer = nn.TransformerEncoderLayer(
                d_model=bottleneck,
                nhead=config.weak_attention_heads,
                dim_feedforward=2 * bottleneck,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.weak_transformer: nn.Module | None = nn.TransformerEncoder(
                weak_layer, num_layers=config.weak_transformer_layers
            )
            self.weak_cls_token = nn.Parameter(torch.zeros(1, 1, bottleneck))
            self.hybrid_fusion: nn.Module | None = nn.Sequential(
                nn.LayerNorm(4 * bottleneck),
                nn.Linear(4 * bottleneck, 2 * bottleneck),
                nn.GELU(),
                nn.Linear(2 * bottleneck, bottleneck),
                nn.GELU(),
            )
            head_size = bottleneck
        else:
            self.external_projector = nn.Identity()
            head_size = config.hidden_size
            self.ordered_encoder = None
            self.birkhoff_encoder = None
            self.weak_token_projection = None
            self.weak_transformer = None
            self.weak_cls_token = None
            self.hybrid_fusion = None
        output_size = self.library_size * config.state_dimension
        self.coefficient_head = nn.Linear(head_size, output_size)
        self.support_head = nn.Linear(head_size, output_size)
        nn.init.normal_(self.coefficient_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.coefficient_head.bias)
        nn.init.normal_(self.support_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.support_head.bias, 2.0)

    @staticmethod
    def _legendre_modes(
        coordinate: torch.Tensor,
        modes: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        values = [torch.ones_like(coordinate)]
        derivatives = [torch.zeros_like(coordinate)]
        if modes == 1:
            return values, derivatives
        values.append(coordinate)
        derivatives.append(torch.ones_like(coordinate))
        for order in range(2, modes):
            values.append(
                ((2 * order - 1) * coordinate * values[-1] - (order - 1) * values[-2])
                / order
            )
            derivatives.append(
                (
                    (2 * order - 1)
                    * (values[-2] + coordinate * derivatives[-1])
                    - (order - 1) * derivatives[-2]
                )
                / order
            )
        return values, derivatives

    @staticmethod
    def _trapezoid_weights(times: torch.Tensor) -> torch.Tensor:
        """Return quadrature weights for strictly increasing batched grids."""
        if times.ndim != 2 or times.shape[1] < 2:
            raise ValueError("quadrature times must have shape [batch,time>=2]")
        weights = torch.zeros_like(times)
        weights[:, 0] = 0.5 * (times[:, 1] - times[:, 0])
        weights[:, -1] = 0.5 * (times[:, -1] - times[:, -2])
        weights[:, 1:-1] = 0.5 * (times[:, 2:] - times[:, :-2])
        return weights

    @classmethod
    def _smooth_birkhoff_weights(cls, times: torch.Tensor) -> torch.Tensor:
        """C-infinity endpoint-zero Birkhoff weights including quadrature."""
        duration = (times[:, -1] - times[:, 0]).clamp_min(
            torch.finfo(times.dtype).eps
        )
        coordinate = (times - times[:, :1]) / duration[:, None]
        interior = (coordinate > 0.0) & (coordinate < 1.0)
        safe = coordinate.clamp(1e-6, 1.0 - 1e-6)
        log_bump = -1.0 / (safe * (1.0 - safe))
        log_bump = log_bump - log_bump.max(dim=1, keepdim=True).values
        bump = torch.where(interior, torch.exp(log_bump), torch.zeros_like(times))
        weighted = cls._trapezoid_weights(times) * bump
        return weighted / weighted.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(times.dtype).eps
        )

    def _birkhoff_statistics(
        self,
        normalized_states: torch.Tensor,
        normalized_times: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-scale sample-level invariant descriptors of the SINDy library."""
        library = polynomial_library(normalized_states, self.exponents)
        quadrature = self._trapezoid_weights(normalized_times)
        quadrature = quadrature / quadrature.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(quadrature.dtype).eps
        )
        blocks = [torch.einsum("bl,blm->bm", quadrature, library)]
        context_length = normalized_states.shape[1]
        for scale in self.config.birkhoff_scales:
            length = max(
                self.config.min_context_steps,
                int(round(context_length * float(scale))),
            )
            length = min(length, context_length)
            local_times = normalized_times[:, -length:]
            local_library = library[:, -length:, :]
            weights = self._smooth_birkhoff_weights(local_times)
            mean = torch.einsum("bl,blm->bm", weights, local_library)
            variance = torch.einsum(
                "bl,blm->bm", weights, (local_library - mean[:, None, :]) ** 2
            )
            blocks.extend((mean, variance))
        statistics = torch.cat(blocks, dim=1)
        return torch.sign(statistics) * torch.log1p(torch.abs(statistics))

    @staticmethod
    def _smooth_test_function(
        coordinate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a max-one smooth bump and its coordinate derivative."""
        unit = 0.5 * (coordinate + 1.0)
        interior = (unit > 0.0) & (unit < 1.0)
        safe = unit.clamp(1e-6, 1.0 - 1e-6)
        log_bump = 4.0 - 1.0 / (safe * (1.0 - safe))
        bump = torch.where(interior, torch.exp(log_bump), torch.zeros_like(unit))
        log_derivative = (1.0 - 2.0 * safe) / (
            2.0 * safe ** 2 * (1.0 - safe) ** 2
        )
        derivative = torch.where(
            interior, bump * log_derivative, torch.zeros_like(unit)
        )
        return bump, derivative

    def _weak_equation_tokens(
        self,
        normalized_states: torch.Tensor,
        normalized_times: torch.Tensor,
    ) -> torch.Tensor:
        """Keep each smooth weak equation as one attention token."""
        _, context_length, _ = normalized_states.shape
        window_length = min(self.config.weak_window_length, context_length)
        if window_length < 5:
            raise ValueError("context is too short for weak-token conditioning")
        last_start = context_length - window_length
        starts = list(range(0, last_start + 1, self.config.weak_stride_steps))
        if starts[-1] != last_start:
            starts.append(last_start)
        tokens = []
        for start in starts:
            stop = start + window_length
            times = normalized_times[:, start:stop]
            states = normalized_states[:, start:stop, :]
            duration = (times[:, -1] - times[:, 0]).clamp_min(
                self.config.min_time_scale
            )
            coordinate = 2.0 * (times - times[:, :1]) / duration[:, None] - 1.0
            bump, bump_derivative = self._smooth_test_function(coordinate)
            polynomials, polynomial_derivatives = self._legendre_modes(
                coordinate, self.config.weak_modes
            )
            trap = self._trapezoid_weights(times)
            library = polynomial_library(states, self.exponents)
            for mode, (polynomial, polynomial_derivative) in enumerate(zip(
                polynomials, polynomial_derivatives
            )):
                phi = bump * polynomial
                derivative = (
                    bump_derivative * polynomial + bump * polynomial_derivative
                ) * (2.0 / duration[:, None])
                features = torch.einsum("bl,blm->bm", trap * phi, library)
                targets = -torch.einsum("bl,bld->bd", trap * derivative, states)
                equation = torch.cat((features, targets), dim=1)
                equation_scale = torch.linalg.vector_norm(
                    equation, dim=1, keepdim=True
                ).clamp_min(self.config.min_time_scale)
                equation = equation / equation_scale
                mode_coordinate = (
                    0.0
                    if self.config.weak_modes == 1
                    else 2.0 * mode / (self.config.weak_modes - 1) - 1.0
                )
                metadata = torch.stack((
                    0.5 * (times[:, 0] + times[:, -1]),
                    duration,
                    torch.full_like(duration, mode_coordinate),
                ), dim=1)
                tokens.append(torch.cat((equation, metadata), dim=1))
        return torch.stack(tokens, dim=1)

    def _hybrid_embedding(
        self,
        normalized_states: torch.Tensor,
        normalized_times: torch.Tensor,
        external_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if any(module is None for module in (
            self.ordered_encoder,
            self.birkhoff_encoder,
            self.weak_token_projection,
            self.weak_transformer,
            self.hybrid_fusion,
        )) or self.weak_cls_token is None:
            raise RuntimeError("hybrid Birkhoff/weak encoder was not initialized")
        foundation = self.external_projector(
            self.external_normalization(external_embedding)
        )
        ordered_input = torch.cat(
            (normalized_states, normalized_times.unsqueeze(-1)), dim=-1
        )
        _, ordered_hidden = self.ordered_encoder(ordered_input)  # type: ignore[misc]
        ordered = ordered_hidden[-1]
        if self.config.hybrid_ablation in {
            "no_birkhoff", "no_birkhoff_no_weak"
        }:
            birkhoff = torch.zeros_like(foundation)
        else:
            birkhoff = self.birkhoff_encoder(  # type: ignore[operator]
                self._birkhoff_statistics(normalized_states, normalized_times)
            )
        if self.config.hybrid_ablation in {"no_weak", "no_birkhoff_no_weak"}:
            weak = torch.zeros_like(foundation)
        else:
            weak_tokens = self.weak_token_projection(  # type: ignore[operator]
                self._weak_equation_tokens(normalized_states, normalized_times)
            )
            cls = self.weak_cls_token.expand(normalized_states.shape[0], -1, -1)
            weak = self.weak_transformer(  # type: ignore[operator]
                torch.cat((cls, weak_tokens), dim=1)
            )[:, 0, :]
        return self.hybrid_fusion(  # type: ignore[operator]
            torch.cat((foundation, ordered, birkhoff, weak), dim=1)
        )

    def _weak_statistic_embedding(
        self,
        normalized_states: torch.Tensor,
        normalized_times: torch.Tensor,
    ) -> torch.Tensor:
        batch, context_length, _ = normalized_states.shape
        window_length = min(self.config.weak_window_length, context_length)
        if window_length < 5:
            raise ValueError("context is too short for weak-statistic conditioning")
        last_start = context_length - window_length
        starts = list(range(0, last_start + 1, self.config.weak_stride_steps))
        if starts[-1] != last_start:
            starts.append(last_start)
        feature_rows = []
        target_rows = []
        for start in starts:
            stop = start + window_length
            times = normalized_times[:, start:stop]
            states = normalized_states[:, start:stop, :]
            duration = (times[:, -1] - times[:, 0]).clamp_min(
                self.config.min_time_scale
            )
            coordinate = 2.0 * (times - times[:, :1]) / duration[:, None] - 1.0
            support = torch.clamp(1.0 - coordinate ** 2, min=0.0)
            bump = support ** 2
            bump_derivative = -4.0 * coordinate * support
            polynomials, polynomial_derivatives = self._legendre_modes(
                coordinate, self.config.weak_modes
            )
            trap = torch.zeros_like(times)
            trap[:, 0] = 0.5 * (times[:, 1] - times[:, 0])
            trap[:, -1] = 0.5 * (times[:, -1] - times[:, -2])
            trap[:, 1:-1] = 0.5 * (times[:, 2:] - times[:, :-2])
            library = polynomial_library(states, self.exponents)
            for polynomial, polynomial_derivative in zip(
                polynomials, polynomial_derivatives
            ):
                phi = bump * polynomial
                derivative = (
                    bump_derivative * polynomial + bump * polynomial_derivative
                ) * (2.0 / duration[:, None])
                feature_rows.append(torch.einsum("bl,blm->bm", trap * phi, library))
                target_rows.append(
                    -torch.einsum("bl,bld->bd", trap * derivative, states)
                )
        features = torch.stack(feature_rows, dim=1)
        targets = torch.stack(target_rows, dim=1)
        row_count = features.shape[1]
        gram = torch.einsum("brm,brn->bmn", features, features) / row_count
        cross = torch.einsum("brm,brd->bmd", features, targets) / row_count
        statistics = torch.cat((gram.flatten(1), cross.flatten(1)), dim=1)
        statistics = torch.sign(statistics) * torch.log1p(torch.abs(statistics))
        if self.stats_encoder is None:
            raise RuntimeError("weak-statistic encoder was not initialized")
        return self.stats_encoder(statistics)

    def _validate_context(
        self,
        states: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if times.ndim == 1:
            times = times.unsqueeze(0)
        if states.ndim != 3 or times.ndim != 2:
            raise ValueError("context states/times must be [batch,time,state] and [batch,time]")
        if states.shape[:2] != times.shape:
            raise ValueError("context state and time shapes do not agree")
        if states.shape[-1] != self.config.state_dimension:
            raise ValueError("context state dimension does not match the model")
        if states.shape[1] < self.config.min_context_steps:
            raise ValueError("context is shorter than min_context_steps")
        if not torch.isfinite(states).all() or not torch.isfinite(times).all():
            raise ValueError("context contains non-finite values")
        if not torch.all(times[:, 1:] > times[:, :-1]):
            raise ValueError("context times must be strictly increasing")
        return states, times

    def infer_dynamics(
        self,
        context_states: torch.Tensor,
        context_times: torch.Tensor,
        *,
        hard_support: bool | None = None,
        context_embedding: torch.Tensor | None = None,
    ) -> InferredDynamics:
        states, times = self._validate_context(context_states, context_times)
        state_mean = states.mean(dim=1, keepdim=True)
        state_scale = states.std(dim=1, unbiased=False, keepdim=True).clamp_min(
            self.config.min_state_scale
        )
        duration = (times[:, -1] - times[:, 0]).clamp_min(
            self.config.min_time_scale
        )
        normalized_states = (states - state_mean) / state_scale
        normalized_times = (
            (times - times[:, -1:]) / duration[:, None]
        ).unsqueeze(-1)
        if self.config.encoder_type == "gru":
            if context_embedding is not None:
                raise ValueError("GRU conditioning does not accept an external embedding")
            encoder_input = torch.cat((normalized_states, normalized_times), dim=-1)
            if self.encoder is None:
                raise RuntimeError("GRU encoder was not initialized")
            _, hidden = self.encoder(encoder_input)  # type: ignore[misc]
            embedding = hidden[-1]
        elif self.config.encoder_type == "weak_stats":
            if context_embedding is not None:
                raise ValueError(
                    "weak-statistic conditioning does not accept an external embedding"
                )
            embedding = self._weak_statistic_embedding(
                normalized_states, normalized_times.squeeze(-1)
            )
        else:
            if context_embedding is None:
                raise ValueError(
                    "external conditioning requires a frozen context embedding"
                )
            embedding = torch.as_tensor(
                context_embedding, dtype=states.dtype, device=states.device
            )
            if embedding.ndim == 1:
                embedding = embedding.unsqueeze(0)
            if embedding.shape != (states.shape[0], self.config.hidden_size):
                raise ValueError(
                    "external context embedding must have shape [batch, hidden_size]"
                )
            if not torch.isfinite(embedding).all():
                raise ValueError("external context embedding contains non-finite values")
            if self.config.encoder_type == "external_birkhoff_weak":
                embedding = self._hybrid_embedding(
                    normalized_states,
                    normalized_times.squeeze(-1),
                    embedding,
                )
            else:
                embedding = self.external_normalization(embedding)
                embedding = self.external_projector(embedding)
        shape = (-1, self.library_size, self.config.state_dimension)
        coefficients = self.coefficient_head(embedding).reshape(shape)
        support_probabilities = torch.sigmoid(
            self.support_head(embedding).reshape(shape)
            / self.config.support_temperature
        )
        if hard_support is None:
            hard_support = not self.training
        if hard_support:
            support = (
                support_probabilities >= self.config.support_threshold
            ).to(coefficients.dtype)
        elif self.config.training_support_mode == "straight_through":
            hard = (
                support_probabilities >= self.config.support_threshold
            ).to(coefficients.dtype)
            support = hard.detach() - support_probabilities.detach() + support_probabilities
        else:
            support = support_probabilities
        return InferredDynamics(
            coefficients=coefficients,
            support_probabilities=support_probabilities,
            effective_coefficients=coefficients * support,
            physical_coefficients=normalized_to_physical_coefficients(
                coefficients * support,
                self.exponents,
                state_mean=state_mean,
                state_scale=state_scale,
                time_scale=duration,
            ),
            state_mean=state_mean,
            state_scale=state_scale,
            time_scale=duration[:, None],
        )

    def normalized_vector_field(
        self,
        normalized_states: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        features = polynomial_library(normalized_states, self.exponents)
        return torch.einsum("bm,bmd->bd", features, coefficients)

    def _rk4_step(
        self,
        states: torch.Tensor,
        step: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        k1 = self.normalized_vector_field(states, coefficients)
        k2 = self.normalized_vector_field(states + 0.5 * step * k1, coefficients)
        k3 = self.normalized_vector_field(states + 0.5 * step * k2, coefficients)
        k4 = self.normalized_vector_field(states + step * k3, coefficients)
        return states + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def rollout(
        self,
        context_states: torch.Tensor,
        context_times: torch.Tensor,
        forecast_offsets: torch.Tensor,
        *,
        initial_states: torch.Tensor | None = None,
        hard_support: bool | None = None,
        context_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, InferredDynamics]:
        """Forecast on offsets from a supplied or context-end initial state."""
        dynamics = self.infer_dynamics(
            context_states,
            context_times,
            hard_support=hard_support,
            context_embedding=context_embedding,
        )
        states, _ = self._validate_context(context_states, context_times)
        offsets = torch.as_tensor(
            forecast_offsets, dtype=states.dtype, device=states.device
        )
        if offsets.ndim == 2:
            if offsets.shape[0] != states.shape[0]:
                raise ValueError("forecast offset batch does not match context batch")
            if not torch.allclose(offsets, offsets[:1].expand_as(offsets)):
                raise ValueError("batched rollout currently requires a shared forecast grid")
            offsets = offsets[0]
        if offsets.ndim != 1 or offsets.numel() < 1:
            raise ValueError("forecast_offsets must be a non-empty vector")
        if not torch.isfinite(offsets).all() or offsets[0] < 0:
            raise ValueError("forecast offsets must be finite and non-negative")
        if offsets.numel() > 1 and not torch.all(offsets[1:] > offsets[:-1]):
            raise ValueError("forecast offsets must be strictly increasing")

        if initial_states is None:
            initial_states = states[:, -1, :]
        else:
            initial_states = torch.as_tensor(
                initial_states, dtype=states.dtype, device=states.device
            )
            if initial_states.ndim == 1:
                initial_states = initial_states.unsqueeze(0)
            if initial_states.shape != (states.shape[0], states.shape[2]):
                raise ValueError("initial state shape does not match the context batch")
            if not torch.isfinite(initial_states).all():
                raise ValueError("initial states contain non-finite values")
        normalized_state = (
            initial_states - dynamics.state_mean.squeeze(1)
        ) / dynamics.state_scale.squeeze(1)
        predictions = []
        previous_offset = offsets.new_zeros(())
        for offset in offsets:
            physical_step = offset - previous_offset
            normalized_step = physical_step / dynamics.time_scale.squeeze(1)
            max_step = float(normalized_step.detach().abs().max().cpu())
            substeps = max(1, int(max_step / self.config.max_normalized_step) + 1)
            step = (normalized_step / substeps).unsqueeze(-1)
            for _ in range(substeps):
                normalized_state = self._rk4_step(
                    normalized_state,
                    step,
                    dynamics.effective_coefficients,
                )
            predictions.append(
                normalized_state * dynamics.state_scale.squeeze(1)
                + dynamics.state_mean.squeeze(1)
            )
            previous_offset = offset
        return torch.stack(predictions, dim=1), dynamics
