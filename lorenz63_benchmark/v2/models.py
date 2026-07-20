from __future__ import annotations

import torch
from torch import nn


def activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation {name!r}")


def make_mlp(input_dim: int, output_dim: int, hidden_dim: int, depth: int, name: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(width, hidden_dim), activation(name)])
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class NormalizedVectorField(nn.Module):
    """Maps normalized state to derivative with respect to Lyapunov time."""

    def __init__(self, hidden_dim: int = 128, depth: int = 3, activation_name: str = "tanh") -> None:
        super().__init__()
        self.net = make_mlp(3, 3, hidden_dim, depth, activation_name)

    def forward(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.net(normalized_state)


class ConditionalFlowMap(nn.Module):
    """Conditional Lorenz flow map with an exact initial-condition ansatz."""

    def __init__(self, hidden_dim: int = 128, depth: int = 3, activation_name: str = "tanh") -> None:
        super().__init__()
        self.net = make_mlp(4, 3, hidden_dim, depth, activation_name)

    def forward(
        self,
        initial_state: torch.Tensor,
        time_seconds: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        time_scale: float,
        maximum_lyapunov_time: float,
    ) -> torch.Tensor:
        if time_seconds.ndim == 2:
            time_seconds = time_seconds.unsqueeze(-1)
        mean = state_mean.reshape(3)
        std = state_std.reshape(3)
        normalized_initial = (initial_state - mean) / std
        if normalized_initial.ndim == 2:
            normalized_initial = normalized_initial[:, None, :]
        normalized_initial = normalized_initial.expand(*time_seconds.shape[:-1], 3)
        lyapunov_time = time_seconds * float(time_scale)
        network_time = lyapunov_time / float(maximum_lyapunov_time)
        velocity = self.net(torch.cat([normalized_initial, network_time], dim=-1))
        normalized_state = normalized_initial + lyapunov_time * velocity
        return mean + std * normalized_state


def rk4_rollout_torch(
    field: nn.Module, initial_state: torch.Tensor, step: float, steps: int
) -> torch.Tensor:
    states = [initial_state]
    state = initial_state
    for _ in range(steps):
        k1 = field(state)
        k2 = field(state + 0.5 * step * k1)
        k3 = field(state + 0.5 * step * k2)
        k4 = field(state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states.append(state)
    return torch.stack(states, dim=1)


def lorenz_rhs_torch(
    state: torch.Tensor, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
) -> torch.Tensor:
    return torch.stack(
        [
            sigma * (state[..., 1] - state[..., 0]),
            state[..., 0] * (rho - state[..., 2]) - state[..., 1],
            state[..., 0] * state[..., 1] - beta * state[..., 2],
        ],
        dim=-1,
    )
