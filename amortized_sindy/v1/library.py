from __future__ import annotations

import math
from itertools import combinations_with_replacement
from typing import Sequence

import torch


def polynomial_exponents(
    dimension: int,
    degree: int,
    *,
    include_bias: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic graded monomial basis."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if degree < 1:
        raise ValueError("degree must be positive")
    exponents: list[tuple[int, ...]] = []
    if include_bias:
        exponents.append((0,) * dimension)
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(dimension), order):
            exponent = [0] * dimension
            for index in indices:
                exponent[index] += 1
            exponents.append(tuple(exponent))
    return tuple(exponents)


def polynomial_library(
    states: torch.Tensor,
    exponents: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Evaluate a monomial library on tensors whose last axis is state."""
    if states.ndim < 1:
        raise ValueError("states must have a state axis")
    dimension = states.shape[-1]
    columns = []
    for exponent in exponents:
        if len(exponent) != dimension:
            raise ValueError("library exponent dimension does not match states")
        column = torch.ones_like(states[..., 0])
        for state_index, power in enumerate(exponent):
            if power < 0:
                raise ValueError("monomial powers must be non-negative")
            if power:
                column = column * states[..., state_index].pow(int(power))
        columns.append(column)
    if not columns:
        raise ValueError("library must contain at least one term")
    return torch.stack(columns, dim=-1)


def exponent_names(
    exponents: Sequence[Sequence[int]],
    variable_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if not exponents:
        return ()
    dimension = len(exponents[0])
    if variable_names is None:
        variable_names = tuple(f"x{index}" for index in range(dimension))
    if len(variable_names) != dimension:
        raise ValueError("variable name count does not match exponent dimension")
    names = []
    for exponent in exponents:
        if len(exponent) != dimension:
            raise ValueError("inconsistent exponent dimensions")
        factors = []
        for name, power in zip(variable_names, exponent):
            if power == 1:
                factors.append(name)
            elif power > 1:
                factors.append(f"{name}^{power}")
        names.append("1" if not factors else " ".join(factors))
    return tuple(names)


def normalized_to_physical_coefficients(
    coefficients: torch.Tensor,
    exponents: Sequence[Sequence[int]],
    *,
    state_mean: torch.Tensor,
    state_scale: torch.Tensor,
    time_scale: torch.Tensor,
) -> torch.Tensor:
    """Convert ``dz/dtau`` coefficients to ``dx/dt`` monomial coefficients.

    Coordinates obey ``z=(x-mean)/scale`` and ``tau=(t-t0)/time_scale``.
    The operation is batched and differentiable with respect to coefficients
    and normalization statistics.
    """
    squeeze_batch = coefficients.ndim == 2
    if squeeze_batch:
        coefficients = coefficients.unsqueeze(0)
    if coefficients.ndim != 3:
        raise ValueError("coefficients must have shape [batch,library,state]")
    batch, library_size, dimension = coefficients.shape
    if library_size != len(exponents):
        raise ValueError("coefficient and exponent library sizes do not match")
    if any(len(exponent) != dimension for exponent in exponents):
        raise ValueError("exponent dimension does not match coefficients")

    def _state_stat(value: torch.Tensor, name: str) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=coefficients.dtype, device=coefficients.device)
        if value.ndim == 3 and value.shape[1] == 1:
            value = value.squeeze(1)
        if value.ndim == 1 and batch == 1 and value.shape[0] == dimension:
            value = value.unsqueeze(0)
        if value.shape != (batch, dimension):
            raise ValueError(f"{name} must have shape [batch,state]")
        return value

    mean = _state_stat(state_mean, "state_mean")
    scale = _state_stat(state_scale, "state_scale")
    time = torch.as_tensor(
        time_scale, dtype=coefficients.dtype, device=coefficients.device
    ).reshape(-1)
    if time.shape != (batch,):
        raise ValueError("time_scale must have one value per batch")
    if torch.any(scale <= 0) or torch.any(time <= 0):
        raise ValueError("state and time scales must be positive")

    exponent_tuples = tuple(tuple(int(power) for power in value) for value in exponents)
    lookup = {value: index for index, value in enumerate(exponent_tuples)}
    if len(lookup) != len(exponent_tuples) or (0,) * dimension not in lookup:
        raise ValueError("exponents must be unique and include the constant term")
    contributions: list[list[torch.Tensor]] = [[] for _ in exponents]
    output_scale = scale / time[:, None]
    for source_index, source_exponent in enumerate(exponent_tuples):
        expansions: list[tuple[tuple[int, ...], torch.Tensor]] = [
            ((0,) * dimension, torch.ones(batch, dtype=coefficients.dtype, device=coefficients.device))
        ]
        for state_index, power in enumerate(source_exponent):
            next_expansions = []
            for partial_exponent, partial_factor in expansions:
                for physical_power in range(power + 1):
                    target_exponent = list(partial_exponent)
                    target_exponent[state_index] = physical_power
                    factor = (
                        math.comb(power, physical_power)
                        * (-mean[:, state_index]).pow(power - physical_power)
                        / scale[:, state_index].pow(power)
                    )
                    next_expansions.append(
                        (tuple(target_exponent), partial_factor * factor)
                    )
            expansions = next_expansions
        for target_exponent, basis_factor in expansions:
            target_index = lookup[target_exponent]
            contributions[target_index].append(
                basis_factor[:, None]
                * coefficients[:, source_index, :]
                * output_scale
            )
    zero = torch.zeros(
        (batch, dimension), dtype=coefficients.dtype, device=coefficients.device
    )
    physical = torch.stack(
        [sum(values, zero) for values in contributions], dim=1
    )
    return physical.squeeze(0) if squeeze_batch else physical
