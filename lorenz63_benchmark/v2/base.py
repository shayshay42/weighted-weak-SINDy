from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .numerics import integrate_rk4


@dataclass(frozen=True)
class Normalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    time_scale: float

    @classmethod
    def from_split(cls, split: dict[str, Any]) -> "Normalization":
        metadata = split["metadata"]["normalization"]
        return cls(
            state_mean=np.asarray(metadata["state_mean"], dtype=np.float64),
            state_std=np.asarray(metadata["state_std"], dtype=np.float64),
            time_scale=float(metadata["time_scale"]),
        )

    def normalize_state(self, states: np.ndarray) -> np.ndarray:
        return (np.asarray(states, dtype=np.float64) - self.state_mean) / self.state_std

    def denormalize_state(self, states: np.ndarray) -> np.ndarray:
        return self.state_mean + self.state_std * np.asarray(states, dtype=np.float64)


class ForecastMethod(ABC):
    method: str
    track: str
    autonomous: bool

    @abstractmethod
    def forecast(self, initial_states: np.ndarray, times: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def vector_field(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.method} does not expose an autonomous vector field")

    def parameters(self) -> dict[str, Any]:
        return {}


class NumpyVectorFieldAdapter(ForecastMethod):
    def __init__(
        self,
        *,
        method: str,
        track: str,
        field: Callable[[np.ndarray], np.ndarray],
        internal_step: float,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.track = track
        self.autonomous = True
        self._field = field
        self.internal_step = float(internal_step)
        self._parameters = parameters or {}
        self.last_vector_field_evaluations = 0

    def vector_field(self, states: np.ndarray) -> np.ndarray:
        return np.asarray(self._field(np.asarray(states, dtype=np.float64)), dtype=np.float64)

    def forecast(self, initial_states: np.ndarray, times: np.ndarray) -> np.ndarray:
        result, evaluations = integrate_rk4(
            self.vector_field, initial_states, times, internal_step=self.internal_step
        )
        self.last_vector_field_evaluations = evaluations
        return result

    def parameters(self) -> dict[str, Any]:
        return dict(self._parameters)
