from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class TabPFNCoefficientSpec:
    """Configuration for a source-only in-context coefficient map."""

    model_path: str = "auto"
    n_estimators: int = 1
    device: str = "cpu"
    random_state: int = 0
    coefficient_scale_floor: float = 1e-8

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if self.coefficient_scale_floor <= 0:
            raise ValueError("coefficient scale floor must be positive")


class TabPFNCoefficientConditioner:
    """Map frozen context embeddings to a SINDy matrix with one TabPFN.

    Source examples are converted to a long table. The final categorical
    column identifies the requested coefficient, which lets one scalar TabPFN
    regressor emit all library-by-state entries. ``fit_source`` is allowed only
    before target evaluation and receives no target rows. Target calls use
    ``predict_coefficients`` and never mutate the source reference set.
    """

    def __init__(
        self,
        spec: TabPFNCoefficientSpec,
        *,
        regressor_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.spec = spec
        self._regressor_factory = regressor_factory
        self._regressor: Any | None = None
        self._coefficient_shape: tuple[int, int] | None = None
        self._coefficient_mean: np.ndarray | None = None
        self._coefficient_scale: np.ndarray | None = None
        self._feature_width: int | None = None
        self._source_fit_complete = False
        self.target_update_count = 0

    def _new_regressor(self, categorical_index: int) -> Any:
        if self._regressor_factory is not None:
            return self._regressor_factory(categorical_index=categorical_index)
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TabPFN is not installed; install the project's 'tabpfn' extra"
            ) from error
        return TabPFNRegressor(
            model_path=self.spec.model_path,
            n_estimators=self.spec.n_estimators,
            device=self.spec.device,
            random_state=self.spec.random_state,
            categorical_features_indices=[categorical_index],
            show_progress_bar=False,
        )

    @staticmethod
    def _validate_features(features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("context embeddings must have shape [sample,feature]")
        if not np.isfinite(values).all():
            raise ValueError("context embeddings contain non-finite values")
        return values

    @staticmethod
    def _long_table(features: np.ndarray, output_size: int) -> np.ndarray:
        repeated = np.repeat(features, output_size, axis=0)
        coefficient_index = np.tile(
            np.arange(output_size, dtype=np.float64), features.shape[0]
        )[:, None]
        return np.concatenate((repeated, coefficient_index), axis=1)

    def fit_source(
        self,
        source_embeddings: np.ndarray,
        source_coefficients: np.ndarray,
    ) -> "TabPFNCoefficientConditioner":
        if self._source_fit_complete:
            raise RuntimeError("source reference set is already frozen")
        features = self._validate_features(source_embeddings)
        coefficients = np.asarray(source_coefficients, dtype=np.float64)
        if coefficients.ndim != 3 or coefficients.shape[0] != features.shape[0]:
            raise ValueError(
                "source coefficients must have shape [sample,library,state]"
            )
        if not np.isfinite(coefficients).all():
            raise ValueError("source coefficients contain non-finite values")
        flattened = coefficients.reshape(coefficients.shape[0], -1)
        mean = flattened.mean(axis=0)
        scale = flattened.std(axis=0)
        scale = np.maximum(scale, self.spec.coefficient_scale_floor)
        normalized = (flattened - mean[None, :]) / scale[None, :]
        table = self._long_table(features, flattened.shape[1])
        targets = normalized.reshape(-1)
        regressor = self._new_regressor(categorical_index=features.shape[1])
        regressor.fit(table, targets)
        self._regressor = regressor
        self._coefficient_shape = (coefficients.shape[1], coefficients.shape[2])
        self._coefficient_mean = mean
        self._coefficient_scale = scale
        self._feature_width = features.shape[1]
        self._source_fit_complete = True
        return self

    def predict_coefficients(self, target_embeddings: np.ndarray) -> np.ndarray:
        if not self._source_fit_complete or self._regressor is None:
            raise RuntimeError("source reference set must be frozen before prediction")
        features = self._validate_features(target_embeddings)
        if features.shape[1] != self._feature_width:
            raise ValueError("target embedding width differs from the source bank")
        assert self._coefficient_shape is not None
        assert self._coefficient_mean is not None
        assert self._coefficient_scale is not None
        output_size = int(np.prod(self._coefficient_shape))
        table = self._long_table(features, output_size)
        normalized = np.asarray(self._regressor.predict(table), dtype=np.float64)
        if normalized.shape != (features.shape[0] * output_size,):
            raise ValueError("TabPFN returned an unexpected prediction shape")
        normalized = normalized.reshape(features.shape[0], output_size)
        coefficients = (
            normalized * self._coefficient_scale[None, :]
            + self._coefficient_mean[None, :]
        )
        if not np.isfinite(coefficients).all():
            raise ValueError("TabPFN coefficient prediction contains non-finite values")
        return coefficients.reshape(features.shape[0], *self._coefficient_shape)

    def protocol_manifest(
        self,
        *,
        source_dataset_ids: list[str],
        excluded_evaluation_dataset_ids: list[str],
    ) -> dict[str, Any]:
        if not self._source_fit_complete:
            raise RuntimeError("source reference set is not frozen")
        return {
            "schema": "tabpfn-sindy-conditioner-v1",
            "source_dataset_ids": list(source_dataset_ids),
            "excluded_evaluation_dataset_ids": list(
                excluded_evaluation_dataset_ids
            ),
            "source_reference_frozen": True,
            "target_optimizer_steps": 0,
            "target_sparse_regression_solves": 0,
            "target_conditioner_updates": self.target_update_count,
            "coefficient_shape": list(self._coefficient_shape or ()),
            "feature_width": self._feature_width,
            "tabpfn": {
                "model_path": self.spec.model_path,
                "n_estimators": self.spec.n_estimators,
                "device": self.spec.device,
                "random_state": self.spec.random_state,
            },
        }
