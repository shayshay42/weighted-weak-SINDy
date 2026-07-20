from __future__ import annotations

import numpy as np


def lorenz_polynomial_coefficients(
    exponents: tuple[tuple[int, ...], ...],
    *,
    sigma: float,
    rho: float,
    beta: float,
) -> np.ndarray:
    truth = np.zeros((len(exponents), 3), dtype=np.float64)
    lookup = {exponent: index for index, exponent in enumerate(exponents)}
    required = {(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0)}
    if not required.issubset(lookup):
        return truth
    truth[lookup[(1, 0, 0)], 0] = -sigma
    truth[lookup[(0, 1, 0)], 0] = sigma
    truth[lookup[(1, 0, 0)], 1] = rho
    truth[lookup[(0, 1, 0)], 1] = -1.0
    truth[lookup[(1, 0, 1)], 1] = -1.0
    truth[lookup[(1, 1, 0)], 2] = 1.0
    truth[lookup[(0, 0, 1)], 2] = -beta
    return truth


def relative_coefficient_error(
    coefficients: np.ndarray,
    exponents: tuple[tuple[int, ...], ...],
    system: dict[str, float],
) -> float:
    truth = lorenz_polynomial_coefficients(exponents, **system)
    denominator = np.linalg.norm(truth)
    return float(np.linalg.norm(np.asarray(coefficients) - truth) / denominator)


def relative_parameter_error(parameters: dict[str, float], system: dict[str, float]) -> float:
    names = ("sigma", "rho", "beta")
    fitted = np.asarray([parameters[name] for name in names], dtype=np.float64)
    truth = np.asarray([system[name] for name in names], dtype=np.float64)
    return float(np.linalg.norm(fitted - truth) / np.linalg.norm(truth))
