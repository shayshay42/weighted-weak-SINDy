from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from amortized_sindy.v1.odeformer_gate import (
    adapt_symbolic_equations_to_sindy,
    condition_odeformer_sindy,
    forecast_odeformer,
    symbolic_polynomial_coefficients,
    symbolic_quadratic_taylor_coefficients,
)


class _Candidate:
    def infix(self) -> str:
        return "x_1 | -x_0 | 0.5*x_2"


class _FakeODEFormer:
    def __init__(self, expected_rerank: bool = False) -> None:
        self.fit_inputs: list[tuple[np.ndarray, np.ndarray]] = []
        self.expected_rerank = expected_rerank

    def fit(self, times, trajectory, *, sort_candidates, verbose):
        assert sort_candidates is self.expected_rerank
        assert verbose is False
        self.fit_inputs.append((np.array(times, copy=True), np.array(trajectory, copy=True)))
        return {0: [_Candidate()]}

    def integrate_prediction(self, times, y0, *, prediction):
        assert isinstance(prediction, _Candidate)
        return np.broadcast_to(y0, (len(times), len(y0))).copy()


class _CandidateBank:
    def fit(self, times, trajectory, *, sort_candidates, verbose):
        assert sort_candidates is False
        return {0: [
            SimpleNamespace(infix=lambda: "1000000*x_0 | 0 | 0"),
            SimpleNamespace(infix=lambda: "0 | 0 | 0"),
        ]}


def test_symbolic_polynomial_conversion_uses_matched_library() -> None:
    coefficients = symbolic_polynomial_coefficients(
        "2 + 3*x_0 - x_1*x_2 | x_0**2 | -4*x_2",
        dimension=3,
    )
    assert coefficients is not None
    # Library order is 1, x, y, z, x^2, xy, xz, y^2, yz, z^2.
    np.testing.assert_allclose(coefficients[0], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(coefficients[1], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(coefficients[8], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(coefficients[4], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(coefficients[3], [0.0, 0.0, -4.0])


def test_symbolic_conversion_rejects_nonpolynomial_output() -> None:
    assert symbolic_polynomial_coefficients(
        "sin(x_0) | x_1 | x_2", dimension=3
    ) is None


def test_symbolic_taylor_adapter_handles_nonpolynomial_equations() -> None:
    coefficients = symbolic_quadratic_taylor_coefficients(
        "sin(x_0) | 1/(2 + x_1) | x_2**3",
        expansion_point=np.array([0.0, 0.0, 1.0]),
    )
    assert coefficients is not None
    # sin(x) around zero is x through second order.
    np.testing.assert_allclose(coefficients[1, 0], 1.0, atol=1e-12)
    # x^3 around x=1 is 1 - 3x + 3x^2 through second order.
    np.testing.assert_allclose(coefficients[[0, 3, 9], 2], [1.0, -3.0, 3.0])


def test_taylor_adapter_rollout_uses_only_equation_and_context() -> None:
    contexts = np.ones((2, 8, 3), dtype=np.float64)
    equations = np.array(["0 | 0 | 0", "0 | 0 | 0"])
    predictions, failed, coefficients = adapt_symbolic_equations_to_sindy(
        equations,
        contexts=contexts,
        forecast_offsets=np.arange(4, dtype=np.float64) * 0.01,
    )
    assert not np.any(failed)
    np.testing.assert_allclose(predictions, 1.0)
    np.testing.assert_allclose(coefficients, 0.0)


def test_forecast_conditioner_has_no_future_truth_route() -> None:
    contexts = np.arange(2 * 8 * 3, dtype=np.float64).reshape(2, 8, 3)
    times = np.arange(8, dtype=np.float64) * 0.01
    offsets = np.arange(4, dtype=np.float64) * 0.01
    regressor = _FakeODEFormer()
    predictions, failed, equations, coefficients, errors = forecast_odeformer(
        contexts,
        context_times=times,
        forecast_offsets=offsets,
        regressor=regressor,
        seed=7,
    )
    assert len(regressor.fit_inputs) == 2
    np.testing.assert_allclose(regressor.fit_inputs[0][1], contexts[0])
    np.testing.assert_allclose(predictions[:, 0], contexts[:, -1])
    assert not np.any(failed)
    assert np.all(equations == "x_1 | -x_0 | 0.5*x_2")
    assert np.all(errors == "")
    assert np.isfinite(coefficients).all()


def test_forecast_can_rerank_only_against_observed_context() -> None:
    contexts = np.arange(8 * 3, dtype=np.float64).reshape(1, 8, 3)
    regressor = _FakeODEFormer(expected_rerank=True)
    forecast_odeformer(
        contexts,
        context_times=np.arange(8, dtype=np.float64) * 0.01,
        forecast_offsets=np.arange(4, dtype=np.float64) * 0.01,
        regressor=regressor,
        seed=7,
        context_rerank=True,
    )
    np.testing.assert_allclose(regressor.fit_inputs[0][1], contexts[0])


def test_post_adapter_reranking_rejects_unstable_sindy_candidate() -> None:
    contexts = np.ones((1, 8, 3), dtype=np.float64)
    predictions, failed, equations, coefficients, scores, errors = (
        condition_odeformer_sindy(
            contexts,
            context_times=np.arange(8, dtype=np.float64) * 0.01,
            forecast_offsets=np.arange(4, dtype=np.float64) * 0.01,
            regressor=_CandidateBank(),
            seed=7,
        )
    )
    assert not failed[0]
    assert equations[0] == "0 | 0 | 0"
    assert errors[0] == ""
    np.testing.assert_allclose(predictions, 1.0)
    np.testing.assert_allclose(coefficients, 0.0)
    assert np.isfinite(scores[0])
