from __future__ import annotations

from lorenz63_benchmark.v2.config import DEFAULT_CONFIG
from lorenz63_benchmark.v2.tune import _candidate_overrides


def test_sindy_search_is_uniform_and_weighted_is_deferred_to_matched_selection() -> None:
    assert len(_candidate_overrides("sindy_strong", DEFAULT_CONFIG, 0.0)) == 24
    assert len(_candidate_overrides("sindy_strong", DEFAULT_CONFIG, 0.01)) == 96
    assert len(_candidate_overrides("sindy_weak", DEFAULT_CONFIG, 0.0)) == 24
    assert _candidate_overrides("sindy_weighted", DEFAULT_CONFIG, 0.0) == []
    assert _candidate_overrides("sindy_weak_weighted", DEFAULT_CONFIG, 0.0) == []
    assert _candidate_overrides("solver_oracle", DEFAULT_CONFIG, 0.0) == [{}]
