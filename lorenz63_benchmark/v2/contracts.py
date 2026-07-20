from __future__ import annotations

from dataclasses import dataclass


PRIMARY_TRACK = "state_only_dynamics"
PARAMETRIC_TRACK = "known_form_hidden_parameters"
PINN_TRACK = "known_physics_forward_surrogate"


@dataclass(frozen=True)
class MethodContract:
    name: str
    label: str
    track: str
    information: tuple[str, ...]
    kind: str
    autonomous: bool
    color: str


METHODS = {
    "node_strong": MethodContract(
        "node_strong", "Strong-form Neural ODE", PRIMARY_TRACK,
        ("observed_states", "observation_times"), "node", True, "#0072B2"
    ),
    "node_soft_dtw": MethodContract(
        "node_soft_dtw", "Soft-DTW Neural ODE", PRIMARY_TRACK,
        ("observed_states", "observation_times"), "node", True, "#E69F00"
    ),
    "node_weak": MethodContract(
        "node_weak", "Weak-form Neural ODE", PRIMARY_TRACK,
        ("observed_states", "observation_times"), "node", True, "#009E73"
    ),
    "sindy_strong": MethodContract(
        "sindy_strong", "Strong-form SINDy", PRIMARY_TRACK,
        ("observed_states", "observation_times"), "sindy", True, "#56B4E9"
    ),
    "sindy_weak": MethodContract(
        "sindy_weak", "Weak-form SINDy", PRIMARY_TRACK,
        ("observed_states", "observation_times"), "sindy", True, "#CC79A7"
    ),
    "sindy_weak_weighted": MethodContract(
        "sindy_weak_weighted", "Endpoint-weighted Weak-form SINDy", PRIMARY_TRACK,
        ("observed_states", "observation_times"), "sindy", True, "#332288"
    ),
    "sindy_weighted": MethodContract(
        "sindy_weighted", "Endpoint-weighted SINDy", PRIMARY_TRACK,
        ("observed_states", "observation_times"), "sindy", True, "#D55E00"
    ),
    "lorenz_ad": MethodContract(
        "lorenz_ad", "AD Lorenz", PARAMETRIC_TRACK,
        ("observed_states", "observation_times", "lorenz_equation_form"), "parametric", True, "#009E73"
    ),
    "lorenz_ad_tapered": MethodContract(
        "lorenz_ad_tapered", "Endpoint-tapered AD Lorenz", PARAMETRIC_TRACK,
        ("observed_states", "observation_times", "lorenz_equation_form"), "parametric", True, "#D55E00"
    ),
    "pinn_strong": MethodContract(
        "pinn_strong", "Strong-form PINN", PINN_TRACK,
        ("initial_states", "lorenz_equation_form", "canonical_parameters"), "pinn", False, "#0072B2"
    ),
    "pinn_weak": MethodContract(
        "pinn_weak", "Weak-form PINN", PINN_TRACK,
        ("initial_states", "lorenz_equation_form", "canonical_parameters"), "pinn", False, "#CC79A7"
    ),
    "pinn_weak_tapered": MethodContract(
        "pinn_weak_tapered", "Endpoint-tapered Weak PINN", PINN_TRACK,
        ("initial_states", "lorenz_equation_form", "canonical_parameters"), "pinn", False, "#E69F00"
    ),
    "solver_oracle": MethodContract(
        "solver_oracle", "RK4 Numerical Oracle", PINN_TRACK,
        ("initial_states", "lorenz_equation_form", "canonical_parameters"), "oracle", True, "#000000"
    ),
}

TRACK_LABELS = {
    PRIMARY_TRACK: "State-only dynamics learning",
    PARAMETRIC_TRACK: "Known equation form, hidden parameters",
    PINN_TRACK: "Known physics forward surrogate",
}

DEFAULT_ORDER = {
    PRIMARY_TRACK: [
        "sindy_strong", "sindy_weighted", "sindy_weak", "sindy_weak_weighted",
        "node_strong", "node_weak", "node_soft_dtw",
    ],
    PARAMETRIC_TRACK: ["lorenz_ad", "lorenz_ad_tapered"],
    PINN_TRACK: ["solver_oracle", "pinn_strong", "pinn_weak", "pinn_weak_tapered"],
}


def contract_for(method: str) -> MethodContract:
    try:
        return METHODS[method]
    except KeyError as exc:
        raise ValueError(f"unknown v2 method {method!r}; choose from {sorted(METHODS)}") from exc
