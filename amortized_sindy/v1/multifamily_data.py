from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .checkpoint import sha256_file
from .io import atomic_save_npz, atomic_write_json
from .library import polynomial_exponents


DATASET_SCHEMA = "amortized-sindy-multifamily-data-v1"
DIMENSION = 3
DEGREE = 2
EXPONENTS = polynomial_exponents(DIMENSION, DEGREE)
EXPONENT_INDEX = {value: index for index, value in enumerate(EXPONENTS)}


@dataclass(frozen=True)
class SystemSpec:
    family: str
    group_id: int
    coefficients: np.ndarray
    initial_sampler: Callable[[np.random.Generator], np.ndarray]
    burn_in_steps: int
    public_parameters: dict[str, float]


def _coefficient_matrix(
    equations: tuple[dict[tuple[int, int, int], float], ...]
) -> np.ndarray:
    coefficients = np.zeros((len(EXPONENTS), DIMENSION), dtype=np.float64)
    for output, terms in enumerate(equations):
        for exponent, value in terms.items():
            coefficients[EXPONENT_INDEX[exponent], output] = float(value)
    return coefficients


def _rossler(group_id: int, rng: np.random.Generator) -> SystemSpec:
    a = rng.uniform(0.12, 0.28)
    b = rng.uniform(0.12, 0.30)
    c = rng.uniform(4.5, 6.0)
    speed = rng.uniform(2.5, 4.0)
    coefficients = speed * _coefficient_matrix((
        {(0, 1, 0): -1.0, (0, 0, 1): -1.0},
        {(1, 0, 0): 1.0, (0, 1, 0): a},
        {(0, 0, 0): b, (1, 0, 1): 1.0, (0, 0, 1): -c},
    ))
    return SystemSpec(
        family="rossler",
        group_id=group_id,
        coefficients=coefficients,
        initial_sampler=lambda local: local.uniform(-3.0, 3.0, size=3),
        burn_in_steps=500,
        public_parameters={"a": a, "b": b, "c": c, "speed": speed},
    )


def _chen(group_id: int, rng: np.random.Generator) -> SystemSpec:
    a = rng.uniform(32.0, 38.0)
    b = rng.uniform(2.6, 3.4)
    c = rng.uniform(26.0, 30.0)
    speed = rng.uniform(0.20, 0.32)
    coefficients = speed * _coefficient_matrix((
        {(1, 0, 0): -a, (0, 1, 0): a},
        {(1, 0, 0): c - a, (0, 1, 0): c, (1, 0, 1): -1.0},
        {(0, 0, 1): -b, (1, 1, 0): 1.0},
    ))
    return SystemSpec(
        family="chen",
        group_id=group_id,
        coefficients=coefficients,
        initial_sampler=lambda local: local.uniform(-12.0, 12.0, size=3),
        burn_in_steps=250,
        public_parameters={"a": a, "b": b, "c": c, "speed": speed},
    )


def _competitive_lv(group_id: int, rng: np.random.Generator) -> SystemSpec:
    growth = rng.uniform(0.6, 1.5, size=3)
    interaction = -rng.uniform(0.08, 0.45, size=(3, 3))
    interaction[np.diag_indices(3)] = -rng.uniform(0.7, 1.3, size=3)
    speed = rng.uniform(1.0, 2.5)
    equations: list[dict[tuple[int, int, int], float]] = []
    for output in range(3):
        terms: dict[tuple[int, int, int], float] = {}
        linear = [0, 0, 0]
        linear[output] = 1
        terms[tuple(linear)] = growth[output]
        for other in range(3):
            quadratic = [0, 0, 0]
            quadratic[output] += 1
            quadratic[other] += 1
            terms[tuple(quadratic)] = interaction[output, other]
        equations.append(terms)
    coefficients = speed * _coefficient_matrix(tuple(equations))
    parameters = {f"growth_{index}": float(value) for index, value in enumerate(growth)}
    parameters["speed"] = speed
    return SystemSpec(
        family="competitive_lv",
        group_id=group_id,
        coefficients=coefficients,
        initial_sampler=lambda local: local.uniform(0.15, 2.0, size=3),
        burn_in_steps=20,
        public_parameters=parameters,
    )


def _random_quadratic(group_id: int, rng: np.random.Generator) -> SystemSpec:
    equations: list[dict[tuple[int, int, int], float]] = []
    skew = rng.uniform(-2.5, 2.5, size=(3, 3))
    skew = skew - skew.T
    damping = rng.uniform(0.4, 1.5, size=3)
    for output in range(3):
        terms: dict[tuple[int, int, int], float] = {
            (0, 0, 0): rng.uniform(-0.5, 0.5)
        }
        for source in range(3):
            exponent = [0, 0, 0]
            exponent[source] = 1
            value = skew[output, source]
            if output == source:
                value -= damping[output]
            terms[tuple(exponent)] = value
        candidate_quadratics = list(EXPONENTS[4:])
        for chosen in rng.choice(len(candidate_quadratics), size=3, replace=False):
            terms[candidate_quadratics[int(chosen)]] = rng.uniform(-0.12, 0.12)
        equations.append(terms)
    coefficients = _coefficient_matrix(tuple(equations))
    return SystemSpec(
        family="random_quadratic",
        group_id=group_id,
        coefficients=coefficients,
        initial_sampler=lambda local: local.uniform(-2.0, 2.0, size=3),
        burn_in_steps=20,
        public_parameters={"draw_seed": float(group_id)},
    )


def _sprott_b(group_id: int, rng: np.random.Generator) -> SystemSpec:
    a = rng.uniform(0.8, 1.2)
    b = rng.uniform(0.8, 1.2)
    c = rng.uniform(0.8, 1.2)
    speed = rng.uniform(1.3, 2.2)
    coefficients = speed * _coefficient_matrix((
        {(0, 1, 1): a},
        {(1, 0, 0): b, (0, 1, 0): -b},
        {(0, 0, 0): c, (1, 1, 0): -1.0},
    ))
    return SystemSpec(
        family="sprott_b",
        group_id=group_id,
        coefficients=coefficients,
        initial_sampler=lambda local: local.uniform(-2.0, 2.0, size=3),
        burn_in_steps=300,
        public_parameters={"a": a, "b": b, "c": c, "speed": speed},
    )


def _lorenz(group_id: int, rng: np.random.Generator) -> SystemSpec:
    sigma = rng.uniform(8.0, 12.0)
    rho = rng.uniform(24.0, 32.0)
    beta = rng.uniform(2.2, 3.1)
    coefficients = _coefficient_matrix((
        {(1, 0, 0): -sigma, (0, 1, 0): sigma},
        {(1, 0, 0): rho, (0, 1, 0): -1.0, (1, 0, 1): -1.0},
        {(0, 0, 1): -beta, (1, 1, 0): 1.0},
    ))
    return SystemSpec(
        family="lorenz_heldout",
        group_id=group_id,
        coefficients=coefficients,
        initial_sampler=lambda local: local.uniform(-15.0, 15.0, size=3),
        burn_in_steps=200,
        public_parameters={"sigma": sigma, "rho": rho, "beta": beta},
    )


def _library_numpy(states: np.ndarray) -> np.ndarray:
    columns = []
    for exponent in EXPONENTS:
        column = np.ones(states.shape[:-1], dtype=np.float64)
        for index, power in enumerate(exponent):
            if power:
                column *= states[..., index] ** power
        columns.append(column)
    return np.stack(columns, axis=-1)


def _rk4_step(state: np.ndarray, dt: float, coefficients: np.ndarray) -> np.ndarray:
    def field(value: np.ndarray) -> np.ndarray:
        return _library_numpy(value) @ coefficients

    k1 = field(state)
    k2 = field(state + 0.5 * dt * k1)
    k3 = field(state + 0.5 * dt * k2)
    k4 = field(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def _simulate(
    spec: SystemSpec,
    *,
    rng: np.random.Generator,
    dt: float,
    retained_steps: int,
    state_bound: float = 1e4,
) -> np.ndarray:
    for _ in range(40):
        state = np.asarray(spec.initial_sampler(rng), dtype=np.float64)
        retained = []
        valid = True
        for step in range(spec.burn_in_steps + retained_steps):
            state = _rk4_step(state, dt, spec.coefficients)
            if not np.isfinite(state).all() or np.max(np.abs(state)) > state_bound:
                valid = False
                break
            if step >= spec.burn_in_steps:
                retained.append(state.copy())
        if valid:
            values = np.asarray(retained)
            if values.shape == (retained_steps, DIMENSION) and np.all(
                np.std(values, axis=0) > 1e-5
            ):
                return values
    raise RuntimeError(f"could not generate bounded informative trajectory for {spec.family}")


def _specs(
    *,
    seed: int,
    source_groups_per_family: int,
    validation_groups: int,
    lorenz_groups: int,
) -> tuple[list[SystemSpec], list[SystemSpec], list[SystemSpec]]:
    rng = np.random.default_rng(seed)
    group_id = 0
    source = []
    for builder in (_rossler, _chen, _competitive_lv, _random_quadratic):
        for _ in range(source_groups_per_family):
            source.append(builder(group_id, rng))
            group_id += 1
    validation = []
    for _ in range(validation_groups):
        validation.append(_sprott_b(group_id, rng))
        group_id += 1
    lorenz = []
    for _ in range(lorenz_groups):
        lorenz.append(_lorenz(group_id, rng))
        group_id += 1
    return source, validation, lorenz


def _generate(
    specs: list[SystemSpec],
    *,
    rng: np.random.Generator,
    trajectories_per_group: int,
    dt: float,
    context_steps: int,
    forecast_steps: int,
) -> dict[str, np.ndarray]:
    total_steps = context_steps + forecast_steps - 1
    contexts = []
    futures = []
    group_ids = []
    trajectory_ids = []
    for spec in specs:
        for trajectory_id in range(trajectories_per_group):
            states = _simulate(
                spec,
                rng=rng,
                dt=dt,
                retained_steps=total_steps,
            )
            contexts.append(states[:context_steps])
            futures.append(states[context_steps - 1 :])
            group_ids.append(spec.group_id)
            trajectory_ids.append(trajectory_id)
    return {
        "context_states": np.asarray(contexts, dtype=np.float64),
        "context_times": np.arange(context_steps, dtype=np.float64) * dt,
        "future_states": np.asarray(futures, dtype=np.float64),
        "forecast_offsets": np.arange(forecast_steps, dtype=np.float64) * dt,
        "group_ids": np.asarray(group_ids, dtype=np.int64),
        "trajectory_ids": np.asarray(trajectory_ids, dtype=np.int64),
    }


def generate_multifamily_data(
    *,
    output_root: str | Path,
    seed: int = 2026,
    source_groups_per_family: int = 8,
    validation_groups: int = 8,
    lorenz_groups: int = 8,
    trajectories_per_group: int = 3,
    dt: float = 0.01,
    context_steps: int = 128,
    forecast_steps: int = 33,
    force: bool = False,
) -> Path:
    if min(
        source_groups_per_family,
        validation_groups,
        lorenz_groups,
        trajectories_per_group,
    ) < 1:
        raise ValueError("all group and trajectory counts must be positive")
    if dt <= 0 or context_steps < 8 or forecast_steps < 2:
        raise ValueError("invalid multifamily time-grid configuration")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"multifamily output directory is not empty: {output_root}")
    sanitized = output_root / "sanitized"
    evaluator = output_root / "evaluator_only"
    sanitized.mkdir(parents=True, exist_ok=True)
    evaluator.mkdir(parents=True, exist_ok=True)
    source_specs, validation_specs, lorenz_specs = _specs(
        seed=seed,
        source_groups_per_family=source_groups_per_family,
        validation_groups=validation_groups,
        lorenz_groups=lorenz_groups,
    )
    rng = np.random.default_rng(seed + 1)
    source = _generate(
        source_specs,
        rng=rng,
        trajectories_per_group=trajectories_per_group,
        dt=dt,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
    )
    validation = _generate(
        validation_specs,
        rng=rng,
        trajectories_per_group=trajectories_per_group,
        dt=dt,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
    )
    lorenz = _generate(
        lorenz_specs,
        rng=rng,
        trajectories_per_group=trajectories_per_group,
        dt=dt,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
    )
    source_path = atomic_save_npz(sanitized / "source_train.npz", **source)
    validation_path = atomic_save_npz(sanitized / "source_validation.npz", **validation)
    lorenz_context_path = atomic_save_npz(
        sanitized / "lorenz_heldout_context.npz",
        context_states=lorenz["context_states"],
        context_times=lorenz["context_times"],
        forecast_offsets=lorenz["forecast_offsets"],
        group_ids=lorenz["group_ids"],
        trajectory_ids=lorenz["trajectory_ids"],
    )
    lorenz_truth_path = atomic_save_npz(
        evaluator / "lorenz_heldout_truth.npz",
        future_states=lorenz["future_states"],
        forecast_offsets=lorenz["forecast_offsets"],
        group_ids=lorenz["group_ids"],
        trajectory_ids=lorenz["trajectory_ids"],
    )
    all_specs = source_specs + validation_specs + lorenz_specs
    hidden_path = atomic_save_npz(
        evaluator / "hidden_systems.npz",
        group_ids=np.asarray([spec.group_id for spec in all_specs], dtype=np.int64),
        coefficients=np.asarray([spec.coefficients for spec in all_specs]),
        family_ids=np.asarray([spec.family for spec in all_specs], dtype="U32"),
    )
    parameters_path = atomic_write_json(
        evaluator / "generator_parameters.json",
        {
            str(spec.group_id): {
                "family": spec.family,
                "parameters": spec.public_parameters,
            }
            for spec in all_specs
        },
    )
    artifacts = {
        "source_train": source_path,
        "source_validation": validation_path,
        "lorenz_heldout_context": lorenz_context_path,
        "lorenz_heldout_truth": lorenz_truth_path,
        "hidden_systems": hidden_path,
        "generator_parameters": parameters_path,
    }
    return atomic_write_json(output_root / "manifest.json", {
        "schema": DATASET_SCHEMA,
        "seed": seed,
        "state_dimension": DIMENSION,
        "library_degree": DEGREE,
        "library_exponents": [list(value) for value in EXPONENTS],
        "dt": dt,
        "context_steps": context_steps,
        "forecast_steps": forecast_steps,
        "trajectories_per_group": trajectories_per_group,
        "source_families": sorted({spec.family for spec in source_specs}),
        "validation_families": sorted({spec.family for spec in validation_specs}),
        "heldout_evaluation_families": sorted({spec.family for spec in lorenz_specs}),
        "lorenz_used_for_training_or_selection": False,
        "artifacts": {
            name: {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate trajectory-only source families and evaluator-isolated Lorenz data."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source-groups-per-family", type=int, default=8)
    parser.add_argument("--validation-groups", type=int, default=8)
    parser.add_argument("--lorenz-groups", type=int, default=8)
    parser.add_argument("--trajectories-per-group", type=int, default=3)
    parser.add_argument("--context-steps", type=int, default=128)
    parser.add_argument("--forecast-steps", type=int, default=33)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(generate_multifamily_data(
        output_root=args.output_root,
        seed=args.seed,
        source_groups_per_family=args.source_groups_per_family,
        validation_groups=args.validation_groups,
        lorenz_groups=args.lorenz_groups,
        trajectories_per_group=args.trajectories_per_group,
        context_steps=args.context_steps,
        forecast_steps=args.forecast_steps,
        dt=args.dt,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
