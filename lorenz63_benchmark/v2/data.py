from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    atomic_save_npz, atomic_write_json, command_line, environment_snapshot,
    sha256_file, sha256_json, source_hash, utc_now,
)
from .config import load_config
from .numerics import (
    LorenzParameters, integrate_scipy, lorenz_jacobian, lorenz_rhs,
    lyapunov_spectrum, spin_up,
)


def _times(duration: float, dt: float) -> np.ndarray:
    steps = int(np.ceil(duration / dt - 1e-12))
    return np.arange(steps + 1, dtype=np.float64) * dt


def _reference(
    initial_states: np.ndarray,
    times: np.ndarray,
    parameters: LorenzParameters,
    reference: dict[str, Any],
    method: str | None = None,
) -> np.ndarray:
    return integrate_scipy(
        initial_states, times, parameters,
        method=method or str(reference["method"]),
        rtol=float(reference["rtol"]), atol=float(reference["atol"]),
        max_step=float(reference["max_step"]),
    )


def _crosscheck_task(
    payload: tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, Any]]
) -> np.ndarray:
    initial_state, times, parameter_values, reference = payload
    parameters = LorenzParameters(**parameter_values)
    return _reference(
        initial_state[None], times, parameters, reference,
        method=str(reference["crosscheck_method"]),
    )[0]


def solver_agreement_times(
    reference_states: np.ndarray,
    crosscheck_states: np.ndarray,
    times: np.ndarray,
    variance_sum: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    disagreement = np.sum((reference_states - crosscheck_states) ** 2, axis=-1) / variance_sum
    agreement = np.full(reference_states.shape[0], float(times[-1]), dtype=np.float64)
    censored = np.ones(reference_states.shape[0], dtype=bool)
    for index, row in enumerate(disagreement):
        crossings = np.flatnonzero(row > threshold)
        if crossings.size:
            agreement[index] = float(times[crossings[0]])
            censored[index] = False
    return agreement, censored


def _public_split_metadata(
    *, role: str, data_seed: int, dt: float, noise_level: float,
    state_mean: np.ndarray, state_std: np.ndarray, lyapunov_exponent: float,
) -> str:
    return json.dumps({
        "schema": "lorenz63-split-v2", "role": role, "data_seed": data_seed,
        "dt": dt, "noise_level": noise_level,
        "normalization": {
            "state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
            "time_scale": lyapunov_exponent,
        },
    }, sort_keys=True)


def generate_split(config: dict[str, Any], output_root: str | Path, data_seed: int) -> Path:
    split_root = Path(output_root) / f"split_seed{data_seed}"
    manifest_path = split_root / "manifest.json"
    config_hash = sha256_json(config)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [existing.get("files", {}).get("validation"), existing.get("files", {}).get("test")]
        records.extend(existing.get("files", {}).get("train", {}).values())
        if (
            existing.get("status") == "complete"
            and existing.get("generator", {}).get("config_hash") == config_hash
            and records
            and all(
                record and (split_root / record["path"]).exists()
                and sha256_file(split_root / record["path"]) == record["sha256"]
                for record in records
            )
        ):
            return split_root
    split_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, {
        "schema": "lorenz63-dataset-manifest-v2", "status": "running",
        "data_seed": data_seed, "started_at": utc_now(), "config_hash": config_hash,
    })
    data_cfg = config["data"]
    reference = data_cfg["reference"]
    parameters = LorenzParameters(**{key: float(value) for key, value in config["system"].items()})
    dt = float(data_cfg["dt"])
    rng = np.random.default_rng(data_seed)
    low = np.asarray(data_cfg["initial_low"], dtype=np.float64)
    high = np.asarray(data_cfg["initial_high"], dtype=np.float64)
    counts = [int(data_cfg["n_train"]), int(data_cfg["n_validation"]), int(data_cfg["n_test"])]
    raw_initial = rng.uniform(low, high, size=(sum(counts), 3))
    print(f"seed {data_seed}: spinning up {sum(counts)} initial conditions", flush=True)
    attractor_initial = spin_up(raw_initial, float(data_cfg["spinup_time"]), parameters, reference)
    train_initial, validation_initial, test_initial = np.split(attractor_initial, np.cumsum(counts)[:-1])

    lyap_cfg = data_cfg["lyapunov"]
    print(f"seed {data_seed}: estimating Lyapunov spectrum", flush=True)
    lyap_start = spin_up(np.asarray([[1.0, 1.0, 1.0]]), float(data_cfg["spinup_time"]), parameters, reference)[0]
    spectrum = lyapunov_spectrum(
        lambda state: lorenz_rhs(state, parameters),
        lambda state: lorenz_jacobian(state, parameters),
        lyap_start,
        duration=float(lyap_cfg["duration"]), burn_in=float(lyap_cfg["burn_in"]),
        step=float(lyap_cfg["step"]), qr_interval=int(lyap_cfg["qr_interval"]),
    )
    largest = float(spectrum[0])
    if not np.isfinite(largest) or largest <= 0:
        raise RuntimeError(f"invalid largest Lyapunov exponent {largest}")

    train_times = _times(float(data_cfg["train_time"]), dt)
    validation_times = _times(float(data_cfg["validation_lyapunov_times"]) / largest, dt)
    maximum_times = _times(float(data_cfg["max_forecast_lyapunov_times"]) / largest, dt)
    print(f"seed {data_seed}: integrating clean training trajectories", flush=True)
    train_clean = _reference(train_initial, train_times, parameters, reference)
    state_mean = train_clean.reshape(-1, 3).mean(axis=0)
    state_std = train_clean.reshape(-1, 3).std(axis=0, ddof=0)
    state_std = np.maximum(state_std, float(config["normalization"]["minimum_std"]))
    variance_sum = float(np.sum(state_std ** 2))
    print(f"seed {data_seed}: integrating validation trajectories", flush=True)
    validation_clean = _reference(validation_initial, validation_times, parameters, reference)
    print(f"seed {data_seed}: integrating DOP853 test reference", flush=True)
    test_full = _reference(test_initial, maximum_times, parameters, reference)

    worker_count = max(1, int(reference.get("crosscheck_workers", 1)))
    print(
        f"seed {data_seed}: integrating {len(test_initial)} independent Radau cross-checks "
        f"with {worker_count} workers",
        flush=True,
    )
    parameter_values = {
        "sigma": parameters.sigma, "rho": parameters.rho, "beta": parameters.beta,
    }
    payloads = [
        (initial_state, maximum_times, parameter_values, reference)
        for initial_state in test_initial
    ]
    if worker_count == 1:
        crosscheck = np.stack([_crosscheck_task(payload) for payload in payloads])
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            crosscheck = np.stack(list(executor.map(_crosscheck_task, payloads, chunksize=1)))
    agreement_times, agreement_censored = solver_agreement_times(
        test_full, crosscheck, maximum_times, variance_sum,
        float(reference["agreement_threshold"]),
    )
    fifth_percentile = float(np.quantile(agreement_times, 0.05, method="linear"))
    requested_horizon = float(data_cfg["max_forecast_lyapunov_times"]) / largest
    forecast_horizon = min(requested_horizon, fifth_percentile)
    horizon_index = int(np.searchsorted(maximum_times, forecast_horizon, side="right") - 1)
    test_times = maximum_times[: horizon_index + 1]
    test_clean = test_full[:, : horizon_index + 1]
    forecast_horizon = float(test_times[-1])

    validation_path = split_root / "validation.npz"
    test_path = split_root / "test.npz"
    atomic_save_npz(
        validation_path, states=validation_clean, times=validation_times,
        initial_states=validation_initial,
        metadata=np.asarray(_public_split_metadata(
            role="validation", data_seed=data_seed, dt=dt, noise_level=0.0,
            state_mean=state_mean, state_std=state_std, lyapunov_exponent=largest,
        )),
    )
    atomic_save_npz(
        test_path, states=test_clean, times=test_times, initial_states=test_initial,
        metadata=np.asarray(_public_split_metadata(
            role="test", data_seed=data_seed, dt=dt, noise_level=0.0,
            state_mean=state_mean, state_std=state_std, lyapunov_exponent=largest,
        )),
    )

    train_paths: dict[str, str] = {}
    noise_records: dict[str, Any] = {}
    for noise_level_value in data_cfg["noise_levels"]:
        noise_level = float(noise_level_value)
        label = noise_label(noise_level)
        noise_rng = np.random.default_rng(np.random.SeedSequence([data_seed, int(round(noise_level * 1e9)), 2026]))
        observed = train_clean + noise_rng.normal(size=train_clean.shape) * (noise_level * state_std)
        observed_mean = observed.reshape(-1, 3).mean(axis=0)
        observed_std = observed.reshape(-1, 3).std(axis=0, ddof=0)
        observed_std = np.maximum(observed_std, float(config["normalization"]["minimum_std"]))
        noise_root = split_root / f"noise_{label}"
        train_path = noise_root / "train.npz"
        atomic_save_npz(
            train_path, states=observed, times=train_times, initial_states=observed[:, 0],
            state_mean=observed_mean, state_std=observed_std,
            metadata=np.asarray(_public_split_metadata(
                role="train", data_seed=data_seed, dt=dt, noise_level=noise_level,
                state_mean=observed_mean, state_std=observed_std, lyapunov_exponent=largest,
            )),
        )
        train_paths[label] = str(train_path.relative_to(split_root))
        noise_records[label] = {
            "level": noise_level, "train_path": train_paths[label], "sha256": sha256_file(train_path),
            "state_mean": observed_mean.tolist(), "state_std": observed_std.tolist(),
        }

    manifest = {
        "schema": "lorenz63-dataset-manifest-v2", "status": "complete",
        "created_at": utc_now(), "data_seed": data_seed,
        "generator": {
            "system": dict(config["system"]), "reference": dict(reference),
            "spinup_time": float(data_cfg["spinup_time"]), "source_hash": source_hash(),
            "config_hash": config_hash,
        },
        "command": command_line(), "environment": environment_snapshot(),
        "counts": {"train": counts[0], "validation": counts[1], "test": counts[2]},
        "dt": dt, "training_time": float(train_times[-1]),
        "lyapunov_spectrum": spectrum.tolist(), "largest_lyapunov_exponent": largest,
        "requested_horizon_lyapunov": float(data_cfg["max_forecast_lyapunov_times"]),
        "forecast_horizon": forecast_horizon,
        "forecast_horizon_lyapunov": largest * forecast_horizon,
        "solver_agreement": {
            "threshold": float(reference["agreement_threshold"]),
            "times": agreement_times.tolist(), "censored": agreement_censored.tolist(),
            "fifth_percentile": fifth_percentile,
        },
        "normalization": {"state_mean": state_mean.tolist(), "state_std": state_std.tolist(), "time_scale": largest},
        "files": {
            "validation": {"path": validation_path.name, "sha256": sha256_file(validation_path)},
            "test": {"path": test_path.name, "sha256": sha256_file(test_path)},
            "train": noise_records,
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(
        f"seed {data_seed}: complete; lambda_max={largest:.8f}, "
        f"valid_horizon={manifest['forecast_horizon_lyapunov']:.3f} LT",
        flush=True,
    )
    return split_root


def noise_label(level: float) -> str:
    if level == 0:
        return "0"
    return f"{level:.6g}".replace(".", "p")


def load_split(path: str | Path, expected_role: str | None = None) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as loaded:
        result = {name: np.asarray(loaded[name]) for name in loaded.files if name != "metadata"}
        metadata = json.loads(str(loaded["metadata"]))
    if metadata.get("schema") != "lorenz63-split-v2":
        raise ValueError(f"not a v2 split: {path}")
    if expected_role is not None and metadata.get("role") != expected_role:
        raise ValueError(f"expected a {expected_role} split, got {metadata.get('role')} from {path}")
    forbidden = {"params", "derivatives", "generator", "test"}
    if metadata.get("role") == "train" and forbidden.intersection(metadata):
        raise ValueError(f"training split exposes forbidden metadata: {forbidden.intersection(metadata)}")
    result["metadata"] = metadata
    result["path"] = path
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate isolated Lorenz63 benchmark v2 splits.")
    parser.add_argument("--config", default="configs/v2/lorenz63_paper.json")
    parser.add_argument("--output-root", default="data/v2")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else [0, *config["final"]["data_seeds"]]
    for seed in seeds:
        path = generate_split(config, args.output_root, seed)
        print(path)


if __name__ == "__main__":
    main()
