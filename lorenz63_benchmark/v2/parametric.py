from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.integrate import trapezoid

from .artifacts import atomic_save_npz, atomic_write_csv
from .base import NumpyVectorFieldAdapter
from .contracts import contract_for
from .models import lorenz_rhs_torch
from .numerics import LorenzParameters, compact_taper, integrate_rk4, lorenz_rhs
from .node import resolve_device


PARAMETRIC_METHODS = {"lorenz_ad", "lorenz_ad_tapered"}


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(values))


def _rollout(
    initial: torch.Tensor,
    raw_parameters: torch.Tensor,
    dt_lyapunov: float,
    steps: int,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    time_scale: float,
) -> torch.Tensor:
    parameters = torch.nn.functional.softplus(raw_parameters) + 1e-8

    def field(normalized_state: torch.Tensor) -> torch.Tensor:
        physical_state = state_mean + state_std * normalized_state
        physical_rhs = lorenz_rhs_torch(
            physical_state, parameters[0], parameters[1], parameters[2]
        )
        return physical_rhs / (state_std * time_scale)

    states = [initial]
    state = initial
    for _ in range(steps):
        k1 = field(state)
        k2 = field(state + 0.5 * dt_lyapunov * k1)
        k3 = field(state + 0.5 * dt_lyapunov * k2)
        k4 = field(state + dt_lyapunov * k3)
        state = state + (dt_lyapunov / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states.append(state)
    return torch.stack(states, dim=1)


def _gather(states: torch.Tensor, trajectory: np.ndarray, starts: np.ndarray, length: int) -> torch.Tensor:
    device = states.device
    trajectory_index = torch.as_tensor(trajectory, dtype=torch.long, device=device)[:, None]
    start_index = torch.as_tensor(starts, dtype=torch.long, device=device)[:, None]
    offsets = torch.arange(length, dtype=torch.long, device=device)[None]
    return states[trajectory_index, start_index + offsets]


def train_parametric(
    method: str,
    train_split: dict[str, Any],
    validation_split: dict[str, Any],
    config: dict[str, Any],
    model_seed: int,
    output_dir: str | Path,
    device_name: str,
) -> tuple[Path, dict[str, Any]]:
    if method not in PARAMETRIC_METHODS:
        raise ValueError(f"not a parametric method: {method}")
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cfg = config["parametric"]
    physical_states = torch.as_tensor(train_split["states"], dtype=torch.float64, device=device)
    state_mean = torch.as_tensor(
        train_split["metadata"]["normalization"]["state_mean"], dtype=torch.float64, device=device
    )
    state_std = torch.as_tensor(
        train_split["metadata"]["normalization"]["state_std"], dtype=torch.float64, device=device
    )
    time_scale = float(train_split["metadata"]["normalization"]["time_scale"])
    states = (physical_states - state_mean) / state_std
    dt = float(train_split["metadata"]["dt"])
    dt_lyapunov = dt * time_scale
    initial = torch.as_tensor(cfg["initial_parameters"], dtype=torch.float64, device=device)
    raw_parameters = torch.nn.Parameter(_inverse_softplus(initial))
    optimizer = torch.optim.Adam(
        [raw_parameters], lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    epochs = int(cfg["epochs"])
    steps_per_epoch = int(cfg["steps_per_epoch"])
    window_steps = int(cfg["window_steps"])
    windows_per_update = int(cfg["windows_per_update"])
    taper = torch.as_tensor(compact_taper(window_steps + 1), dtype=torch.float64, device=device)
    uniform = torch.ones_like(taper)
    weights = taper if method == "lorenz_ad_tapered" else uniform
    weights = weights / weights.mean()
    rng = np.random.default_rng(np.random.SeedSequence([model_seed, 9167]))
    history: list[dict[str, Any]] = []
    updates = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for _ in range(steps_per_epoch):
            trajectory = rng.integers(0, states.shape[0], size=windows_per_update)
            starts = rng.integers(0, states.shape[1] - window_steps, size=windows_per_update)
            target = _gather(states, trajectory, starts, window_steps + 1)
            prediction = _rollout(
                target[:, 0], raw_parameters, dt_lyapunov, window_steps,
                state_mean, state_std, time_scale,
            )
            squared = (prediction - target).square().mean(dim=-1)
            loss = (squared * weights[None]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw_parameters], float(cfg["grad_clip"]))
            optimizer.step()
            total += float(loss.detach().cpu())
            updates += 1
        values = torch.nn.functional.softplus(raw_parameters).detach().cpu().numpy()
        history.append({
            "epoch": epoch, "method": method, "model_seed": model_seed,
            "loss": total / steps_per_epoch, "sigma": float(values[0]),
            "rho": float(values[1]), "beta": float(values[2]),
            "elapsed_seconds": time.perf_counter() - started,
        })

    fitted = torch.nn.functional.softplus(raw_parameters).detach().cpu().numpy() + 1e-8
    parameters = LorenzParameters(*[float(value) for value in fitted])
    validation_states = np.asarray(validation_split["states"], dtype=np.float64)
    validation_times = np.asarray(validation_split["times"], dtype=np.float64)
    maximum = min(validation_times[-1], 2.0 / float(train_split["metadata"]["normalization"]["time_scale"]))
    keep = validation_times <= maximum + 1e-12
    prediction, validation_evaluations = integrate_rk4(
        lambda state: lorenz_rhs(state, parameters), validation_states[:, 0], validation_times[keep],
        float(config["evaluation"]["internal_step"]),
    )
    scale = np.asarray(train_split["metadata"]["normalization"]["state_std"], dtype=np.float64)
    nrmse = np.sqrt(np.mean(((prediction - validation_states[:, keep]) / scale) ** 2, axis=-1))
    validation_score = float(trapezoid(np.median(nrmse, axis=0), validation_times[keep]) / maximum)
    metadata = {
        "parameters": {"sigma": fitted[0], "rho": fitted[1], "beta": fitted[2]},
        "validation_nrmse_auc_0_2LT": validation_score,
        "endpoint_tapered": method == "lorenz_ad_tapered",
    }
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "model.npz"
    contract = contract_for(method)
    atomic_save_npz(
        checkpoint_path, schema=np.asarray("lorenz63-model-v2"), method=np.asarray(method),
        track=np.asarray(contract.track), information_contract=np.asarray(contract.information),
        fitted_parameters=fitted, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    training = {
        "optimizer_updates": updates, "wall_time_seconds": time.perf_counter() - started,
        "parameter_count": 3,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "vector_field_evaluations_validation": validation_evaluations,
        "vector_field_evaluations": 4 * window_steps * windows_per_update * updates,
        "windows_per_update": windows_per_update, "window_steps": window_steps,
    }
    atomic_write_csv(output_dir / "history.csv", history)
    return checkpoint_path, training


def load_parametric(path: str | Path, internal_step: float) -> NumpyVectorFieldAdapter:
    with np.load(path, allow_pickle=False) as loaded:
        if str(loaded["schema"]) != "lorenz63-model-v2":
            raise ValueError(f"not a v2 checkpoint: {path}")
        method = str(loaded["method"])
        if method not in PARAMETRIC_METHODS:
            raise ValueError(f"not a parametric checkpoint: {path}")
        track = str(loaded["track"])
        fitted = np.asarray(loaded["fitted_parameters"], dtype=np.float64)
        metadata = json.loads(str(loaded["metadata"]))
    parameters = LorenzParameters(*[float(value) for value in fitted])
    return NumpyVectorFieldAdapter(
        method=method, track=track, field=lambda states: lorenz_rhs(states, parameters),
        internal_step=internal_step,
        parameters=dict(metadata["parameters"]),
    )
