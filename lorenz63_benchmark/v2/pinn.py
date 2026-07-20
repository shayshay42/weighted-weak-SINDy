from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import atomic_write_csv
from .base import ForecastMethod
from .contracts import contract_for
from .losses import time_derivative, weak_residual_loss
from .models import ConditionalFlowMap, lorenz_rhs_torch
from .node import resolve_device


PINN_METHODS = {"pinn_strong", "pinn_weak", "pinn_weak_tapered"}


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def train_pinn(
    method: str,
    train_split: dict[str, Any],
    validation_split: dict[str, Any],
    config: dict[str, Any],
    model_seed: int,
    output_dir: str | Path,
    device_name: str,
) -> tuple[Path, dict[str, Any]]:
    if method not in PINN_METHODS:
        raise ValueError(f"not a PINN method: {method}")
    if validation_split["metadata"]["role"] != "validation":
        raise ValueError("PINN requires an explicit validation split contract")
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cfg = config["pinn"]
    dtype = torch.float32 if str(cfg.get("dtype", "float32")) == "float32" else torch.float64
    system = {key: float(value) for key, value in config["system"].items()}
    initial_states = torch.as_tensor(train_split["initial_states"], dtype=dtype, device=device)
    normalization = train_split["metadata"]["normalization"]
    state_mean = torch.as_tensor(normalization["state_mean"], dtype=dtype, device=device)
    state_std = torch.as_tensor(normalization["state_std"], dtype=dtype, device=device)
    time_scale = float(normalization["time_scale"])
    maximum_lyapunov_time = float(cfg["train_lyapunov_times"])
    maximum_time = maximum_lyapunov_time / time_scale
    model = ConditionalFlowMap(
        hidden_dim=int(cfg["hidden_dim"]), depth=int(cfg["depth"]),
        activation_name=str(cfg["activation"]),
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    epochs = int(cfg["epochs"])
    steps_per_epoch = int(cfg["steps_per_epoch"])
    windows_per_update = int(cfg["windows_per_update"])
    quadrature_points = int(cfg["quadrature_points"])
    if quadrature_points < 3:
        raise ValueError("PINN quadrature_points must be at least three")
    physical_step = maximum_time / (quadrature_points - 1)
    base_times = torch.linspace(
        0.0, maximum_time, quadrature_points, dtype=dtype, device=device
    ).view(1, quadrature_points, 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(model_seed + 4513)
    history: list[dict[str, Any]] = []
    updates = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for _ in range(steps_per_epoch):
            indices = torch.randint(
                initial_states.shape[0], (windows_per_update,), generator=generator, device=device
            )
            x0 = initial_states[indices]
            times = base_times.expand(windows_per_update, -1, -1).clone().detach().requires_grad_(
                method == "pinn_strong"
            )
            prediction = model(
                x0, times, state_mean, state_std, time_scale, maximum_lyapunov_time
            )
            rhs = lorenz_rhs_torch(prediction, **system)
            if method == "pinn_strong":
                residual = time_derivative(prediction, times) - rhs
                loss = (residual / state_std).square().mean()
            else:
                loss = weak_residual_loss(
                    (prediction - state_mean) / state_std, rhs / state_std, physical_step,
                    modes=int(cfg["weak_modes"]), power=2,
                    tapered=method == "pinn_weak_tapered",
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            total += float(loss.detach().cpu())
            updates += 1
        history.append({
            "epoch": epoch, "method": method, "model_seed": model_seed,
            "physics_loss": total / steps_per_epoch,
            "elapsed_seconds": time.perf_counter() - started,
        })

    contract = contract_for(method)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "model.pt"
    training = {
        "optimizer_updates": updates, "wall_time_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "collocation_points_per_update": windows_per_update * quadrature_points,
        "trajectory_target_values_used": 0,
        "initial_condition_ansatz": "exact",
        "train_lyapunov_times": maximum_lyapunov_time,
        "training_dtype": str(dtype).replace("torch.", ""),
        "vector_field_evaluations": updates * windows_per_update * quadrature_points,
    }
    checkpoint = {
        "schema": "lorenz63-model-v2", "method": method, "track": contract.track,
        "information_contract": list(contract.information), "model_seed": model_seed,
        "architecture": {
            "hidden_dim": int(cfg["hidden_dim"]), "depth": int(cfg["depth"]),
            "activation": str(cfg["activation"]),
        },
        "normalization": {
            "state_mean": state_mean.detach().cpu().tolist(),
            "state_std": state_std.detach().cpu().tolist(), "time_scale": time_scale,
        },
        "maximum_lyapunov_time": maximum_lyapunov_time,
        "system": system,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": training,
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    atomic_write_csv(output_dir / "history.csv", history)
    return checkpoint_path, training


class PINNAdapter(ForecastMethod):
    def __init__(self, checkpoint: dict[str, Any], device: torch.device) -> None:
        self.method = str(checkpoint["method"])
        self.track = str(checkpoint["track"])
        self.autonomous = False
        architecture = checkpoint["architecture"]
        self.model = ConditionalFlowMap(
            hidden_dim=int(architecture["hidden_dim"]), depth=int(architecture["depth"]),
            activation_name=str(architecture["activation"]),
        ).to(device=device, dtype=torch.float64)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.device = device
        normalization = checkpoint["normalization"]
        self.mean = torch.as_tensor(normalization["state_mean"], dtype=torch.float64, device=device)
        self.std = torch.as_tensor(normalization["state_std"], dtype=torch.float64, device=device)
        self.time_scale = float(normalization["time_scale"])
        self.maximum_lyapunov_time = float(checkpoint["maximum_lyapunov_time"])

    def forecast(self, initial_states: np.ndarray, times: np.ndarray) -> np.ndarray:
        x0 = torch.as_tensor(initial_states, dtype=torch.float64, device=self.device)
        time = torch.as_tensor(times, dtype=torch.float64, device=self.device)
        time = time.view(1, -1, 1).expand(x0.shape[0], -1, -1)
        with torch.no_grad():
            prediction = self.model(
                x0, time, self.mean, self.std, self.time_scale, self.maximum_lyapunov_time
            )
        self.last_surrogate_evaluations = int(x0.shape[0] * time.shape[1])
        return prediction.cpu().numpy()


def load_pinn(path: str | Path, device_name: str = "cpu") -> PINNAdapter:
    device = resolve_device(device_name)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "lorenz63-model-v2" or checkpoint.get("method") not in PINN_METHODS:
        raise ValueError(f"not a v2 PINN checkpoint: {path}")
    return PINNAdapter(checkpoint, device)
