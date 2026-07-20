from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .artifacts import atomic_write_csv
from .base import ForecastMethod, Normalization
from .contracts import contract_for
from .losses import soft_dtw_divergence, weak_residual_loss
from .models import NormalizedVectorField, rk4_rollout_torch
from .numerics import observed_derivatives


NODE_METHODS = {"node_strong", "node_soft_dtw", "node_weak"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


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


def numerical_derivatives(
    states: np.ndarray, dt: float, noise_level: float, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, int]:
    return observed_derivatives(
        states, dt, noise_level,
        savgol_window=int(cfg["savgol_window"]),
        savgol_polyorder=int(cfg["savgol_polyorder"]),
    )


def _window_indices(
    rng: np.random.Generator,
    n_trajectories: int,
    n_times: int,
    window_steps: int,
    margin: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    maximum_start = n_times - margin - window_steps - 1
    if maximum_start < margin:
        raise ValueError("training trajectories are shorter than the configured NODE window")
    trajectories = rng.integers(0, n_trajectories, size=batch_size)
    starts = rng.integers(margin, maximum_start + 1, size=batch_size)
    return trajectories, starts


def _gather_windows(states: torch.Tensor, trajectories: np.ndarray, starts: np.ndarray, length: int) -> torch.Tensor:
    device = states.device
    trajectory_index = torch.as_tensor(trajectories, device=device, dtype=torch.long)[:, None]
    start_index = torch.as_tensor(starts, device=device, dtype=torch.long)[:, None]
    offsets = torch.arange(length, device=device, dtype=torch.long)[None, :]
    return states[trajectory_index, start_index + offsets]


def node_sampling_margin(
    noise_level: float, derivative_margin: int, cfg: dict[str, Any]
) -> int:
    margin = 2 if noise_level == 0.0 else int(cfg["sampling_margin_noisy"])
    if derivative_margin > margin:
        raise ValueError(
            "NODE sampling margin must cover the selected derivative estimator"
        )
    return margin


def train_node(
    method: str,
    train_split: dict[str, Any],
    validation_split: dict[str, Any],
    config: dict[str, Any],
    model_seed: int,
    output_dir: str | Path,
    device_name: str,
) -> tuple[Path, dict[str, Any]]:
    if method not in NODE_METHODS:
        raise ValueError(f"not a NODE method: {method}")
    if train_split["metadata"]["role"] != "train" or validation_split["metadata"]["role"] != "validation":
        raise ValueError("NODE fit requires explicit train and validation splits")
    set_seed(model_seed)
    cfg = config["node"]
    device = resolve_device(device_name)
    dtype = torch.float32 if str(cfg.get("dtype", "float32")) == "float32" else torch.float64
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    normalization = Normalization.from_split(train_split)
    states_np = np.asarray(train_split["states"], dtype=np.float64)
    normalized_np = normalization.normalize_state(states_np)
    states = torch.as_tensor(normalized_np, dtype=dtype, device=device)
    dt = float(train_split["metadata"]["dt"])
    dt_lyapunov = dt * normalization.time_scale
    noise_level = float(train_split["metadata"]["noise_level"])
    derivative_margin = 0
    derivative_tensor: torch.Tensor | None = None
    if method == "node_strong":
        derivative_states, derivatives, derivative_margin = numerical_derivatives(states_np, dt, noise_level, cfg)
        normalized_derivatives = derivatives / (normalization.state_std * normalization.time_scale)
        derivative_tensor = torch.as_tensor(normalized_derivatives, dtype=dtype, device=device)
        expected = states_np[:, derivative_margin : states_np.shape[1] - derivative_margin]
        if not np.array_equal(derivative_states, expected):
            raise AssertionError("derivative/state alignment failed")

    model = NormalizedVectorField(
        hidden_dim=int(cfg["hidden_dim"]), depth=int(cfg["depth"]),
        activation_name=str(cfg["activation"]),
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    epochs = int(cfg["epochs"])
    steps_per_epoch = int(cfg["steps_per_epoch"])
    windows_per_update = int(cfg["windows_per_update"])
    window_steps = int(cfg["window_steps"])
    margin = node_sampling_margin(noise_level, derivative_margin, cfg)
    rng = np.random.default_rng(np.random.SeedSequence([model_seed, 7321]))
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    updates = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            trajectory_indices, starts = _window_indices(
                rng, states.shape[0], states.shape[1], window_steps, margin, windows_per_update
            )
            observed_window = _gather_windows(states, trajectory_indices, starts, window_steps + 1)
            if method == "node_strong":
                assert derivative_tensor is not None
                shifted_starts = starts - derivative_margin
                derivative_window = _gather_windows(
                    derivative_tensor, trajectory_indices, shifted_starts, window_steps + 1
                )
                loss = nn.functional.mse_loss(model(observed_window), derivative_window)
            elif method == "node_soft_dtw":
                prediction = rk4_rollout_torch(model, observed_window[:, 0], dt_lyapunov, window_steps)
                loss = soft_dtw_divergence(
                    prediction, observed_window, gamma=float(cfg["gamma"])
                ).mean()
            else:
                rhs = model(observed_window)
                loss = weak_residual_loss(
                    observed_window, rhs, dt_lyapunov,
                    modes=int(cfg["weak_modes"]), power=int(cfg["weak_power"]), tapered=False,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            updates += 1
        history.append({
            "epoch": epoch, "method": method, "model_seed": model_seed,
            "loss": epoch_loss / steps_per_epoch,
            "elapsed_seconds": time.perf_counter() - started,
        })

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "model.pt"
    contract = contract_for(method)
    checkpoint = {
        "schema": "lorenz63-model-v2", "method": method, "track": contract.track,
        "information_contract": list(contract.information), "model_seed": model_seed,
        "architecture": {
            "hidden_dim": int(cfg["hidden_dim"]), "depth": int(cfg["depth"]),
            "activation": str(cfg["activation"]),
        },
        "normalization": {
            "state_mean": normalization.state_mean.tolist(),
            "state_std": normalization.state_std.tolist(),
            "time_scale": normalization.time_scale,
        },
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": {
            "optimizer_updates": updates,
            "wall_time_seconds": time.perf_counter() - started,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "windows_per_update": windows_per_update,
            "window_steps": window_steps,
            "sampling_margin": margin,
            "validation_used_for_optimization": False,
            "training_dtype": str(dtype).replace("torch.", ""),
            "vector_field_evaluations": int(
                updates * windows_per_update
                * (4 * window_steps if method == "node_soft_dtw" else window_steps + 1)
            ),
        },
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    history_path = output_dir / "history.csv"
    atomic_write_csv(history_path, history)
    return checkpoint_path, checkpoint["training"]


class TorchNodeAdapter(ForecastMethod):
    def __init__(
        self,
        *,
        method: str,
        track: str,
        model: NormalizedVectorField,
        device: torch.device,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        time_scale: float,
        internal_step: float,
    ) -> None:
        self.method = method
        self.track = track
        self.autonomous = True
        self.model = model
        self.device = device
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float64, device=device)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float64, device=device)
        self.time_scale = float(time_scale)
        self.internal_step = float(internal_step)
        self.last_vector_field_evaluations = 0

    def _field_tensor(self, states: torch.Tensor) -> torch.Tensor:
        normalized = (states - self.state_mean) / self.state_std
        return self.state_std * self.time_scale * self.model(normalized)

    def vector_field(self, states: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(states, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            values = self._field_tensor(tensor)
        return values.cpu().numpy()

    def forecast(self, initial_states: np.ndarray, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) < 0.0):
            raise ValueError("forecast times must be a nonempty increasing vector")
        state = torch.as_tensor(initial_states, dtype=torch.float64, device=self.device)
        result = torch.empty(
            (state.shape[0], times.size, 3), dtype=torch.float64, device=self.device
        )
        result[:, 0] = state
        evaluations = 0
        with torch.no_grad():
            for index in range(1, times.size):
                interval = float(times[index] - times[index - 1])
                substeps = max(1, int(np.ceil(abs(interval) / self.internal_step)))
                step = interval / substeps
                for _ in range(substeps):
                    k1 = self._field_tensor(state)
                    k2 = self._field_tensor(state + 0.5 * step * k1)
                    k3 = self._field_tensor(state + 0.5 * step * k2)
                    k4 = self._field_tensor(state + step * k3)
                    state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                    evaluations += 4 * state.shape[0]
                result[:, index] = state
        self.last_vector_field_evaluations = evaluations
        return result.cpu().numpy()


def load_node(path: str | Path, internal_step: float, device_name: str = "cpu") -> TorchNodeAdapter:
    device = resolve_device(device_name)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "lorenz63-model-v2" or checkpoint.get("method") not in NODE_METHODS:
        raise ValueError(f"not a v2 NODE checkpoint: {path}")
    architecture = checkpoint["architecture"]
    model = NormalizedVectorField(
        hidden_dim=int(architecture["hidden_dim"]), depth=int(architecture["depth"]),
        activation_name=str(architecture["activation"]),
    ).to(device=device, dtype=torch.float64)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    norm = checkpoint["normalization"]
    mean = np.asarray(norm["state_mean"], dtype=np.float64)
    std = np.asarray(norm["state_std"], dtype=np.float64)
    time_scale = float(norm["time_scale"])

    return TorchNodeAdapter(
        method=str(checkpoint["method"]), track=str(checkpoint["track"]), model=model,
        device=device, state_mean=mean, state_std=std, time_scale=time_scale,
        internal_step=internal_step,
    )
