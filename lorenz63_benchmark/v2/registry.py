from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ForecastMethod
from .contracts import contract_for


@dataclass(frozen=True)
class MethodInterface:
    """Uniform fit/load surface; loaded objects provide forecast and vector_field."""

    name: str

    def fit(
        self,
        train: dict[str, Any],
        validation: dict[str, Any],
        config: dict[str, Any],
        model_seed: int,
        output_dir: str | Path,
        device: str = "auto",
        trajectory_count: int | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        return fit_method(
            self.name, train, validation, config, model_seed, output_dir, device,
            trajectory_count=trajectory_count,
        )

    def load(
        self, checkpoint_path: str | Path, config: dict[str, Any], device: str = "cpu"
    ) -> ForecastMethod:
        return load_method(checkpoint_path, self.name, config, device=device)


def method_interface(method: str) -> MethodInterface:
    contract_for(method)
    return MethodInterface(method)


def fit_method(
    method: str,
    train_split: dict[str, Any],
    validation_split: dict[str, Any],
    config: dict[str, Any],
    model_seed: int,
    output_dir: str | Path,
    device: str,
    trajectory_count: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    contract_for(method)
    kind = contract_for(method).kind
    if kind == "node":
        from .node import train_node
        return train_node(method, train_split, validation_split, config, model_seed, output_dir, device)
    if kind == "sindy":
        from .sindy import train_sindy
        return train_sindy(
            method, train_split, validation_split, config, model_seed, output_dir,
            trajectory_count=trajectory_count,
        )
    if kind == "parametric":
        from .parametric import train_parametric
        return train_parametric(method, train_split, validation_split, config, model_seed, output_dir, device)
    if kind == "pinn":
        from .pinn import train_pinn
        return train_pinn(method, train_split, validation_split, config, model_seed, output_dir, device)
    if kind == "pretrained":
        from .panda import create_panda_checkpoint
        return create_panda_checkpoint(train_split, config, output_dir)
    if method == "solver_oracle":
        from .oracle import create_oracle
        return create_oracle(config, output_dir)
    raise ValueError(f"unsupported method {method}")


def load_method(
    checkpoint_path: str | Path,
    method: str,
    config: dict[str, Any],
    device: str = "cpu",
) -> ForecastMethod:
    internal_step = float(config["evaluation"]["internal_step"])
    kind = contract_for(method).kind
    if kind == "node":
        from .node import load_node
        return load_node(checkpoint_path, internal_step, device)
    if kind == "sindy":
        from .sindy import load_sindy
        return load_sindy(checkpoint_path, internal_step)
    if kind == "parametric":
        from .parametric import load_parametric
        return load_parametric(checkpoint_path, internal_step)
    if kind == "pinn":
        from .pinn import load_pinn
        return load_pinn(checkpoint_path, device)
    if kind == "pretrained":
        from .panda import load_panda
        return load_panda(checkpoint_path, device)
    if method == "solver_oracle":
        from .oracle import load_oracle
        return load_oracle(checkpoint_path, internal_step)
    raise ValueError(f"unsupported method {method}")
