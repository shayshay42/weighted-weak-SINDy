from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_save_npz
from .base import NumpyVectorFieldAdapter
from .contracts import PINN_TRACK, contract_for
from .numerics import LorenzParameters, lorenz_rhs


ORACLE_INTERNAL_STEP = 1e-4


def create_oracle(config: dict[str, Any], output_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    contract = contract_for("solver_oracle")
    path = Path(output_dir) / "model.npz"
    parameters = np.asarray([
        config["system"]["sigma"], config["system"]["rho"], config["system"]["beta"]
    ], dtype=np.float64)
    atomic_save_npz(
        path, schema=np.asarray("lorenz63-model-v2"), method=np.asarray("solver_oracle"),
        track=np.asarray(contract.track), information_contract=np.asarray(contract.information),
        parameters=parameters, internal_step=np.asarray(ORACLE_INTERNAL_STEP, dtype=np.float64),
    )
    return path, {
        "optimizer_updates": 0, "wall_time_seconds": 0.0, "parameter_count": 0,
        "peak_gpu_memory_bytes": 0, "vector_field_evaluations": 0,
        "forecast_internal_step": ORACLE_INTERNAL_STEP,
    }


def load_oracle(path: str | Path, internal_step: float) -> NumpyVectorFieldAdapter:
    with np.load(path, allow_pickle=False) as loaded:
        if str(loaded["method"]) != "solver_oracle":
            raise ValueError(f"not an oracle checkpoint: {path}")
        values = np.asarray(loaded["parameters"], dtype=np.float64)
        checkpoint_step = (
            float(loaded["internal_step"]) if "internal_step" in loaded.files
            else ORACLE_INTERNAL_STEP
        )
    oracle_step = min(float(internal_step), checkpoint_step)
    if not np.isfinite(oracle_step) or oracle_step <= 0.0:
        raise ValueError(f"invalid oracle integration step {oracle_step}")
    parameters = LorenzParameters(*[float(value) for value in values])
    return NumpyVectorFieldAdapter(
        method="solver_oracle", track=PINN_TRACK, field=lambda state: lorenz_rhs(state, parameters),
        internal_step=oracle_step, parameters={
            "sigma": parameters.sigma, "rho": parameters.rho, "beta": parameters.beta,
        },
    )
