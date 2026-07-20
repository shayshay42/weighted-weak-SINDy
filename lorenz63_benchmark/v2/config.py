from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "schema": "lorenz63-benchmark-v2",
    "system": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    "data": {
        "dt": 0.01,
        "spinup_time": 100.0,
        "train_time": 20.0,
        "validation_lyapunov_times": 5.0,
        "max_forecast_lyapunov_times": 30.0,
        "n_train": 64,
        "n_validation": 32,
        "n_test": 128,
        "initial_low": [-20.0, -30.0, 5.0],
        "initial_high": [20.0, 30.0, 45.0],
        "noise_levels": [0.0, 0.001, 0.01, 0.05],
        "reference": {
            "method": "DOP853", "rtol": 1e-12, "atol": 1e-14, "max_step": 0.001,
            "crosscheck_method": "Radau", "agreement_threshold": 1e-6,
            "crosscheck_workers": 16,
        },
        "lyapunov": {"duration": 200.0, "burn_in": 20.0, "step": 0.001, "qr_interval": 10},
    },
    "normalization": {"time_scale": "largest_lyapunov_exponent", "minimum_std": 1e-12},
    "node": {
        "hidden_dim": 128, "depth": 3, "activation": "tanh", "dtype": "float32",
        "epochs": 100, "steps_per_epoch": 50, "windows_per_update": 32,
        "window_steps": 40, "learning_rate": 1e-3, "weight_decay": 1e-6,
        "grad_clip": 1.0, "gamma": 0.1, "weak_modes": 4, "weak_power": 2,
        "savgol_window": 11, "savgol_polyorder": 3, "sampling_margin_noisy": 10,
    },
    "sindy": {
        "degree": 2, "threshold": 0.05, "ridge": 1e-10, "max_iter": 20,
        "weak_window_steps": 80, "weak_stride_steps": 20, "weak_modes": 4,
        "savgol_window": 11, "savgol_polyorder": 3,
        "ablation_trajectory_counts": [1, 4, 16, 64],
    },
    "parametric": {
        "initial_parameters": [8.0, 25.0, 2.0], "epochs": 200,
        "steps_per_epoch": 50, "windows_per_update": 32, "window_steps": 25,
        "learning_rate": 0.05, "weight_decay": 0.0, "grad_clip": 10.0,
    },
    "pinn": {
        "hidden_dim": 128, "depth": 3, "activation": "tanh", "dtype": "float32",
        "epochs": 100, "steps_per_epoch": 50, "windows_per_update": 32, "quadrature_points": 16,
        "weak_modes": 4, "learning_rate": 1e-3, "weight_decay": 0.0,
        "grad_clip": 10.0, "train_lyapunov_times": 2.0,
        "evaluation_lyapunov_times": 5.0,
    },
    "evaluation": {
        "internal_step": 0.001, "vpt_threshold": 0.4,
        "auc_lyapunov_times": [1.0, 2.0, 5.0], "long_run_subsample": 1000,
        "stability_bound": 1000.0, "lyapunov_duration": 20.0,
        "lyapunov_step": 0.005, "lyapunov_qr_interval": 5,
        "error_plot_max": 100.0,
    },
    "tuning": {
        "development_seed": 0,
        "objective": "median_nrmse_auc_0_2LT",
        "soft_dtw_gamma": [0.01, 0.05, 0.1, 0.5],
        "savgol_window": [7, 11, 15, 21],
        "sindy_degree": [2, 3],
        "sindy_threshold": [0.0, 0.01, 0.05, 0.1],
        "sindy_ridge": [0.0, 1e-10, 1e-8],
        "candidate_epochs": 20,
        "candidate_steps_per_epoch": 25,
    },
    "noise_overrides": {},
    "method_overrides": {},
    "aggregation": {
        "bootstrap_resamples": 10000, "bootstrap_seed": 2026,
        "require_complete_matrix": True, "require_complete_ablation": True,
    },
    "acceptance": {
        "largest_lyapunov_min": 0.8, "largest_lyapunov_max": 1.0,
        "sindy_coefficient_relative_error": 1e-3,
        "ad_parameter_relative_error": 1e-3,
    },
    "final": {"data_seeds": [1, 2, 3, 4, 5], "model_seeds": [0, 1, 2]},
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with Path(path).open("r", encoding="utf-8") as handle:
        return deep_update(DEFAULT_CONFIG, json.load(handle))


def config_for_noise(config: dict[str, Any], noise_level: float) -> dict[str, Any]:
    from .data import noise_label
    override = config.get("noise_overrides", {}).get(noise_label(float(noise_level)), {})
    return deep_update(config, override)


def config_for_method(config: dict[str, Any], noise_level: float, method: str) -> dict[str, Any]:
    from .data import noise_label
    configured = config_for_noise(config, noise_level)
    method_overrides = config.get("method_overrides", {}).get(
        noise_label(float(noise_level)), {}
    )
    matched_parent = {
        "sindy_weighted": "sindy_strong",
        "sindy_weak_weighted": "sindy_weak",
    }.get(method)
    override = method_overrides.get(method)
    if override is None and matched_parent is not None:
        override = method_overrides.get(matched_parent, {})
    if override is None:
        override = {}
    return deep_update(configured, override)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    from .artifacts import atomic_write_json
    atomic_write_json(path, config)
