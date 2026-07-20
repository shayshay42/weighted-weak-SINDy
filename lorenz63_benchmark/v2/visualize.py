from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .aggregate import paired_hierarchical_bootstrap
from .artifacts import (
    atomic_write_csv,
    atomic_write_json,
    command_line,
    environment_snapshot,
    sha256_file,
    source_hash,
    utc_now,
)
from .config import load_config
from .contracts import (
    METHODS,
    PARAMETRIC_TRACK,
    PINN_TRACK,
    PRIMARY_TRACK,
    TRACK_LABELS,
)
from .data import noise_label
from .plot import _method_order


TRACKS = (PRIMARY_TRACK, PARAMETRIC_TRACK, PINN_TRACK)
KIND_MARKERS = {"node": "o", "sindy": "s", "parametric": "D", "pinn": "^", "oracle": "*"}
NOISE_DISPLAY = {0.0: "0", 0.001: "0.1%", 0.01: "1%", 0.05: "5%"}
TRACK_LINESTYLES = {
    PRIMARY_TRACK: "-",
    PARAMETRIC_TRACK: (0, (6, 2.2)),
    PINN_TRACK: (0, (1.2, 1.7)),
}


def _style() -> None:
    plt.rcParams.update({
        "axes.grid": False,
        "axes.titleweight": "semibold",
        "font.size": 9,
        "figure.dpi": 130,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    })


def _despine(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)


def _noise_values(frame: pd.DataFrame) -> list[float]:
    return sorted(float(value) for value in frame["noise_level"].unique())


def _ordered_methods(
    config: dict[str, Any], track: str, noise: float, available: Iterable[str]
) -> list[str]:
    return _method_order(track, sorted(set(available)), config, noise)


def _bootstrap_row(
    bootstrap: pd.DataFrame, track: str, noise: float, method: str, metric: str
) -> pd.Series:
    selected = bootstrap[
        (bootstrap["track"] == track)
        & np.isclose(bootstrap["noise_level"], noise)
        & (bootstrap["method"] == method)
        & (bootstrap["metric"] == metric)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one bootstrap row for {track}/{noise:g}/{method}/{metric}, "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
    *,
    title: str,
    role: str,
    description: str,
) -> None:
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=240 if extension == "png" else None, bbox_inches="tight")
        catalog.append({
            "figure": stem,
            "format": extension,
            "path": str(path.resolve()),
            "title": title,
            "role": role,
            "description": description,
        })
    plt.close(figure)


def _method_point(
    axis: plt.Axes,
    *,
    y: float,
    method: str,
    estimate: float,
    lower: float,
    upper: float,
) -> None:
    color = METHODS[method].color
    marker = KIND_MARKERS[METHODS[method].kind]
    axis.plot([lower, upper], [y, y], color=color, linewidth=2.0, solid_capstyle="round")
    axis.scatter(
        [estimate], [y], color=color, edgecolor="black", linewidth=0.5,
        marker=marker, s=58 if marker != "*" else 90, zorder=3,
    )


def _raw_run_means(trajectory_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    return trajectory_metrics.groupby(
        ["run_id", "method", "track", "data_seed", "model_seed", "noise_level"],
        as_index=False,
    )[metric].mean()


def _common_auc_metric(
    bootstrap: pd.DataFrame, run_metrics: pd.DataFrame, config: dict[str, Any]
) -> tuple[str, float]:
    expected = len(run_metrics[["track", "method", "noise_level"]].drop_duplicates())
    for horizon in sorted(
        (float(value) for value in config["evaluation"]["auc_lyapunov_times"]),
        reverse=True,
    ):
        metric = f"nrmse_auc_0_{horizon:g}LT"
        observed = bootstrap[bootstrap["metric"] == metric][
            ["track", "method", "noise_level"]
        ].drop_duplicates()
        if len(observed) == expected:
            return metric, horizon
    raise ValueError("no normalized-RMSE AUC horizon is complete across all method tracks")


def plot_benchmark_overview(
    *,
    run_metrics: pd.DataFrame,
    trajectory_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
    noise: float = 0.0,
) -> None:
    auc_metric, auc_horizon = _common_auc_metric(bootstrap, run_metrics, config)
    metrics = [
        ("vpt_restricted_lt", "restricted_mean_vpt_lt", "Restricted mean VPT (LT)", False),
        (
            auc_metric,
            auc_metric,
            f"Mean normalized-RMSE AUC, 0-{auc_horizon:g} LT",
            True,
        ),
    ]
    raw_by_metric = {
        "vpt_restricted_lt": _raw_run_means(trajectory_metrics, "vpt_restricted_lt"),
        auc_metric: _raw_run_means(trajectory_metrics, auc_metric),
    }
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    for column, track in enumerate(TRACKS):
        available = run_metrics[
            (run_metrics["track"] == track) & np.isclose(run_metrics["noise_level"], noise)
        ]["method"].unique()
        methods = _ordered_methods(config, track, noise, available)
        for row, (bootstrap_metric, _run_metric, xlabel, logarithmic) in enumerate(metrics):
            axis = axes[row, column]
            raw = raw_by_metric[bootstrap_metric]
            for position, method in enumerate(methods):
                record = _bootstrap_row(bootstrap, track, noise, method, bootstrap_metric)
                method_raw = raw[
                    (raw["method"] == method) & np.isclose(raw["noise_level"], noise)
                ][bootstrap_metric].to_numpy(dtype=float)
                jitter = np.linspace(-0.12, 0.12, max(method_raw.size, 1))[: method_raw.size]
                axis.scatter(
                    method_raw,
                    position + jitter,
                    color=METHODS[method].color,
                    alpha=0.28,
                    s=13,
                    linewidth=0,
                    zorder=1,
                )
                _method_point(
                    axis,
                    y=position,
                    method=method,
                    estimate=float(record["estimate"]),
                    lower=float(record["ci_lower"]),
                    upper=float(record["ci_upper"]),
                )
            axis.set_yticks(np.arange(len(methods)), [METHODS[method].label for method in methods])
            axis.invert_yaxis()
            axis.set_xlabel(xlabel)
            if logarithmic:
                axis.set_xscale("log")
            if row == 0:
                axis.set_title(TRACK_LABELS[track], fontsize=11)
            _despine(axis)
    figure.suptitle(
        "Lorenz63 benchmark overview | noiseless training",
        fontsize=15,
    )
    figure.subplots_adjust(wspace=0.58, hspace=0.34, top=0.91)
    _save_figure(
        figure,
        output_dir,
        "benchmark_overview__noise_0",
        formats,
        catalog,
        title="Noiseless benchmark overview",
        role="main",
        description="Run points and paired hierarchical-bootstrap 95% intervals for VPT and 0-5 LT error.",
    )


def plot_noise_robustness(
    *,
    bootstrap: pd.DataFrame,
    run_metrics: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
) -> None:
    noise_values = _noise_values(run_metrics)
    x = np.arange(len(noise_values))
    auc_metric, auc_horizon = _common_auc_metric(bootstrap, run_metrics, config)
    panels = [
        ("vpt_restricted_lt", "Restricted mean VPT (LT)", False),
        (auc_metric, f"Mean normalized-RMSE AUC, 0-{auc_horizon:g} LT", True),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, squeeze=False)
    for column, track in enumerate(TRACKS):
        available = run_metrics[run_metrics["track"] == track]["method"].unique()
        methods = _ordered_methods(config, track, noise_values[0], available)
        for row, (metric, ylabel, logarithmic) in enumerate(panels):
            axis = axes[row, column]
            for method in methods:
                records = [
                    _bootstrap_row(bootstrap, track, noise, method, metric)
                    for noise in noise_values
                ]
                estimate = np.asarray([float(record["estimate"]) for record in records])
                lower = np.asarray([float(record["ci_lower"]) for record in records])
                upper = np.asarray([float(record["ci_upper"]) for record in records])
                color = METHODS[method].color
                marker = KIND_MARKERS[METHODS[method].kind]
                axis.fill_between(x, lower, upper, color=color, alpha=0.10, linewidth=0)
                axis.plot(
                    x, estimate, color=color, marker=marker, linewidth=1.8,
                    markersize=5.5, label=METHODS[method].label,
                )
            if logarithmic:
                axis.set_yscale("log")
            axis.set_ylabel(ylabel)
            axis.set_xticks(x, [NOISE_DISPLAY.get(value, f"{100 * value:g}%") for value in noise_values])
            if row == 0:
                axis.set_title(TRACK_LABELS[track], fontsize=11)
                axis.tick_params(labelbottom=False)
                axis.legend(
                    loc="upper center", bbox_to_anchor=(0.5, -0.08),
                    fontsize=8, ncol=2,
                )
            else:
                axis.set_xlabel("Training noise relative to coordinate standard deviation")
            _despine(axis)
    figure.suptitle("Noise robustness across information-equivalent tracks", fontsize=15)
    figure.subplots_adjust(wspace=0.34, hspace=0.48, top=0.91)
    _save_figure(
        figure,
        output_dir,
        "noise_robustness",
        formats,
        catalog,
        title="Noise robustness",
        role="main",
        description="VPT and forecast-error response to deterministic training noise with 95% intervals.",
    )


def hierarchical_survival_curves(
    frame: pd.DataFrame,
    times_lt: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    required = {
        "method", "data_seed", "model_seed", "trajectory_id",
        "vpt_restricted_lt", "vpt_censored",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"survival input misses columns: {sorted(missing)}")
    methods = sorted(str(value) for value in frame["method"].unique())
    data_seeds = sorted(int(value) for value in frame["data_seed"].unique())
    model_seeds = sorted(int(value) for value in frame["model_seed"].unique())
    curves = np.empty(
        (len(methods), len(data_seeds), len(model_seeds), len(times_lt)), dtype=np.float64
    )
    for method_index, method in enumerate(methods):
        for data_index, data_seed in enumerate(data_seeds):
            for model_index, model_seed in enumerate(model_seeds):
                selected = frame[
                    (frame["method"] == method)
                    & (frame["data_seed"] == data_seed)
                    & (frame["model_seed"] == model_seed)
                ].sort_values("trajectory_id")
                if selected.empty:
                    raise ValueError(f"incomplete survival cell for {method}/{data_seed}/{model_seed}")
                vpt = selected["vpt_restricted_lt"].to_numpy(dtype=float)
                censored = selected["vpt_censored"].astype(bool).to_numpy()
                valid = (vpt[:, None] > times_lt[None]) | (
                    censored[:, None] & (vpt[:, None] >= times_lt[None])
                )
                curves[method_index, data_index, model_index] = valid.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(methods), len(times_lt)), dtype=np.float32)
    chunk_size = 128
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        count = stop - start
        split_indices = rng.integers(
            len(data_seeds), size=(count, len(data_seeds))
        )
        model_indices = rng.integers(
            len(model_seeds), size=(count, len(data_seeds), len(model_seeds))
        )
        selected = np.empty(
            (
                len(methods), count, len(data_seeds), len(model_seeds),
                len(times_lt),
            ),
            dtype=np.float64,
        )
        for split_draw in range(len(data_seeds)):
            sampled_split = split_indices[:, split_draw]
            for model_draw in range(len(model_seeds)):
                sampled_model = model_indices[:, split_draw, model_draw]
                selected[:, :, split_draw, model_draw] = curves[
                    :, sampled_split, sampled_model, :
                ]
        draws[start:stop] = selected.mean(axis=(2, 3)).transpose(1, 0, 2)
    point = curves.mean(axis=(1, 2))
    return {
        method: {
            "estimate": point[index],
            "ci_lower": np.quantile(draws[:, index], 0.025, axis=0),
            "ci_upper": np.quantile(draws[:, index], 0.975, axis=0),
        }
        for index, method in enumerate(methods)
    }


def plot_forecast_survival(
    *,
    trajectory_metrics: pd.DataFrame,
    run_metrics: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> None:
    for noise in _noise_values(run_metrics):
        maximum = min(
            5.0,
            float(run_metrics[np.isclose(run_metrics["noise_level"], noise)]["forecast_horizon_lt"].min()),
        )
        times = np.linspace(0.0, maximum, 121)
        figure, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True)
        for column, track in enumerate(TRACKS):
            axis = axes[column]
            group = trajectory_metrics[
                (trajectory_metrics["track"] == track)
                & np.isclose(trajectory_metrics["noise_level"], noise)
            ]
            summaries = hierarchical_survival_curves(
                group, times, resamples=resamples, seed=seed
            )
            methods = _ordered_methods(config, track, noise, summaries)
            for method in methods:
                summary = summaries[method]
                color = METHODS[method].color
                axis.fill_between(
                    times, summary["ci_lower"], summary["ci_upper"],
                    color=color, alpha=0.11, linewidth=0,
                )
                axis.plot(
                    times, summary["estimate"], color=color, linewidth=2.0,
                    label=METHODS[method].label,
                )
            axis.set_xlim(0.0, maximum)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlabel("Forecast time (Lyapunov times)")
            axis.set_title(TRACK_LABELS[track], fontsize=11)
            axis.legend(loc="best", fontsize=8)
            _despine(axis)
        axes[0].set_ylabel("Fraction of forecasts remaining below E(t) = 0.4")
        figure.suptitle(
            f"Valid-forecast survival | training noise {NOISE_DISPLAY.get(noise, noise)}",
            fontsize=15,
        )
        figure.subplots_adjust(wspace=0.18, top=0.86)
        label = noise_label(noise)
        _save_figure(
            figure,
            output_dir,
            f"forecast_survival__noise_{label}",
            formats,
            catalog,
            title=f"Valid-forecast survival at noise {noise:g}",
            role="main" if noise == 0.0 else "supplement",
            description="Fraction of trajectories not yet crossing the VPT threshold; bands resample data and model seeds.",
        )


def _survival_overlay_noises(run_metrics: pd.DataFrame) -> list[float]:
    nonzero = [noise for noise in _noise_values(run_metrics) if noise > 0.0]
    if len(nonzero) != 3:
        raise ValueError(
            "the three-panel survival overlay requires exactly three nonzero noise levels; "
            f"found {nonzero}"
        )
    return nonzero


def plot_forecast_survival_overlay(
    *,
    trajectory_metrics: pd.DataFrame,
    run_metrics: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> None:
    noises = _survival_overlay_noises(run_metrics)
    maximum = min(
        5.0,
        *(float(run_metrics[np.isclose(run_metrics["noise_level"], noise)]["forecast_horizon_lt"].min())
          for noise in noises),
    )
    times = np.linspace(0.0, maximum, 121)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.1), sharex=True, sharey=True)
    plotted_methods: list[str] = []
    for axis, noise in zip(axes, noises):
        group = trajectory_metrics[np.isclose(trajectory_metrics["noise_level"], noise)]
        summaries = hierarchical_survival_curves(
            group, times, resamples=resamples, seed=seed
        )
        methods: list[str] = []
        for track in TRACKS:
            available = [
                method for method in summaries
                if METHODS[method].track == track
            ]
            methods.extend(_ordered_methods(config, track, noise, available))
        for method in methods:
            summary = summaries[method]
            contract = METHODS[method]
            axis.plot(
                times,
                summary["estimate"],
                color=contract.color,
                linestyle=TRACK_LINESTYLES[contract.track],
                linewidth=2.05,
                solid_capstyle="round",
                dash_capstyle="round",
                alpha=0.96,
                zorder=3 if contract.track == PINN_TRACK else 2,
            )
            if method not in plotted_methods:
                plotted_methods.append(method)
        axis.set_xlim(0.0, maximum)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel(r"Forecast time $\lambda_{\max}t$ (Lyapunov times)")
        axis.set_title(f"Training noise {NOISE_DISPLAY.get(noise, noise)}", fontsize=11)
        _despine(axis)
    axes[0].set_ylabel(r"Fraction of forecasts with $E(t) \leq 0.4$")

    track_handles = [
        Line2D(
            [0], [0], color="#222222", linewidth=2.2,
            linestyle=TRACK_LINESTYLES[track], label=TRACK_LABELS[track],
        )
        for track in TRACKS
    ]
    method_handles = [
        Line2D(
            [0], [0], color=METHODS[method].color, linewidth=2.2,
            linestyle=TRACK_LINESTYLES[METHODS[method].track],
            label=METHODS[method].label,
        )
        for method in plotted_methods
    ]
    track_legend = figure.legend(
        handles=track_handles,
        title="Information track (line style)",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        fontsize=8.5,
        title_fontsize=8.5,
    )
    figure.add_artist(track_legend)
    figure.legend(
        handles=method_handles,
        title="Method (color and track line style)",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        fontsize=8,
        title_fontsize=8.5,
        columnspacing=1.5,
        handlelength=3.0,
    )
    figure.suptitle(
        "Valid-prediction-time survival across methods and noisy training sets",
        fontsize=15,
        y=0.985,
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.27, wspace=0.14)
    _save_figure(
        figure,
        output_dir,
        "forecast_survival__all_tracks__noisy_levels",
        formats,
        catalog,
        title="All-method VPT survival across noisy training levels",
        role="main",
        description=(
            "All methods overlaid at 0.1%, 1%, and 5% training noise; color identifies "
            "method and line style identifies information-equivalent track."
        ),
    )


def _load_curves(path: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as loaded:
        index = json.loads(str(loaded["index_json"]))
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files if name != "index_json"}
    return index, arrays


def plot_error_heatmaps(
    *,
    curves_path: Path,
    bootstrap: pd.DataFrame,
    run_metrics: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
) -> None:
    curve_index, arrays = _load_curves(curves_path)
    norm = mcolors.LogNorm(vmin=1e-8, vmax=10.0)
    for noise in _noise_values(run_metrics):
        maximum = min(
            5.0,
            float(run_metrics[np.isclose(run_metrics["noise_level"], noise)]["forecast_horizon_lt"].min()),
        )
        common = np.linspace(0.0, maximum, 301)
        figure, axes = plt.subplots(1, 3, figsize=(18, 6.2))
        image = None
        for column, track in enumerate(TRACKS):
            axis = axes[column]
            records = [
                record for record in curve_index
                if record["track"] == track
                and np.isclose(float(record["noise_level"]), noise)
            ]
            methods = _ordered_methods(config, track, noise, [record["method"] for record in records])
            matrix = []
            for method in methods:
                record = next(record for record in records if record["method"] == method)
                prefix = record["prefix"]
                matrix.append(np.interp(
                    common,
                    arrays[f"{prefix}_times_lt"],
                    arrays[f"{prefix}_median"],
                ))
            matrix_array = np.clip(np.asarray(matrix), 1e-12, 10.0)
            image = axis.imshow(
                matrix_array,
                aspect="auto",
                origin="upper",
                extent=(0.0, maximum, len(methods) - 0.5, -0.5),
                cmap="viridis",
                norm=norm,
                interpolation="nearest",
            )
            for position, method in enumerate(methods):
                vpt = float(_bootstrap_row(
                    bootstrap, track, noise, method, "vpt_restricted_lt"
                )["estimate"])
                marker_x = min(vpt, maximum)
                if vpt > maximum:
                    axis.scatter(
                        [marker_x], [position], marker=">", color="white",
                        edgecolor="black", linewidth=0.5, s=34, zorder=3,
                    )
                else:
                    axis.plot(
                        [marker_x], [position], marker="|", color="white",
                        markersize=7, markeredgewidth=1.2, zorder=3,
                    )
            axis.set_yticks(np.arange(len(methods)), [METHODS[method].label for method in methods])
            axis.set_ylim(len(methods) - 0.5, -0.5)
            axis.set_xlabel("Forecast time (Lyapunov times)")
            axis.set_title(TRACK_LABELS[track], fontsize=11)
            axis.grid(False)
        assert image is not None
        figure.subplots_adjust(wspace=0.52, right=0.89, top=0.86)
        colorbar_axis = figure.add_axes([0.92, 0.15, 0.012, 0.69])
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Median normalized squared error E(t)")
        figure.suptitle(
            f"Forecast-error growth | training noise {NOISE_DISPLAY.get(noise, noise)}",
            fontsize=15,
        )
        label = noise_label(noise)
        _save_figure(
            figure,
            output_dir,
            f"error_growth_heatmap__noise_{label}",
            formats,
            catalog,
            title=f"Forecast-error heatmap at noise {noise:g}",
            role="main" if noise == 0.0 else "supplement",
            description="Median log-scale error over time; white markers show restricted-mean VPT and arrows exceed 5 LT.",
        )


def _paired_contrast(
    trajectory_metrics: pd.DataFrame,
    left: str,
    right: str,
    noise: float,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float, np.ndarray]:
    keys = ["data_seed", "model_seed", "trajectory_id", "noise_level"]
    left_frame = trajectory_metrics[
        (trajectory_metrics["method"] == left)
        & np.isclose(trajectory_metrics["noise_level"], noise)
    ][keys + ["vpt_restricted_lt"]].rename(columns={"vpt_restricted_lt": "left"})
    right_frame = trajectory_metrics[
        (trajectory_metrics["method"] == right)
        & np.isclose(trajectory_metrics["noise_level"], noise)
    ][keys + ["vpt_restricted_lt"]].rename(columns={"vpt_restricted_lt": "right"})
    paired = left_frame.merge(right_frame, on=keys, validate="one_to_one")
    paired["delta"] = paired["left"] - paired["right"]
    paired["method"] = "contrast"
    result = paired_hierarchical_bootstrap(
        paired[["method", "data_seed", "model_seed", "trajectory_id", "delta"]],
        "delta",
        resamples=resamples,
        seed=seed,
    ).iloc[0]
    run_delta = paired.groupby(["data_seed", "model_seed"])["delta"].mean().to_numpy()
    return (
        float(result["estimate"]),
        float(result["ci_lower"]),
        float(result["ci_upper"]),
        run_delta,
    )


def plot_paired_effects(
    *,
    trajectory_metrics: pd.DataFrame,
    run_metrics: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> None:
    contrasts = [
        ("node_strong", "node_soft_dtw", "Strong NODE - Soft-DTW NODE"),
        ("node_strong", "node_weak", "Strong NODE - Weak NODE"),
        ("sindy_strong", "sindy_weighted", "Standard SINDy - Endpoint-weighted SINDy"),
        (
            "sindy_weak", "sindy_weak_weighted",
            "Weak-form SINDy - Endpoint-weighted Weak-form SINDy",
        ),
        ("lorenz_ad", "lorenz_ad_tapered", "AD Lorenz - Endpoint-tapered AD Lorenz"),
    ]
    noises = _noise_values(run_metrics)
    noise_colors = ["#000000", "#0072B2", "#009E73", "#D55E00"]
    offsets = np.linspace(-0.24, 0.24, len(noises))
    figure, axis = plt.subplots(figsize=(12.5, 8.2))
    for contrast_index, (left, right, label) in enumerate(contrasts):
        for noise_index, noise in enumerate(noises):
            estimate, lower, upper, raw = _paired_contrast(
                trajectory_metrics, left, right, noise,
                resamples=resamples, seed=seed,
            )
            y = contrast_index + offsets[noise_index]
            jitter = np.linspace(-0.06, 0.06, max(raw.size, 1))[: raw.size]
            axis.scatter(
                raw, y + jitter, color=noise_colors[noise_index], alpha=0.22,
                s=13, linewidth=0,
            )
            axis.plot([lower, upper], [y, y], color=noise_colors[noise_index], linewidth=2.1)
            axis.scatter(
                [estimate], [y], color=noise_colors[noise_index], edgecolor="black",
                linewidth=0.45, s=42, zorder=3,
                label=NOISE_DISPLAY.get(noise, str(noise)) if contrast_index == 0 else None,
            )
    axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    axis.set_yticks(np.arange(len(contrasts)), [item[2] for item in contrasts])
    axis.invert_yaxis()
    axis.set_xlabel("Paired difference in trajectory VPT (Lyapunov times); positive favors first method")
    axis.set_title("Matched method effects across training-noise levels", fontsize=14)
    axis.legend(title="Training noise", ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.20))
    _despine(axis)
    figure.subplots_adjust(left=0.34, bottom=0.22, top=0.90)
    _save_figure(
        figure,
        output_dir,
        "paired_vpt_effects",
        formats,
        catalog,
        title="Paired VPT effects",
        role="main",
        description="Trajectory-paired VPT differences with hierarchical-bootstrap 95% intervals and per-run points.",
    )


def plot_cost_accuracy(
    *,
    run_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
    noise: float = 0.0,
) -> None:
    clean = run_metrics[np.isclose(run_metrics["noise_level"], noise)]
    maximum_parameters = max(float(clean["parameter_count"].max()), 1.0)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for column, track in enumerate(TRACKS):
        axis = axes[column]
        group = clean[clean["track"] == track]
        methods = _ordered_methods(config, track, noise, group["method"].unique())
        for method in methods:
            method_frame = group[group["method"] == method]
            training_time = float(method_frame["training_wall_time_seconds"].median())
            parameter_count = float(method_frame["parameter_count"].median())
            size = 55.0 + 180.0 * np.sqrt(np.log10(parameter_count + 1.0) / np.log10(maximum_parameters + 1.0))
            record = _bootstrap_row(bootstrap, track, noise, method, "vpt_restricted_lt")
            estimate = float(record["estimate"])
            lower = float(record["ci_lower"])
            upper = float(record["ci_upper"])
            color = METHODS[method].color
            marker = KIND_MARKERS[METHODS[method].kind]
            axis.plot([training_time, training_time], [lower, upper], color=color, linewidth=1.7)
            axis.scatter(
                [training_time], [estimate], s=size, marker=marker, color=color,
                edgecolor="black", linewidth=0.6, label=METHODS[method].label, zorder=3,
            )
        training_times = group.groupby("method")["training_wall_time_seconds"].median().to_numpy(dtype=float)
        positive = training_times[training_times > 0.0]
        if positive.size == training_times.size and positive.max() / positive.min() < 10.0:
            axis.set_xlabel("Training wall time (seconds)")
        elif np.any(training_times == 0.0):
            axis.set_xscale("symlog", linthresh=0.1, linscale=0.7)
            axis.set_xlabel("Training wall time (seconds, symlog scale)")
        else:
            axis.set_xscale("log")
            axis.set_xlabel("Training wall time (seconds, log scale)")
        axis.set_ylabel("Restricted mean VPT (LT)")
        axis.set_title(TRACK_LABELS[track], fontsize=11)
        axis.legend(loc="best", fontsize=8)
        _despine(axis)
    figure.suptitle("Forecast skill versus training cost | noiseless training", fontsize=15)
    figure.subplots_adjust(wspace=0.28, top=0.86)
    _save_figure(
        figure,
        output_dir,
        "cost_accuracy_pareto__noise_0",
        formats,
        catalog,
        title="Cost-accuracy comparison",
        role="main",
        description="VPT versus measured training wall time; marker area encodes parameter count.",
    )


def _mean_wasserstein_bootstrap(
    run_metrics: pd.DataFrame,
    track: str,
    noise: float,
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    group = run_metrics[
        (run_metrics["track"] == track)
        & np.isclose(run_metrics["noise_level"], noise)
    ].copy()
    group["mean_coordinate_wasserstein"] = group[
        ["wasserstein_x", "wasserstein_y", "wasserstein_z"]
    ].mean(axis=1)
    group = group[group["mean_coordinate_wasserstein"].notna()].copy()
    group["trajectory_id"] = 0
    return paired_hierarchical_bootstrap(
        group[
            ["method", "data_seed", "model_seed", "trajectory_id", "mean_coordinate_wasserstein"]
        ],
        "mean_coordinate_wasserstein",
        resamples=resamples,
        seed=seed,
    )


def plot_post_divergence_statistics(
    *,
    run_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> None:
    metric_columns = [
        ("stability_fraction", "Stable fraction", False),
        ("rq_mmd", "RQ-MMD", True),
        ("mean_coordinate_wasserstein", "Mean Wasserstein-1", True),
        ("covariance_relative_error", "Covariance error", True),
        ("lyapunov_spectrum_relative_error", "Lyapunov-spectrum error", True),
    ]
    for noise in _noise_values(run_metrics):
        figure, axes = plt.subplots(3, 5, figsize=(22, 11), squeeze=False)
        for row, track in enumerate(TRACKS):
            group = run_metrics[
                (run_metrics["track"] == track)
                & np.isclose(run_metrics["noise_level"], noise)
                & run_metrics["stability_fraction"].notna()
            ]
            methods = _ordered_methods(config, track, noise, group["method"].unique())
            wasserstein = _mean_wasserstein_bootstrap(
                run_metrics, track, noise, resamples=resamples, seed=seed
            )
            for column, (metric, title, logarithmic) in enumerate(metric_columns):
                axis = axes[row, column]
                scale_values = []
                for position, method in enumerate(methods):
                    if metric == "mean_coordinate_wasserstein":
                        selected = wasserstein[wasserstein["method"] == method]
                        if selected.empty:
                            continue
                        record = selected.iloc[0]
                    else:
                        selected = bootstrap[
                            (bootstrap["track"] == track)
                            & np.isclose(bootstrap["noise_level"], noise)
                            & (bootstrap["method"] == method)
                            & (bootstrap["metric"] == metric)
                        ]
                        if selected.empty:
                            continue
                        record = selected.iloc[0]
                    scale_values.extend([
                        float(record["ci_lower"]),
                        float(record["estimate"]),
                        float(record["ci_upper"]),
                    ])
                    _method_point(
                        axis,
                        y=position,
                        method=method,
                        estimate=float(record["estimate"]),
                        lower=float(record["ci_lower"]),
                        upper=float(record["ci_upper"]),
                    )
                axis.set_yticks(
                    np.arange(len(methods)),
                    [METHODS[method].label for method in methods] if column == 0 else [],
                )
                axis.invert_yaxis()
                if logarithmic:
                    finite_scale = np.asarray(scale_values, dtype=float)
                    finite_scale = finite_scale[np.isfinite(finite_scale)]
                    if finite_scale.size and np.all(finite_scale > 0.0):
                        axis.set_xscale("log")
                        axis.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
                        axis.xaxis.set_minor_formatter(mticker.NullFormatter())
                    else:
                        positive_scale = finite_scale[finite_scale > 0.0]
                        threshold = max(
                            float(positive_scale.min()) * 0.1 if positive_scale.size else 1e-8,
                            1e-12,
                        )
                        axis.set_xscale("symlog", linthresh=threshold, linscale=0.6)
                else:
                    axis.set_xlim(0.95, 1.002)
                if row == 0:
                    axis.set_title(title, fontsize=10)
                if column == 0:
                    axis.set_ylabel(TRACK_LABELS[track])
                _despine(axis)
        figure.suptitle(
            f"Post-divergence autonomous-dynamics statistics | training noise {NOISE_DISPLAY.get(noise, noise)}",
            fontsize=15,
        )
        figure.subplots_adjust(wspace=0.36, hspace=0.42, left=0.13, top=0.91)
        label = noise_label(noise)
        _save_figure(
            figure,
            output_dir,
            f"post_divergence_statistics__noise_{label}",
            formats,
            catalog,
            title=f"Post-divergence statistics at noise {noise:g}",
            role="main" if noise == 0.0 else "supplement",
            description="Aligned bootstrap intervals for stability, attractor distributions, covariance, and Lyapunov spectrum.",
        )


def _ablation_summary(
    frame: pd.DataFrame, *, resamples: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (noise, count), group in frame.groupby(["noise_level", "trajectory_count"]):
        methods = sorted(group["method"].unique())
        data_seeds = sorted(int(value) for value in group["data_seed"].unique())
        pivot = group.pivot(index="data_seed", columns="method", values="restricted_mean_vpt_lt")
        pivot = pivot.reindex(index=data_seeds, columns=methods)
        if pivot.isna().any().any():
            raise ValueError("incomplete paired SINDy ablation matrix")
        values = pivot.to_numpy(dtype=float)
        indices = rng.integers(len(data_seeds), size=(resamples, len(data_seeds)))
        draws = values[indices].mean(axis=1)
        point = values.mean(axis=0)
        for method_index, method in enumerate(methods):
            rows.append({
                "noise_level": float(noise),
                "trajectory_count": int(count),
                "method": method,
                "estimate": float(point[method_index]),
                "ci_lower": float(np.quantile(draws[:, method_index], 0.025)),
                "ci_upper": float(np.quantile(draws[:, method_index], 0.975)),
            })
    return pd.DataFrame(rows)


def plot_sindy_sample_efficiency(
    *,
    ablation_metrics: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> None:
    frame = ablation_metrics.copy()
    frame["trajectory_count"] = frame["run_id"].str.extract(r"__k(\d+)$").astype(int)
    summary = _ablation_summary(frame, resamples=resamples, seed=seed)
    noises = sorted(summary["noise_level"].unique())
    figure, axes = plt.subplots(1, len(noises), figsize=(18, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, noise in zip(axes, noises):
        group = summary[np.isclose(summary["noise_level"], noise)]
        for method in ("sindy_strong", "sindy_weighted"):
            selected = group[group["method"] == method].sort_values("trajectory_count")
            x = selected["trajectory_count"].to_numpy(dtype=float)
            estimate = selected["estimate"].to_numpy(dtype=float)
            lower = selected["ci_lower"].to_numpy(dtype=float)
            upper = selected["ci_upper"].to_numpy(dtype=float)
            color = METHODS[method].color
            axis.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
            axis.plot(
                x, estimate, color=color, marker=KIND_MARKERS[METHODS[method].kind],
                linewidth=2.0, label=METHODS[method].label,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        axis.set_xlabel("Training trajectories")
        axis.set_title(f"Noise {NOISE_DISPLAY.get(float(noise), noise)}")
        _despine(axis)
    axes[0].set_ylabel("Restricted mean VPT (LT)")
    axes[-1].legend(loc="best", fontsize=8)
    figure.suptitle("Matched SINDy sample-efficiency ablation", fontsize=15)
    figure.subplots_adjust(wspace=0.12, top=0.83)
    _save_figure(
        figure,
        output_dir,
        "sindy_sample_efficiency",
        formats,
        catalog,
        title="SINDy sample efficiency",
        role="main",
        description="Standard and endpoint-weighted SINDy across 1, 4, 16, and 64 training trajectories.",
    )


def _load_attractor_summary(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as loaded:
        metadata = json.loads(str(loaded["metadata_json"]))
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files if name != "metadata_json"}
    return metadata, arrays


def _track_band_axes(
    figure: plt.Figure,
    method_counts: list[int],
    *,
    include_truth: bool,
) -> list[list[plt.Axes]]:
    maximum = max(count + int(include_truth) for count in method_counts)
    grid = figure.add_gridspec(3, maximum, hspace=0.42, wspace=0.30)
    result: list[list[plt.Axes]] = []
    for row, count in enumerate(method_counts):
        axes = [figure.add_subplot(grid[row, column]) for column in range(count + int(include_truth))]
        for column in range(count + int(include_truth), maximum):
            hidden = figure.add_subplot(grid[row, column])
            hidden.axis("off")
        result.append(axes)
    return result


def plot_attractor_density(
    *,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
) -> None:
    noise = float(metadata["noise_level"])
    methods_by_track = [
        _ordered_methods(
            config, track, noise,
            [method for method in metadata["methods"] if METHODS[method].track == track],
        )
        for track in TRACKS
    ]
    figure = plt.figure(figsize=(20, 10.5))
    rows = _track_band_axes(figure, [len(methods) for methods in methods_by_track], include_truth=True)
    x_edges = arrays["x_edges"]
    z_edges = arrays["z_edges"]
    truth = arrays["density_truth"]
    all_density = [truth, *[arrays[f"density__{method}"] for method in metadata["methods"]]]
    positive = np.concatenate([value[value > 0] for value in all_density if np.any(value > 0)])
    norm = mcolors.LogNorm(vmin=max(float(np.quantile(positive, 0.03)), 1e-8), vmax=float(np.max(positive)))
    image = None
    for row, (track, methods, axes) in enumerate(zip(TRACKS, methods_by_track, rows)):
        panels = [
            (
                "Reference truth",
                truth,
                float(arrays["density_in_range_fraction__truth"]),
            ),
            *[
                (
                    METHODS[method].label,
                    arrays[f"density__{method}"],
                    float(arrays[f"density_in_range_fraction__{method}"]),
                )
                for method in methods
            ],
        ]
        for axis, (title, density, in_range) in zip(axes, panels):
            image = axis.pcolormesh(
                x_edges, z_edges, density.T, cmap="magma", norm=norm,
                shading="auto", rasterized=True,
            )
            centers_x = 0.5 * (x_edges[:-1] + x_edges[1:])
            centers_z = 0.5 * (z_edges[:-1] + z_edges[1:])
            truth_max = float(truth.max())
            axis.contour(
                centers_x, centers_z, truth.T,
                levels=[0.08 * truth_max, 0.25 * truth_max],
                colors="white", linewidths=0.55, alpha=0.75,
            )
            axis.set_title(f"{title}\n{100.0 * in_range:.1f}% within truth bounds", fontsize=8.5)
            axis.set_xlabel("x")
            if axis is axes[0]:
                axis.set_ylabel(f"{TRACK_LABELS[track]}\nz")
            axis.grid(False)
    assert image is not None
    colorbar = figure.colorbar(image, ax=[axis for row in rows for axis in row], fraction=0.015, pad=0.015)
    colorbar.set_label("Balanced forecast-state probability density")
    start, stop = metadata["density_window_lt"]
    figure.suptitle(
        f"Forecast-state density from {start:g} to {stop:g} Lyapunov times | noiseless training",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.07)
    _save_figure(
        figure,
        output_dir,
        "forecast_state_density__noise_0",
        formats,
        catalog,
        title="Forecast-state density",
        role="diagnostic",
        description="Equal-run-weight x-z densities for every method over the common 2-5 LT forecast window.",
    )


def plot_return_maps(
    *,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
) -> None:
    autonomous = list(metadata["autonomous_methods"])
    noise = float(metadata["noise_level"])
    tracks = [
        track for track in TRACKS if any(METHODS[method].track == track for method in autonomous)
    ]
    methods_by_track = [
        _ordered_methods(
            config, track, noise,
            [method for method in autonomous if METHODS[method].track == track],
        )
        for track in tracks
    ]
    maximum = max(len(methods) + 1 for methods in methods_by_track)
    figure = plt.figure(figsize=(20, 3.5 * len(tracks)))
    grid = figure.add_gridspec(len(tracks), maximum, hspace=0.43, wspace=0.30)
    truth = arrays["return_truth"]
    limits = np.quantile(truth.reshape(-1), [0.002, 0.998]) if truth.size else np.asarray([25.0, 50.0])
    span = float(limits[1] - limits[0])
    lower, upper = float(limits[0] - 0.04 * span), float(limits[1] + 0.04 * span)
    for row, (track, methods) in enumerate(zip(tracks, methods_by_track)):
        panels: list[tuple[str, str | None]] = [("Reference truth", None)] + [
            (METHODS[method].label, method) for method in methods
        ]
        for column, (title, method) in enumerate(panels):
            axis = figure.add_subplot(grid[row, column])
            axis.scatter(
                truth[:, 0], truth[:, 1], s=4, color="#777777", alpha=0.10,
                linewidth=0, rasterized=True,
            )
            if method is not None:
                pairs = arrays[f"return__{method}"]
                axis.scatter(
                    pairs[:, 0], pairs[:, 1], s=5, color=METHODS[method].color,
                    alpha=0.20, linewidth=0, rasterized=True,
                )
            axis.plot([lower, upper], [lower, upper], color="#BBBBBB", linewidth=0.7)
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_title(title, fontsize=8.5)
            axis.set_xlabel("z maximum n")
            if column == 0:
                axis.set_ylabel(f"{TRACK_LABELS[track]}\nz maximum n+1")
            axis.grid(False)
        for column in range(len(panels), maximum):
            hidden = figure.add_subplot(grid[row, column])
            hidden.axis("off")
    figure.suptitle(
        f"Post-divergence Lorenz return maps | time >= {metadata['return_start_lt']:g} LT",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.08)
    _save_figure(
        figure,
        output_dir,
        "post_divergence_return_maps__noise_0",
        formats,
        catalog,
        title="Post-divergence Lorenz return maps",
        role="diagnostic",
        description="Consecutive z-maximum return maps for autonomous models after 5 LT; PINN flow maps are excluded by contract.",
    )


def plot_representative_forecasts(
    *,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
    catalog: list[dict[str, Any]],
) -> None:
    noise = float(metadata["noise_level"])
    methods_by_track = [
        _ordered_methods(
            config, track, noise,
            [method for method in metadata["methods"] if METHODS[method].track == track],
        )
        for track in TRACKS
    ]
    figure = plt.figure(figsize=(20, 9.5))
    rows = _track_band_axes(figure, [len(methods) for methods in methods_by_track], include_truth=False)
    times = arrays["representative_times_lt"]
    truth = arrays["representative_truth_x"]
    representative = metadata["representative"]
    all_predictions = np.concatenate([
        arrays[f"representative_prediction_x__{method}"] for method in metadata["methods"]
    ])
    finite = all_predictions[np.isfinite(all_predictions)]
    combined = np.concatenate([truth, finite])
    lower, upper = np.quantile(combined, [0.005, 0.995])
    padding = 0.08 * (upper - lower)
    for track, methods, axes in zip(TRACKS, methods_by_track, rows):
        for axis, method in zip(axes, methods):
            prediction = arrays[f"representative_prediction_x__{method}"]
            method_metadata = representative["methods"][method]
            vpt = float(method_metadata["vpt_restricted_lt"])
            axis.plot(times, truth, color="#222222", linewidth=1.25, label="Truth")
            axis.plot(times, prediction, color=METHODS[method].color, linewidth=1.15, label="Prediction")
            axis.axvline(min(vpt, float(times[-1])), color=METHODS[method].color, linestyle="--", linewidth=0.9)
            axis.set_xlim(float(times[0]), float(times[-1]))
            axis.set_ylim(float(lower - padding), float(upper + padding))
            axis.set_title(f"{METHODS[method].label}\nVPT {vpt:.2f} LT", fontsize=8.5)
            axis.set_xlabel("Forecast time (LT)")
            if axis is axes[0]:
                axis.set_ylabel(f"{TRACK_LABELS[track]}\nx")
            _despine(axis)
    rows[0][0].legend(loc="lower left", fontsize=8)
    figure.suptitle(
        "Paired representative forecast selected by median primary-track difficulty | noiseless training",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.07)
    _save_figure(
        figure,
        output_dir,
        "representative_forecasts__noise_0",
        formats,
        catalog,
        title="Paired representative forecasts",
        role="diagnostic",
        description=(
            f"Shared test cell d{representative['data_seed']}/m{representative['model_seed']}/"
            f"trajectory{representative['trajectory_id']}; dashed lines mark each method's VPT."
        ),
    )


def generate_survival_overlay(
    *,
    config_path: str | Path,
    results_dir: str | Path,
    output_dir: str | Path,
    formats: tuple[str, ...] = ("png", "pdf"),
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int | None = None,
) -> Path:
    invalid_formats = sorted(set(formats).difference({"png", "pdf"}))
    if invalid_formats:
        raise ValueError(f"unsupported figure formats: {invalid_formats}")
    config = load_config(config_path)
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics_path = results_dir / "run_metrics.csv"
    trajectory_metrics_path = results_dir / "trajectory_metrics.csv"
    run_metrics = pd.read_csv(run_metrics_path)
    trajectory_metrics = pd.read_csv(trajectory_metrics_path)
    resamples = int(
        bootstrap_resamples
        if bootstrap_resamples is not None
        else config["aggregation"]["bootstrap_resamples"]
    )
    seed = int(
        bootstrap_seed if bootstrap_seed is not None else config["aggregation"]["bootstrap_seed"]
    )
    _style()
    catalog: list[dict[str, Any]] = []
    plot_forecast_survival_overlay(
        trajectory_metrics=trajectory_metrics,
        run_metrics=run_metrics,
        config=config,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
        resamples=resamples,
        seed=seed,
    )
    artifacts = {
        Path(row["path"]).name: {
            "path": row["path"], "sha256": sha256_file(row["path"])
        }
        for row in catalog
    }
    manifest = {
        "schema": "lorenz63-survival-overlay-manifest-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "source_hash": source_hash(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "formats": list(formats),
        "noise_levels": _survival_overlay_noises(run_metrics),
        "track_line_styles": {
            PRIMARY_TRACK: "solid",
            PARAMETRIC_TRACK: "dashed",
            PINN_TRACK: "dotted",
        },
        "bootstrap": {"resamples": resamples, "seed": seed},
        "inputs": {
            "config": sha256_file(config_path),
            "run_metrics": sha256_file(run_metrics_path),
            "trajectory_metrics": sha256_file(trajectory_metrics_path),
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "forecast_survival_overlay.manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def generate_visualizations(
    *,
    config_path: str | Path,
    results_dir: str | Path,
    output_dir: str | Path,
    attractor_summary_path: str | Path,
    formats: tuple[str, ...] = ("png", "pdf"),
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int | None = None,
) -> Path:
    invalid_formats = sorted(set(formats).difference({"png", "pdf"}))
    if invalid_formats:
        raise ValueError(f"unsupported figure formats: {invalid_formats}")
    config = load_config(config_path)
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics_path = results_dir / "run_metrics.csv"
    trajectory_metrics_path = results_dir / "trajectory_metrics.csv"
    bootstrap_path = results_dir / "bootstrap_summary.csv"
    ablation_path = results_dir / "sindy_ablation_metrics.csv"
    curves_path = results_dir / "aggregate_curves.npz"
    attractor_summary_path = Path(attractor_summary_path)
    run_metrics = pd.read_csv(run_metrics_path)
    trajectory_metrics = pd.read_csv(trajectory_metrics_path)
    bootstrap = pd.read_csv(bootstrap_path)
    try:
        ablation = pd.read_csv(ablation_path)
    except pd.errors.EmptyDataError:
        ablation = pd.DataFrame()
    if ablation.empty and bool(config["aggregation"].get("require_complete_ablation", True)):
        raise ValueError("the configured paper visualization requires the SINDy ablation matrix")
    metadata, attractor_arrays = _load_attractor_summary(attractor_summary_path)
    resamples = int(
        bootstrap_resamples
        if bootstrap_resamples is not None
        else config["aggregation"]["bootstrap_resamples"]
    )
    seed = int(
        bootstrap_seed if bootstrap_seed is not None else config["aggregation"]["bootstrap_seed"]
    )
    _style()
    catalog: list[dict[str, Any]] = []
    plot_benchmark_overview(
        run_metrics=run_metrics,
        trajectory_metrics=trajectory_metrics,
        bootstrap=bootstrap,
        config=config,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
    )
    plot_noise_robustness(
        bootstrap=bootstrap,
        run_metrics=run_metrics,
        config=config,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
    )
    plot_forecast_survival(
        trajectory_metrics=trajectory_metrics,
        run_metrics=run_metrics,
        config=config,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
        resamples=resamples,
        seed=seed,
    )
    plot_error_heatmaps(
        curves_path=curves_path,
        bootstrap=bootstrap,
        run_metrics=run_metrics,
        config=config,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
    )
    plot_paired_effects(
        trajectory_metrics=trajectory_metrics,
        run_metrics=run_metrics,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
        resamples=resamples,
        seed=seed,
    )
    plot_cost_accuracy(
        run_metrics=run_metrics,
        bootstrap=bootstrap,
        config=config,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
    )
    plot_post_divergence_statistics(
        run_metrics=run_metrics,
        bootstrap=bootstrap,
        config=config,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
        resamples=resamples,
        seed=seed,
    )
    if not ablation.empty:
        plot_sindy_sample_efficiency(
            ablation_metrics=ablation,
            output_dir=output_dir,
            formats=formats,
            catalog=catalog,
            resamples=resamples,
            seed=seed,
        )
    plot_attractor_density(
        metadata=metadata,
        arrays=attractor_arrays,
        config=config,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
    )
    plot_return_maps(
        metadata=metadata,
        arrays=attractor_arrays,
        config=config,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
    )
    plot_representative_forecasts(
        metadata=metadata,
        arrays=attractor_arrays,
        config=config,
        output_dir=output_dir,
        formats=formats,
        catalog=catalog,
    )
    catalog_path = output_dir / "figure_catalog.csv"
    atomic_write_csv(catalog_path, catalog)
    artifact_records = {
        Path(row["path"]).name: {
            "path": row["path"], "sha256": sha256_file(row["path"])
        }
        for row in catalog
    }
    artifact_records["figure_catalog.csv"] = {
        "path": str(catalog_path.resolve()), "sha256": sha256_file(catalog_path)
    }
    manifest = {
        "schema": "lorenz63-visualization-manifest-v2",
        "status": "complete",
        "completed_at": utc_now(),
        "source_hash": source_hash(),
        "command": command_line(),
        "environment": environment_snapshot(),
        "formats": list(formats),
        "bootstrap": {"resamples": resamples, "seed": seed},
        "inputs": {
            "config": sha256_file(config_path),
            "run_metrics": sha256_file(run_metrics_path),
            "trajectory_metrics": sha256_file(trajectory_metrics_path),
            "bootstrap_summary": sha256_file(bootstrap_path),
            "sindy_ablation_metrics": sha256_file(ablation_path),
            "aggregate_curves": sha256_file(curves_path),
            "attractor_summary": sha256_file(attractor_summary_path),
        },
        "figure_count": len({row["figure"] for row in catalog}),
        "artifact_count": len(artifact_records),
        "artifacts": artifact_records,
    }
    manifest_path = output_dir / "visualization_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Lorenz63 v2 paper visualization package."
    )
    parser.add_argument("--config", default="configs/v2/lorenz63_frozen.json")
    parser.add_argument("--results-dir", default="results/v2")
    parser.add_argument("--output-dir", default="results/v2/visualizations")
    parser.add_argument(
        "--attractor-summary", default="results/v2/visualizations/attractor_summary.npz"
    )
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--survival-overlay-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.survival_overlay_only:
        path = generate_survival_overlay(
            config_path=args.config,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            formats=tuple(args.formats),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        path = generate_visualizations(
            config_path=args.config,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            attractor_summary_path=args.attractor_summary,
            formats=tuple(args.formats),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    print(path)


if __name__ == "__main__":
    main()
