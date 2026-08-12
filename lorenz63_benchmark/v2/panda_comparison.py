from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aggregate import paired_hierarchical_bootstrap
from .artifacts import atomic_write_csv
from .contracts import METHODS
from .visualize import hierarchical_survival_curves


COMPARISON_METHODS = (
    "sindy_weak_weighted",
    "sindy_weak",
    "panda_zero_shot",
)
LINESTYLES = {
    "sindy_weak_weighted": "-",
    "sindy_weak": "-",
    "panda_zero_shot": (0, (6, 2.2)),
}


def _despine(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)


def _run_means(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return frame.groupby(
        ["run_id", "method", "data_seed", "model_seed"], as_index=False
    )[metric].mean()


def _violin(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    methods: tuple[str, ...],
    ylabel: str,
) -> None:
    finite_values = frame[metric].to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    overflow = 1.1 * float(finite_values.max()) if finite_values.size else 1.0
    for position, method in enumerate(methods):
        raw = frame.loc[frame["method"] == method, metric].to_numpy(dtype=float)
        values = raw[np.isfinite(raw)]
        if values.size >= 2 and np.ptp(values) > 1e-14:
            parts = axis.violinplot(
                [values], positions=[position], widths=0.7,
                showmeans=False, showmedians=True, showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(METHODS[method].color)
                body.set_edgecolor("#202020")
                body.set_alpha(0.72)
        jitter = np.linspace(-0.13, 0.13, max(values.size, 1))[: values.size]
        axis.scatter(
            position + jitter, values, color=METHODS[method].color,
            edgecolor="black", linewidth=0.35, s=25, zorder=3,
        )
        nonfinite = ~np.isfinite(raw)
        if np.any(nonfinite):
            bad_jitter = np.linspace(-0.13, 0.13, max(raw.size, 1))[nonfinite]
            axis.scatter(
                position + bad_jitter, np.full(np.count_nonzero(nonfinite), overflow),
                marker="x", color=METHODS[method].color, s=40, linewidth=1.3,
            )
    axis.set_xticks(
        np.arange(len(methods)),
        [METHODS[method].label for method in methods],
        rotation=17,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    _despine(axis)


def _load_curves(path: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as loaded:
        index = json.loads(str(loaded["index_json"]))
        arrays = {
            key: np.asarray(loaded[key]) for key in loaded.files if key != "index_json"
        }
    return index, arrays


def _bootstrap_summary(
    trajectory_metrics: pd.DataFrame,
    run_metrics: pd.DataFrame,
    methods: tuple[str, ...],
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {
        method: {"method": method, "label": METHODS[method].label}
        for method in methods
    }
    trajectory_columns = {
        "vpt_restricted_lt": "restricted_mean_vpt_lt",
        "nrmse_auc_0_1LT": "nrmse_auc_0_1LT",
        "nrmse_auc_0_2LT": "nrmse_auc_0_2LT",
        "nrmse_auc_0_5LT": "nrmse_auc_0_5LT",
    }
    for source, label in trajectory_columns.items():
        if source not in trajectory_metrics:
            continue
        summary = paired_hierarchical_bootstrap(
            trajectory_metrics, source, resamples=resamples, seed=seed
        ).set_index("method")
        for method in methods:
            rows[method][label] = float(summary.loc[method, "estimate"])
            rows[method][f"{label}_ci_lower"] = float(summary.loc[method, "ci_lower"])
            rows[method][f"{label}_ci_upper"] = float(summary.loc[method, "ci_upper"])

    run_frame = run_metrics.copy()
    run_frame["trajectory_id"] = 0
    run_frame["mean_coordinate_wasserstein"] = run_frame[
        ["wasserstein_x", "wasserstein_y", "wasserstein_z"]
    ].mean(axis=1)
    for metric in ("rq_mmd", "mean_coordinate_wasserstein", "covariance_relative_error"):
        observed = run_frame[run_frame[metric].notna()]
        summary = paired_hierarchical_bootstrap(
            observed, metric, resamples=resamples, seed=seed
        ).set_index("method")
        for method in methods:
            rows[method][metric] = float(summary.loc[method, "estimate"])
            rows[method][f"{metric}_ci_lower"] = float(summary.loc[method, "ci_lower"])
            rows[method][f"{metric}_ci_upper"] = float(summary.loc[method, "ci_upper"])
    for method in methods:
        selected = run_metrics[run_metrics["method"] == method]
        rows[method]["vpt_censoring_fraction"] = float(
            selected["vpt_censoring_fraction"].mean()
        )
        rows[method]["forecast_context_steps"] = int(
            selected["forecast_context_steps"].iloc[0]
        )
        rows[method]["run_count"] = int(len(selected))
    return pd.DataFrame([rows[method] for method in methods])


def make_panda_comparison_figures(
    *,
    run_metrics: pd.DataFrame,
    trajectory_metrics: pd.DataFrame,
    curves_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> tuple[list[Path], Path]:
    available = set(run_metrics["method"].astype(str))
    missing = set(COMPARISON_METHODS).difference(available)
    if missing:
        raise ValueError(f"Panda comparison is missing methods: {sorted(missing)}")
    noise_values = sorted(float(value) for value in run_metrics["noise_level"].unique())
    if len(noise_values) != 1:
        raise ValueError("the Panda comparison artifact currently requires one noise level")
    noise = noise_values[0]
    selected_runs = run_metrics[run_metrics["method"].isin(COMPARISON_METHODS)].copy()
    selected_trajectories = trajectory_metrics[
        trajectory_metrics["method"].isin(COMPARISON_METHODS)
    ].copy()
    context_values = selected_runs["forecast_context_steps"].unique()
    if len(context_values) != 1:
        raise ValueError(f"comparison methods used different context lengths: {context_values}")

    resamples = int(config["aggregation"]["bootstrap_resamples"])
    seed = int(config["aggregation"]["bootstrap_seed"])
    summary = _bootstrap_summary(
        selected_trajectories,
        selected_runs,
        COMPARISON_METHODS,
        resamples=resamples,
        seed=seed,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "panda_vs_weak_sindy_summary.csv"
    atomic_write_csv(summary_path, summary.to_dict(orient="records"))

    plt.rcParams.update({
        "axes.grid": False,
        "axes.titleweight": "semibold",
        "font.size": 9,
        "legend.frameon": False,
        "figure.dpi": 130,
    })
    figure, axes = plt.subplots(2, 3, figsize=(18, 10.5))

    run_vpt = _run_means(selected_trajectories, "vpt_restricted_lt")
    _violin(axes[0, 0], run_vpt, "vpt_restricted_lt", COMPARISON_METHODS, "Restricted mean VPT (LT)")
    censor_labels = {
        row.method: f"{100.0 * row.vpt_censoring_fraction:.0f}% censored"
        for row in summary.itertuples(index=False)
    }
    axes[0, 0].set_xticklabels([
        f"{METHODS[method].label}\n{censor_labels[method]}" for method in COMPARISON_METHODS
    ], rotation=17, ha="right")
    axes[0, 0].set_title("A  Valid prediction time")

    maximum = min(5.0, float(selected_runs["forecast_horizon_lt"].min()))
    survival_times = np.linspace(0.0, maximum, 121)
    survival = hierarchical_survival_curves(
        selected_trajectories,
        survival_times,
        resamples=resamples,
        seed=seed,
    )
    for method in COMPARISON_METHODS:
        record = survival[method]
        axes[0, 1].fill_between(
            survival_times, record["ci_lower"], record["ci_upper"],
            color=METHODS[method].color, alpha=0.11, linewidth=0,
        )
        axes[0, 1].plot(
            survival_times, record["estimate"], color=METHODS[method].color,
            linestyle=LINESTYLES[method], linewidth=2.1, label=METHODS[method].label,
        )
    axes[0, 1].set_xlim(0.0, maximum)
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_xlabel("Forecast time (Lyapunov times)")
    axes[0, 1].set_ylabel("Fraction below E(t) = 0.4")
    axes[0, 1].set_title("B  Valid-forecast survival")
    axes[0, 1].legend(loc="best", fontsize=8)
    _despine(axes[0, 1])

    curve_index, curve_arrays = _load_curves(Path(curves_path))
    plot_max = float(config["evaluation"].get("error_plot_max", 100.0))
    for method in COMPARISON_METHODS:
        records = [
            record for record in curve_index
            if record["method"] == method and np.isclose(float(record["noise_level"]), noise)
        ]
        if len(records) != 1:
            raise ValueError(f"expected one aggregate curve for {method}, found {len(records)}")
        prefix = records[0]["prefix"]
        times = curve_arrays[f"{prefix}_times_lt"]
        keep = times <= maximum + 1e-12
        median = np.clip(
            np.nan_to_num(curve_arrays[f"{prefix}_median"], nan=plot_max, posinf=plot_max),
            1e-10,
            plot_max,
        )
        q25 = np.clip(
            np.nan_to_num(curve_arrays[f"{prefix}_q25"], nan=plot_max, posinf=plot_max),
            1e-10,
            plot_max,
        )
        q75 = np.clip(
            np.nan_to_num(curve_arrays[f"{prefix}_q75"], nan=plot_max, posinf=plot_max),
            1e-10,
            plot_max,
        )
        axes[0, 2].plot(
            times[keep], median[keep], color=METHODS[method].color,
            linestyle=LINESTYLES[method], linewidth=2.1, label=METHODS[method].label,
        )
        axes[0, 2].fill_between(
            times[keep], q25[keep], q75[keep], color=METHODS[method].color,
            alpha=0.11, linewidth=0,
        )
    axes[0, 2].axhline(0.4, color="#555555", linestyle=":", linewidth=1.2)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlim(0.0, maximum)
    axes[0, 2].set_ylim(1e-8, plot_max * 1.1)
    axes[0, 2].set_xlabel("Forecast time (Lyapunov times)")
    axes[0, 2].set_ylabel("Normalized squared error E(t)")
    axes[0, 2].set_title("C  Error growth")
    _despine(axes[0, 2])

    run_auc = _run_means(selected_trajectories, "nrmse_auc_0_2LT")
    _violin(axes[1, 0], run_auc, "nrmse_auc_0_2LT", COMPARISON_METHODS, "Normalized-RMSE AUC, 0-2 LT")
    axes[1, 0].set_title("D  Early-horizon error")

    selected_runs["mean_coordinate_wasserstein"] = selected_runs[
        ["wasserstein_x", "wasserstein_y", "wasserstein_z"]
    ].mean(axis=1)
    _violin(axes[1, 1], selected_runs, "rq_mmd", COMPARISON_METHODS, "Rational-quadratic MMD")
    axes[1, 1].set_title("E  Post-divergence distribution")
    _violin(
        axes[1, 2], selected_runs, "mean_coordinate_wasserstein",
        COMPARISON_METHODS, "Mean coordinate Wasserstein-1",
    )
    axes[1, 2].set_title("F  Attractor marginals")

    context_steps = int(context_values[0])
    figure.suptitle(
        "Panda versus weak-form SINDy on Lorenz63\n"
        f"Noiseless, context-aligned forecasts ({context_steps} observed samples; 95% hierarchical-bootstrap bands)",
        fontsize=15,
    )
    figure.subplots_adjust(wspace=0.31, hspace=0.47, top=0.88, bottom=0.12)
    outputs = []
    stem = output_dir / "panda_vs_weak_sindy__noise_0"
    for suffix in ("png", "svg"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs, summary_path
