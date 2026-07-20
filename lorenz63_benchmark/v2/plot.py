from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contracts import DEFAULT_ORDER, METHODS, TRACK_LABELS


def _method_order(
    track: str, available: list[str], config: dict[str, Any], noise_level: float
) -> list[str]:
    from .data import noise_label
    configured = (
        config.get("plot_order_by_noise", {}).get(noise_label(noise_level), {}).get(track)
        or config.get("plot_order", {}).get(track)
        or DEFAULT_ORDER.get(track, [])
    )
    ordered = [method for method in configured if method in available]
    default_tail = [
        method for method in DEFAULT_ORDER.get(track, [])
        if method in available and method not in ordered
    ]
    remaining = sorted(
        method for method in available if method not in ordered and method not in default_tail
    )
    return ordered + default_tail + remaining


def _violin(
    axis: plt.Axes,
    frame: pd.DataFrame,
    methods: list[str],
    metric: str,
    ylabel: str,
    *,
    censor: bool = False,
) -> None:
    positions = np.arange(len(methods))
    all_values = frame[metric].to_numpy(dtype=float)
    all_finite = all_values[np.isfinite(all_values)]
    overflow_level = max(1.0, 1.15 * float(np.max(all_finite))) if all_finite.size else 1.0
    for position, method in zip(positions, methods):
        raw_values = frame.loc[frame["method"] == method, metric].dropna().to_numpy(dtype=float)
        values = raw_values[np.isfinite(raw_values)]
        if values.size >= 2 and np.ptp(values) > 1e-14:
            parts = axis.violinplot(
                [values], positions=[position], widths=0.72,
                showmeans=False, showmedians=True, showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(METHODS[method].color)
                body.set_edgecolor("#202020")
                body.set_alpha(0.8)
        jitter = np.linspace(-0.12, 0.12, max(values.size, 1))[: values.size]
        axis.scatter(
            position + jitter, values, s=22, color=METHODS[method].color,
            edgecolor="black", linewidth=0.4, zorder=3,
        )
        infinite = np.isinf(raw_values)
        if np.any(infinite):
            infinite_jitter = np.linspace(-0.12, 0.12, max(raw_values.size, 1))[infinite]
            axis.scatter(
                position + infinite_jitter, np.full(np.count_nonzero(infinite), overflow_level),
                marker="x", s=42, color=METHODS[method].color, linewidth=1.4, zorder=4,
            )
        if censor and values.size:
            method_frame = frame.loc[frame["method"] == method]
            censoring = method_frame["vpt_censoring_fraction"].to_numpy(dtype=float)
            horizons = method_frame["forecast_horizon_lt"].to_numpy(dtype=float)
            censored = censoring > 0.0
            if np.any(censored):
                censor_jitter = np.linspace(-0.12, 0.12, max(censoring.size, 1))
                axis.scatter(
                    position + censor_jitter[censored], horizons[censored], marker="^",
                    s=30.0 + 30.0 * censoring[censored], facecolor="white",
                    edgecolor=METHODS[method].color, linewidth=1.2, zorder=4,
                )
    labels = []
    for method in methods:
        label = METHODS[method].label
        if censor:
            rate = frame.loc[frame["method"] == method, "vpt_censoring_fraction"].mean()
            label += f"\n({100.0 * rate:.0f}% censored)"
        infinite_count = int(np.isinf(frame.loc[frame["method"] == method, metric]).sum())
        if infinite_count:
            label += f"\n({infinite_count} infinite)"
        labels.append(label)
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)


def _load_curves(curves_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    with np.load(curves_path, allow_pickle=False) as loaded:
        index = json.loads(str(loaded["index_json"]))
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files if name != "index_json"}
    return index, arrays


def make_track_figures(
    run_metrics_path: str | Path,
    curves_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    plt.rcParams.update({
        "axes.grid": False, "figure.dpi": 130, "font.size": 10,
        "axes.titleweight": "semibold", "legend.frameon": False,
    })
    metrics = pd.read_csv(run_metrics_path)
    curve_index, curve_arrays = _load_curves(curves_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for (track, noise), frame in metrics.groupby(["track", "noise_level"]):
        available = sorted(frame["method"].astype(str).unique())
        methods = _method_order(str(track), available, config, float(noise))
        auc_columns = [
            column for column in frame.columns
            if column.startswith("median_nrmse_auc_0_") and not frame[column].isna().all()
        ]
        if not auc_columns:
            raise ValueError("no finite normalized-RMSE AUC is available for plotting")
        metric_panels = [(auc_columns[-1], "Median normalized-RMSE AUC", "Forecast error per trained run")]
        if str(track) == "known_physics_forward_surrogate":
            domain_columns = [
                ("median_pinn_in_domain_auc_0_2LT", "In-domain normalized-RMSE AUC (0-2 LT)"),
                ("median_pinn_extrapolation_auc_2_5LT", "Extrapolation normalized-RMSE AUC (2-5 LT)"),
            ]
            finite_domain = [
                (column, label, label) for column, label in domain_columns
                if column in frame and not frame[column].isna().all()
            ]
            if finite_domain:
                metric_panels = finite_domain
        row_count = 2 + len(metric_panels)
        figure, axes = plt.subplots(
            row_count, 1, figsize=(max(10.5, 1.8 * len(methods)), 4.5 * row_count)
        )
        figure.suptitle(
            f"{TRACK_LABELS.get(str(track), str(track))} | training noise {100.0 * float(noise):g}%",
            fontsize=15,
        )
        _violin(
            axes[0], frame, methods, "restricted_mean_vpt_lt",
            "Restricted mean VPT (Lyapunov times)", censor=True,
        )
        axes[0].set_title("Short-horizon predictability")
        for panel_index, (column, ylabel, title) in enumerate(metric_panels, start=1):
            _violin(axes[panel_index], frame, methods, column, ylabel)
            axes[panel_index].set_title(title)
        axis = axes[-1]
        plot_max = float(config["evaluation"].get("error_plot_max", 100.0))
        for method in methods:
            records = [
                record for record in curve_index
                if record["track"] == track and abs(float(record["noise_level"]) - float(noise)) < 1e-15
                and record["method"] == method
            ]
            if not records:
                continue
            prefix = records[0]["prefix"]
            times = curve_arrays[f"{prefix}_times_lt"]
            median = np.clip(np.nan_to_num(
                curve_arrays[f"{prefix}_median"], nan=plot_max, posinf=plot_max
            ), 1e-12, plot_max)
            q25 = np.clip(np.nan_to_num(
                curve_arrays[f"{prefix}_q25"], nan=plot_max, posinf=plot_max
            ), 1e-12, plot_max)
            q75 = np.clip(np.nan_to_num(
                curve_arrays[f"{prefix}_q75"], nan=plot_max, posinf=plot_max
            ), 1e-12, plot_max)
            color = METHODS[method].color
            axis.plot(times, median, color=color, linewidth=2.0, label=METHODS[method].label)
            axis.fill_between(times, q25, q75, color=color, alpha=0.15, linewidth=0)
        axis.axhline(
            float(config["evaluation"]["vpt_threshold"]), color="#555555",
            linestyle="--", linewidth=1.0, label="VPT threshold",
        )
        axis.set_yscale("log")
        axis.set_ylim(1e-12, plot_max * 1.1)
        axis.set_xlabel("Forecast time (Lyapunov times)")
        axis.set_ylabel(f"Normalized squared error (visual clipping at {plot_max:g})")
        axis.set_title("Error versus forecast time")
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            ncol=min(3, len(methods)), loc="upper center",
            bbox_to_anchor=(0.5, -0.18), borderaxespad=0.0,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.975))
        noise_label = f"{float(noise):.6g}".replace(".", "p")
        path = output_dir / f"{track}__noise_{noise_label}.png"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)

        long_frame = frame.copy()
        wasserstein_columns = [
            column for column in ("wasserstein_x", "wasserstein_y", "wasserstein_z")
            if column in long_frame
        ]
        if wasserstein_columns:
            long_frame["mean_coordinate_wasserstein"] = long_frame[wasserstein_columns].mean(axis=1)
        long_metrics = [
            ("rq_mmd", "Rational-quadratic MMD"),
            ("mean_coordinate_wasserstein", "Mean coordinate Wasserstein-1"),
            ("covariance_relative_error", "Covariance relative error"),
            ("lyapunov_spectrum_relative_error", "Lyapunov-spectrum relative error"),
        ]
        long_metrics = [
            (column, label) for column, label in long_metrics
            if column in long_frame and long_frame.groupby("method")[column].count().gt(0).sum() >= 2
        ]
        if long_metrics:
            long_figure, long_axes = plt.subplots(
                len(long_metrics), 1,
                figsize=(max(10.5, 1.8 * len(methods)), 4.2 * len(long_metrics)),
            )
            long_axes = np.atleast_1d(long_axes)
            long_figure.suptitle(
                f"{TRACK_LABELS.get(str(track), str(track))} | long-run dynamics",
                fontsize=15,
            )
            for long_axis, (column, label) in zip(long_axes, long_metrics):
                _violin(long_axis, long_frame, methods, column, label)
                long_axis.set_title(label)
            long_figure.tight_layout(rect=(0, 0, 1, 0.975))
            long_path = output_dir / f"{track}__noise_{noise_label}__long_run.png"
            long_figure.savefig(long_path, dpi=220, bbox_inches="tight")
            plt.close(long_figure)
            outputs.append(long_path)
    return outputs
