"""Report/dashboard figure export (Sprint 5.1).

Four pure plot functions, each takes already-loaded DataFrames/Series and
returns a Matplotlib `Figure` -- no file I/O, no `plt.show()`, no global
backend/style side effects (those belong to the thin I/O layer,
`scripts/export_report_assets.py`). `plot_coverage_forest`, `plot_fan_chart`,
and `plot_reliability` are ported verbatim (minus one unused parameter) from
the plot-library cell in `04_regime_risk.ipynb`, where they were notebook-local,
not importable from the package (see the corrected Decision 3 in the Sprint-5.1
spec). `plot_forecast_vs_actual` is a new, minimal function styled after the
"Forecast vs. Actual" section of `02_baselines.ipynb`, now parametrised for the
production model rather than the Sprint-2 baselines.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

VARIANT_COLORS: dict[str, str] = {
    "raw": "#9e9e9e",
    "calibrated": "#1f77b4",
    "fhs": "#ff7f0e",
}
SIDE_MARKERS: dict[str, str] = {"long": "o", "short": "s"}
MODEL_COLORS: dict[str, str] = {
    "similarday_naive": "#9e9e9e",
    "lasso": "#2ca02c",
    "lightgbm": "#1f77b4",
    "arimax": "#d62728",
}
LOW_SUPPORT_LABEL = "open marker = low support; the CI is illustrative, not inferential"


def plot_coverage_forest(coverage: pd.DataFrame, *, alpha: float) -> Figure:
    """The signature figure of the model validation (Notebook 04 / model_validation_report.md).
    Forest plot of breach_rate with bootstrap CI, one row per (variant, side, subset),
    vertical reference line at `alpha`.

    Rows whose CI excludes alpha are highlighted. `low_support` rows get an OPEN
    marker and a DASHED CI bar -- never dropped. `raw` rows are drawn greyed
    and labelled as the counterfactual they are.
    """
    subsets = list(dict.fromkeys(coverage["subset"]))
    combos = [("calibrated", "long"), ("calibrated", "short"), ("fhs", "long"), ("fhs", "short")]
    n_sub = len(subsets)
    fig, ax = plt.subplots(figsize=(9, 0.42 * n_sub * len(combos) + 1.5))

    y: float = 0
    yticks: list[float] = []
    yticklabels: list[str] = []
    for subset in subsets:
        y0 = y
        for variant, side in combos:
            row = coverage[
                (coverage["subset"] == subset)
                & (coverage["variant"] == variant)
                & (coverage["side"] == side)
            ]
            if row.empty:
                y += 1
                continue
            r = row.iloc[0]
            low_support = bool(r["low_support"])
            excludes = r.get("breach_rate_ci_excludes_alpha")
            color = VARIANT_COLORS[variant]
            marker = SIDE_MARKERS[side]
            facecolor = color if not low_support else "none"
            linestyle = "dashed" if low_support else "solid"
            if pd.notna(r["breach_rate_ci_low"]):
                ax.plot(
                    [r["breach_rate_ci_low"], r["breach_rate_ci_high"]],
                    [y, y],
                    color=color,
                    linewidth=2.2 if excludes is True else 1.2,
                    linestyle=linestyle,
                    alpha=0.9,
                )
            ax.scatter(
                [r["breach_rate"]],
                [y],
                marker=marker,
                facecolor=facecolor,
                edgecolor=color,
                s=32,
                zorder=3,
            )
            raw_row = coverage[
                (coverage["subset"] == subset)
                & (coverage["variant"] == "raw")
                & (coverage["side"] == side)
            ]
            if not raw_row.empty and variant == "calibrated":
                ax.scatter(
                    [raw_row.iloc[0]["breach_rate"]],
                    [y],
                    marker=marker,
                    facecolor="none",
                    edgecolor=VARIANT_COLORS["raw"],
                    s=20,
                    zorder=2,
                    linestyle="dotted",
                )
            y += 1
        yticks.append((y0 + y - 1) / 2)
        yticklabels.append(subset)
        y += 0.6

    ax.axvline(alpha, color="black", linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()
    ax.set_xlabel("breach_rate")
    ax.set_title(f"Coverage forest plot (reference line at alpha={alpha:.2f})")
    handles = [
        plt.Line2D(
            [0], [0], color=VARIANT_COLORS["calibrated"], marker="o", label="calibrated, long"
        ),
        plt.Line2D(
            [0], [0], color=VARIANT_COLORS["calibrated"], marker="s", label="calibrated, short"
        ),
        plt.Line2D([0], [0], color=VARIANT_COLORS["fhs"], marker="o", label="fhs, long"),
        plt.Line2D([0], [0], color=VARIANT_COLORS["fhs"], marker="s", label="fhs, short"),
        plt.Line2D(
            [0],
            [0],
            color=VARIANT_COLORS["raw"],
            marker="o",
            linestyle="dotted",
            label="raw (counterfactual)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="grey",
            marker="o",
            markerfacecolor="none",
            linestyle="dashed",
            label=LOW_SUPPORT_LABEL,
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncols=1)
    fig.tight_layout()
    return fig


def plot_fan_chart(
    quantiles: dict[float, pd.Series], y_true: pd.Series, *, title: str, ax: Axes | None = None
) -> Figure:
    """Nested quantile bands (0.05-0.95, 0.10-0.90, 0.25-0.75) with the realised
    price on top. Accepts an optional `ax` so a before/after grid can reuse it.
    Returns the parent Figure.
    """
    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.2))
    else:
        assert isinstance(ax.figure, Figure)
        fig = ax.figure
    idx = y_true.index
    bands = [(0.05, 0.95, 0.15), (0.10, 0.90, 0.25), (0.25, 0.75, 0.4)]
    for lo, hi, band_alpha in bands:
        if lo in quantiles and hi in quantiles:
            ax.fill_between(
                idx,
                quantiles[lo].reindex(idx),
                quantiles[hi].reindex(idx),
                color=VARIANT_COLORS["calibrated"],
                alpha=band_alpha,
                linewidth=0,
            )
    if 0.5 in quantiles:
        ax.plot(idx, quantiles[0.5].reindex(idx), color=VARIANT_COLORS["calibrated"], linewidth=1)
    ax.plot(idx, y_true, color="black", linewidth=1.3, label="realised price")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("EUR/MWh")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    if owns_fig:
        fig.tight_layout()
    return fig


def plot_reliability(curves: dict[str, pd.DataFrame]) -> Figure:
    """Reliability diagram: nominal vs empirical coverage, ideal diagonal.

    `curves` maps artefact label ("raw", "calibrated_sorted", "arimax") -> a
    frame indexed by `level` with an `empirical` column. ARIMAX is drawn as a
    pale reference curve, never as a competitor.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1, label="ideal")
    styles: dict[str, dict[str, str | float]] = {
        "raw": {"color": VARIANT_COLORS["raw"], "marker": "o"},
        "calibrated_sorted": {"color": VARIANT_COLORS["calibrated"], "marker": "o"},
        "arimax": {"color": MODEL_COLORS["arimax"], "marker": "^", "alpha": 0.45},
    }
    for label, frame in curves.items():
        style: dict[str, str | float] = styles.get(label, {})
        ax.plot(
            frame.index,
            frame["empirical"],
            label=label,
            linewidth=2 if label != "arimax" else 1.5,
            **style,  # type: ignore[arg-type]
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal one-sided coverage")
    ax.set_ylabel("empirical one-sided coverage")
    ax.set_title("Reliability: nominal vs. empirical coverage")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_forecast_vs_actual(
    y_true: pd.Series, y_pred: pd.Series, *, model_label: str = "LightGBM"
) -> Figure:
    """Daily-mean actual price vs. forecast median over the full test period.

    Minimal, parametrised version of `02_baselines.ipynb`'s "Forecast vs.
    Actual -- Time Series View" section, now for the production model
    instead of the Sprint-2 baselines. Makes the crisis-period regime shift
    visually legible.
    """
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    daily = frame.resample("D").mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["y_true"], color="black", linewidth=0.8, label="Actual")
    ax.plot(
        daily.index,
        daily["y_pred"],
        color=VARIANT_COLORS["calibrated"],
        linewidth=0.8,
        alpha=0.85,
        label=model_label,
    )
    ax.set_ylabel("Day-ahead price (EUR/MWh, daily mean)")
    ax.set_title("Actual vs. forecast -- daily means")
    ax.legend()
    fig.tight_layout()
    return fig
