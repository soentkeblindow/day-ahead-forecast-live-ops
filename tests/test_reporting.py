from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.figure import Figure

from energy_price_forecast.evaluation.regimes import MACRO_REGIME_COLUMN, REGIME_FLAG_COLUMNS
from energy_price_forecast.reporting.assets import (
    plot_coverage_forest,
    plot_fan_chart,
    plot_forecast_vs_actual,
    plot_reliability,
)
from energy_price_forecast.reporting.snapshot import SNAPSHOT_COLUMNS, build_snapshot
from energy_price_forecast.reporting.tables import (
    assert_model_comparison_reconciles,
    backtest_coverage_export,
    coverage_summary,
    model_comparison,
    model_comparison_by_regime,
    risk_headline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def _regime_flags(idx: pd.DatetimeIndex) -> pd.DataFrame:
    data: dict[str, object] = {col: [False] * len(idx) for col in REGIME_FLAG_COLUMNS}
    data["normal"] = [True] * len(idx)
    data[MACRO_REGIME_COLUMN] = pd.Categorical(["calm"] * len(idx))
    return pd.DataFrame(data, index=idx)


def _clean_snapshot_inputs(
    idx: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    n = len(idx)
    return {
        "price_actual": pd.Series([50.0 + i for i in range(n)], index=idx),
        "forecast_median": pd.Series([50.0 + i for i in range(n)], index=idx),
        "lo_raw": pd.Series([40.0 + i for i in range(n)], index=idx),
        "hi_raw": pd.Series([60.0 + i for i in range(n)], index=idx),
        "lo_calibrated": pd.Series([35.0 + i for i in range(n)], index=idx),
        "hi_calibrated": pd.Series([65.0 + i for i in range(n)], index=idx),
    }


# ---------------------------------------------------------------------------
# build_snapshot
# ---------------------------------------------------------------------------


def test_build_snapshot_basic_schema_and_dtypes() -> None:
    idx = _hourly_utc("2021-01-01", 6)
    inputs = _clean_snapshot_inputs(idx)
    frame, stats = build_snapshot(**inputs, regime_flags=_regime_flags(idx))

    assert list(frame.columns) == list(SNAPSHOT_COLUMNS)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert str(frame.index.tz) == "UTC"
    assert frame.index.is_unique
    for col in REGIME_FLAG_COLUMNS:
        assert frame[col].dtype == bool
    assert isinstance(frame[MACRO_REGIME_COLUMN].dtype, pd.CategoricalDtype)
    assert stats.n_rows == 6
    assert stats.n_dropped_index_mismatch == 0
    assert stats.n_dropped_missing_required == 0
    assert stats.crossing_rate_raw == 0.0
    assert stats.crossing_rate_calibrated == 0.0


def test_build_snapshot_drops_small_index_mismatch() -> None:
    idx = _hourly_utc("2021-01-01", 100)
    inputs = _clean_snapshot_inputs(idx)
    # Drop the first hour from one series only -- a small (<1%) mismatch.
    inputs["hi_raw"] = inputs["hi_raw"].iloc[1:]

    frame, stats = build_snapshot(**inputs, regime_flags=_regime_flags(idx))

    assert stats.n_dropped_index_mismatch == 1
    assert stats.n_rows == 99
    assert frame.index.min() == idx[1]


def test_build_snapshot_raises_on_large_index_mismatch() -> None:
    idx = _hourly_utc("2021-01-01", 10)
    inputs = _clean_snapshot_inputs(idx)
    inputs["hi_raw"] = inputs["hi_raw"].iloc[:5]  # 50% mismatch

    with pytest.raises(ValueError, match="Index mismatch"):
        build_snapshot(**inputs, regime_flags=_regime_flags(idx))


def test_build_snapshot_drops_rows_missing_required_columns() -> None:
    idx = _hourly_utc("2021-01-01", 5)
    inputs = _clean_snapshot_inputs(idx)
    inputs["price_actual"] = inputs["price_actual"].copy()
    inputs["price_actual"].iloc[2] = float("nan")

    frame, stats = build_snapshot(**inputs, regime_flags=_regime_flags(idx))

    assert stats.n_dropped_missing_required == 1
    assert stats.n_rows == 4
    assert idx[2] not in frame.index


def test_build_snapshot_measures_crossing_without_repairing() -> None:
    idx = _hourly_utc("2021-01-01", 4)
    inputs = _clean_snapshot_inputs(idx)
    # Force a crossing violation on the raw band for one hour.
    inputs["lo_raw"] = inputs["lo_raw"].copy()
    inputs["lo_raw"].iloc[0] = 999.0  # lo_raw > forecast_median -> crossing

    frame, stats = build_snapshot(**inputs, regime_flags=_regime_flags(idx))

    assert stats.crossing_rate_raw == pytest.approx(0.25)
    assert stats.crossing_rate_calibrated == 0.0
    # Not repaired: the crossing value survives untouched in the output.
    assert frame.loc[idx[0], "lo_raw"] == 999.0


def test_build_snapshot_raises_on_non_utc_index() -> None:
    idx_naive = pd.date_range("2021-01-01", periods=4, freq="h")
    inputs = _clean_snapshot_inputs(idx_naive)
    with pytest.raises(ValueError, match="UTC"):
        build_snapshot(**inputs, regime_flags=_regime_flags(idx_naive))


def test_build_snapshot_raises_on_non_hourly_index() -> None:
    idx = pd.DatetimeIndex(
        ["2021-01-01T00:00Z", "2021-01-01T01:00Z", "2021-01-01T03:00Z"]
    ).tz_convert("UTC")
    inputs = _clean_snapshot_inputs(idx)
    with pytest.raises(ValueError, match="hourly grid"):
        build_snapshot(**inputs, regime_flags=_regime_flags(idx))


# ---------------------------------------------------------------------------
# model_comparison
# ---------------------------------------------------------------------------


def _pred_frame(y_true: list[float], y_pred: list[float]) -> pd.DataFrame:
    idx = _hourly_utc("2021-01-01", len(y_true))
    return pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "delivery_day": idx.tz_convert("Europe/Berlin").normalize(),
        },
        index=idx,
    )


def test_model_comparison_order_and_values() -> None:
    predictions = {
        "ARIMAX": _pred_frame([50.0, 52.0], [52.0, 50.0]),
        "Naive": _pred_frame([50.0, 52.0], [50.0, 52.0]),
    }
    table = model_comparison(predictions)

    assert list(table["model"]) == ["Naive", "ARIMAX"]
    naive_row = table[table["model"] == "Naive"].iloc[0]
    assert naive_row["mae"] == 0.0
    assert naive_row["rmse"] == 0.0
    arimax_row = table[table["model"] == "ARIMAX"].iloc[0]
    assert arimax_row["mae"] == 2.0


def test_model_comparison_appends_unknown_models_after_known_order() -> None:
    predictions = {
        "Mystery": _pred_frame([50.0], [51.0]),
        "LightGBM": _pred_frame([50.0], [50.0]),
    }
    table = model_comparison(predictions)
    assert list(table["model"]) == ["LightGBM", "Mystery"]


# ---------------------------------------------------------------------------
# model_comparison_by_regime / assert_model_comparison_reconciles (Sprint 5.3.1)
# ---------------------------------------------------------------------------


def _mixed_regime_flags(idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(idx)
    half = n // 2
    data: dict[str, object] = {col: [False] * n for col in REGIME_FLAG_COLUMNS}
    data["renewable_scarcity"] = [True] * half + [False] * (n - half)
    data["normal"] = [False] * half + [True] * (n - half)
    data[MACRO_REGIME_COLUMN] = pd.Categorical(["crisis"] * half + ["calm"] * (n - half))
    return pd.DataFrame(data, index=idx)


def test_model_comparison_by_regime_reconciles_and_computes_regimes() -> None:
    idx = _hourly_utc("2021-01-01", 4)
    predictions = {
        "Naive": _pred_frame([10.0, 20.0, 50.0, 52.0], [12.0, 24.0, 50.0, 52.0]),
        "LightGBM": _pred_frame([10.0, 20.0, 50.0, 52.0], [10.0, 20.0, 50.0, 52.0]),
    }
    flags = _mixed_regime_flags(idx)

    comparison = model_comparison(predictions)
    by_regime = model_comparison_by_regime(predictions, flags)

    assert set(by_regime.columns) == {"model", "regime", "axis", "n", "mae", "rmse", "wape"}
    assert_model_comparison_reconciles(comparison, by_regime)  # must not raise

    naive = by_regime[by_regime["model"] == "Naive"].set_index("regime")
    assert naive.loc["overall", "n"] == 4
    assert naive.loc["overall", "mae"] == pytest.approx(1.5)
    assert naive.loc["renewable_scarcity", "n"] == 2
    assert naive.loc["renewable_scarcity", "mae"] == pytest.approx(3.0)
    assert naive.loc["renewable_scarcity", "axis"] == "flag"
    assert naive.loc["normal", "mae"] == pytest.approx(0.0)
    assert naive.loc["crisis", "mae"] == pytest.approx(3.0)
    assert naive.loc["crisis", "axis"] == "macro"
    assert naive.loc["calm", "mae"] == pytest.approx(0.0)
    assert naive.loc["post_crisis", "n"] == 0
    assert pd.isna(naive.loc["post_crisis", "mae"])


def test_assert_model_comparison_reconciles_raises_on_mismatch() -> None:
    idx = _hourly_utc("2021-01-01", 4)
    predictions = {"Naive": _pred_frame([10.0, 20.0, 50.0, 52.0], [12.0, 24.0, 50.0, 52.0])}
    flags = _mixed_regime_flags(idx)
    comparison = model_comparison(predictions)
    by_regime = model_comparison_by_regime(predictions, flags)
    tampered = by_regime.copy()
    tampered.loc[tampered["regime"] == "overall", "mae"] = 999.0

    with pytest.raises(ValueError, match="Naive"):
        assert_model_comparison_reconciles(comparison, tampered)


# ---------------------------------------------------------------------------
# coverage_summary
# ---------------------------------------------------------------------------


def test_coverage_summary_combines_three_sources() -> None:
    conformal_raw = pd.DataFrame(
        {
            "bucket": ["overall", "negative"],
            "level": [0.05, 0.05],
            "coverage": [0.17, 0.20],
            "pinball": [1.0, 1.0],
            "n": [100, 10],
        }
    )
    conformal_sorted = pd.DataFrame(
        {
            "bucket": ["overall", "negative"],
            "level": [0.05, 0.05],
            "coverage": [0.05, 0.06],
            "pinball": [0.9, 0.9],
            "n": [100, 10],
        }
    )
    reliability_curve = pd.DataFrame(
        {
            "bucket": ["overall", "overall"],
            "level": [0.05, 0.05],
            "coverage": [0.30, 0.05],
            "pinball": [1.5, 1.0],
            "n": [100, 100],
            "model": ["arimax", "lightgbm"],
        }
    )

    table = coverage_summary(conformal_raw, conformal_sorted, reliability_curve)

    assert list(table["nominal_level"]) == [0.05]
    row = table.iloc[0]
    assert row["lightgbm_raw_coverage"] == 0.17
    assert row["lightgbm_calibrated_coverage"] == 0.05
    assert row["arimax_raw_coverage"] == 0.30


# ---------------------------------------------------------------------------
# backtest_coverage_export
# ---------------------------------------------------------------------------


def test_backtest_coverage_export_selects_reported_columns() -> None:
    df = pd.DataFrame(
        {
            "variant": ["calibrated"],
            "side": ["long"],
            "subset": ["overall"],
            "n": [100],
            "n_breach": [5],
            "breach_rate": [0.05],
            "breach_rate_ci_low": [0.04],
            "breach_rate_ci_high": [0.06],
            "breach_rate_ci_excludes_alpha": [True],
            "kupiec_pvalue": [0.5],
            "chris_ind_lr_day": [1.0],
            "chris_ind_pvalue_day": [0.3],
            "z1": [0.1],
            "z1_ci_low": [-0.1],
            "z1_ci_high": [0.3],
            "low_support": [False],
            "block_days": [1],
            "basel_n_windows": [6.0],
        }
    )
    out = backtest_coverage_export(df)
    assert "block_days" not in out.columns
    assert "basel_n_windows" not in out.columns
    assert "breach_rate" in out.columns


def test_backtest_coverage_export_raises_on_missing_columns() -> None:
    df = pd.DataFrame({"variant": ["raw"]})
    with pytest.raises(ValueError, match="missing expected columns"):
        backtest_coverage_export(df)


# ---------------------------------------------------------------------------
# risk_headline
# ---------------------------------------------------------------------------


def test_risk_headline_computes_buffer_factor_for_calibrated_only() -> None:
    backtest_coverage = pd.DataFrame(
        {
            "variant": ["raw", "calibrated", "fhs"],
            "side": ["long", "long", "long"],
            "subset": ["overall", "overall", "overall"],
            "n": [100, 100, 100],
            "n_breach": [18, 5, 5],
            "breach_rate": [0.18, 0.05, 0.05],
            "breach_rate_ci_low": [None, 0.04, 0.04],
            "breach_rate_ci_high": [None, 0.06, 0.06],
            "kupiec_pvalue": [0.0, 0.9, 0.9],
            "z1": [None, 0.1, 0.1],
            "z1_ci_low": [None, -0.1, -0.1],
            "z1_ci_high": [None, 0.3, 0.3],
        }
    )
    risk_summary = pd.DataFrame(
        {
            "variant": ["raw", "calibrated", "fhs"],
            "side": ["long", "long", "long"],
            "mean_var": [100.0, 150.0, 140.0],
        }
    )

    out = risk_headline(backtest_coverage, risk_summary)

    calibrated_row = out[out["variant"] == "calibrated"].iloc[0]
    assert calibrated_row["buffer_factor"] == pytest.approx(1.5)
    raw_row = out[out["variant"] == "raw"].iloc[0]
    assert pd.isna(raw_row["buffer_factor"])


# ---------------------------------------------------------------------------
# Plot functions (smoke-level: returns a Figure, no exception, English labels)
# ---------------------------------------------------------------------------


def test_plot_coverage_forest_smoke() -> None:
    rows = []
    for subset in ("overall", "evening_ramp"):
        for variant in ("raw", "calibrated", "fhs"):
            for side in ("long", "short"):
                rows.append(
                    {
                        "subset": subset,
                        "variant": variant,
                        "side": side,
                        "breach_rate": 0.05,
                        "breach_rate_ci_low": 0.04 if variant != "raw" else None,
                        "breach_rate_ci_high": 0.06 if variant != "raw" else None,
                        "breach_rate_ci_excludes_alpha": False,
                        "low_support": False,
                    }
                )
    coverage = pd.DataFrame(rows)

    fig = plot_coverage_forest(coverage, alpha=0.05)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "breach_rate"


def test_plot_fan_chart_smoke() -> None:
    idx = _hourly_utc("2021-01-01", 10)
    y_true = pd.Series(range(10), index=idx, dtype=float)
    quantiles = {
        a: pd.Series([float(v) + a * 10 for v in range(10)], index=idx)
        for a in (0.05, 0.25, 0.5, 0.75, 0.95)
    }

    fig = plot_fan_chart(quantiles, y_true, title="Test window")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_ylabel() == "EUR/MWh"


def test_plot_reliability_smoke() -> None:
    curves = {
        "raw": pd.DataFrame({"empirical": [0.1, 0.5, 0.9]}, index=[0.05, 0.5, 0.95]),
        "calibrated_sorted": pd.DataFrame(
            {"empirical": [0.05, 0.5, 0.95]}, index=[0.05, 0.5, 0.95]
        ),
        "arimax": pd.DataFrame({"empirical": [0.15, 0.55, 0.85]}, index=[0.05, 0.5, 0.95]),
    }

    fig = plot_reliability(curves)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "nominal one-sided coverage"


def test_plot_forecast_vs_actual_smoke() -> None:
    idx = _hourly_utc("2021-01-01", 48)
    y_true = pd.Series(range(48), index=idx, dtype=float)
    y_pred = pd.Series([v + 1 for v in range(48)], index=idx, dtype=float)

    fig = plot_forecast_vs_actual(y_true, y_pred, model_label="LightGBM")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert "Actual" in [line.get_label() for line in ax.get_lines()]
    assert "LightGBM" in [line.get_label() for line in ax.get_lines()]
