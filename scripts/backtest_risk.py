"""Thin wiring for Sprint 4.4b: backtest validation of the 4.4a risk measures.

Loads risk_hourly.parquet / book_hourly.parquet (4.4a), attaches the ex-ante
conditioning fields (forecast-level bucket, hour-of-day phase,
forecast_dunkelflaute, forecast_renewable_surplus), and runs the three
validation tests (Kupiec, Christoffersen, Acerbi-Szekely) plus the
month-stratified day-block bootstrap CIs, per (variant, side, subset). All
logic lives in evaluation.backtest / evaluation.bootstrap; this script only
loads data, wires the calls together, and writes artefacts out.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.evaluation.backtest import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    basel_traffic_light,
    christoffersen_independence,
    day_breach_series,
    kupiec_pof,
)
from energy_price_forecast.evaluation.bootstrap import (
    cell_occupancy,
    stratified_day_block_bootstrap,
)
from energy_price_forecast.evaluation.config import (
    CALIBRATION_EXPERIMENT_NAME,
    BacktestConfig,
    RegimeConfig,
)
from energy_price_forecast.evaluation.regimes import tag_regimes
from energy_price_forecast.evaluation.reliability import FORECAST_LEVEL_BUCKETS
from energy_price_forecast.evaluation.risk import es_ratio_conditional
from energy_price_forecast.market_time import LOCAL_TZ

_SIDES: tuple[str, ...] = ("long", "short")
_VALIDATED_VARIANTS: tuple[str, ...] = ("calibrated", "fhs")
_ALL_VARIANTS: tuple[str, ...] = ("raw", "calibrated", "fhs")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 4.4b: backtest validation of the trading-book risk measures."
    )
    p.add_argument(
        "--risk-hourly-path", default=Path("data/processed/risk_hourly.parquet"), type=Path
    )
    p.add_argument(
        "--book-hourly-path", default=Path("data/processed/book_hourly.parquet"), type=Path
    )
    p.add_argument("--features-path", default=Path("data/processed/features.parquet"), type=Path)
    p.add_argument("--data-path", default=Path("data/interim/hourly.parquet"), type=Path)
    p.add_argument("--out-dir", default=Path("data/processed"), type=Path)
    p.add_argument(
        "--max-bootstrap",
        default=None,
        type=int,
        help="Overrides BacktestConfig.max_bootstrap (Nachtrag 1, part A).",
    )
    p.add_argument(
        "--block-days",
        default=None,
        type=int,
        help="Overrides BacktestConfig.block_days (Nachtrag 2, part A). Passed to every "
        "stratified_day_block_bootstrap call.",
    )
    p.add_argument(
        "--tag",
        default="",
        help="Suffix for backtest_coverage.csv / backtest_variant_contrast.csv and the "
        "MLflow run name (Nachtrag 2, part A). Empty (default) keeps the historical "
        "filenames/run name unchanged; a non-empty tag also skips writing the "
        "bootstrap-independent descriptive/Basel outputs (they would just be identical "
        "copies across the block_days grid).",
    )
    p.add_argument("--study", default="backtest_risk")
    p.add_argument("--note", default="", help="MLflow tag: free-text run note.")
    return p.parse_args()


def _load_risk_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"timestamp", "variant", "side", "es", "breach"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def _validate_level_against_4_4a(level: float) -> None:
    """Fail-fast check: BacktestConfig.level must match the 4.4a risk_measures run."""
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(CALIBRATION_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(
            f"MLflow experiment {CALIBRATION_EXPERIMENT_NAME!r} not found -- "
            "run scripts/compute_risk.py first."
        )
    runs = client.search_runs(
        experiment.experiment_id,
        filter_string="tags.mlflow.runName = 'risk_measures'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            "No 'risk_measures' MLflow run found -- run scripts/compute_risk.py first."
        )
    logged_level = float(runs[0].data.params["level"])
    if abs(logged_level - level) > 1e-9:
        raise ValueError(
            f"BacktestConfig.level={level} does not match the 4.4a risk_measures run's "
            f"logged level={logged_level}. Keep the two in sync before validating."
        )


def _height_bucket(mark: pd.Series) -> pd.Series:
    bucket = pd.Series("unassigned", index=mark.index, dtype="object")
    for name, low, high in FORECAST_LEVEL_BUCKETS:
        bucket[(mark >= low) & (mark < high)] = name
    return bucket


def _build_subsets(
    common_index: pd.DatetimeIndex,
    mark: pd.Series,
    features: pd.DataFrame,
    config: BacktestConfig,
    regime_config: RegimeConfig,
) -> dict[str, pd.Series]:
    """Ex-ante-known conditioning masks, keyed by subset name. "overall" is all-True."""
    subsets: dict[str, pd.Series] = {"overall": pd.Series(True, index=common_index)}

    local_hour = common_index.tz_convert(LOCAL_TZ).hour
    for phase, hours in config.hour_phases.items():
        subsets[phase] = pd.Series(local_hour.isin(hours), index=common_index)

    bucket = _height_bucket(mark)
    for name, _, _ in FORECAST_LEVEL_BUCKETS:
        subsets[f"bucket_{name}"] = bucket == name

    dunkelflaute_threshold = 1.0 - regime_config.renewable_scarcity_residual_share
    subsets["forecast_dunkelflaute"] = features["renewable_share_forecast"] < dunkelflaute_threshold
    subsets["forecast_renewable_surplus"] = (
        features["residual_load_forecast"] < config.surplus_forecast_residual_load_threshold
    )
    return subsets


def _variant_frame(
    risk_hourly: pd.DataFrame, book: pd.DataFrame, *, variant: str, side: str
) -> pd.DataFrame:
    rows = risk_hourly[(risk_hourly["variant"] == variant) & (risk_hourly["side"] == side)]
    rows = rows.set_index("timestamp").sort_index()
    if not rows.index.equals(book.index):
        raise ValueError(
            f"risk_hourly rows for variant={variant!r}, side={side!r} do not share book_hourly's index."
        )
    pnl = book["pnl_long"] if side == "long" else book["pnl_short"]
    return pd.DataFrame(
        {"loss": -pnl, "es": rows["es"], "breach": rows["breach"]}, index=book.index
    )


def _bootstrap_ci(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    config: BacktestConfig,
) -> dict[str, float | int | bool]:
    return stratified_day_block_bootstrap(
        frame,
        statistic,
        local_tz=LOCAL_TZ,
        seed=config.bootstrap_seed,
        min_bootstrap=config.min_bootstrap,
        max_bootstrap=config.max_bootstrap,
        check_every=config.check_every,
        mc_tol=config.mc_tol,
        n_stable=config.n_stable,
        block_days=config.block_days,
    )


def _combine_bootstrap_meta(*results: dict[str, float | int | bool]) -> tuple[int, bool]:
    """Strictest-wins aggregation for cells backed by several bootstrap calls.

    A coverage row draws on 3 independent bootstrap calls (breach_rate, z1,
    z2); a contrast row on 2 (d_es_ratio, d_breach_rate). Reported as ONE
    shared (n_bootstrap_used, converged) pair per row: n_bootstrap_used is the
    MAX across the calls (the slowest-converging one), converged is the AND
    (owner decision -- always report the strictest behaviour of the group).
    """
    n_bootstrap_used = max(int(r["n_bootstrap_used"]) for r in results)
    converged = all(bool(r["converged"]) for r in results)
    return n_bootstrap_used, converged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger(__name__)
    args = _parse_args()
    config = BacktestConfig()
    if args.max_bootstrap is not None:
        config = dataclasses.replace(config, max_bootstrap=args.max_bootstrap)
    if args.block_days is not None:
        config = dataclasses.replace(config, block_days=args.block_days)
    regime_config = RegimeConfig()

    _validate_level_against_4_4a(config.level)

    risk_hourly = _load_risk_hourly(args.risk_hourly_path)
    book = pd.read_parquet(args.book_hourly_path)
    common_index = pd.DatetimeIndex(book.index)

    features = load_processed_features(args.features_path).reindex(common_index)
    missing_features = features[["residual_load_forecast", "renewable_share_forecast"]].isna().all()
    if missing_features.any():
        raise ValueError(
            "features.parquet has no overlap with book_hourly's index for the forecast "
            "conditioning columns -- check --features-path / --book-hourly-path."
        )

    subsets = _build_subsets(common_index, book["mark"], features, config, regime_config)
    log.info("built %d ex-ante conditioning subsets", len(subsets))

    # ------------------------------------------------------------------
    # Independence family (Christoffersen), per (variant, side) ONLY --
    # not subset-conditioned (spec 5.3 step 5) -- then broadcast into
    # every subset row of the coverage table.
    # ------------------------------------------------------------------
    chris_by_variant_side: dict[tuple[str, str], dict[str, float]] = {}
    variant_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for variant in _ALL_VARIANTS:
        for side in _SIDES:
            vf = _variant_frame(risk_hourly, book, variant=variant, side=side)
            variant_frames[(variant, side)] = vf
            if variant == "raw":
                continue
            valid = vf["breach"].notna()
            hourly_ind = christoffersen_independence(vf.loc[valid, "breach"])
            day_series = day_breach_series(vf["breach"], k=config.day_breach_k, local_tz=LOCAL_TZ)
            day_ind = christoffersen_independence(day_series)
            chris_by_variant_side[(variant, side)] = {
                "chris_ind_lr_hourly": hourly_ind["lr"],
                "chris_ind_pvalue_hourly": hourly_ind["pvalue"],
                "chris_ind_lr_day": day_ind["lr"],
                "chris_ind_pvalue_day": day_ind["pvalue"],
            }

    # ------------------------------------------------------------------
    # Basel traffic light (Nachtrag 1, part B), per (variant, side) ONLY --
    # a CALENDAR construct on the full hourly series, never subset-sliced.
    # basel_windows_rows feeds the new backtest_basel_windows.csv; the
    # coverage table only carries the summary (basel_*), and only on the
    # subset == "overall" row.
    # ------------------------------------------------------------------
    basel_by_variant_side: dict[tuple[str, str], dict[str, object]] = {}
    basel_window_rows: list[dict[str, object]] = []
    for variant in _ALL_VARIANTS:
        for side in _SIDES:
            basel = basel_traffic_light(
                variant_frames[(variant, side)]["breach"],
                level=config.level,
                window_days=config.basel_window_days,
                drop_partial=config.basel_drop_partial_window,
                local_tz=LOCAL_TZ,
            )
            basel_by_variant_side[(variant, side)] = basel
            for w in basel["windows"]:  # type: ignore[union-attr]
                basel_window_rows.append({"variant": variant, "side": side, **w})

    # ------------------------------------------------------------------
    # Coverage / magnitude family, per (variant, side, subset).
    # ------------------------------------------------------------------
    coverage_rows: list[dict[str, object]] = []
    for variant in _ALL_VARIANTS:
        for side in _SIDES:
            vf = variant_frames[(variant, side)]
            chris = chris_by_variant_side.get((variant, side), {})
            for subset_name, mask in subsets.items():
                sub = vf.loc[mask]
                valid = sub["breach"].notna()
                sub_valid = sub.loc[valid]
                n = int(sub_valid.shape[0])
                n_breach = int((sub_valid["breach"] == 1.0).sum())
                kupiec = kupiec_pof(sub_valid["breach"], level=config.level)
                occ = cell_occupancy(
                    sub_valid, local_tz=LOCAL_TZ, min_cell_days=config.min_cell_days
                )

                row: dict[str, object] = {
                    "variant": variant,
                    "side": side,
                    "subset": subset_name,
                    "block_days": config.block_days,
                    "n": n,
                    "n_breach": n_breach,
                    "breach_rate": kupiec["breach_rate"],
                    "breach_rate_ci_low": float("nan"),
                    "breach_rate_ci_high": float("nan"),
                    "kupiec_lr": kupiec["lr"],
                    "kupiec_pvalue": kupiec["pvalue"],
                    "chris_ind_lr_hourly": chris.get("chris_ind_lr_hourly", float("nan")),
                    "chris_ind_pvalue_hourly": chris.get("chris_ind_pvalue_hourly", float("nan")),
                    "chris_ind_lr_day": chris.get("chris_ind_lr_day", float("nan")),
                    "chris_ind_pvalue_day": chris.get("chris_ind_pvalue_day", float("nan")),
                    "z1": float("nan"),
                    "z1_ci_low": float("nan"),
                    "z1_ci_high": float("nan"),
                    "z2": float("nan"),
                    "z2_ci_low": float("nan"),
                    "z2_ci_high": float("nan"),
                    "breach_rate_ci_excludes_alpha": None,
                    "z1_excludes_zero": None,
                    "z2_excludes_zero": None,
                    "n_bootstrap_used": float("nan"),
                    "converged": None,
                    # basel_* is a CALENDAR construct (Nachtrag 1, B.2c) --
                    # filled only on the subset == "overall" row, NaN/None
                    # everywhere else.
                    "basel_latest_zone": None,
                    "basel_n_windows": float("nan"),
                    "basel_n_yellow": float("nan"),
                    "basel_n_red": float("nan"),
                    "low_support": occ["low_support"],
                }

                if subset_name == "overall":
                    basel = basel_by_variant_side[(variant, side)]
                    row["basel_latest_zone"] = basel["latest_zone"]
                    row["basel_n_windows"] = basel["n_windows"]
                    row["basel_n_yellow"] = basel["n_yellow"]
                    row["basel_n_red"] = basel["n_red"]

                if variant in _VALIDATED_VARIANTS and n > 0:
                    row["z1"] = acerbi_szekely_z1(
                        sub_valid["loss"], sub_valid["es"], sub_valid["breach"]
                    )
                    row["z2"] = acerbi_szekely_z2(
                        sub_valid["loss"], sub_valid["es"], sub_valid["breach"], level=config.level
                    )
                    breach_ci = _bootstrap_ci(
                        sub_valid,
                        lambda df: float((df["breach"] == 1.0).mean()),
                        config=config,
                    )
                    z1_ci = _bootstrap_ci(
                        sub_valid,
                        lambda df: acerbi_szekely_z1(df["loss"], df["es"], df["breach"]),
                        config=config,
                    )
                    z2_ci = _bootstrap_ci(
                        sub_valid,
                        lambda df: acerbi_szekely_z2(
                            df["loss"], df["es"], df["breach"], level=config.level
                        ),
                        config=config,
                    )
                    row["breach_rate_ci_low"] = breach_ci["ci_low"]
                    row["breach_rate_ci_high"] = breach_ci["ci_high"]
                    # The honest, day-correlation-corrected counterpart to
                    # kupiec_pvalue < 0.05: does the bootstrap CI exclude the
                    # nominal target rate (1 - level)? Kupiec's hourly chi2 is
                    # anti-conservative (intra-day breaches are correlated),
                    # so it rejects strictly more often than this does -- see
                    # docs/backtest_results.md section 1 for the real-data gap.
                    alpha = 1.0 - config.level
                    row["breach_rate_ci_excludes_alpha"] = not (
                        breach_ci["ci_low"] <= alpha <= breach_ci["ci_high"]
                    )
                    row["z1_ci_low"] = z1_ci["ci_low"]
                    row["z1_ci_high"] = z1_ci["ci_high"]
                    row["z2_ci_low"] = z2_ci["ci_low"]
                    row["z2_ci_high"] = z2_ci["ci_high"]
                    # Same "honest verdict" logic as breach_rate_ci_excludes_alpha,
                    # for the severity (Z1/Z2) side: does the bootstrap CI exclude 0
                    # (Nachtrag 2 spec, 2.6)? Boundary touch does NOT count as
                    # exclusion -- the conservative convention.
                    row["z1_excludes_zero"] = not (z1_ci["ci_low"] <= 0.0 <= z1_ci["ci_high"])
                    row["z2_excludes_zero"] = not (z2_ci["ci_low"] <= 0.0 <= z2_ci["ci_high"])
                    n_bootstrap_used, converged = _combine_bootstrap_meta(breach_ci, z1_ci, z2_ci)
                    row["n_bootstrap_used"] = n_bootstrap_used
                    row["converged"] = converged

                coverage_rows.append(row)
    coverage_table = pd.DataFrame(coverage_rows)

    # ------------------------------------------------------------------
    # Variant contrast: calibrated - fhs, one bootstrap-CI'd difference
    # per (side, subset).
    # ------------------------------------------------------------------
    contrast_rows: list[dict[str, object]] = []
    for side in _SIDES:
        cal = variant_frames[("calibrated", side)]
        fhs = variant_frames[("fhs", side)]
        paired = pd.DataFrame(
            {
                "loss": cal["loss"],
                "es_cal": cal["es"],
                "breach_cal": cal["breach"],
                "es_fhs": fhs["es"],
                "breach_fhs": fhs["breach"],
            },
            index=common_index,
        )
        for subset_name, mask in subsets.items():
            sub = paired.loc[mask]
            valid = sub["breach_cal"].notna() & sub["breach_fhs"].notna()
            sub_valid = sub.loc[valid]

            def _d_es_ratio(df: pd.DataFrame) -> float:
                return acerbi_szekely_z1(
                    df["loss"], df["es_cal"], df["breach_cal"]
                ) - acerbi_szekely_z1(df["loss"], df["es_fhs"], df["breach_fhs"])

            def _d_breach_rate(df: pd.DataFrame) -> float:
                return float((df["breach_cal"] == 1.0).mean() - (df["breach_fhs"] == 1.0).mean())

            es_ci = _bootstrap_ci(sub_valid, _d_es_ratio, config=config)
            rate_ci = _bootstrap_ci(sub_valid, _d_breach_rate, config=config)
            occ = cell_occupancy(sub_valid, local_tz=LOCAL_TZ, min_cell_days=config.min_cell_days)
            es_separates = not (es_ci["ci_low"] <= 0.0 <= es_ci["ci_high"])
            rate_separates = not (rate_ci["ci_low"] <= 0.0 <= rate_ci["ci_high"])
            n_bootstrap_used, converged = _combine_bootstrap_meta(es_ci, rate_ci)
            contrast_rows.append(
                {
                    "side": side,
                    "subset": subset_name,
                    "block_days": config.block_days,
                    "d_es_ratio": es_ci["point"],
                    "d_es_ratio_ci_low": es_ci["ci_low"],
                    "d_es_ratio_ci_high": es_ci["ci_high"],
                    "d_breach_rate": rate_ci["point"],
                    "d_breach_rate_ci_low": rate_ci["ci_low"],
                    "d_breach_rate_ci_high": rate_ci["ci_high"],
                    "separates": bool(es_separates and rate_separates),
                    "low_support": occ["low_support"],
                    "n_bootstrap_used": n_bootstrap_used,
                    "converged": converged,
                }
            )
    contrast_table = pd.DataFrame(contrast_rows)

    # ------------------------------------------------------------------
    # Descriptive ex-post table: spike / negative_price conditioning
    # (realised, NOT ex-ante -- inferential=False), plus the two
    # forecast-subset descriptive numbers from spec 2.7 / section 6.
    # ------------------------------------------------------------------
    interim = load_interim_hourly(args.data_path)
    regimes = tag_regimes(interim).reindex(common_index)

    descriptive_rows: list[dict[str, object]] = []
    for variant in _VALIDATED_VARIANTS:
        for side in _SIDES:
            vf = variant_frames[(variant, side)]
            for expost_name in ("spike", "negative_price"):
                col = "price_spike" if expost_name == "spike" else "negative_price"
                mask = regimes[col].fillna(False).astype(bool)
                sub = vf.loc[mask]
                risk_like = pd.DataFrame(
                    {"breach": sub["breach"], "es": sub["es"]}, index=sub.index
                )
                ratio = es_ratio_conditional(-sub["loss"], risk_like)
                descriptive_rows.append(
                    {
                        "kind": "expost_es_ratio",
                        "variant": variant,
                        "side": side,
                        "subset": expost_name,
                        "value_name": "es_ratio_conditional",
                        "value": ratio,
                        "n": int(sub.shape[0]),
                        "inferential": False,
                    }
                )

    dunkelflaute_mask = subsets["forecast_dunkelflaute"]
    surplus_mask = subsets["forecast_renewable_surplus"]
    negative_bucket_mask = subsets["bucket_negative"]

    intersection = int((surplus_mask & negative_bucket_mask).sum())
    union = int((surplus_mask | negative_bucket_mask).sum())
    jaccard = intersection / union if union > 0 else float("nan")
    descriptive_rows.append(
        {
            "kind": "overlap",
            "variant": "n/a",
            "side": "n/a",
            "subset": "forecast_renewable_surplus_vs_negative_bucket",
            "value_name": "jaccard",
            "value": jaccard,
            "n": union,
            "inferential": False,
        }
    )
    descriptive_rows.append(
        {
            "kind": "overlap",
            "variant": "n/a",
            "side": "n/a",
            "subset": "forecast_renewable_surplus_vs_negative_bucket",
            "value_name": "surplus_share_in_negative_bucket",
            "value": intersection / int(negative_bucket_mask.sum())
            if negative_bucket_mask.sum() > 0
            else float("nan"),
            "n": int(negative_bucket_mask.sum()),
            "inferential": False,
        }
    )

    realised_negative = (interim["day_ahead_price"] < 0.0).reindex(common_index).fillna(False)
    realised_spike = regimes["price_spike"].fillna(False).astype(bool)
    surplus_proxy_hit_rate = (
        float((realised_negative & surplus_mask).sum() / surplus_mask.sum())
        if surplus_mask.sum() > 0
        else float("nan")
    )
    dunkelflaute_proxy_hit_rate = (
        float((realised_spike & dunkelflaute_mask).sum() / dunkelflaute_mask.sum())
        if dunkelflaute_mask.sum() > 0
        else float("nan")
    )
    descriptive_rows.append(
        {
            "kind": "proxy_hit_rate",
            "variant": "n/a",
            "side": "n/a",
            "subset": "forecast_renewable_surplus_vs_realised_negative_price",
            "value_name": "hit_rate",
            "value": surplus_proxy_hit_rate,
            "n": int(surplus_mask.sum()),
            "inferential": False,
        }
    )
    descriptive_rows.append(
        {
            "kind": "proxy_hit_rate",
            "variant": "n/a",
            "side": "n/a",
            "subset": "forecast_dunkelflaute_vs_realised_price_spike",
            "value_name": "hit_rate",
            "value": dunkelflaute_proxy_hit_rate,
            "n": int(dunkelflaute_mask.sum()),
            "inferential": False,
        }
    )
    descriptive_table = pd.DataFrame(descriptive_rows)

    log.warning(
        "variant='raw' rows in backtest_coverage.csv are a COUNTERFACTUAL, not a "
        "validated risk number (4.4a spec 2.3): frequency/count columns are filled, "
        "but z1/z2 and their bootstrap CIs are NaN by design."
    )

    basel_windows_table = pd.DataFrame(basel_window_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    coverage_path = args.out_dir / f"backtest_coverage{suffix}.csv"
    contrast_path = args.out_dir / f"backtest_variant_contrast{suffix}.csv"
    coverage_table.to_csv(coverage_path, index=False)
    contrast_table.to_csv(contrast_path, index=False)
    written_paths = [coverage_path, contrast_path]

    if args.tag:
        log.info(
            "tag=%r set: skipping backtest_descriptive_expost.csv / "
            "backtest_basel_windows.csv -- they are bootstrap-independent (unaffected by "
            "--block-days), so writing them per tag would just produce identical copies "
            "(Nachtrag 2, part A).",
            args.tag,
        )
    else:
        log.warning(
            "backtest_descriptive_expost.csv conditions on REALISED price outcomes "
            "(spike, negative_price) -- these rows are descriptive only "
            "(inferential=False), never a coverage verdict (spec 2.7)."
        )
        log.warning(
            "backtest_basel_windows.csv (Nachtrag 1, part B): non-overlapping windows on "
            "the FULL hourly series only (never a conditioning subset); illustrative, not "
            "a regulatory verdict -- zone bounds assume independent hours, which this "
            "book's intra-day breach clustering violates (see file header, spec B.3)."
        )
        descriptive_path = args.out_dir / "backtest_descriptive_expost.csv"
        basel_windows_path = args.out_dir / "backtest_basel_windows.csv"
        descriptive_table.to_csv(descriptive_path, index=False)
        with open(basel_windows_path, "w", encoding="utf-8", newline="") as f:
            f.write(
                "# ILLUSTRATIVE, not a regulatory verdict (spec 4.4b 2.9, Nachtrag 1 B).\n"
                "# Non-overlapping windows of BacktestConfig.basel_window_days delivery days\n"
                "# on the FULL hourly series only (never a conditioning subset, Nachtrag 1\n"
                "# B.2c). An incomplete trailing window is dropped when\n"
                "# BacktestConfig.basel_drop_partial_window (default True).\n"
                "# Zone boundaries assume INDEPENDENT hours (binomial); intra-day breaches\n"
                "# cluster in this book (hourly chris_ind_lr is large by construction, spec\n"
                "# 2.4), so the breach count is overdispersed relative to binomial and these\n"
                "# zone bounds are too tight -- a perfectly calibrated model will show\n"
                "# yellow too often (Nachtrag 1, B.3).\n"
            )
            basel_windows_table.to_csv(f, index=False)
        written_paths += [descriptive_path, basel_windows_path]

    for path in written_paths:
        log.info("written to %s", path)

    n_subsets_kupiec_rejected = int(
        (
            (coverage_table["variant"].isin(_VALIDATED_VARIANTS))
            & (coverage_table["kupiec_pvalue"] < 0.05)
        ).sum()
    )
    n_subsets_bootstrap_rejected = int(
        (
            coverage_table["breach_rate_ci_excludes_alpha"].notna()
            & coverage_table["breach_rate_ci_excludes_alpha"].astype("boolean")
        ).sum()
    )
    n_subsets_z1_excludes_zero = int(
        (
            coverage_table["z1_excludes_zero"].notna()
            & coverage_table["z1_excludes_zero"].astype("boolean")
        ).sum()
    )
    ramp_short_row = contrast_table[
        (contrast_table["side"] == "short") & (contrast_table["subset"] == "evening_ramp")
    ]
    ramp_short_separates = (
        bool(ramp_short_row["separates"].iloc[0]) if len(ramp_short_row) else False
    )

    run_name = f"backtest_risk_{args.tag}" if args.tag else "backtest_risk"

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(CALIBRATION_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {"study": args.study, "note": args.note, "block_days": str(config.block_days)}
        )
        mlflow.log_params(
            {
                "level": config.level,
                "day_breach_k": config.day_breach_k,
                "min_bootstrap": config.min_bootstrap,
                "max_bootstrap": config.max_bootstrap,
                "check_every": config.check_every,
                "mc_tol": config.mc_tol,
                "n_stable": config.n_stable,
                "bootstrap_seed": config.bootstrap_seed,
                "basel_window_days": config.basel_window_days,
                "basel_drop_partial_window": config.basel_drop_partial_window,
                "min_cell_days": config.min_cell_days,
                "block_days": config.block_days,
            }
        )
        n_cells_not_converged = int(
            (
                coverage_table["converged"].notna() & ~coverage_table["converged"].astype("boolean")
            ).sum()
        )
        mlflow.log_metrics(
            {
                "n_subsets_kupiec_rejected": float(n_subsets_kupiec_rejected),
                "n_subsets_bootstrap_rejected": float(n_subsets_bootstrap_rejected),
                "n_subsets_z1_excludes_zero": float(n_subsets_z1_excludes_zero),
                "ramp_short_separates": float(ramp_short_separates),
                "surplus_proxy_hit_rate": surplus_proxy_hit_rate,
                "n_cells_not_converged": float(n_cells_not_converged),
                "block_days": float(config.block_days),
            }
        )
        for path in written_paths:
            mlflow.log_artifact(str(path))

    log.info("MLflow backtest_risk run logged.")


if __name__ == "__main__":
    main()
