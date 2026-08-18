from pathlib import Path

import pandas as pd
import pytest

from scripts.backtest_block_sensitivity import (
    _load_grid,
    build_contrast_sensitivity,
    build_coverage_sensitivity,
)


def _coverage_row(
    *,
    variant: str = "calibrated",
    side: str = "short",
    subset: str = "overall",
    block_days: int,
    breach_rate: float = 0.10,
    breach_rate_ci_low: float,
    breach_rate_ci_high: float,
    breach_rate_ci_excludes_alpha: bool | None,
    z1: float = 0.05,
    z1_ci_low: float = -0.01,
    z1_ci_high: float = 0.10,
    z1_excludes_zero: bool | None = False,
    low_support: bool = False,
    converged: bool | None = True,
    n_bootstrap_used: float = 5000.0,
) -> dict[str, object]:
    return {
        "variant": variant,
        "side": side,
        "subset": subset,
        "block_days": block_days,
        "breach_rate": breach_rate,
        "breach_rate_ci_low": breach_rate_ci_low,
        "breach_rate_ci_high": breach_rate_ci_high,
        "breach_rate_ci_excludes_alpha": breach_rate_ci_excludes_alpha,
        "z1": z1,
        "z1_ci_low": z1_ci_low,
        "z1_ci_high": z1_ci_high,
        "z1_excludes_zero": z1_excludes_zero,
        "low_support": low_support,
        "converged": converged,
        "n_bootstrap_used": n_bootstrap_used,
    }


# ---------------------------------------------------------------------------
# 13. Point-estimate invariance
# ---------------------------------------------------------------------------


def test_point_estimate_must_be_invariant_across_block_days() -> None:
    coverage = pd.DataFrame(
        [
            _coverage_row(
                block_days=1,
                breach_rate_ci_low=0.04,
                breach_rate_ci_high=0.06,
                breach_rate_ci_excludes_alpha=False,
            ),
            _coverage_row(
                block_days=3,
                breach_rate=0.11,  # BUG: point estimate drifted with block_days
                breach_rate_ci_low=0.03,
                breach_rate_ci_high=0.07,
                breach_rate_ci_excludes_alpha=False,
            ),
        ]
    )

    with pytest.raises(ValueError, match="breach_rate varies with block_days"):
        build_coverage_sensitivity(coverage)


def test_point_estimate_invariance_holds_for_consistent_data() -> None:
    coverage = pd.DataFrame(
        [
            _coverage_row(
                block_days=1,
                breach_rate_ci_low=0.04,
                breach_rate_ci_high=0.06,
                breach_rate_ci_excludes_alpha=False,
            ),
            _coverage_row(
                block_days=3,
                breach_rate_ci_low=0.03,
                breach_rate_ci_high=0.07,
                breach_rate_ci_excludes_alpha=False,
            ),
        ]
    )

    result = build_coverage_sensitivity(coverage)
    assert result["breach_rate"].nunique() == 1


# ---------------------------------------------------------------------------
# 14. ci_width_ratio / verdict_flipped, hand-checked
# ---------------------------------------------------------------------------


def test_ci_width_ratio_and_verdict_flipped_hand_checked() -> None:
    coverage = pd.DataFrame(
        [
            _coverage_row(
                block_days=1,
                breach_rate_ci_low=0.040,
                breach_rate_ci_high=0.060,  # width 0.020, excludes 0.05? no -> False
                breach_rate_ci_excludes_alpha=False,
            ),
            _coverage_row(
                block_days=5,
                breach_rate_ci_low=0.020,
                breach_rate_ci_high=0.045,  # width 0.025, excludes 0.05 -> True
                breach_rate_ci_excludes_alpha=True,
            ),
        ]
    )

    result = build_coverage_sensitivity(coverage).set_index("block_days")

    assert result.loc[1, "ci_width"] == pytest.approx(0.020)
    assert result.loc[1, "ci_width_ratio"] == pytest.approx(1.0)
    assert result.loc[1, "verdict_flipped"] == False  # noqa: E712

    assert result.loc[5, "ci_width"] == pytest.approx(0.025)
    assert result.loc[5, "ci_width_ratio"] == pytest.approx(0.025 / 0.020)
    assert bool(result.loc[5, "verdict_flipped"]) is True


def test_verdict_flipped_is_na_when_base_verdict_unknown() -> None:
    # variant='raw'-like situation is excluded upstream, but a validated cell
    # can still have n=0 (breach_rate_ci_excludes_alpha stays None at K=1) --
    # verdict_flipped must stay <NA>, not silently become False.
    coverage = pd.DataFrame(
        [
            _coverage_row(
                block_days=1,
                breach_rate_ci_low=float("nan"),
                breach_rate_ci_high=float("nan"),
                breach_rate_ci_excludes_alpha=None,
            ),
            _coverage_row(
                block_days=3,
                breach_rate_ci_low=float("nan"),
                breach_rate_ci_high=float("nan"),
                breach_rate_ci_excludes_alpha=None,
            ),
        ]
    )
    result = build_coverage_sensitivity(coverage).set_index("block_days")
    assert result.loc[3, "verdict_flipped"] is pd.NA


def test_contrast_sensitivity_ci_width_ratio_and_separates_flipped() -> None:
    contrast = pd.DataFrame(
        [
            {
                "side": "short",
                "subset": "evening_ramp",
                "block_days": 1,
                "d_es_ratio": -0.08,
                "d_es_ratio_ci_low": -0.128,
                "d_es_ratio_ci_high": -0.032,  # width 0.096, separates (0 excluded) -> True
                "d_breach_rate": 0.01,
                "d_breach_rate_ci_low": -0.02,
                "d_breach_rate_ci_high": 0.04,
                "separates": True,
                "low_support": False,
            },
            {
                "side": "short",
                "subset": "evening_ramp",
                "block_days": 5,
                "d_es_ratio": -0.08,
                "d_es_ratio_ci_low": -0.150,
                "d_es_ratio_ci_high": 0.010,  # width 0.16, 0 included -> False
                "d_breach_rate": 0.01,
                "d_breach_rate_ci_low": -0.03,
                "d_breach_rate_ci_high": 0.05,
                "separates": False,
                "low_support": False,
            },
        ]
    )

    result = build_contrast_sensitivity(contrast).set_index("block_days")

    assert result.loc[1, "d_es_ratio_ci_width"] == pytest.approx(0.096)
    assert result.loc[1, "d_es_ratio_ci_width_ratio"] == pytest.approx(1.0)
    assert result.loc[1, "separates_flipped"] == False  # noqa: E712

    assert result.loc[5, "d_es_ratio_ci_width"] == pytest.approx(0.16)
    assert result.loc[5, "d_es_ratio_ci_width_ratio"] == pytest.approx(0.16 / 0.096)
    assert bool(result.loc[5, "separates_flipped"]) is True


# ---------------------------------------------------------------------------
# 15. Fail-fast on missing / mismatched grid files
# ---------------------------------------------------------------------------


def test_load_grid_fails_fast_with_the_generating_command_line(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        _load_grid(
            lambda out_dir, k: (
                out_dir / f"backtest_coverage_block{k}.csv"
                if k != 1
                else out_dir / "backtest_coverage.csv"
            ),
            tmp_path,
            (1, 5),
        )
    message = str(exc_info.value)
    assert "backtest_coverage.csv" in message
    assert "uv run python scripts/backtest_risk.py --block-days 1" in message


def test_load_grid_fails_fast_on_missing_tagged_file(tmp_path: Path) -> None:
    (tmp_path / "backtest_coverage.csv").write_text("variant,block_days\ncalibrated,1\n")

    def path_fn(out_dir: Path, k: int) -> Path:
        return (
            out_dir / "backtest_coverage.csv"
            if k == 1
            else out_dir / f"backtest_coverage_block{k}.csv"
        )

    with pytest.raises(FileNotFoundError) as exc_info:
        _load_grid(path_fn, tmp_path, (1, 5))
    message = str(exc_info.value)
    assert "backtest_coverage_block5.csv" in message
    assert "--block-days 5 --tag block5" in message


def test_load_grid_fails_fast_on_block_days_mismatch(tmp_path: Path) -> None:
    pd.DataFrame({"variant": ["calibrated"], "block_days": [3]}).to_csv(
        tmp_path / "backtest_coverage.csv", index=False
    )

    def path_fn(out_dir: Path, k: int) -> Path:
        return out_dir / "backtest_coverage.csv"

    with pytest.raises(ValueError, match="expected to hold block_days=1"):
        _load_grid(path_fn, tmp_path, (1,))
