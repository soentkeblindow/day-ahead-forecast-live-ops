"""ARIMAX as dynamic harmonic regression — Sprint 3.4 classical benchmark."""

from __future__ import annotations

import logging
import warnings
from typing import Any, cast

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

# Structural configuration for the harmonic-regression benchmark. Fixed and
# documented (NOT auto-selected): decision 3 fixes the ARMA order, decision 2
# the Fourier harmonics. All reversible via constructor / CLI args.
_DEFAULT_ORDER: tuple[int, int, int] = (2, 0, 0)  # (p, d, q); d=0, q=0 -> AR(2) errors
_FOURIER_DAILY_K: int = 4  # harmonics for the 24h cycle
_FOURIER_WEEKLY_K: int = (
    0  # weekly Fourier dropped: is_weekend + AR(1)≈1 already capture the weekly pattern
)
_QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.5, 0.95)
_RESID_WARMUP: int = 48  # drop the first residuals (Kalman filter warm-up)
_OPTIM_GTOL: float = 1e-3  # L-BFGS-B gradient tolerance (scipy default 1e-5); relaxed because
# we need a good forecast, not numerically perfect MLE coefficients

# Curated exogenous regressor set (decision 4). Linear MLE is collinearity- and
# dimension-sensitive, so we keep ONE fundamental (residual load forecast, NOT
# its load/wind/solar components -- they are its exact linear combination), the
# commodities, and the binary calendar/regime flags. Everything else from
# features.parquet is dropped: price lags and rolling means (the ARMA part models
# price dynamics), forecast-error and cross-border lags, the calendar sin/cos
# (replaced by the richer Fourier basis), and renewable_share (collinear).
ARIMAX_EXOG_COLUMNS: tuple[str, ...] = (
    "residual_load_forecast",
    "ttf_gas_lag_48h",
    "eua_co2_lag_48h",
    "eua_missing",
    "is_weekend",
    "is_holiday",
    "is_regional_holiday",
    "is_crisis",
    "is_post_crisis",
)


def select_arimax_exog(x: pd.DataFrame) -> pd.DataFrame:
    """Subset the full feature matrix to the curated ARIMAX exog set (decision 4)."""
    missing = [c for c in ARIMAX_EXOG_COLUMNS if c not in x.columns]
    if missing:
        raise ValueError(f"feature matrix is missing ARIMAX exog columns: {missing}")
    return x.loc[:, list(ARIMAX_EXOG_COLUMNS)]


def fourier_terms(index: pd.DatetimeIndex, *, period_hours: int, k: int) -> pd.DataFrame:
    """Deterministic Fourier (harmonic) terms for one seasonal period.

    Returns 2*k columns (one sin/cos pair per harmonic h = 1..k). The cycle
    position is computed in LOCAL time (Europe/Berlin) so peaks line up with the
    local daily/weekly rhythm and stay aligned across DST -- matching the sprint-2
    calendar-feature convention. Daily period -> local hour-of-day; weekly period
    -> local hour-of-week. A single harmonic (k=1) is exactly the existing
    hour_sin/cos calendar feature; higher harmonics resolve the non-sinusoidal
    intraday shape (morning + evening peaks, midday solar dip).
    """
    local = index.tz_convert("Europe/Berlin")
    if period_hours == 24:
        pos = local.hour.to_numpy(dtype=float)
    elif period_hours == 168:
        pos = (local.dayofweek * 24 + local.hour).to_numpy(dtype=float)
    else:
        raise ValueError(f"unsupported period_hours={period_hours}")
    cols: dict[str, np.ndarray] = {}
    for h in range(1, k + 1):
        ang = 2.0 * np.pi * h * pos / period_hours
        cols[f"sin_{period_hours}_{h}"] = np.sin(ang)
        cols[f"cos_{period_hours}_{h}"] = np.cos(ang)
    return pd.DataFrame(cols, index=index)


class ARIMAXForecaster:
    """ARIMAX as a dynamic harmonic regression (abstract decision 4).

    Model:  y_t = mu_t + eta_t
      mu_t  = regression on Fourier terms (daily K=4 + weekly K=1) + standardised
              curated exog (decision 4); zero-variance exog columns are dropped
              dynamically per training window to avoid rank deficiency
      eta_t = AR(2) residual process; order (2,0,0), d=0 because the
              regression already removes level/trend/season (decision 3)
    statsmodels SARIMAX fits everything jointly by maximum likelihood.

    Quantiles (decision 5, lean one-step path): point + frozen empirical residual
    offset. After each fit we take the in-sample one-step residuals (warm-up
    trimmed) and freeze their 0.05/0.5/0.95 empirical quantiles as ADDITIVE
    offsets: q_alpha(t) = point(t) + Q_alpha(resid). Distribution-free (captures
    fat tails / skew) but (a) homoskedastic -- a constant band -- and (b) built on
    one-step residuals, so day-ahead intervals may under-cover. Both are FINDINGS
    the 3.3 calibration table measures, not bugs. The rolling-90 window gives the
    offsets a slowly adapting scale. Because the three alpha runs share one
    deterministic point and monotone offsets, ARIMAX cannot cross
    (crossing_rate == 0 by construction), unlike the independent LightGBM fits.

    Refit cadence (decision 7): the harness re-calls fit only every refit_every
    folds (the expensive MLE step). predict runs on EVERY fold and advances the
    state to the current gate-closure origin by applying the frozen parameters to
    the current rolling-90 window via SARIMAXResults.apply(..., refit=False) -- a
    cheap Kalman pass, no re-optimisation. ARIMAX is the first model that actually
    CONSUMES the protocol's `history` argument.

    Leakage: fit sees only the training window; all exog are gate-closure-available
    (2.3) and Fourier terms are deterministic, so forecasting day D uses nothing
    from day D or later. The harness enforces train_index.max() < test_index.min().
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        order: tuple[int, int, int] = _DEFAULT_ORDER,
        fourier_daily_k: int = _FOURIER_DAILY_K,
        fourier_weekly_k: int = _FOURIER_WEEKLY_K,
        levels: tuple[float, ...] = _QUANTILE_LEVELS,
        resid_warmup: int = _RESID_WARMUP,
    ) -> None:
        self.alpha = alpha
        self.order = order
        self.fourier_daily_k = fourier_daily_k
        self.fourier_weekly_k = fourier_weekly_k
        self.levels = levels
        self.resid_warmup = resid_warmup
        self._res: Any = None  # fitted SARIMAXResults (frozen params)
        self._scaler_mean: pd.Series | None = None  # exog standardiser, train-only
        self._scaler_std: pd.Series | None = None
        self._zero_var_cols: set[str] = set()  # exog columns constant in training window
        self._start_params: np.ndarray | None = None  # warm-start seed from previous fit
        self._offsets: dict[float, float] = {}  # frozen empirical resid quantiles
        # Running exog buffer so predict can rebuild the current window's design.
        # Verified (sec 3): history = y.loc[fold.train_index] is the full current
        # rolling-90-day window on every fold. fit() seeds the buffer; predict()
        # appends x_test day-by-day, so reindex(history.index) always resolves cleanly.
        self._exog_buf: pd.DataFrame | None = None

    def _design(self, index: pd.DatetimeIndex, exog_raw: pd.DataFrame) -> pd.DataFrame:
        """Regression design = Fourier terms + standardised exog (zero-var cols dropped)."""
        assert self._scaler_mean is not None and self._scaler_std is not None
        f_d = fourier_terms(index, period_hours=24, k=self.fourier_daily_k)
        f_w = fourier_terms(index, period_hours=168, k=self.fourier_weekly_k)
        active_cols = [c for c in exog_raw.columns if c not in self._zero_var_cols]
        exog_active = exog_raw[active_cols]
        exog_std = (exog_active - self._scaler_mean[active_cols]) / self._scaler_std[active_cols]
        # Missing exog (e.g. eua before Oct-2021) -> 0 == the standardised mean;
        # MLE cannot ingest NaN exog, and the eua_missing flag still carries the
        # "was missing" signal. (Endog NaN is fine -- the Kalman filter skips it.)
        exog_std = exog_std.fillna(0.0)
        return pd.concat([f_d, f_w, exog_std], axis=1)

    def fit(self, y_train: pd.Series, x_train: pd.DataFrame | None = None) -> None:
        if x_train is None:
            raise ValueError("ARIMAXForecaster requires an exog matrix (x_train)")
        # Freeze the standardiser on the training window only (leakage-safe).
        self._scaler_mean = x_train.mean()
        raw_std = x_train.std(ddof=0)
        self._zero_var_cols = set(raw_std[raw_std == 0.0].index.tolist())
        if self._zero_var_cols:
            logger.debug("dropping zero-variance exog columns: %s", sorted(self._zero_var_cols))
        self._scaler_std = raw_std.replace(0.0, 1.0)
        design = self._design(cast(pd.DatetimeIndex, y_train.index), x_train)
        # Warm start: reuse previous fit's params if the parameter vector length
        # matches (it changes when _zero_var_cols changes between folds). AR(p) +
        # sigma2 = order[0]+1 extra params on top of the design matrix columns.
        n_params_expected = design.shape[1] + self.order[0] + 1
        start_params = (
            self._start_params
            if self._start_params is not None and len(self._start_params) == n_params_expected
            else None
        )
        # Set freq="h" so statsmodels does not emit ValueWarning (freq is lost after
        # .loc[] slicing). Suppress statsmodels ConvergenceWarning noise; instead
        # read mle_retvals after the fit for a clean, single log line per fold.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = SARIMAX(
                endog=y_train.asfreq("h"),
                exog=design.asfreq("h"),
                order=self.order,
                enforce_stationarity=True,
                enforce_invertibility=True,
            ).fit(disp=False, maxiter=200, start_params=start_params, pgtol=_OPTIM_GTOL)
        self._start_params = np.asarray(self._res.params).copy()
        retvals: dict[str, Any] = getattr(self._res, "mle_retvals", {}) or {}
        warnflag = retvals.get("warnflag", 0)
        if warnflag != 0:
            logger.warning(
                "SARIMAX MLE did not converge (warnflag=%d, iterations=%s); best params used",
                warnflag,
                retvals.get("iterations", "?"),
            )
        # Freeze empirical residual-quantile offsets (decision 5): one-step
        # in-sample residuals, warm-up trimmed so init outliers don't inflate tails.
        resid = pd.Series(self._res.resid).iloc[self.resid_warmup :]
        self._offsets = {a: float(resid.quantile(a)) for a in self.levels}
        self._exog_buf = x_train.copy()
        logger.debug(
            "fit complete: order=%s offsets=%s",
            self.order,
            {a: f"{v:.3f}" for a, v in self._offsets.items()},
        )

    @property
    def exog_coefficients(self) -> pd.Series:
        """Return the fitted standardised exogenous coefficients (named Series).

        Subsets the SARIMAX parameter vector to the curated exog columns
        (ARIMAX_EXOG_COLUMNS) that survived the per-fit zero-variance drop. Exog are
        z-scored at fit (decision 4), so each coefficient is in EUR/MWh per one
        standard deviation of that regressor -- magnitudes are directly comparable
        across regressors (e.g. gas pass-through vs. merit-order slope). A regime
        dummy is constant within its own regime window and is therefore absent here
        (dropped as zero-variance); that absence is expected, not an error. Raises
        RuntimeError if called before `fit`.
        """
        if self._res is None:
            raise RuntimeError("model is not fitted; call fit() first")
        params = self._res.params
        present = [c for c in ARIMAX_EXOG_COLUMNS if c in params.index]
        return params.loc[present]

    def predict(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.Series:
        if self._res is None:
            raise RuntimeError("predict called before fit")
        if x_test is None:
            raise ValueError("ARIMAXForecaster requires an exog matrix (x_test)")
        assert self._exog_buf is not None
        # Advance the state to the current gate-closure origin: apply the frozen
        # params to the current rolling window (endog from `history`, exog from the
        # buffer). Verified (sec 3): history = y.loc[fold.train_index] is the full
        # 90-day window on every fold. With --refit-every 1 this step is unnecessary.
        endog_now = history
        exog_now = self._exog_buf.reindex(endog_now.index)
        endog_idx = cast(pd.DatetimeIndex, endog_now.index)
        applied = self._res.apply(
            endog=endog_now.asfreq("h"),
            exog=self._design(endog_idx, exog_now).asfreq("h"),
            refit=False,
        )
        design_fc = self._design(test_index, x_test)
        point = applied.get_forecast(steps=len(test_index), exog=design_fc).predicted_mean
        # Cache this fold's exog so later folds can rebuild their rolling window.
        self._exog_buf = pd.concat([self._exog_buf, x_test])
        self._exog_buf = self._exog_buf[~self._exog_buf.index.duplicated(keep="last")]
        out = np.asarray(point) + self._offsets[self.alpha]
        return pd.Series(out, index=test_index, name="y_pred")
