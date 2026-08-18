from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..market_time import LOCAL_TZ


@dataclass(frozen=True)
class FeatureConfig:
    """Central, immutable configuration of feature hyperparameters.

    One place for every tunable number in the feature pipeline. Passed into the
    builders (a default instance is used when none is given). Step 2.3.3 extends
    this with its own lag and rolling-window settings, so all feature knobs stay
    collected here rather than scattered across modules.
    """

    # Commodities: 48h lands on D-2 local at gate closure (leakage-safe).
    commodity_lag_hours: int = 48

    # Crisis-regime boundaries (Europe/Berlin local dates), from the EDA (3.0, D2).
    crisis_start: pd.Timestamp = pd.Timestamp("2021-09-01", tz=LOCAL_TZ)
    post_crisis_start: pd.Timestamp = pd.Timestamp("2023-04-01", tz=LOCAL_TZ)

    # --- Added in step 2.3.3 ---
    # DA_FIXED quantities (price, scheduled flows): >= 24h is leakage-safe.
    price_lags_hours: tuple[int, ...] = (24, 48, 168)
    scheduled_flow_lags_hours: tuple[int, ...] = (24,)

    # RT_ACTUAL-bound quantities (actuals, physical flows, forecast errors):
    # >= 48h required. A 24h lag leaks in the afternoon target rows (the
    # ~1h real-time publication delay pushes knowledge time past gate closure).
    actual_lags_hours: tuple[int, ...] = (48, 168)
    physical_flow_lags_hours: tuple[int, ...] = (48,)
    forecast_error_lags_hours: tuple[int, ...] = (48,)

    # Trailing rolling means of the price. The leading (most recent) window edge
    # sits `rolling_base_lag_hours` before each target (24h -> DA_FIXED-safe).
    rolling_windows_hours: tuple[int, ...] = (24, 168)
    rolling_base_lag_hours: int = 24

    def max_lookback_hours(self) -> int:
        """Longest history any feature reaches back. Used for warm-up trimming.

        Plain lags reach back `lag` hours; a rolling mean of window W with leading
        edge at `rolling_base_lag_hours` reaches back base_lag + W - 1 hours.
        """
        lag_max = max(
            max(self.price_lags_hours),
            max(self.actual_lags_hours),
            max(self.scheduled_flow_lags_hours),
            max(self.physical_flow_lags_hours),
            max(self.forecast_error_lags_hours),
            self.commodity_lag_hours,
        )
        rolling_max = max(self.rolling_base_lag_hours + w - 1 for w in self.rolling_windows_hours)
        return max(lag_max, rolling_max)
