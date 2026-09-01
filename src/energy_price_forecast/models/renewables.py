"""Label, model, clipping, and MW conversion for the three renewables
capacity-factor models (spec 6.5.2, section 5.6).

Label: capacity_factor(target, t) = forecast_14_1_D(target, t) /
installed_capacity_at(target, t). Fail-fast on any missing target value --
the target series is never interpolated (plan section 8, rule 3).

Model: `models/lgbm.py::LGBMForecaster`, reused unmodified (it already
accepts external LightGBM parameters and both objectives this step needs --
confirmed in the section 3.1 stand-abgleich, no wrapper needed).

Post-processing: predictions are clipped at 0 (capacity factors cannot be
negative); values above 1 are counted and logged, never clipped away (spec
2.9 -- a value above 1 is a sign of a wrong denominator, worth seeing, not
hiding). Solar predictions at night are forced to exactly 0 (spec 2.3), and
night-hour rows never enter solar training in the first place.

Back-conversion to MW uses the identical `installed_capacity_at` call as the
label -- one rule, one code path (spec 2.5, decision 10): if the label
denominator and the prediction denominator ever diverged, their errors
would not cancel the way they do when both come from the same function.
"""

from __future__ import annotations

import logging

import pandas as pd

from energy_price_forecast.data.capacity import (
    CapacityExtrapolation,
    CapacitySource,
    ProductionType,
    installed_capacity_at,
)
from energy_price_forecast.features.solar_geometry import is_daylight_hour
from energy_price_forecast.models.lgbm import LGBMForecaster, Objective

logger = logging.getLogger(__name__)

# Untuned, derived from plan E1 (many correlated features, scarce effective
# sample size -- spec 6.5.2 section 11), not searched. feature_fraction is
# the one knob that directly addresses that situation; min_data_in_leaf is
# raised well above LightGBM's default of 20 so no leaf memorises a single
# weather pattern.
DEFAULT_LGBM_PARAMS: dict[str, object] = {
    "feature_fraction": 0.7,
    "min_data_in_leaf": 100,
}
DEFAULT_SEED = 0


def capacity_factor_label(
    target_values: pd.Series,
    valid_time_utc: pd.DatetimeIndex,
    production_type: ProductionType,
    *,
    source: CapacitySource = CapacitySource.PUBLIC_REGISTRY,
    method: CapacityExtrapolation = CapacityExtrapolation.LAST_INCREMENT,
) -> pd.Series:
    """capacity_factor(target, t) = forecast_14_1_D(target, t) / installed_capacity_at(target, t).

    ``target_values`` and ``valid_time_utc`` must be the same length and in
    the same order; the returned Series keeps ``target_values``'s own
    index, so callers may pass either a flat or a MultiIndex series (e.g.
    (run_init_utc, valid_time_utc)) -- only the point in *calendar* time
    (``valid_time_utc``) matters for the capacity lookup.

    Raises if any target value is missing (no interpolation of the target
    series, plan section 8 rule 3). Capacity factors above 1 are counted
    and logged, never removed (spec 2.9).
    """
    if target_values.isna().any():
        missing = target_values[target_values.isna()]
        raise ValueError(
            f"target series has {len(missing)} missing value(s), first at {missing.index[0]!r}"
        )
    if len(target_values) != len(valid_time_utc):
        raise ValueError("target_values and valid_time_utc must be the same length")

    capacity_mw = installed_capacity_at(
        production_type, valid_time_utc, source=source, method=method
    )
    capacity_factor = target_values.to_numpy(dtype="float64") / capacity_mw.to_numpy()

    over_one = capacity_factor > 1.0
    if over_one.any():
        logger.info(
            "%d of %d label capacity factors above 1.0 for %s (max %.4f) -- kept, not removed",
            int(over_one.sum()),
            len(capacity_factor),
            production_type.value,
            float(capacity_factor.max()),
        )

    return pd.Series(capacity_factor, index=target_values.index, name=f"{production_type.value}_cf")


def to_mw(
    capacity_factor: pd.Series,
    valid_time_utc: pd.DatetimeIndex,
    production_type: ProductionType,
    *,
    source: CapacitySource = CapacitySource.PUBLIC_REGISTRY,
    method: CapacityExtrapolation = CapacityExtrapolation.LAST_INCREMENT,
) -> pd.Series:
    """prediction_mw = capacity_factor * installed_capacity_at(target, t).

    The same `installed_capacity_at` call as `capacity_factor_label` --
    same source, same method -- so a systematic capacity error cancels
    rather than compounding (spec 2.5, 2.11).
    """
    if len(capacity_factor) != len(valid_time_utc):
        raise ValueError("capacity_factor and valid_time_utc must be the same length")
    capacity_mw = installed_capacity_at(
        production_type, valid_time_utc, source=source, method=method
    )
    return pd.Series(
        capacity_factor.to_numpy(dtype="float64") * capacity_mw.to_numpy(),
        index=capacity_factor.index,
        name=f"{production_type.value}_mw",
    )


def daylight_mask_for_training(valid_time_utc: pd.DatetimeIndex) -> pd.Series:
    """True for daylight hours (spec 2.3) -- solar rows outside this mask
    are excluded from both training and evaluation, never included as
    structural zeros.
    """
    return is_daylight_hour(pd.DatetimeIndex(valid_time_utc))


class RenewablesModel:
    """One LightGBM capacity-factor regressor for one production type."""

    def __init__(
        self,
        production_type: ProductionType,
        *,
        objective: Objective = "l2",
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.production_type = production_type
        self.forecaster = LGBMForecaster(
            objective=objective, params=dict(DEFAULT_LGBM_PARAMS), random_state=seed
        )

    def fit(self, x_train: pd.DataFrame, y_train_capacity_factor: pd.Series) -> None:
        self.forecaster.fit(y_train_capacity_factor, x_train)

    def predict_capacity_factor(self, test_index: pd.Index, x_test: pd.DataFrame) -> pd.Series:
        """Clipped-at-0 capacity-factor prediction (spec 2.9).

        ``test_index`` is typically the shared feature matrix's own
        (run_init_utc, valid_time_utc) MultiIndex, not a plain
        DatetimeIndex -- `LGBMForecaster.predict` only uses it as the
        output Series' row label, so this is safe even though its own type
        hint (unchanged, spec 3.3) says DatetimeIndex specifically.

        Solar night-hour zeroing (spec 2.3) is applied separately by the
        caller via `apply_solar_night_zero`, once it knows which target
        this prediction is for -- this method has no target-specific
        behaviour of its own.
        """
        raw = self.forecaster.predict(
            test_index,  # type: ignore[arg-type]
            history=pd.Series(dtype="float64"),
            x_test=x_test,
        )
        clipped = raw.clip(lower=0.0)

        over_one = clipped > 1.0
        if over_one.any():
            logger.info(
                "%d of %d predicted capacity factors above 1.0 for %s (max %.4f) -- kept, not clipped",
                int(over_one.sum()),
                len(clipped),
                self.production_type.value,
                float(clipped.max()),
            )
        return clipped


def apply_solar_night_zero(
    predicted_capacity_factor: pd.Series, valid_time_utc: pd.DatetimeIndex
) -> pd.Series:
    """Forces the solar prediction to exactly 0 outside daylight hours (spec 2.3)."""
    daylight = daylight_mask_for_training(valid_time_utc).to_numpy()
    values = predicted_capacity_factor.to_numpy(dtype="float64").copy()
    values[~daylight] = 0.0
    return pd.Series(
        values, index=predicted_capacity_factor.index, name=predicted_capacity_factor.name
    )
