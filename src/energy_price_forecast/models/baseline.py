from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from energy_price_forecast.evaluation.walkforward import LOCAL_TZ

_Fn = Callable[..., np.ndarray]

# forward (apply before fit) and inverse (apply after predict)
_TARGET_TRANSFORMS: dict[str, tuple[_Fn, _Fn]] = {
    "asinh": (np.arcsinh, np.sinh),
    "identity": (lambda a: a, lambda a: a),
}

# Weekdays (Mon=0 … Sun=6) for which the 24-hour lag applies (Tue–Fri).
# All other weekdays (Mon, Sat, Sun) use the 168-hour (7-day) lag.
_LAG_24H_WEEKDAYS: frozenset[int] = frozenset({1, 2, 3, 4})


class SimilarDayNaive:
    """Same-hour lag forecast: 24 h for Tue–Fri, 168 h (1 week) otherwise.

    A no-fit benchmark after Lago et al. 2021 and the harness smoke test.
    fit is a no-op (no learned state); predict reads the lag from history.
    Hours whose lag is missing from history yield NaN, which the metrics drop.

    DST note: the 168-h lag is a fixed UTC offset. On days immediately after
    a clock change the lag may miss by one local hour, and some timestamps
    (e.g. the extra autumn hour) may be absent 7 days earlier → NaN.
    Accepted benchmark artefact; not over-engineered.
    """

    def fit(self, y_train: pd.Series, x_train: pd.DataFrame | None = None) -> None:
        return None

    def predict(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.Series:
        # All hours belong to the same delivery day; weekday is uniform.
        weekday = test_index.tz_convert(LOCAL_TZ)[0].weekday()
        lag = pd.Timedelta(hours=24) if weekday in _LAG_24H_WEEKDAYS else pd.Timedelta(hours=168)
        lagged_utc = test_index - lag
        values = history.reindex(lagged_utc).to_numpy()
        return pd.Series(values, index=test_index, name="y_pred")


class _BaseLinearForecaster:
    """Shared fit/predict logic for linear forecasters (Lasso, Ridge, OLS).

    Subclasses implement _make_estimator() and call super().__init__() to
    initialise the shared state (target_transform, _pipeline, _columns).
    """

    def __init__(self, *, target_transform: str = "asinh") -> None:
        if target_transform not in _TARGET_TRANSFORMS:
            raise ValueError(f"unknown target_transform: {target_transform!r}")
        self.target_transform = target_transform
        self._pipeline: Pipeline | None = None
        self._columns: pd.Index | None = None

    def _make_estimator(self) -> object:
        raise NotImplementedError

    def fit(self, y_train: pd.Series, x_train: pd.DataFrame | None = None) -> None:
        if x_train is None:
            raise ValueError(f"{type(self).__name__} requires a feature matrix (x_train)")
        mask = y_train.notna()
        x = x_train.loc[mask]
        y = y_train.loc[mask]
        self._columns = x.columns
        self._pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            self._make_estimator(),
        )
        forward, _ = _TARGET_TRANSFORMS[self.target_transform]
        self._pipeline.fit(x.to_numpy(), forward(y.to_numpy()))

    def predict(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.Series:
        if self._pipeline is None or self._columns is None:
            raise RuntimeError("predict called before fit")
        if x_test is None:
            raise ValueError(f"{type(self).__name__} requires a feature matrix (x_test)")
        x = x_test.reindex(columns=self._columns)
        _, inverse = _TARGET_TRANSFORMS[self.target_transform]
        return pd.Series(
            inverse(self._pipeline.predict(x.to_numpy())), index=test_index, name="y_pred"
        )


class LassoForecaster(_BaseLinearForecaster):
    """Lasso (L1) baseline: SimpleImputer → StandardScaler → LassoCV.

    Alpha selected via TimeSeriesSplit cross-validation on training data only.
    """

    def __init__(
        self,
        *,
        target_transform: str = "asinh",
        cv_splits: int = 5,
        max_iter: int = 5000,
        random_state: int = 0,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(target_transform=target_transform)
        self.cv_splits = cv_splits
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _make_estimator(self) -> LassoCV:
        return LassoCV(
            cv=TimeSeriesSplit(n_splits=self.cv_splits),
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )


class RidgeForecaster(_BaseLinearForecaster):
    """Ridge (L2) baseline: SimpleImputer → StandardScaler → RidgeCV.

    Alpha selected via efficient leave-one-out CV (no TimeSeriesSplit needed;
    RidgeCV uses the full analytical solution per alpha candidate).
    """

    def __init__(
        self,
        *,
        target_transform: str = "asinh",
        alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    ) -> None:
        super().__init__(target_transform=target_transform)
        self.alphas = alphas

    def _make_estimator(self) -> RidgeCV:
        return RidgeCV(alphas=self.alphas)


class OLSForecaster(_BaseLinearForecaster):
    """OLS baseline: SimpleImputer → StandardScaler → LinearRegression.

    No regularisation; included to show how much L1/L2 shrinkage matters.
    """

    def __init__(self, *, target_transform: str = "asinh") -> None:
        super().__init__(target_transform=target_transform)

    def _make_estimator(self) -> LinearRegression:
        return LinearRegression()
