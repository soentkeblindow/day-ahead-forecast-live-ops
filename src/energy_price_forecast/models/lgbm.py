from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Untuned placeholder hyperparameters for the 3.1 end-to-end probe. These are
# deliberately NOT tuned: step 3.2 replaces them with optuna-frozen structural
# parameters injected via the `params` argument, so this dict is the documented
# "before" baseline.
_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 1.0,  # no row subsampling -> no bagging randomness
    "colsample_bytree": 1.0,  # no column subsampling
}

Objective = Literal["quantile", "l2"]

# Mapping from this project's loss names to LightGBM's objective strings.
# LightGBM calls squared error "regression", which does not name the loss;
# "l2" does. The indirection exists only here.
_LGBM_OBJECTIVE: dict[str, str] = {
    "quantile": "quantile",
    "l2": "regression",
}


class LGBMForecaster:
    """LightGBM forecaster on the engineered feature matrix.

    Block-forecast (one prediction per delivery day) consistent with D3. Two
    loss functions are supported, and the choice must match how the forecast
    is scored:

    - ``objective="quantile"`` with ``alpha`` (default 0.5) minimises the
      pinball loss, whose minimiser is the conditional alpha-quantile. At
      alpha=0.5 this is the conditional median and the associated point score
      is MAE (pinball at 0.5 equals 0.5 * MAE). This is the inherited Sprint 3
      behaviour and remains the default.
    - ``objective="l2"`` minimises squared error, whose minimiser is the
      conditional MEAN. This is the loss to use when the forecast is scored by
      RMSE, as the Energy-Arena point challenge is. Day-ahead prices are
      right-skewed, so the conditional mean sits systematically above the
      conditional median; a median forecast therefore carries a structural
      negative bias under RMSE.

    ``alpha`` applies to the quantile objective only. Passing it together with
    ``objective="l2"`` raises ValueError rather than being silently ignored --
    a silently ignored alpha would let a caller believe it holds a 95% quantile
    while holding a mean forecast.

    No target transform (decision D5): gradient-boosted trees are scale-invariant
    and absorb skew natively, so unlike the Lasso baseline there is no asinh/sinh.
    Missing feature values are passed through unchanged -- LightGBM routes NaN to a
    learned default split direction, so no imputation is needed.

    Leakage contract (spec section 4): fit only sees x_train / y_train; predict
    applies the fitted trees to x_test. The harness enforces train_index < test
    per fold. The `history` argument of predict is unused (feature-based model,
    like the Lasso baseline) but kept to satisfy the Forecaster protocol.
    """

    def __init__(
        self,
        *,
        objective: Objective = "quantile",
        alpha: float | None = None,
        params: dict[str, Any] | None = None,
        random_state: int = 0,
        n_jobs: int = 1,
    ) -> None:
        if objective not in _LGBM_OBJECTIVE:
            raise ValueError(
                f"objective must be one of {sorted(_LGBM_OBJECTIVE)}, got {objective!r}"
            )
        if objective == "l2" and alpha is not None:
            raise ValueError(
                "alpha applies to objective='quantile' only; "
                "objective='l2' predicts the conditional mean and has no alpha"
            )
        self.objective = objective
        self.alpha = 0.5 if (objective == "quantile" and alpha is None) else alpha
        self.params = dict(_DEFAULT_PARAMS if params is None else params)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model: LGBMRegressor | None = None
        self._columns: pd.Index | None = None

    def fit(self, y_train: pd.Series, x_train: pd.DataFrame | None = None) -> None:
        if x_train is None:
            raise ValueError("LGBMForecaster requires a feature matrix (x_train)")
        # Drop rows with a missing target; missing features are LightGBM's job.
        mask = y_train.notna()
        x = x_train.loc[mask]
        y = y_train.loc[mask]
        self._columns = x.columns  # freeze column order for predict-time reindex
        # alpha is only meaningful (and only accepted by LightGBM) under the
        # quantile objective -- under "regression" it would be misread as a
        # Huber scale parameter, so it is omitted entirely in the l2 branch.
        objective_kwargs: dict[str, Any] = (
            {"alpha": self.alpha} if self.objective == "quantile" else {}
        )
        self._model = LGBMRegressor(
            objective=_LGBM_OBJECTIVE[self.objective],
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            deterministic=True,  # reproducible given a fixed seed
            force_col_wise=True,  # removes a source of run-to-run variation
            verbose=-1,  # silence LightGBM's training chatter
            **objective_kwargs,
            **self.params,
        )
        self._model.fit(x, y)  # trained directly in EUR/MWh (no target transform)

    @property
    def fitted_estimator(self) -> LGBMRegressor:
        """Return the fitted underlying LightGBM estimator (for SHAP/inspection).

        Read-only accessor so downstream analysis (step 3.5 SHAP) explains exactly
        the delivered model rather than a separately refitted one. Raises RuntimeError
        if called before `fit`, consistent with the predict-before-fit contract.
        """
        if self._model is None:
            raise RuntimeError("estimator is not fitted; call fit() first")
        return self._model

    def predict(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.Series:
        if self._model is None or self._columns is None:
            raise RuntimeError("predict called before fit")
        if x_test is None:
            raise ValueError("LGBMForecaster requires a feature matrix (x_test)")
        # Defensive: identical column order as during fit.
        x = x_test.reindex(columns=self._columns)
        preds = self._model.predict(x)
        return pd.Series(np.asarray(preds), index=test_index, name="y_pred")
