"""Post-hoc monotonicity fix for a multi-level quantile grid (Sprint 4.3c).

Scaled conformal calibration (``evaluation.conformal``) shifts each of the
seven quantile levels independently (``q_tilde_a = q_a + Q_a * sigma``). That
fixes marginal coverage but does not guarantee the seven calibrated series
stay ordered within an hour: at some hours ``q_tilde_0.05 > q_tilde_0.10``,
etc. ``conformal.py`` reports this rate but deliberately does not correct it
-- crossing is a real signal about noise in the Q_alpha estimates, not merely
a cosmetic defect, and must stay visible on the calibrated-but-unsorted
artifact.

This module is the opposite, explicit step: the classical
Chernozhukov/Fenton/Galichon (2010) rearrangement operator. For each hour,
sort the seven values and reassign them to the levels in ascending order --
the i-th smallest value becomes the forecast for the i-th smallest level.
This guarantees ``crossing_rate == 0`` by construction, at the cost of
relabelling which level a given value belongs to.

Caveat kept here, not just in the spec: rearrangement does not change which
values occur at a given hour, only which level each is assigned to. The
average pinball loss across all seven levels is invariant under it, but an
individual level's pinball / coverage can shift slightly (a value that used
to be "the q_0.10 forecast" may become "the q_0.05 forecast" at a crossing
hour). It is a sharpness/consistency fix, not a fresh calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rearrange_quantiles(preds: dict[float, pd.Series]) -> dict[float, pd.Series]:
    """Sort each hour's values across levels so the grid is rank-consistent.

    ``preds`` maps each level to its forecast series; all series must share
    an identical index. Returns a same-shaped dict where, for every hour, the
    i-th smallest input value is assigned to the i-th smallest level -- i.e.
    every output row is monotonically increasing in the level. Pure and
    vectorized (one ``np.sort`` over the whole grid, no per-hour Python
    loop).
    """
    levels = sorted(preds)
    indices = [preds[a].index for a in levels]
    if not all(indices[0].equals(idx) for idx in indices[1:]):
        raise ValueError("all levels in preds must share an identical index")
    index = indices[0]

    grid = np.column_stack([preds[a].to_numpy() for a in levels])
    sorted_grid = np.sort(grid, axis=1)

    return {
        a: pd.Series(sorted_grid[:, i], index=index, name=preds[a].name)
        for i, a in enumerate(levels)
    }
