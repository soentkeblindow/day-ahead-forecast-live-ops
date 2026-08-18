"""One-off generator for the Beleg C golden-fixture regression anchor (Sprint 6 / 6.1).

Run once from the repo root: `uv run python scripts/freeze_golden_fixture.py`.
Regenerating overwrites tests/fixtures/golden_lgbm_predictions.parquet; the
corresponding test (tests/test_provenance.py) re-derives predictions from the
same in-code synthetic dataset and asserts bit-identical equality against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from energy_price_forecast.models.lgbm import LGBMForecaster  # noqa: E402
from tests.test_provenance import FIXTURE_PATH, make_golden_dataset  # noqa: E402


def main() -> None:
    x_train, x_test, y_train, test_index = make_golden_dataset()

    model = LGBMForecaster()
    model.fit(y_train, x_train)
    preds = model.predict(test_index, history=y_train, x_test=x_test)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_frame().to_parquet(FIXTURE_PATH)
    print(f"Wrote {len(preds)} golden predictions to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
