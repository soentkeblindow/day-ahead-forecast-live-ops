"""Smoke check for load_all_data — manual verification only, not run in CI.

Pulls one month of real data via all eight fetchers and validates that the
merged result is non-empty and has the expected shape.

Usage:
    uv run python scripts/smoke_check_loader.py
"""

import logging

import pandas as pd

from energy_price_forecast.data.loaders import load_all_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2024-01-31 23:00", tz="UTC")

print(f"\n--- load_all_data ({START.date()} to {END.date()}) ---")
df = load_all_data(START, END)

if df.empty:
    print(">>> WARNING: result is empty — check API key and individual fetchers.")
else:
    print(f"Shape:      {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Index:      {df.index[0]}  →  {df.index[-1]}")
    print(f"Timezone:   {pd.DatetimeIndex(df.index).tz}")
    print("\nNaN summary (columns with missing values):")
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if nan_cols.empty:
        print("  none")
    else:
        for col, count in nan_cols.items():
            pct = 100 * count / len(df)
            print(f"  {col}: {count} ({pct:.1f}%)")
    print(
        f"\nday_ahead_price — min: {df['day_ahead_price'].min():.2f}  "
        f"max: {df['day_ahead_price'].max():.2f}  "
        f"mean: {df['day_ahead_price'].mean():.2f}"
    )
