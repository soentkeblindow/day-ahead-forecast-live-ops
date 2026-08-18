"""Smoke check for the commodities client — manual verification only, not run in CI.

Usage:
    uv run python scripts/smoke_check_commodities.py

Expected ranges (as of 2024):
    TTF Gas:  ~20–200 EUR/MWh
    EUA CO2:  ~20–100 EUR/t
"""

import pandas as pd

from energy_price_forecast.data.commodities_client import fetch_eua_co2, fetch_ttf_gas

START = pd.Timestamp("2023-01-01", tz="UTC")
END = pd.Timestamp.now(tz="UTC").normalize()

CHECKS = [
    ("TTF Gas", fetch_ttf_gas, "ttf_gas_eur_per_mwh"),
    ("EUA CO2", fetch_eua_co2, "eua_co2_eur_per_t"),
]

for name, fetch_fn, col in CHECKS:
    print(f"\n--- {name} ({START.date()} to {END.date()}) ---")
    df = fetch_fn(START, END)

    if df.empty:
        print(f">>> WARNING: {name} returned empty DataFrame — ticker may be broken or renamed.")
        continue

    print(f"Rows:  {len(df)}")
    print(f"First: {df.index[0].date()}  {df[col].iloc[0]:.2f}")
    print(f"Last:  {df.index[-1].date()}  {df[col].iloc[-1]:.2f}")
    print(f"Min:   {df[col].min():.2f}")
    print(f"Max:   {df[col].max():.2f}")
