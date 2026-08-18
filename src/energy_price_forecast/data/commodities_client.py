# Deliberate design choices vs. the ENTSO-E client (see spec §1.1):
# 1. Daily granularity only — yfinance futures data has no intra-day history.
#    Forward-fill to hourly grid happens in Sprint 2 (feature engineering).
# 2. No cache — a single yfinance call fetches years of data in seconds.
#    Add a simple cache in step 7 if EDA reveals it is worth the complexity.
# 3. No retry logic — yfinance has built-in retries; wrapping them adds nothing.

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TTF_TICKER = "TTF=F"  # Dutch TTF Natural Gas Futures
EUA_TICKER = "CO2.L"  # ICE EUA Futures (London)


def _fetch_yahoo_history(
    ticker: str,
    col_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch daily close prices from Yahoo Finance for one futures ticker.

    Returns a single-column DataFrame with a UTC DatetimeIndex (trading days
    only; weekends and holidays are absent). No forward-fill is applied here —
    that belongs in Sprint 2 feature engineering.

    Empty response: logs a warning and returns an empty DataFrame with the
    correct column and index type. yfinance network exceptions propagate to the
    caller unmodified. This asymmetry is intentional: missing data is not a
    code error; a network failure should fail loudly so the caller can retry.
    """
    start_utc = start.tz_convert("UTC") if start.tzinfo is not None else start.tz_localize("UTC")
    end_utc = end.tz_convert("UTC") if end.tzinfo is not None else end.tz_localize("UTC")

    # yfinance uses a half-open [start, end) interval; add one day to make end inclusive.
    yf_end = end_utc + pd.Timedelta(days=1)

    raw: pd.DataFrame = yf.Ticker(ticker).history(start=start_utc, end=yf_end)

    if raw.empty:
        logger.warning(
            "yfinance returned no data for ticker %s for %s–%s", ticker, start_utc, end_utc
        )
        return pd.DataFrame(columns=[col_name], index=pd.DatetimeIndex([], tz="UTC"))

    # Normalize timezone: distinguish naive vs. tz-aware explicitly (not via try/except).
    # Cast to DatetimeIndex so mypy can resolve .tz, .tz_localize, .tz_convert.
    dt_index = pd.DatetimeIndex(raw.index)
    if dt_index.tz is None:
        raw.index = dt_index.tz_localize("UTC")
    else:
        raw.index = dt_index.tz_convert("UTC")

    result = raw[["Close"]].rename(columns={"Close": col_name})

    # Warn if the result is suspiciously sparse — may indicate a broken or renamed ticker.
    expected_trading_days = (end_utc - start_utc).days * 5 / 7
    if expected_trading_days > 0 and len(result) < 0.5 * expected_trading_days:
        logger.warning(
            "Ticker %s returned only %d rows for %s–%s (expected ~%d trading days). "
            "The ticker may be broken or renamed.",
            ticker,
            len(result),
            start_utc,
            end_utc,
            int(expected_trading_days),
        )

    return result


def fetch_ttf_gas(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch daily TTF Natural Gas close prices in EUR/MWh.

    Returns a DataFrame with a UTC DatetimeIndex (daily, trading days only) and
    a single column ``ttf_gas_eur_per_mwh``. Weekends and holidays are absent at
    source; forward-fill to hourly granularity (4-day limit) is applied in
    ``load_all_data`` after merging.

    Args:
        start: Inclusive start date. Any timezone is accepted; converted to UTC.
        end: Inclusive end date. Any timezone is accepted; converted to UTC.

    Returns:
        DataFrame with columns [``ttf_gas_eur_per_mwh``] and UTC DatetimeIndex.
    """
    return _fetch_yahoo_history(TTF_TICKER, "ttf_gas_eur_per_mwh", start, end)


def fetch_eua_co2(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch daily EUA CO2 allowance close prices in EUR/t.

    Returns a DataFrame with a UTC DatetimeIndex (daily, trading days only) and
    a single column ``eua_co2_eur_per_t``. Weekends and holidays are absent at
    source; forward-fill to hourly granularity (4-day limit) is applied in
    ``load_all_data`` after merging. Data is only available from October 2021
    onward — earlier requests return an empty DataFrame.

    Args:
        start: Inclusive start date. Any timezone is accepted; converted to UTC.
        end: Inclusive end date. Any timezone is accepted; converted to UTC.

    Returns:
        DataFrame with columns [``eua_co2_eur_per_t``] and UTC DatetimeIndex.
    """
    return _fetch_yahoo_history(EUA_TICKER, "eua_co2_eur_per_t", start, end)
