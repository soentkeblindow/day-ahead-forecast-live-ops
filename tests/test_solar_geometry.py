import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.features.solar_geometry import solar_position

_FIXTURE_PATH = "tests/fixtures/solar_position_reference.csv"
_TOLERANCE_DEG = 0.5

# he_th from data/weather_grid.py (spec 6.5.1 F7 reference point).
_REFERENCE_LAT, _REFERENCE_LON = 50.8, 9.8


def _local_noon_utc(date: str) -> pd.Timestamp:
    return pd.Timestamp(f"{date}T12:00:00", tz="Europe/Berlin").tz_convert("UTC")


def test_matches_pvlib_reference_fixture() -> None:
    ref = pd.read_csv(_FIXTURE_PATH)
    ref["timestamp_utc"] = pd.to_datetime(ref["timestamp_utc"], utc=True)

    for (lat, lon), group in ref.groupby(["latitude", "longitude"]):
        timestamps = pd.DatetimeIndex(group["timestamp_utc"])
        # groupby() keys are typed Hashable by the pandas stubs, but these two
        # columns are float64 in the fixture -- the cast is always safe here.
        mine = solar_position(timestamps, float(lat), float(lon))  # type: ignore[arg-type]

        elevation_diff = np.abs(mine["elevation"].to_numpy() - group["elevation"].to_numpy())
        assert elevation_diff.max() < _TOLERANCE_DEG, f"elevation mismatch at ({lat}, {lon})"

        # Azimuth is circular -- wrap the difference into [-180, 180] before comparing.
        azimuth_diff = (mine["azimuth"].to_numpy() - group["azimuth"].to_numpy() + 180) % 360 - 180
        assert np.abs(azimuth_diff).max() < _TOLERANCE_DEG, f"azimuth mismatch at ({lat}, {lon})"


def test_elevation_higher_at_summer_solstice_than_equinox_at_local_noon() -> None:
    summer = solar_position(
        pd.DatetimeIndex([_local_noon_utc("2025-06-21")]), _REFERENCE_LAT, _REFERENCE_LON
    )
    equinox = solar_position(
        pd.DatetimeIndex([_local_noon_utc("2025-03-20")]), _REFERENCE_LAT, _REFERENCE_LON
    )
    assert summer["elevation"].iloc[0] > equinox["elevation"].iloc[0]


def test_elevation_lower_at_winter_solstice_than_equinox_at_local_noon() -> None:
    winter = solar_position(
        pd.DatetimeIndex([_local_noon_utc("2025-12-21")]), _REFERENCE_LAT, _REFERENCE_LON
    )
    equinox = solar_position(
        pd.DatetimeIndex([_local_noon_utc("2025-09-22")]), _REFERENCE_LAT, _REFERENCE_LON
    )
    assert winter["elevation"].iloc[0] < equinox["elevation"].iloc[0]


def test_south_points_higher_than_north_points_at_local_noon() -> None:
    # rp_sl (south, 49.8N) vs sh_west (north, 54.4N) from data/weather_grid.py.
    noon = pd.DatetimeIndex([_local_noon_utc("2025-06-21")])
    south = solar_position(noon, 49.8, 7.5)
    north = solar_position(noon, 54.4, 9.0)
    assert south["elevation"].iloc[0] > north["elevation"].iloc[0]


@pytest.mark.parametrize("date", ["2026-03-29", "2026-10-25"])
def test_night_elevation_is_negative_on_both_dst_changeover_days(date: str) -> None:
    midnight_utc = pd.Timestamp(f"{date}T00:00:00Z")
    result = solar_position(pd.DatetimeIndex([midnight_utc]), _REFERENCE_LAT, _REFERENCE_LON)
    assert result["elevation"].iloc[0] < 0.0


def test_naive_timestamp_raises() -> None:
    naive = pd.DatetimeIndex([pd.Timestamp("2025-06-21T12:00:00")])
    with pytest.raises(ValueError, match="UTC"):
        solar_position(naive, _REFERENCE_LAT, _REFERENCE_LON)


def test_non_utc_timestamp_raises() -> None:
    berlin = pd.DatetimeIndex([pd.Timestamp("2025-06-21T12:00:00", tz="Europe/Berlin")])
    with pytest.raises(ValueError, match="UTC"):
        solar_position(berlin, _REFERENCE_LAT, _REFERENCE_LON)
