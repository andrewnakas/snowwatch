"""Unit tests for ensemble-member construction edge cases (app/forecast.py)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.forecast import _build_nbm_wx_for_snow17


def _wx_fcst(n, start=date(2026, 1, 10)):
    return pd.DataFrame({
        "date": [start + timedelta(days=i) for i in range(n)],
        "precipitation_sum": [5.0] * n,
        "temperature_2m_mean": [-2.0] * n,
        "temperature_2m_max": [0.0] * n,
        "temperature_2m_min": [-5.0] * n,
        "snowfall_sum": [4.0] * n,
        "shortwave_radiation_sum": [8.0] * n,
        "windspeed_10m_max": [12.0] * n,
    })


def _nbm_fcst(n, start=date(2026, 1, 10)):
    return pd.DataFrame({
        "date": [start + timedelta(days=i) for i in range(n)],
        "nbm_snowfall_sum": [6.0] * n,
        "nbm_precip_sum": [7.0] * n,
        "nbm_tmean": [-3.0] * n,
        "nbm_tmax": [-1.0] * n,
        "nbm_tmin": [-6.0] * n,
    })


class TestBuildNbmWxForSnow17:
    def test_no_nbm_returns_empty(self):
        # Empty NBM must NOT fall back to Open-Meteo wholesale: that would
        # make nbm_snow17 a duplicate of the snow17 member and double
        # SNOW-17's weight in the blend (every AK station, any NBM outage).
        out = _build_nbm_wx_for_snow17(pd.DataFrame(), _wx_fcst(7), horizon=7)
        assert out.empty
        out = _build_nbm_wx_for_snow17(None, _wx_fcst(7), horizon=7)
        assert out.empty

    def test_nbm_values_win_over_wx(self):
        out = _build_nbm_wx_for_snow17(_nbm_fcst(7), _wx_fcst(7), horizon=7)
        assert len(out) == 7
        assert (out["precipitation_sum"] == 7.0).all()
        assert (out["temperature_2m_mean"] == -3.0).all()
        # Radiation/wind come from Open-Meteo (NBM doesn't carry them).
        assert (out["shortwave_radiation_sum"] == 8.0).all()

    def test_short_nbm_tail_fills_from_wx(self):
        out = _build_nbm_wx_for_snow17(_nbm_fcst(3), _wx_fcst(7), horizon=7)
        assert len(out) == 7
        assert (out["precipitation_sum"].iloc[:3] == 7.0).all()
        assert (out["precipitation_sum"].iloc[3:] == 5.0).all()

    def test_missing_nbm_precip_backfilled_from_snowfall(self):
        nbm = _nbm_fcst(2)
        nbm["nbm_precip_sum"] = [None, None]
        out = _build_nbm_wx_for_snow17(nbm, _wx_fcst(2), horizon=2)
        # 6 cm snow at 10:1 SLR -> 6 mm SWE
        assert (out["precipitation_sum"] == 6.0).all()
