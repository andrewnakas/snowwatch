"""NOAA NBM (National Blend of Models) CONUS daily snowfall + temp forecast.

Open-Meteo exposes NBM via the `ncep_nbm_conus` model. Returns calibrated
daily snowfall and precip out to 11 days — the upper bound NBM supports. Days
12-14 stay NaN and are downstream-filled by the blend.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

NO_FETCH = os.environ.get("SW_NO_FETCH") == "1"
NBM_OFF = os.environ.get("SW_NBM_OFF") == "1"

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "nbm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NBM_URL = "https://api.open-meteo.com/v1/forecast"
NBM_MAX_DAYS = 11

DAILY_VARS = (
    "snowfall_sum",
    "precipitation_sum",
    "precipitation_probability_mean",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
)

OUT_COLS = (
    "nbm_snowfall_sum",
    "nbm_precip_sum",
    "nbm_pop_mean",
    "nbm_tmean",
    "nbm_tmax",
    "nbm_tmin",
)
_RENAME = dict(zip(DAILY_VARS, OUT_COLS))


def _cache_path(lat: float, lon: float, days: int) -> Path:
    return CACHE_DIR / f"nbm_{lat:.3f}_{lon:.3f}_{days}.json"


def _http_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"User-Agent": "snowwatch/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_forecast(lat: float, lon: float, days: int = NBM_MAX_DAYS, *, max_age_hours: int = 3) -> pd.DataFrame:
    if NBM_OFF:
        return pd.DataFrame(columns=["date"] + list(OUT_COLS))
    days = min(days, NBM_MAX_DAYS)
    cache = _cache_path(lat, lon, days)
    if NO_FETCH:
        if cache.exists():
            return _to_df(json.loads(cache.read_text()))
        candidates = sorted(CACHE_DIR.glob(f"nbm_{lat:.3f}_{lon:.3f}_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return _to_df(json.loads(candidates[0].read_text()))
        return pd.DataFrame(columns=["date"] + list(OUT_COLS))
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        payload = json.loads(cache.read_text())
    else:
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "daily": ",".join(DAILY_VARS),
            "forecast_days": days,
            "timezone": "UTC",
            "models": "ncep_nbm_conus",
        }
        try:
            payload = _http_json(NBM_URL + "?" + urlencode(params))
            cache.write_text(json.dumps(payload))
        except Exception:
            return pd.DataFrame(columns=["date"] + list(OUT_COLS))
    return _to_df(payload)


def _to_df(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        return pd.DataFrame(columns=["date"] + list(OUT_COLS))
    df = pd.DataFrame(daily).rename(columns={"time": "date", **_RENAME})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keep = ["date"] + [c for c in OUT_COLS if c in df.columns]
    return df[keep]
