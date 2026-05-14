"""Open-Meteo daily weather (historical archive + short-range blend forecast).

Returns SNOTEL-relevant covariates: snowfall, precipitation, temperature
min/mean/max, shortwave radiation, wind, snow_depth. Persistent incremental
caching mirrors `app/snotel.py`.
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

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "openmeteo"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "openmeteo_records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "shortwave_radiation_sum",
    "windspeed_10m_max",
    "snow_depth_max",
]

SCHEMA_VERSION = "sw_v1"


def _cache_path(lat: float, lon: float, start: date, end: date, kind: str) -> Path:
    key = f"{kind}_{lat:.3f}_{lon:.3f}_{start.isoformat()}_{end.isoformat()}.json"
    return CACHE_DIR / key


def _http_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"User-Agent": "snowwatch/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _record_path(lat: float, lon: float) -> Path:
    return RECORDS_DIR / f"hist_{lat:.3f}_{lon:.3f}.json"


def fetch_history(lat: float, lon: float, start: date, end: date, *, max_age_hours: int = 24) -> pd.DataFrame:
    rp = _record_path(lat, lon)
    rec: dict = {}
    if rp.exists():
        try:
            rec = json.loads(rp.read_text())
        except Exception:
            rec = {}
    have: dict = rec.get("rows", {})
    last_known = rec.get("last_known")
    first_known = rec.get("first_known") or (min(have.keys()) if have else None)
    fetched_at = rec.get("fetched_at", 0)
    age = time.time() - float(fetched_at) if fetched_at else float("inf")
    schema_stale = rec.get("schema_version") != SCHEMA_VERSION

    if NO_FETCH:
        return _slice(have, start, end)

    needs_backward = bool(first_known) and first_known > start.isoformat()
    needs_forward = (not last_known) or (last_known < end.isoformat())

    if (
        not schema_stale
        and not needs_backward
        and last_known
        and last_known >= end.isoformat()
        and age < max_age_hours * 3600
    ):
        return _slice(have, start, end)

    fwd_from = start
    if last_known:
        try:
            ld = date.fromisoformat(last_known)
            if ld >= start:
                fwd_from = ld - timedelta(days=2)
        except ValueError:
            pass

    fetched_any = False
    if needs_backward:
        bwd_from = start
        bwd_to = date.fromisoformat(first_known) - timedelta(days=1) if first_known else end
        if bwd_from <= bwd_to:
            fetched_any = _fetch_and_merge(lat, lon, have, bwd_from, bwd_to) or fetched_any
    if needs_forward and fwd_from <= end:
        fetched_any = _fetch_and_merge(lat, lon, have, fwd_from, end) or fetched_any

    if fetched_any and have:
        rp.write_text(json.dumps({
            "lat": lat, "lon": lon, "rows": have,
            "first_known": min(have.keys()), "last_known": max(have.keys()),
            "fetched_at": time.time(), "schema_version": SCHEMA_VERSION,
        }, separators=(",", ":")))

    return _slice(have, start, end)


def _fetch_and_merge(lat: float, lon: float, have: dict, start: date, end: date) -> bool:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC",
    }
    try:
        payload = _http_json(ARCHIVE_URL + "?" + urlencode(params))
    except Exception:
        return False
    df_new = _to_df(payload)
    added = False
    for _, r in df_new.iterrows():
        k = r["date"].isoformat() if hasattr(r["date"], "isoformat") else str(r["date"])
        cell = {col: (None if pd.isna(r[col]) else float(r[col])) for col in df_new.columns if col != "date"}
        if k not in have:
            added = True
        have[k] = cell
    return added or not df_new.empty


def _slice(have: dict, start: date, end: date) -> pd.DataFrame:
    if not have:
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    s, e = start.isoformat(), end.isoformat()
    rows = []
    for d_iso, vals in have.items():
        if not (s <= d_iso <= e):
            continue
        row = {"date": d_iso}
        for v in DAILY_VARS:
            row[v] = vals.get(v) if isinstance(vals, dict) else None
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def fetch_forecast(lat: float, lon: float, days: int = 14, *, max_age_hours: int = 3) -> pd.DataFrame:
    today = date.today()
    end = today + timedelta(days=days)
    cache = _cache_path(lat, lon, today, end, "fcst")
    if NO_FETCH:
        if cache.exists():
            return _to_df(json.loads(cache.read_text()))
        candidates = sorted(
            CACHE_DIR.glob(f"fcst_{lat:.3f}_{lon:.3f}_*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if candidates:
            return _to_df(json.loads(candidates[0].read_text()))
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        payload = json.loads(cache.read_text())
    else:
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "daily": ",".join(DAILY_VARS),
            "forecast_days": min(days, 16),
            "timezone": "UTC",
        }
        payload = _http_json(FORECAST_URL + "?" + urlencode(params))
        cache.write_text(json.dumps(payload))
    return _to_df(payload)


def _to_df(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
