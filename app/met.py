"""Multi-model Open-Meteo forecast layer.

One module owns every Open-Meteo *forecast* call the pipeline makes:

  fetch_multimodel(lat, lon)  — NBM + HRRR + GFS + ECMWF IFS + AIFS daily
                                snow/precip/temp in ONE request, wide columns
                                `{model}_{var}` (model ∈ MODEL_KEYS).
  fetch_ensemble(lat, lon)    — GEFS + AIFS-ENS member snowfall/precip reduced
                                immediately to per-day spread stats.
  fetch_profile(lat, lon)     — hourly pressure-level temp/wind/geopotential
                                for the SLR random forest (storm-gated by the
                                caller; expensive otherwise).

All calls run through a per-process weighted budget with a degradation
ladder: when the budget is exhausted (or the API 429s repeatedly) the
fetchers return stale cache or empty frames instead of failing the shard.
`SW_NO_FETCH=1` serves cache only, mirroring app/weather.py semantics.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def _no_fetch() -> bool:
    return os.environ.get("SW_NO_FETCH") == "1"


CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "met"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Open-Meteo model id → short column prefix. Order matters only for display.
# NOTE: `gfs025`/`ecmwf_aifs025` look valid but return all-null daily values
# on this endpoint; `gfs_global` and `ecmwf_aifs025_single` are the working
# ids (verified 2026-06).
MODELS = {
    "ncep_nbm_conus": "nbm",
    "ncep_hrrr_conus": "hrrr",
    "gfs_global": "gfs",
    "ecmwf_ifs025": "ifs",
    "ecmwf_aifs025_single": "aifs",
}
MODEL_KEYS = tuple(MODELS.values())

DAILY_VARS = (
    "snowfall_sum",                    # cm
    "precipitation_sum",               # mm
    "temperature_2m_mean",             # C
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_mean",  # %
)
VAR_SHORT = {
    "snowfall_sum": "snowfall",
    "precipitation_sum": "precip",
    "temperature_2m_mean": "tmean",
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "precipitation_probability_mean": "pop",
}

ENSEMBLE_MODELS = "gfs025,ecmwf_aifs025"
ENSEMBLE_VARS = ("snowfall_sum", "precipitation_sum")

PROFILE_LEVELS = (1000, 925, 850, 700, 600, 500)
# relative_humidity is the key addition: with temperature it gives wet-bulb
# (phase) and locates RH-weighted crystal growth in the dendritic zone (SLR).
PROFILE_VARS = ("temperature", "relative_humidity", "wind_speed", "geopotential_height")

# ---------- call budget ------------------------------------------------------
# Open-Meteo's free tier weights a call by variables×days×models. These are
# deliberately conservative estimates; the point is graceful degradation, not
# precise accounting. Reset per process (one shard build = one process).
_COST = {"multimodel": 7.0, "ensemble": 22.0, "profile": 4.0, "other": 1.0}
_DEFAULT_BUDGET = float(os.environ.get("SW_MET_BUDGET", "9000"))
_spent = 0.0
_consecutive_429 = 0


def budget_left() -> float:
    return _DEFAULT_BUDGET - _spent


def _charge(kind: str) -> bool:
    """Reserve budget for one call; False means degrade (serve cache/empty)."""
    global _spent
    if _consecutive_429 >= 5:
        return False
    cost = _COST.get(kind, 1.0)
    if _spent + cost > _DEFAULT_BUDGET:
        return False
    _spent += cost
    return True


def _http_json(url: str, timeout: int = 60, *, retries: int = 2) -> Optional[dict]:
    global _consecutive_429
    req = Request(url, headers={"User-Agent": "snowwatch/0.2"})
    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                _consecutive_429 = 0
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                _consecutive_429 += 1
                if attempt < retries:
                    time.sleep(5 * (attempt + 1))
                    continue
                return None
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def _cached_or_empty(prefix: str, lat: float, lon: float, to_df, empty_cols: list[str]) -> pd.DataFrame:
    candidates = sorted(CACHE_DIR.glob(f"{prefix}_{lat:.3f}_{lon:.3f}*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            return to_df(json.loads(p.read_text()))
        except Exception:
            continue
    return pd.DataFrame(columns=empty_cols)


# ---------- multi-model deterministic ---------------------------------------

def _mm_cols() -> list[str]:
    return [f"{m}_{VAR_SHORT[v]}" for m in MODEL_KEYS for v in DAILY_VARS]


def _mm_to_df(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        return pd.DataFrame(columns=["date"] + _mm_cols())
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    rename = {}
    for om_id, short in MODELS.items():
        for v in DAILY_VARS:
            rename[f"{v}_{om_id}"] = f"{short}_{VAR_SHORT[v]}"
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keep = ["date"] + [c for c in _mm_cols() if c in df.columns]
    return df[keep]


def fetch_multimodel(lat: float, lon: float, days: int = 7, *, max_age_hours: float = 5.0) -> pd.DataFrame:
    """Daily forecasts from all MODELS in one request, columns {model}_{var}.

    HRRR covers only ~2 days and NBM ~11; missing model-days are NaN, which
    downstream consumers (LightGBM, blend) tolerate natively.
    """
    cache = CACHE_DIR / f"mm_{lat:.3f}_{lon:.3f}_{days}.json"
    if _no_fetch():
        return _cached_or_empty("mm", lat, lon, _mm_to_df, ["date"] + _mm_cols())
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        try:
            return _mm_to_df(json.loads(cache.read_text()))
        except Exception:
            pass
    if not _charge("multimodel"):
        return _cached_or_empty("mm", lat, lon, _mm_to_df, ["date"] + _mm_cols())
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily": ",".join(DAILY_VARS),
        "forecast_days": days,
        "timezone": "UTC",
        "models": ",".join(MODELS.keys()),
    }
    payload = _http_json(FORECAST_URL + "?" + urlencode(params))
    if payload is None or "daily" not in payload:
        return _cached_or_empty("mm", lat, lon, _mm_to_df, ["date"] + _mm_cols())
    cache.write_text(json.dumps(payload, separators=(",", ":")))
    return _mm_to_df(payload)


# ---------- ensemble spread stats --------------------------------------------

ENS_COLS = [
    "ens_snow_mean", "ens_snow_std", "ens_snow_p10", "ens_snow_p50",
    "ens_snow_p90", "ens_snow_prob_pos", "ens_precip_mean", "ens_precip_std",
    "ens_n_members",
]


def _ens_reduce(payload: dict) -> pd.DataFrame:
    """Reduce raw per-member ensemble daily values to spread stats per day."""
    daily = payload.get("daily") or {}
    times = daily.get("time")
    if not times:
        return pd.DataFrame(columns=["date"] + ENS_COLS)
    snow_cols = [k for k in daily if k.startswith("snowfall_sum")]
    precip_cols = [k for k in daily if k.startswith("precipitation_sum")]
    snow = np.array([daily[k] for k in snow_cols], dtype=float)    # (members, days)
    precip = np.array([daily[k] for k in precip_cols], dtype=float)
    rows = []
    for i, t in enumerate(times):
        s = snow[:, i] if snow.size else np.array([])
        p = precip[:, i] if precip.size else np.array([])
        s = s[np.isfinite(s)]
        p = p[np.isfinite(p)]
        rows.append({
            "date": t,
            "ens_snow_mean": round(float(np.mean(s)), 3) if s.size else None,
            "ens_snow_std": round(float(np.std(s)), 3) if s.size else None,
            "ens_snow_p10": round(float(np.percentile(s, 10)), 3) if s.size else None,
            "ens_snow_p50": round(float(np.percentile(s, 50)), 3) if s.size else None,
            "ens_snow_p90": round(float(np.percentile(s, 90)), 3) if s.size else None,
            "ens_snow_prob_pos": round(float(np.mean(s > 0.5)), 3) if s.size else None,  # >0.5cm
            "ens_precip_mean": round(float(np.mean(p)), 3) if p.size else None,
            "ens_precip_std": round(float(np.std(p)), 3) if p.size else None,
            "ens_n_members": int(s.size),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def fetch_ensemble(lat: float, lon: float, days: int = 7, *, max_age_hours: float = 11.0) -> pd.DataFrame:
    """GEFS + AIFS-ENS member snowfall/precip reduced to daily spread stats.

    The raw payload is large (~80 members × vars × days) so only the reduced
    stats are cached. Callers gate this to the 00Z/12Z builds via
    SW_ENSEMBLE_BUILD; on other builds the (≤11h old) cache is reused.
    """
    cache = CACHE_DIR / f"ens_{lat:.3f}_{lon:.3f}_{days}.json"

    def _from_cache(payload: dict) -> pd.DataFrame:
        df = pd.DataFrame(payload)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    if _no_fetch() or os.environ.get("SW_ENSEMBLE_BUILD") != "1":
        out = _cached_or_empty("ens", lat, lon, _from_cache, ["date"] + ENS_COLS)
        return out
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        try:
            return _from_cache(json.loads(cache.read_text()))
        except Exception:
            pass
    if not _charge("ensemble"):
        return _cached_or_empty("ens", lat, lon, _from_cache, ["date"] + ENS_COLS)
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily": ",".join(ENSEMBLE_VARS),
        "forecast_days": days,
        "timezone": "UTC",
        "models": ENSEMBLE_MODELS,
    }
    payload = _http_json(ENSEMBLE_URL + "?" + urlencode(params), timeout=90)
    if payload is None or "daily" not in payload:
        return _cached_or_empty("ens", lat, lon, _from_cache, ["date"] + ENS_COLS)
    df = _ens_reduce(payload)
    try:
        ser = df.copy()
        ser["date"] = ser["date"].astype(str)
        cache.write_text(ser.to_json(orient="records"))
    except Exception:
        pass
    return df


def fetch_ensemble_cached_records(payload_text: str) -> pd.DataFrame:
    df = pd.read_json(payload_text, orient="records")
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------- pressure-level profiles (for the SLR random forest) -------------

def _profile_vars() -> list[str]:
    return [f"{v}_{lev}hPa" for v in PROFILE_VARS for lev in PROFILE_LEVELS]


def fetch_profile(lat: float, lon: float, days: int = 7, *, max_age_hours: float = 5.0) -> pd.DataFrame:
    """Hourly pressure-level temperature/wind/geopotential (GFS) for SLR
    prediction. Callers should storm-gate this: only fetch when meaningful
    QPF is forecast, which keeps it out of ~75% of station-builds."""
    cache = CACHE_DIR / f"prof_{lat:.3f}_{lon:.3f}_{days}.json"

    def _to_df(payload: dict) -> pd.DataFrame:
        hourly = payload.get("hourly") or {}
        if not hourly.get("time"):
            return pd.DataFrame()
        df = pd.DataFrame(hourly).rename(columns={"time": "datetime"})
        return df

    if _no_fetch():
        return _cached_or_empty("prof", lat, lon, _to_df, [])
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        try:
            return _to_df(json.loads(cache.read_text()))
        except Exception:
            pass
    if not _charge("profile"):
        return _cached_or_empty("prof", lat, lon, _to_df, [])
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": ",".join(_profile_vars()),
        "forecast_days": days,
        "timezone": "UTC",
        "models": "gfs025",
        "temporal_resolution": "hourly_6",
    }
    payload = _http_json(FORECAST_URL + "?" + urlencode(params), timeout=90)
    if payload is None or "hourly" not in payload:
        return _cached_or_empty("prof", lat, lon, _to_df, [])
    cache.write_text(json.dumps(payload, separators=(",", ":")))
    return _to_df(payload)


# ---------- compatibility shim ----------------------------------------------

NBM_COMPAT_RENAME = {
    "nbm_snowfall": "nbm_snowfall_sum",
    "nbm_precip": "nbm_precip_sum",
    "nbm_pop": "nbm_pop_mean",
    # nbm_tmean / nbm_tmax / nbm_tmin already match app/nbm.py's contract.
}


def nbm_compat(mm: pd.DataFrame) -> pd.DataFrame:
    """Slice the multi-model frame down to app/nbm.py's column contract so
    existing members (nbm_snow17, nbm_snowfall, daily_forecast) are untouched."""
    if mm is None or mm.empty:
        return pd.DataFrame(columns=["date", "nbm_snowfall_sum", "nbm_precip_sum",
                                     "nbm_pop_mean", "nbm_tmean", "nbm_tmax", "nbm_tmin"])
    cols = ["date"] + [c for c in mm.columns if c.startswith("nbm_")]
    out = mm[cols].rename(columns=NBM_COMPAT_RENAME)
    # NBM rows past its ~11-day horizon are NaN in the merged frame; drop
    # all-NaN rows so consumers see the same shape app/nbm.py produced.
    val_cols = [c for c in out.columns if c != "date"]
    return out.dropna(subset=val_cols, how="all").reset_index(drop=True)
