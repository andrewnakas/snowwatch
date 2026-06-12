"""NWS api.weather.gov forecast fetcher.

NBM (`app/nbm.py`) is a raw model output; this is the *forecaster-touched*
gridded forecast — the official NWS public-facing prediction. Each WFO
office can hand-edit the grid before publication, so this can disagree with
NBM for the same point.

Endpoint pattern:
  /points/{lat},{lon}                   -> resolves to (office, gridX, gridY)
  /gridpoints/{office}/{gridX},{gridY}  -> snowfallAmount, QPF, temperature,
                                            snowLevel, POP, ...

Forecast bins are variable-width ISO durations (`PT1H`, `PT5H`, `PT6H`...).
We aggregate to UTC daily so `divergence_vs_nws()` can compare apples to
apples with our daily-step ensemble.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

NO_FETCH = os.environ.get("SW_NO_FETCH") == "1"
NWS_OFF = os.environ.get("SW_NWS_OFF") == "1"

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "nws"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UA = "snowwatch (https://andrewnakas.github.io/snowwatch)"
POINTS_URL = "https://api.weather.gov/points"

# Daily totals/aggregates we surface
OUT_COLS = (
    "nws_snowfall_in",     # daily total snowfall (in)
    "nws_precip_in",       # daily QPF (in)
    "nws_tmean_f",         # mean of hourly temperature (°F)
    "nws_tmax_f",          # max
    "nws_tmin_f",          # min
    "nws_snow_level_ft",   # daily mean snow level (ft AMSL)
    "nws_pop_pct",         # max POP across the day
    "nws_office",          # WFO that issued the forecast
)


def _http_json(url: str, timeout: int = 60, *, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA, "Accept": "application/geo+json"})
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt + 1)
    return None


_DURATION_RE = re.compile(r"PT(\d+)H")


def _parse_bin(valid: str) -> Optional[Tuple[datetime, datetime]]:
    """`2026-05-15T13:00:00+00:00/PT6H` -> (start_dt, end_dt) in UTC."""
    if not valid or "/" not in valid:
        return None
    start_s, dur_s = valid.split("/", 1)
    try:
        start = datetime.fromisoformat(start_s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None
    m = _DURATION_RE.match(dur_s)
    if not m:
        return None
    return start, start + timedelta(hours=int(m.group(1)))


def _accumulate_per_day(values: list, *, day_index: dict, mode: str = "sum"):
    """Distribute each variable-width bin across the UTC days it spans.

    `mode` controls aggregation: "sum" partitions the bin's value
    proportionally to hours-in-day (right for snowfall/precip totals);
    "mean" averages the value weighted by hours-in-day; "max"/"min" pick
    extremes per day.
    """
    for v in values:
        bin_window = _parse_bin(v.get("validTime", ""))
        if bin_window is None or v.get("value") is None:
            continue
        start, end = bin_window
        val = float(v["value"])
        # Walk one day at a time, clipping bin to the day boundary.
        cur = start
        while cur < end:
            day_start = datetime(cur.year, cur.month, cur.day, tzinfo=timezone.utc)
            day_end = day_start + timedelta(days=1)
            slice_start = max(cur, day_start)
            slice_end = min(end, day_end)
            hours_in_slice = (slice_end - slice_start).total_seconds() / 3600.0
            if hours_in_slice <= 0:
                cur = day_end
                continue
            d = day_start.date()
            cell = day_index.setdefault(d, {})
            if mode == "sum":
                # Proportional split — preserves total accumulation.
                bin_hours = (end - start).total_seconds() / 3600.0
                cell.setdefault("_sum", 0.0)
                cell.setdefault("_n", 0.0)
                cell["_sum"] += val * (hours_in_slice / max(bin_hours, 1e-6))
            elif mode == "mean":
                cell.setdefault("_w_val", 0.0)
                cell.setdefault("_w", 0.0)
                cell["_w_val"] += val * hours_in_slice
                cell["_w"] += hours_in_slice
            elif mode == "max":
                cur_max = cell.get("_max")
                cell["_max"] = val if cur_max is None else max(cur_max, val)
            elif mode == "min":
                cur_min = cell.get("_min")
                cell["_min"] = val if cur_min is None else min(cur_min, val)
            cur = day_end
    # Reduce
    out = {}
    for d, cell in day_index.items():
        if mode == "sum":
            out[d] = cell.get("_sum")
        elif mode == "mean":
            out[d] = (cell["_w_val"] / cell["_w"]) if cell.get("_w") else None
        elif mode == "max":
            out[d] = cell.get("_max")
        elif mode == "min":
            out[d] = cell.get("_min")
    return out


def _resolve_point(lat: float, lon: float) -> Optional[dict]:
    cache = CACHE_DIR / f"point_{lat:.3f}_{lon:.3f}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400 * 7:
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    if NO_FETCH:
        return None
    payload = _http_json(f"{POINTS_URL}/{lat:.4f},{lon:.4f}")
    if not payload:
        return None
    props = payload.get("properties") or {}
    out = {
        "office": props.get("gridId"),
        "gridX": props.get("gridX"),
        "gridY": props.get("gridY"),
        "forecastGridData": props.get("forecastGridData"),
    }
    if out["forecastGridData"]:
        cache.write_text(json.dumps(out))
    return out


def fetch_forecast(lat: float, lon: float, days: int = 7, *, max_age_hours: int = 3) -> pd.DataFrame:
    """Return a daily NWS forecast DataFrame with `OUT_COLS` columns + `date`."""
    if NWS_OFF:
        return pd.DataFrame(columns=["date"] + list(OUT_COLS))

    cache = CACHE_DIR / f"fcst_{lat:.3f}_{lon:.3f}.json"
    payload = None
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        try:
            payload = json.loads(cache.read_text())
        except Exception:
            payload = None
    if payload is None:
        if NO_FETCH:
            if cache.exists():
                try:
                    payload = json.loads(cache.read_text())
                except Exception:
                    payload = None
            if payload is None:
                return pd.DataFrame(columns=["date"] + list(OUT_COLS))
        else:
            point = _resolve_point(lat, lon)
            if not point or not point.get("forecastGridData"):
                return pd.DataFrame(columns=["date"] + list(OUT_COLS))
            grid = _http_json(point["forecastGridData"])
            if not grid:
                return pd.DataFrame(columns=["date"] + list(OUT_COLS))
            payload = {"office": point.get("office"), "grid": grid}
            cache.write_text(json.dumps(payload))

    return _to_daily_df(payload, days=days)


def _to_daily_df(payload: dict, *, days: int) -> pd.DataFrame:
    grid = (payload or {}).get("grid") or {}
    props = grid.get("properties") or {}
    if not props:
        return pd.DataFrame(columns=["date"] + list(OUT_COLS))
    office = (payload.get("office") or props.get("gridId") or "")

    snow_mm = _accumulate_per_day((props.get("snowfallAmount") or {}).get("values", []) or [], day_index={}, mode="sum")
    qpf_mm = _accumulate_per_day((props.get("quantitativePrecipitation") or {}).get("values", []) or [], day_index={}, mode="sum")
    tmean_c = _accumulate_per_day((props.get("temperature") or {}).get("values", []) or [], day_index={}, mode="mean")
    tmax_c = _accumulate_per_day((props.get("maxTemperature") or {}).get("values", []) or [], day_index={}, mode="max")
    tmin_c = _accumulate_per_day((props.get("minTemperature") or {}).get("values", []) or [], day_index={}, mode="min")
    snow_lvl_m = _accumulate_per_day((props.get("snowLevel") or {}).get("values", []) or [], day_index={}, mode="mean")
    pop_pct = _accumulate_per_day((props.get("probabilityOfPrecipitation") or {}).get("values", []) or [], day_index={}, mode="max")

    # tmax/tmin as separate variables aren't always populated (some offices
    # only emit the hourly `temperature` series). Fill from the hourly grid.
    if not tmax_c and tmean_c:
        tmax_c = _accumulate_per_day((props.get("temperature") or {}).get("values", []) or [], day_index={}, mode="max")
    if not tmin_c and tmean_c:
        tmin_c = _accumulate_per_day((props.get("temperature") or {}).get("values", []) or [], day_index={}, mode="min")

    all_dates = sorted(set().union(snow_mm, qpf_mm, tmean_c, tmax_c, tmin_c, snow_lvl_m, pop_pct))
    today = datetime.now(timezone.utc).date()
    rows = []
    for d in all_dates:
        if d < today:
            continue
        if (d - today).days >= days:
            break
        rows.append({
            "date": d,
            "nws_snowfall_in": (snow_mm.get(d) / 25.4) if snow_mm.get(d) is not None else None,
            "nws_precip_in":   (qpf_mm.get(d)  / 25.4) if qpf_mm.get(d)  is not None else None,
            "nws_tmean_f":     (tmean_c[d] * 9 / 5 + 32) if tmean_c.get(d) is not None else None,
            "nws_tmax_f":      (tmax_c[d]  * 9 / 5 + 32) if tmax_c.get(d)  is not None else None,
            "nws_tmin_f":      (tmin_c[d]  * 9 / 5 + 32) if tmin_c.get(d)  is not None else None,
            "nws_snow_level_ft": (snow_lvl_m[d] * 3.28084) if snow_lvl_m.get(d) is not None else None,
            "nws_pop_pct":     pop_pct.get(d),
            "nws_office":      office,
        })
    if not rows:
        return pd.DataFrame(columns=["date"] + list(OUT_COLS))
    return pd.DataFrame(rows)
