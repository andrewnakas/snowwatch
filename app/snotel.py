"""NRCS SNOTEL fetcher (AWDB REST API).

Two layers:

  - `list_active_stations()` — pull the SNTL network catalogue and filter to
    sites with current lat/lon, elevation, and an active end date.
  - `fetch_history(triplet, start, end)` — daily SNWD (snow depth, in),
    WTEQ (snow-water-equivalent, in), PRCPSA (precip accumulation increment),
    and TAVG (mean air temp, F). Persistent incremental cache on disk so CI
    shards only ever pull a single daily delta.

AWDB endpoint: ``https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1``.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


def _no_fetch() -> bool:
    return os.environ.get("SW_NO_FETCH") == "1"


ROOT = Path(__file__).resolve().parents[1] / "data" / "cache"
SITES_DIR = ROOT / "snotel_sites"
SITES_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR = ROOT / "snotel_records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

API = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"

# Elements we want on every fetch. WTEQ + SNWD are the forecast targets; PRCPSA
# (precip, daily accumulation increment) and TAVG (mean temp F) provide the
# in-situ covariates the ridge/lgbm members lean on.
ELEMENTS = ("WTEQ", "SNWD", "PRCPSA", "TAVG")

_ELT_TO_KEY = {
    "WTEQ": "swe_in",
    "SNWD": "snow_depth_in",
    "PRCPSA": "precip_in",
    "TAVG": "tavg_f",
}


def _http_json(url: str, timeout: int = 60, *, retries: int = 3) -> object:
    """GET JSON with bounded retries. AWDB intermittently SSL-handshake-times-out
    under load (especially during CI fan-out); a couple of short retries with
    backoff turns those transient failures into successes."""
    req = Request(url, headers={"User-Agent": "snowwatch/0.1"})
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt + 1)  # 2s, 3s, 5s
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


def _catalog_path() -> Path:
    return SITES_DIR / "sntl_stations.json"


def list_active_stations(*, max_age_days: int = 7) -> list[dict]:
    """Return the active SNOTEL station list as list of dicts with keys:
       id, triplet, name, state, lat, lon, elevation_ft, start_date.

    Cached on disk for `max_age_days`. The NRCS catalogue only changes when a
    station goes on/offline so weekly refresh is plenty.
    """
    p = _catalog_path()
    if p.exists() and (time.time() - p.stat().st_mtime) < max_age_days * 86400:
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    if _no_fetch():
        return json.loads(p.read_text()) if p.exists() else []

    url = (
        f"{API}/stations?networkCds=SNTL"
        "&returnForecastPointMetadata=false"
        "&returnReservoirMetadata=false"
        "&returnStationElements=false"
    )
    try:
        payload = _http_json(url)
    except Exception:
        return json.loads(p.read_text()) if p.exists() else []

    out: list[dict] = []
    today = date.today().isoformat()
    for s in payload if isinstance(payload, list) else []:
        if s.get("networkCode") != "SNTL":
            continue
        lat = s.get("latitude")
        lon = s.get("longitude")
        if lat is None or lon is None:
            continue
        end_date = (s.get("endDate") or "")[:10]
        # The AWDB convention is to mark active sites with a future "end"
        # date (typically 2100-01-01). Tolerate both that and a true blank.
        if end_date and end_date < today:
            continue
        out.append({
            "id": str(s.get("stationId") or s.get("triplet") or "").split(":")[0],
            "triplet": s.get("stationTriplet") or s.get("triplet"),
            "name": s.get("name"),
            "state": s.get("stateCode") or s.get("state"),
            "lat": float(lat),
            "lon": float(lon),
            "elevation_ft": s.get("elevation"),
            "start_date": (s.get("beginDate") or "")[:10],
        })
    if out:
        p.write_text(json.dumps(out, separators=(",", ":")))
    return out


def _record_path(triplet: str) -> Path:
    safe = triplet.replace(":", "_")
    return RECORDS_DIR / f"{safe}.json"


def fetch_history(triplet: str, start: date, end: date, *, max_age_hours: int = 12) -> pd.DataFrame:
    """Return a daily DataFrame for `triplet` between [start, end] with columns:
        date, swe_in, snow_depth_in, precip_in, tavg_f

    Maintains an on-disk incremental record per station; subsequent calls only
    request [last_known + 1, end].
    """
    rp = _record_path(triplet)
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

    needs_backward = bool(first_known) and first_known > start.isoformat()
    needs_forward = (not last_known) or (last_known < end.isoformat())

    if (
        not needs_backward
        and last_known
        and last_known >= end.isoformat()
        and age < max_age_hours * 3600
    ):
        return _slice(have, start, end)
    if _no_fetch():
        return _slice(have, start, end)

    fwd_from = start
    if last_known:
        try:
            ld = date.fromisoformat(last_known)
            if ld >= start:
                # Re-fetch the last 3 days in case AWDB revises provisional rows.
                fwd_from = ld - timedelta(days=3)
        except ValueError:
            pass

    fetched_any = False
    if needs_backward:
        bwd_from = start
        bwd_to = (date.fromisoformat(first_known) - timedelta(days=1)) if first_known else end
        if bwd_from <= bwd_to:
            fetched_any = _fetch_and_merge(triplet, have, bwd_from, bwd_to) or fetched_any
    if needs_forward and fwd_from <= end:
        fetched_any = _fetch_and_merge(triplet, have, fwd_from, end) or fetched_any

    if fetched_any and have:
        rec = {
            "stationTriplet": triplet,
            "rows": have,
            "first_known": min(have.keys()),
            "last_known": max(have.keys()),
            "fetched_at": time.time(),
        }
        rp.write_text(json.dumps(rec, separators=(",", ":")))
    return _slice(have, start, end)


def _fetch_and_merge(triplet: str, have: dict, start: date, end: date) -> bool:
    params = {
        "stationTriplets": triplet,
        "elements": ",".join(ELEMENTS),
        "duration": "DAILY",
        "beginDate": start.isoformat(),
        "endDate": end.isoformat(),
    }
    url = f"{API}/data?" + urlencode(params)
    try:
        payload = _http_json(url)
    except Exception:
        return False
    if not isinstance(payload, list) or not payload:
        return False
    added = False
    for station_payload in payload:
        for elt in station_payload.get("data", []):
            code = elt.get("stationElement", {}).get("elementCode")
            if code not in _ELT_TO_KEY:
                continue
            key = _ELT_TO_KEY[code]
            for row in elt.get("values", []) or []:
                d = row.get("date")
                v = row.get("value")
                if d is None or v is None:
                    continue
                d = d[:10]
                cell = have.setdefault(d, {})
                cell[key] = float(v)
                added = True
    return added


def _slice(have: dict, start: date, end: date) -> pd.DataFrame:
    cols = ["date", "swe_in", "snow_depth_in", "precip_in", "tavg_f"]
    if not have:
        return pd.DataFrame(columns=cols)
    s, e = start.isoformat(), end.isoformat()
    rows = []
    for d_iso, vals in have.items():
        if not (s <= d_iso <= e):
            continue
        rows.append({
            "date": d_iso,
            "swe_in": vals.get("swe_in") if isinstance(vals, dict) else None,
            "snow_depth_in": vals.get("snow_depth_in") if isinstance(vals, dict) else None,
            "precip_in": vals.get("precip_in") if isinstance(vals, dict) else None,
            "tavg_f": vals.get("tavg_f") if isinstance(vals, dict) else None,
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)
