#!/usr/bin/env python3
"""Bulk-backfill HRRR lead-1 history from the dynamical.org Zarr archive.

https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr — 3 km
CONUS, inits every 6h back to 2018-07-13, hourly leads 0–48. From each 00Z
init the (24, 48] window gives the full lead-1 day; HRRR's 48h horizon can't
cover lead 2+.

Two things make this archive special for SnowWatch:
  - snowfall_surface is HRRR's NATIVE snow accumulation (m s⁻¹ step rate,
    variable-density microphysics) — a different, richer quantity than the
    fixed 0.7 cm/mm convention in the Open-Meteo tree. It therefore goes to
    its own tree (data/prevruns/hrrr_zarr/) and its own feature column at
    wiring time — never mixed into the existing hrrr_* columns.
  - 8 winters × 3 km: nearest-gridpoint at 3 km is more representative than
    bilinear at 25 km, so plain nearest via a cached KD-tree index is used.

Output: data/prevruns/hrrr_zarr/<triplet>.csv.gz, headerless:
  valid_date, lead_days(=1), snowfall_cm, precip_mm, tmean_c
Resumable via data/prevruns/hrrr_zarr_state.json.

Usage:
    python scripts/backfill_hrrr_zarr.py --start-month 2018-11
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backfill_previous_runs import SUMMER_MONTHS  # noqa: E402
from scripts.backfill_gfs_zarr import _month_days  # noqa: E402

ZARR_URL = "https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr"
STATE_PATH = ROOT / "data" / "prevruns" / "hrrr_zarr_state.json"
INDEX_PATH = ROOT / "data" / "prevruns" / "hrrr_grid_index.json"
OUT_ROOT = ROOT / "data" / "prevruns" / "hrrr_zarr"
EMAIL = "nakas@tree60weather.com"

VARS = ["snowfall_surface", "precipitation_surface", "temperature_2m"]
OUT_COLS = ["valid_date", "lead_days", "snowfall_cm", "precip_mm", "tmean_c"]


def grid_index(ds, stations: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest (y, x) per station on the Lambert grid, cached to disk.
    Returns (y_idx, x_idx, in_domain mask) — AK stations fall outside CONUS."""
    key = [s["triplet"] for s in stations]
    if INDEX_PATH.exists():
        try:
            c = json.loads(INDEX_PATH.read_text())
            if c.get("triplets") == key:
                return (np.array(c["y"]), np.array(c["x"]), np.array(c["ok"], dtype=bool))
        except Exception:
            pass
    from scipy.spatial import cKDTree
    lat2d = ds.latitude.values
    lon2d = ds.longitude.values
    tree = cKDTree(np.column_stack([lat2d.ravel(), lon2d.ravel()]))
    pts = np.column_stack([[s["lat"] for s in stations],
                           [s["lon"] for s in stations]])
    dist, flat = tree.query(pts)
    y_idx, x_idx = np.unravel_index(flat, lat2d.shape)
    ok = dist < 0.05          # ~3–5 km in degrees; beyond that = off-grid (AK)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps({
        "triplets": key, "y": y_idx.tolist(), "x": x_idx.tolist(),
        "ok": ok.tolist()}))
    return y_idx, x_idx, ok


def extract_month(ds, ym: str, stations: list[dict], *, today: date) -> pd.DataFrame:
    import xarray as xr
    inits = [d for d in _month_days(ym) if d < today]
    init_ts = pd.to_datetime([d.isoformat() for d in inits])   # 00Z
    init_ts = init_ts[init_ts.isin(pd.to_datetime(ds.init_time.values))]
    if len(init_ts) == 0:
        return pd.DataFrame(columns=["triplet", *OUT_COLS])
    y_idx, x_idx, ok = grid_index(ds, stations)
    triplets = np.array([s["triplet"] for s in stations])[ok]
    y_da = xr.DataArray(y_idx[ok], dims="station")
    x_da = xr.DataArray(x_idx[ok], dims="station")

    lead_h_all = (ds.lead_time.values / np.timedelta64(1, "h")).astype(float)
    keep = (lead_h_all >= 24) & (lead_h_all <= 48)
    sub = ds[VARS].sel(init_time=init_ts).isel(
        lead_time=np.flatnonzero(keep), y=y_da, x=x_da)
    data = sub.compute()
    lead_h = lead_h_all[keep]

    acc = lead_h > 24          # (24, 48]
    ins = lead_h < 48          # [24, 48)
    snow_rate = data["snowfall_surface"].values       # m s-1, (init, lead, st)
    prec_rate = data["precipitation_surface"].values  # kg m-2 s-1
    t2m = data["temperature_2m"].values               # °C
    snow_cm = np.nansum(snow_rate[:, acc, :] * 3600.0, axis=1) * 100.0
    precip_mm = np.nansum(prec_rate[:, acc, :] * 3600.0, axis=1)
    all_nan = np.isnan(snow_rate[:, acc, :]).all(axis=1)
    snow_cm[all_nan] = np.nan
    precip_mm[all_nan] = np.nan
    tmean = np.nanmean(t2m[:, ins, :], axis=1)

    rows = []
    for k, it in enumerate(init_ts):
        vd = (it + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        rows.append(pd.DataFrame({
            "triplet": triplets, "valid_date": vd, "lead_days": 1,
            "snowfall_cm": np.round(snow_cm[k], 3),
            "precip_mm": np.round(precip_mm[k], 3),
            "tmean_c": np.round(tmean[k], 2),
        }))
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["snowfall_cm", "precip_mm", "tmean_c"], how="all")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-month", default="2018-11")
    ap.add_argument("--end-month", default=None)
    ap.add_argument("--email", default=EMAIL)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import xarray as xr
    stations = json.loads((ROOT / "data" / "stations.json").read_text())["stations"]
    stations.sort(key=lambda s: s["triplet"])
    if args.limit:
        stations = stations[: args.limit]
    ds = xr.open_zarr(f"{ZARR_URL}?email={args.email}", decode_timedelta=True)

    today = date.today()
    end_month = args.end_month or (today.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"months": {}}

    ym = args.start_month
    t0 = time.time()
    while ym <= end_month:
        nxt = (date.fromisoformat(ym + "-01") + timedelta(days=32)).strftime("%Y-%m")
        if int(ym[5:7]) in SUMMER_MONTHS or state["months"].get(ym) == "ok":
            ym = nxt
            continue
        t1 = time.time()
        month_df = extract_month(ds, ym, stations, today=today)
        n = 0
        for triplet, g in month_df.groupby("triplet"):
            p = OUT_ROOT / f"{str(triplet).replace(':', '_')}.csv.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            buf = io.StringIO()
            g[OUT_COLS].to_csv(buf, header=False, index=False)
            with gzip.open(p, "at") as f:
                f.write(buf.getvalue())
            n += len(g)
        state["months"][ym] = "ok"
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=0))
        print(f"{ym}: {n} rows in {time.time()-t1:.0f}s (total {time.time()-t0:.0f}s)",
              flush=True)
        ym = nxt
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
