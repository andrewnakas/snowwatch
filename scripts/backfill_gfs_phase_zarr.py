#!/usr/bin/env python3
"""Backfill GFS wet-bulb phase features from the dynamical.org Zarr archive
(v1.6 Tier 1) — the training-side counterpart of app/met.py
fetch_phase_hourly.

Pulls HOURLY t2m + rh2m per (station, init, lead-day window) and reduces to
daily phase aggregates through app/phase_features.daily_phase_aggregates —
the SAME function the live path uses, so train and serve are consistent by
construction. Source model (GFS) and lead alignment also match the live
fetch. freezing_level_height is not in this archive; the wet-bulb trio
carries the phase signal (freezing-level stays live-only and OUT of
FEATURES until an archived source exists).

Output: data/prevruns/gfs_phase/<triplet>.csv.gz, headerless:
  valid_date, lead_days, wb_mean_c, wb_min_c, hours_wb_below_0
Resumable via data/prevruns/gfs_phase_state.json.

Memory: hourly × 169 leads × 4·912 pts × 2 vars ≈ 0.5 GB per 6-init batch —
same batching as the GEFS extractor.

Usage:
    python scripts/backfill_gfs_phase_zarr.py --start-month 2021-11
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

from app.snowphysics import wet_bulb_c  # noqa: E402
from scripts.backfill_previous_runs import SUMMER_MONTHS  # noqa: E402
from scripts.backfill_gfs_zarr import (  # noqa: E402
    LEADS, ZARR_URL, EMAIL, _bilinear_weights, _month_days,
)

STATE_PATH = ROOT / "data" / "prevruns" / "gfs_phase_state.json"
OUT_ROOT = ROOT / "data" / "prevruns" / "gfs_phase"
# wind10: mean 10m speed — with temperature and humidity it's the third
# top predictor in the published SLR work (Veals et al. 2025).
VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m"]
OUT_COLS = ["valid_date", "lead_days", "wb_mean_c", "wb_min_c",
            "hours_wb_below_0", "wind10_mean_ms"]
INIT_BATCH = 6


def _wb_vec(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Vectorized Stull wet-bulb — identical formula to snowphysics.wet_bulb_c
    (kept in lockstep by tests/test_phase_features.py::test_vec_matches_scalar)."""
    t = np.asarray(t_c, dtype=float)
    r = np.clip(np.asarray(rh, dtype=float), 0.0, 100.0)
    return (t * np.arctan(0.151977 * np.sqrt(r + 8.313659))
            + np.arctan(t + r) - np.arctan(r - 1.676331)
            + 0.00391838 * r ** 1.5 * np.arctan(0.023101 * r)
            - 4.686035)


def _extract_inits(ds, init_ts, stations) -> pd.DataFrame:
    import xarray as xr
    lats = np.array([s["lat"] for s in stations], dtype=float)
    lons = np.array([s["lon"] for s in stations], dtype=float)
    triplets = np.array([s["triplet"] for s in stations])
    corners = _bilinear_weights(ds, lats, lons)
    n_st = len(stations)
    lat_all = xr.DataArray(np.concatenate([c[0].values for c in corners]), dims="pt")
    lon_all = xr.DataArray(np.concatenate([c[1].values for c in corners]), dims="pt")
    weights = np.stack([c[2] for c in corners])

    lead_h_all = (ds.lead_time.values / np.timedelta64(1, "h")).astype(float)
    keep = (lead_h_all >= 24) & (lead_h_all < 192)
    sub = ds[VARS].sel(init_time=init_ts).isel(
        lead_time=np.flatnonzero(keep), latitude=lat_all, longitude=lon_all)
    data = sub.compute()
    lead_h = lead_h_all[keep]

    def _interp(v):
        raw = data[v].values
        raw = raw.reshape(raw.shape[0], raw.shape[1], 4, n_st)
        return np.einsum("ilcs,cs->ils", raw, weights)

    t2m = _interp("temperature_2m")
    rh = _interp("relative_humidity_2m")
    wb = _wb_vec(t2m, rh)                       # (init, lead, station)
    wb[np.isnan(t2m) | np.isnan(rh)] = np.nan
    wind = np.hypot(_interp("wind_u_10m"), _interp("wind_v_10m"))

    rows = []
    for n in LEADS:
        ins = (lead_h >= 24 * n) & (lead_h < 24 * n + 24)
        if ins.sum() < 4:
            continue
        w = wb[:, ins, :]
        n_valid = np.sum(np.isfinite(w), axis=1)          # (init, st)
        with np.errstate(all="ignore"):
            wb_mean = np.nanmean(w, axis=1)
            wb_min = np.nanmin(w, axis=1)
            hrs = np.nansum(w <= 0.0, axis=1) * 24.0 / np.maximum(n_valid, 1)
        with np.errstate(all="ignore"):
            wind_mean = np.nanmean(wind[:, ins, :], axis=1)
        bad = n_valid < 4
        wb_mean[bad] = np.nan
        wb_min[bad] = np.nan
        hrs = hrs.astype(float)
        hrs[bad] = np.nan
        wind_mean[bad] = np.nan
        for k, it in enumerate(init_ts):
            vd = (it + pd.Timedelta(days=n)).strftime("%Y-%m-%d")
            rows.append(pd.DataFrame({
                "triplet": triplets, "valid_date": vd, "lead_days": n,
                "wb_mean_c": np.round(wb_mean[k], 2),
                "wb_min_c": np.round(wb_min[k], 2),
                "hours_wb_below_0": np.round(hrs[k], 1),
                "wind10_mean_ms": np.round(wind_mean[k], 2),
            }))
    if not rows:
        return pd.DataFrame(columns=["triplet", *OUT_COLS])
    return pd.concat(rows, ignore_index=True).dropna(subset=["wb_mean_c"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-month", default="2021-11")
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
        inits = [d for d in _month_days(ym) if d < today]
        init_all = pd.to_datetime([d.isoformat() for d in inits])
        init_all = init_all[init_all.isin(pd.to_datetime(ds.init_time.values))]
        frames = []
        failed = False
        for i in range(0, len(init_all), INIT_BATCH):
            for attempt in range(4):
                try:
                    frames.append(_extract_inits(ds, init_all[i:i + INIT_BATCH], stations))
                    break
                except Exception as exc:   # transient S3 resets kill hours
                    print(f"{ym} batch {i}: attempt {attempt+1} failed "
                          f"({type(exc).__name__}: {str(exc)[:100]})", flush=True)
                    time.sleep(30 * (attempt + 1))
            else:
                failed = True
                break
        if failed:
            print(f"{ym}: giving up — rerun later")
            ym = nxt
            continue
        frames = [f for f in frames if not f.empty]
        n = 0
        if frames:
            month_df = pd.concat(frames, ignore_index=True)
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
