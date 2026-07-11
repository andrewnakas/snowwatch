#!/usr/bin/env python3
"""Backfill GFS 10m wind COMPONENTS from the dynamical.org Zarr archive —
the upslope-flow predictor tree (release-3).

The phase tree (backfill_gfs_phase_zarr.py) reads hourly wind_u/v_10m but
collapses them to speed; direction is where the orographic signal lives —
upslope flow into a favorable aspect is the classic big-snow discriminator,
and NBM's documented failure mode is upslope under-forecasting. Signed
daily-mean u/v per lead let the trees learn each station's favorable
directions via (u, v) x (lat, lon) interactions; the daily VECTOR mean also
carries steadiness (a wobbling wind averages toward zero, a locked-in
upslope flow doesn't).

The live counterpart is app/met.py fetch_phase_hourly + wind_direction_10m
-> u/v in phase_features.live_phase_daily (same GFS model via Open-Meteo);
storm-gating happens at pairs-build/live-fetch time exactly like the phase
columns, so train and serve see the same masked signal.

Output: data/prevruns/gfs_wind/<triplet>.csv.gz, headerless:
  valid_date, lead_days, u10_mean_ms, v10_mean_ms
Resumable via data/prevruns/gfs_wind_state.json.

Usage:
    python scripts/backfill_gfs_wind_zarr.py --start-month 2021-11
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
from scripts.backfill_gfs_zarr import (  # noqa: E402
    LEADS, ZARR_URL, EMAIL, _bilinear_weights, _month_days,
)

STATE_PATH = ROOT / "data" / "prevruns" / "gfs_wind_state.json"
OUT_ROOT = ROOT / "data" / "prevruns" / "gfs_wind"
VARS = ["wind_u_10m", "wind_v_10m"]
OUT_COLS = ["valid_date", "lead_days", "u10_mean_ms", "v10_mean_ms"]
INIT_BATCH = 6


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

    u = _interp("wind_u_10m")                    # (init, lead, station), m/s
    v = _interp("wind_v_10m")

    rows = []
    for n in LEADS:
        ins = (lead_h >= 24 * n) & (lead_h < 24 * n + 24)
        if ins.sum() < 4:
            continue
        uu, vv = u[:, ins, :], v[:, ins, :]
        n_valid = np.sum(np.isfinite(uu) & np.isfinite(vv), axis=1)  # (init, st)
        with np.errstate(all="ignore"):
            u_mean = np.nanmean(uu, axis=1)
            v_mean = np.nanmean(vv, axis=1)
        bad = n_valid < 4
        u_mean[bad] = np.nan
        v_mean[bad] = np.nan
        for k, it in enumerate(init_ts):
            vd = (it + pd.Timedelta(days=n)).strftime("%Y-%m-%d")
            rows.append(pd.DataFrame({
                "triplet": triplets, "valid_date": vd, "lead_days": n,
                "u10_mean_ms": np.round(u_mean[k], 2),
                "v10_mean_ms": np.round(v_mean[k], 2),
            }))
    if not rows:
        return pd.DataFrame(columns=["triplet", *OUT_COLS])
    return pd.concat(rows, ignore_index=True).dropna(subset=["u10_mean_ms"])


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
