#!/usr/bin/env python3
"""Bulk-backfill GEFS ensemble spread history from the dynamical.org Zarr
archive — the training-side counterpart of app/met.py fetch_ensemble.

Source: https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr
(31 members, inits 2020-10 → present, 3-hourly to 240h, 0.25°). Per
(station, valid_date, lead) this reduces the member cloud to the same
spread stats the live path computes, PLUS the ensemble-mean 500hPa height —
upper-air signal the current model has never seen. Snowfall follows the
Open-Meteo convention (snow-water mm × 0.7 → cm, member-wise categorical
phase), prob_pos threshold 0.5 cm — mirroring app/met.py _ens_reduce.

Output: data/prevruns/gefs_ens/<triplet>.csv.gz, headerless, columns:
  valid_date, lead_days, ens_snow_mean_cm, ens_snow_std_cm, ens_snow_p10_cm,
  ens_snow_p50_cm, ens_snow_p90_cm, ens_prob_pos, ens_precip_mean_mm,
  ens_precip_std_mm, ens_n_members, z500_mean_m

NOTE live-vs-training member sets differ (live fetch_ensemble mixes GEFS +
AIFS-ENS, ~80 members; this archive is GEFS-only, 31). ens_n_members is
stored so the model can condition on it; the Phase-3 wiring decides whether
to restrict the live reduction to GEFS members instead.

COST: the member dimension multiplies transfer ~31× vs the GFS extraction —
run AFTER backfill_gfs_zarr.py finishes, and expect ~overnight for 3+
winters. Resumable via data/prevruns/gefs_zarr_state.json.

Usage:
    python scripts/backfill_gefs_zarr.py --start-month 2022-11
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
    LEADS, SWE_MM_TO_SNOW_CM, _bilinear_weights, _month_days, _step_seconds,
)

ZARR_URL = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr"
STATE_PATH = ROOT / "data" / "prevruns" / "gefs_zarr_state.json"
OUT_ROOT = ROOT / "data" / "prevruns" / "gefs_ens"
EMAIL = "nakas@tree60weather.com"

VARS = ["precipitation_surface", "categorical_snow_surface",
        "geopotential_height_500hpa"]
PROB_POS_CM = 0.5      # mirrors app/met.py _ens_reduce (> 0.5 cm)

OUT_COLS = ["valid_date", "lead_days", "ens_snow_mean_cm", "ens_snow_std_cm",
            "ens_snow_p10_cm", "ens_snow_p50_cm", "ens_snow_p90_cm",
            "ens_prob_pos", "ens_precip_mean_mm", "ens_precip_std_mm",
            "ens_n_members", "z500_mean_m"]


def extract_month(ds, ym: str, stations: list[dict], *, today: date) -> pd.DataFrame:
    import xarray as xr
    inits = [d for d in _month_days(ym) if d < today]
    init_ts = pd.to_datetime([d.isoformat() for d in inits])
    init_ts = init_ts[init_ts.isin(pd.to_datetime(ds.init_time.values))]
    if len(init_ts) == 0:
        return pd.DataFrame(columns=["triplet", *OUT_COLS])

    lats = np.array([s["lat"] for s in stations], dtype=float)
    lons = np.array([s["lon"] for s in stations], dtype=float)
    triplets = np.array([s["triplet"] for s in stations])
    corners = _bilinear_weights(ds, lats, lons)
    n_st = len(stations)
    lat_all = xr.DataArray(np.concatenate([c[0].values for c in corners]), dims="pt")
    lon_all = xr.DataArray(np.concatenate([c[1].values for c in corners]), dims="pt")
    weights = np.stack([c[2] for c in corners])            # (4, n_st)

    lead_h_all = (ds.lead_time.values / np.timedelta64(1, "h")).astype(float)
    keep = (lead_h_all >= 24) & (lead_h_all <= 192)
    sub = ds[VARS].sel(init_time=init_ts).isel(
        lead_time=np.flatnonzero(keep), latitude=lat_all, longitude=lon_all)
    data = sub.compute()
    lead_h = lead_h_all[keep]
    step_s = np.where(lead_h <= 240, 10800.0, 21600.0)   # GEFS: 3-hourly ≤240h

    def _interp(v):                          # (init, member?, lead, 4*n) → (…, n)
        raw = data[v].values
        raw = raw.reshape(*raw.shape[:-1], 4, n_st)
        return np.einsum("...cs,cs->...s", raw, weights)

    rate = _interp("precipitation_surface")          # (init, mem, lead, st)
    csnow = _interp("categorical_snow_surface")
    z500 = _interp("geopotential_height_500hpa")

    rows = []
    for n in LEADS:
        acc = (lead_h > 24 * n) & (lead_h <= 24 * n + 24)
        ins = (lead_h >= 24 * n) & (lead_h < 24 * n + 24)
        if not acc.any():
            continue
        w = step_s[None, None, acc, None]
        precip_mm = np.nansum(rate[:, :, acc, :] * w, axis=2)        # (init, mem, st)
        snow_cm = np.nansum(rate[:, :, acc, :] * csnow[:, :, acc, :] * w,
                            axis=2) * SWE_MM_TO_SNOW_CM
        all_nan = np.isnan(rate[:, :, acc, :]).all(axis=2)
        precip_mm[all_nan] = np.nan
        snow_cm[all_nan] = np.nan
        z5 = np.nanmean(np.nanmean(z500[:, :, ins, :], axis=2), axis=1)  # (init, st)
        n_mem = np.sum(~np.isnan(snow_cm), axis=1)                   # (init, st)
        with np.errstate(all="ignore"):
            stats = {
                "ens_snow_mean_cm": np.nanmean(snow_cm, axis=1),
                "ens_snow_std_cm": np.nanstd(snow_cm, axis=1),
                "ens_snow_p10_cm": np.nanpercentile(snow_cm, 10, axis=1),
                "ens_snow_p50_cm": np.nanpercentile(snow_cm, 50, axis=1),
                "ens_snow_p90_cm": np.nanpercentile(snow_cm, 90, axis=1),
                "ens_prob_pos": np.nanmean((snow_cm > PROB_POS_CM), axis=1),
                "ens_precip_mean_mm": np.nanmean(precip_mm, axis=1),
                "ens_precip_std_mm": np.nanstd(precip_mm, axis=1),
            }
        for k, it in enumerate(init_ts):
            vd = (it + pd.Timedelta(days=n)).strftime("%Y-%m-%d")
            rows.append(pd.DataFrame({
                "triplet": triplets, "valid_date": vd, "lead_days": n,
                **{c: np.round(v[k], 3) for c, v in stats.items()},
                "ens_n_members": n_mem[k],
                "z500_mean_m": np.round(z5[k], 1),
            }))
    if not rows:
        return pd.DataFrame(columns=["triplet", *OUT_COLS])
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["ens_snow_mean_cm", "ens_precip_mean_mm"], how="all")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-month", default="2022-11")
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
