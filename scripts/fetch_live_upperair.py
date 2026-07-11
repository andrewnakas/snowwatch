#!/usr/bin/env python3
"""Bulk live Zarr prefetch for the release-3 upperair features.

ONE read per store per build, all 912 stations at once — the same access
pattern as a single backfill batch, so it finishes in minutes and costs
zero Open-Meteo budget:
  * GEFS ens-mean 500hPa height (z500_mean_m) for the next 7 valid days,
    from the latest available init in the live-updating archive;
  * HRRR native snowfall (hrrr_native_snowfall_cm) for the latest 00Z
    init's (24, 48]h window — the lead-1 valid day, matching the training
    tree exactly.

Output: data/cache/live_upperair.json
  {"created_utc", "gefs_init", "hrrr_init",
   "stations": {triplet: {"z500_by_valid": {"YYYY-MM-DD": m, ...},
                          "hrrr_native_by_valid": {"YYYY-MM-DD": cm}}}}

Idempotent: skips the fetch when the cache already holds the latest inits.
Each CI shard runs it at start; concurrent shards racing the write is
harmless (same content). ANY failure leaves the previous cache in place
and exits 0 — the postproc member reads what exists and NaN-routes the
rest; this prefetch must never break a build.

Usage:
    python scripts/fetch_live_upperair.py [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backfill_gfs_zarr import EMAIL, _bilinear_weights  # noqa: E402
from scripts.backfill_gefs_zarr import ZARR_URL as GEFS_URL  # noqa: E402
from scripts.backfill_hrrr_zarr import ZARR_URL as HRRR_URL, grid_index  # noqa: E402

OUT_PATH = ROOT / "data" / "cache" / "live_upperair.json"
Z500_LEADS = range(1, 8)


def _stations() -> list[dict]:
    sts = json.loads((ROOT / "data" / "stations.json").read_text())["stations"]
    sts.sort(key=lambda s: s["triplet"])
    return sts


def fetch_gefs_z500(stations: list[dict]) -> tuple[str, dict[str, dict[str, float]]]:
    """{triplet: {valid_date: z500_mean_m}} from the latest GEFS init."""
    import xarray as xr
    ds = xr.open_zarr(f"{GEFS_URL}?email={EMAIL}", decode_timedelta=True)
    init = pd.Timestamp(ds.init_time.values[-1])

    lats = np.array([s["lat"] for s in stations], dtype=float)
    lons = np.array([s["lon"] for s in stations], dtype=float)
    corners = _bilinear_weights(ds, lats, lons)
    n_st = len(stations)
    lat_all = xr.DataArray(np.concatenate([c[0].values for c in corners]), dims="pt")
    lon_all = xr.DataArray(np.concatenate([c[1].values for c in corners]), dims="pt")
    weights = np.stack([c[2] for c in corners])

    lead_h_all = (ds.lead_time.values / np.timedelta64(1, "h")).astype(float)
    keep = (lead_h_all >= 24) & (lead_h_all < 192)
    sub = ds["geopotential_height_500hpa"].sel(init_time=[init]).isel(
        lead_time=np.flatnonzero(keep), latitude=lat_all, longitude=lon_all)
    raw = sub.compute().values                     # (init, mem, lead, 4*n)
    raw = raw.reshape(*raw.shape[:-1], 4, n_st)
    z = np.einsum("...cs,cs->...s", raw, weights)  # (init, mem, lead, st)
    lead_h = lead_h_all[keep]

    out: dict[str, dict[str, float]] = {s["triplet"]: {} for s in stations}
    for n in Z500_LEADS:
        ins = (lead_h >= 24 * n) & (lead_h < 24 * n + 24)
        if not ins.any():
            continue
        with np.errstate(all="ignore"):
            z5 = np.nanmean(np.nanmean(z[0][:, ins, :], axis=1), axis=0)  # (st,)
        vd = (init + pd.Timedelta(days=n)).strftime("%Y-%m-%d")
        for k, s in enumerate(stations):
            if np.isfinite(z5[k]):
                out[s["triplet"]][vd] = round(float(z5[k]), 1)
    return init.isoformat(), out


def fetch_hrrr_native(stations: list[dict]) -> tuple[str, dict[str, dict[str, float]]]:
    """{triplet: {valid_date: native snowfall cm}} from the latest 00Z HRRR
    init's (24, 48] window — the training tree's exact convention."""
    import xarray as xr
    ds = xr.open_zarr(f"{HRRR_URL}?email={EMAIL}", decode_timedelta=True)
    inits = pd.to_datetime(ds.init_time.values)
    z00 = inits[inits.hour == 0]
    init = pd.Timestamp(z00[-1])

    y_idx, x_idx, ok = grid_index(ds, stations)
    lead_h_all = (ds.lead_time.values / np.timedelta64(1, "h")).astype(float)
    acc = (lead_h_all > 24) & (lead_h_all <= 48)
    import xarray as xr2
    pt_y = xr2.DataArray(y_idx, dims="pt")
    pt_x = xr2.DataArray(x_idx, dims="pt")
    sub = ds["snowfall_surface"].sel(init_time=init).isel(
        lead_time=np.flatnonzero(acc), y=pt_y, x=pt_x)
    raw = sub.compute().values                     # (lead, st)
    snow_cm = np.nansum(raw * 3600.0, axis=0) * 100.0
    all_nan = np.isnan(raw).all(axis=0)
    snow_cm[all_nan | ~ok] = np.nan

    vd = (init + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out = {s["triplet"]: ({vd: round(float(snow_cm[k]), 2)}
                          if np.isfinite(snow_cm[k]) else {})
           for k, s in enumerate(stations)}
    return init.isoformat(), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    stations = _stations()
    prev = {}
    if OUT_PATH.exists():
        try:
            prev = json.loads(OUT_PATH.read_text())
        except Exception:
            prev = {}

    payload = {"created_utc": pd.Timestamp.now("UTC").isoformat(),
               "gefs_init": prev.get("gefs_init"),
               "hrrr_init": prev.get("hrrr_init"),
               "stations": prev.get("stations", {})}
    changed = False
    for name, fn, key in (("gefs", fetch_gefs_z500, "z500_by_valid"),
                          ("hrrr", fetch_hrrr_native, "hrrr_native_by_valid")):
        t0 = time.time()
        try:
            init_iso, by_st = fn(stations)
        except Exception as exc:   # degrade, never break the build
            print(f"{name}: fetch failed ({type(exc).__name__}: {str(exc)[:120]}) "
                  f"— keeping previous cache", flush=True)
            continue
        if not args.force and payload.get(f"{name}_init") == init_iso:
            print(f"{name}: init {init_iso} already cached", flush=True)
            continue
        payload[f"{name}_init"] = init_iso
        for t, vals in by_st.items():
            payload["stations"].setdefault(t, {})[key] = vals
        changed = True
        print(f"{name}: init {init_iso}, {len(by_st)} stations in "
              f"{time.time()-t0:.0f}s", flush=True)
    if changed or not OUT_PATH.exists():
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
