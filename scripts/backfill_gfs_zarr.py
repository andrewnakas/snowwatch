#!/usr/bin/env python3
"""Bulk-backfill GFS forecast history from the dynamical.org Zarr archive.

Point-extracts the full 912-station catalogue from
https://data.dynamical.org/noaa/gfs/forecast/latest.zarr (inits 2021-05-01 →
present, 0.25°) and writes daily per-lead rows in the exact
data/prevruns/gfs/<triplet>.csv.gz schema the Open-Meteo backfill uses, so
scripts/build_training_data.py consumes them unchanged. This bypasses the
Open-Meteo API budget entirely — 5+ winters × 912 stations in one local run
instead of months of nightly CI.

Conventions that must match the Open-Meteo tree (validated via --validate):
  - valid_date is the UTC calendar day; lead_days N from the 00Z init covers
    lead hours (24N, 24(N+1)].
  - precip_mm = Σ rate·Δt over the day (precipitation_surface is kg m⁻² s⁻¹,
    step-averaged; Δt is 1h out to 120h, 3h beyond).
  - snowfall_cm = snow-water mm × 0.7 — Open-Meteo's fixed 7:1-in-cm
    convention, with snow-water = Σ rate·Δt·categorical_snow. The model
    consumes this as a *feature*, so matching the live convention matters
    more than the constant's physical realism.
  - tmean_c = daily mean of instantaneous temperature_2m.

Run in month shards with resumable state (data/prevruns/gfs_zarr_state.json):
    python scripts/backfill_gfs_zarr.py --start-month 2021-11
    python scripts/backfill_gfs_zarr.py --validate   # no writes; overlap
        regression vs existing Open-Meteo rows — require slope≈1, r²≈1
        BEFORE the first real run.
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

from scripts.backfill_previous_runs import (  # noqa: E402
    OUT_COLS, SUMMER_MONTHS, _paths, read_station_model,
)

ZARR_URL = "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr"
STATE_PATH = ROOT / "data" / "prevruns" / "gfs_zarr_state.json"
EMAIL = "nakas@tree60weather.com"   # dynamical.org usage tracking

LEADS = list(range(1, 8))
SWE_MM_TO_SNOW_CM = 0.7             # Open-Meteo's snowfall convention
VARS = ["temperature_2m", "precipitation_surface", "categorical_snow_surface"]


def _open(email: str):
    import xarray as xr
    return xr.open_zarr(f"{ZARR_URL}?email={email}", decode_timedelta=True)


def _bilinear_weights(ds, lats: np.ndarray, lons: np.ndarray):
    """Index/weight sets for bilinear interpolation on the 0.25° grid.

    Nearest-gridpoint sampling loses badly to Open-Meteo's interpolated
    values in complex terrain (validated 2026-07-06: snowfall r 0.77 vs 0.86
    bilinear); four-corner bilinear is the cheap fix — same chunks touched.
    Returns [(lat_idx_da, lon_idx_da, weight_array)] × 4 corners.
    """
    import xarray as xr
    lat0 = ds.latitude.values[0]      # 90.0, descending
    lon0 = ds.longitude.values[0]     # -180.0, ascending
    fi = (lat0 - lats) / 0.25         # fractional row (lat descending)
    fj = (lons - lon0) / 0.25
    i0 = np.clip(np.floor(fi).astype(int), 0, ds.sizes["latitude"] - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, ds.sizes["longitude"] - 2)
    di = np.clip(fi - i0, 0.0, 1.0)
    dj = np.clip(fj - j0, 0.0, 1.0)
    corners = []
    for oi, oj, w in ((0, 0, (1 - di) * (1 - dj)), (0, 1, (1 - di) * dj),
                      (1, 0, di * (1 - dj)), (1, 1, di * dj)):
        corners.append((xr.DataArray(i0 + oi, dims="station"),
                        xr.DataArray(j0 + oj, dims="station"), w))
    return corners


def _month_days(ym: str) -> list[date]:
    d0 = date.fromisoformat(ym + "-01")
    out = []
    d = d0
    while d.month == d0.month:
        out.append(d)
        d += timedelta(days=1)
    return out


def _step_seconds(lead_hours: np.ndarray) -> np.ndarray:
    """Accumulation window per step: hourly to 120h, 3-hourly beyond."""
    return np.where(lead_hours <= 120, 3600.0, 10800.0)


def extract_month(ds, ym: str, stations: list[dict], *,
                  today: date) -> pd.DataFrame:
    """All (station, valid_date, lead) rows for one init month."""
    inits = [d for d in _month_days(ym) if d < today]
    if not inits:
        return pd.DataFrame(columns=["triplet", *OUT_COLS])
    init_ts = pd.to_datetime([d.isoformat() for d in inits])
    avail = pd.to_datetime(ds.init_time.values)
    init_ts = init_ts[init_ts.isin(avail)]
    if len(init_ts) == 0:
        return pd.DataFrame(columns=["triplet", *OUT_COLS])

    lats = np.array([s["lat"] for s in stations], dtype=float)
    lons = np.array([s["lon"] for s in stations], dtype=float)
    triplets = np.array([s["triplet"] for s in stations])
    corners = _bilinear_weights(ds, lats, lons)
    import xarray as xr
    n_st = len(stations)
    # One fetch for all four corners: stack them along the station dim so the
    # underlying chunks are pulled once, then reshape (…, 4, n_st) and weight.
    lat_all = xr.DataArray(np.concatenate([c[0].values for c in corners]), dims="pt")
    lon_all = xr.DataArray(np.concatenate([c[1].values for c in corners]), dims="pt")
    weights = np.stack([c[2] for c in corners])        # (4, n_st)

    lead_h_all = (ds.lead_time.values / np.timedelta64(1, "h")).astype(float)
    # Lead-7 accumulations need hours up to 192; lead-1 temp needs from 24.
    keep = (lead_h_all >= 24) & (lead_h_all <= 192)
    sub = ds[VARS].sel(init_time=init_ts).isel(
        lead_time=np.flatnonzero(keep), latitude=lat_all, longitude=lon_all)
    data = sub.compute()

    def _interp(v: str) -> np.ndarray:
        raw = data[v].values                            # (init, lead, 4*n_st)
        raw = raw.reshape(raw.shape[0], raw.shape[1], 4, n_st)
        return np.einsum("ilcs,cs->ils", raw, weights)

    lead_h = lead_h_all[keep]
    step_s = _step_seconds(lead_h)

    t2m = _interp("temperature_2m")          # (init, lead, station) °C
    rate = _interp("precipitation_surface")  # kg m-2 s-1, step avg
    csnow = _interp("categorical_snow_surface")

    rows = []
    for n in LEADS:
        # valid day at lead n (00Z init) spans lead hours [24n, 24n+24):
        # accumulations take (24n, 24n+24] (step-avg covers the hour before
        # its timestamp); instantaneous temp takes [24n, 24n+24).
        acc = (lead_h > 24 * n) & (lead_h <= 24 * n + 24)
        ins = (lead_h >= 24 * n) & (lead_h < 24 * n + 24)
        if not acc.any() or not ins.any():
            continue
        precip_mm = np.nansum(rate[:, acc, :] * step_s[None, acc, None], axis=1)
        snow_mm = np.nansum(rate[:, acc, :] * csnow[:, acc, :]
                            * step_s[None, acc, None], axis=1)
        # nansum returns 0 for all-NaN slices — mask those to NaN explicitly
        all_nan = np.isnan(rate[:, acc, :]).all(axis=1)
        precip_mm[all_nan] = np.nan
        snow_mm[all_nan] = np.nan
        tmean = np.nanmean(t2m[:, ins, :], axis=1)
        for k, it in enumerate(init_ts):
            vd = (it + pd.Timedelta(days=n)).strftime("%Y-%m-%d")
            rows.append(pd.DataFrame({
                "triplet": triplets,
                "valid_date": vd,
                "lead_days": n,
                "snowfall_cm": snow_mm[k] * SWE_MM_TO_SNOW_CM,
                "precip_mm": precip_mm[k],
                "tmean_c": tmean[k],
            }))
    if not rows:
        return pd.DataFrame(columns=["triplet", *OUT_COLS])
    out = pd.concat(rows, ignore_index=True)
    out["snowfall_cm"] = out["snowfall_cm"].astype(float).round(3)
    out["precip_mm"] = out["precip_mm"].astype(float).round(3)
    out["tmean_c"] = out["tmean_c"].astype(float).round(2)
    return out.dropna(subset=["snowfall_cm", "precip_mm", "tmean_c"], how="all")


def append_station_rows(month_df: pd.DataFrame) -> int:
    """Append a month's rows to the per-station gzip CSVs (headerless)."""
    n = 0
    for triplet, g in month_df.groupby("triplet"):
        csv_p, _ = _paths("gfs", str(triplet))
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        g[OUT_COLS].to_csv(buf, header=False, index=False)
        with gzip.open(csv_p, "at") as f:
            f.write(buf.getvalue())
        n += len(g)
    return n


def validate_overlap(ds, stations: list[dict], *, n_stations: int = 12) -> int:
    """Regression of Zarr-derived features vs existing Open-Meteo rows on the
    overlap period. Gate: slope≈1 before any real writes."""
    have = [s for s in stations if _paths("gfs", s["triplet"])[0].exists()]
    if not have:
        print("no existing Open-Meteo gfs files to validate against")
        return 1
    sample = have[:n_stations]
    om = []
    for s in sample:
        df = read_station_model("gfs", s["triplet"])
        df["triplet"] = s["triplet"]
        om.append(df)
    om = pd.concat(om, ignore_index=True)
    om = om[om["valid_date"] >= "2025-01-01"]
    months = sorted({d[:7] for d in om["valid_date"]})[:3]
    zarr_rows = pd.concat(
        [extract_month(ds, ym, sample, today=date.today()) for ym in months],
        ignore_index=True)
    merged = om.merge(zarr_rows, on=["triplet", "valid_date", "lead_days"],
                      suffixes=("_om", "_zr"))
    if merged.empty:
        print("no overlap rows — nothing to validate")
        return 1
    print(f"validate: {len(merged)} overlap rows, {len(sample)} stations, months {months}")
    ok = True
    for col in ("snowfall_cm", "precip_mm", "tmean_c"):
        a = pd.to_numeric(merged[f"{col}_om"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_zr"], errors="coerce")
        m = a.notna() & b.notna()
        if m.sum() < 30:
            print(f"  {col}: only {m.sum()} rows, skipped")
            continue
        r = float(np.corrcoef(a[m], b[m])[0, 1])
        slope = float(np.polyfit(a[m], b[m], 1)[0]) if a[m].std() > 0 else float("nan")
        mad = float((b[m] - a[m]).abs().mean())
        print(f"  {col}: r={r:.3f} slope={slope:.3f} mean|Δ|={mad:.3f} n={int(m.sum())}")
        if col == "snowfall_cm":
            # Cold-day slice: with phase unambiguous, any residual snowfall
            # difference is interpolation/init-mix, not phase attribution.
            cold = m & (pd.to_numeric(merged["tmean_c_zr"], errors="coerce") < -2.0)
            if cold.sum() >= 30:
                rc = float(np.corrcoef(a[cold], b[cold])[0, 1])
                sc = float(np.polyfit(a[cold], b[cold], 1)[0])
                print(f"    (cold days tmean<-2C): r={rc:.3f} slope={sc:.3f} "
                      f"n={int(cold.sum())}")
        if col != "tmean_c" and (r < 0.9 or not 0.8 <= slope <= 1.25):
            ok = False
    print("VALIDATION " + ("PASSED" if ok else "FAILED — do not write; "
          "reconcile units/conventions first"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-month", default="2021-11")
    ap.add_argument("--end-month", default=None,
                    help="default: last fully archived month")
    ap.add_argument("--email", default=EMAIL)
    ap.add_argument("--limit", type=int, default=0, help="station limit (debug)")
    ap.add_argument("--validate", action="store_true",
                    help="overlap regression vs Open-Meteo rows; no writes")
    args = ap.parse_args()

    stations = json.loads((ROOT / "data" / "stations.json").read_text())["stations"]
    stations.sort(key=lambda s: s["triplet"])
    if args.limit:
        stations = stations[: args.limit]

    ds = _open(args.email)
    if args.validate:
        return validate_overlap(ds, stations)

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
        n = append_station_rows(month_df)
        state["months"][ym] = "ok"
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=0))
        print(f"{ym}: {n} rows appended in {time.time()-t1:.0f}s "
              f"(total {time.time()-t0:.0f}s)", flush=True)
        ym = nxt
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
