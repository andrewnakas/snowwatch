#!/usr/bin/env python3
"""Join backfilled Previous-Runs forecasts with QC'd SNOTEL targets into the
training-pairs table used by the pooled post-processor and the baseline
benchmarks.

Output: data/training/pairs.csv.gz, one row per (station, valid_date, lead):

  triplet, valid_date, lead_days,
  nbm_snowfall_cm, nbm_precip_mm, nbm_tmean_c,
  hrrr_snowfall_cm, hrrr_precip_mm, hrrr_tmean_c,         (leads 1-2 only)
  gfs_snowfall_cm, gfs_precip_mm, gfs_tmean_c,
  ifs_precip_mm, ifs_tmean_c, aifs_precip_mm, aifs_tmean_c,
  mm_snow_mean_cm, mm_snow_std_cm, mm_precip_mean_mm, mm_precip_std_mm, mm_n,
  obs_snowfall_in, obs_dswe_in, obs_depth_prev_in, obs_swe_prev_in, quality,
  doy, elevation_ft, lat, lon, median_slr, snow_class, nbm_version

Usage:
    python scripts/build_training_data.py [--limit N] [--out PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import snotel, targets  # noqa: E402

_spec = importlib.util.spec_from_file_location("bf", ROOT / "scripts" / "backfill_previous_runs.py")
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

MODEL_KEYS = ("nbm", "hrrr", "gfs", "ifs", "aifs")
SNOW_MODELS = ("nbm", "hrrr", "gfs")  # models with archived snowfall


def nbm_version_for(d: str) -> str:
    """NBM version epoch by valid date (training feature / stratifier)."""
    if d >= "2026-05-05":
        return "v5.0"
    if d >= "2025-05-28":
        return "v4.3"
    return "v4.2"


def build_station(st: dict, *, hist_days: int = 900) -> pd.DataFrame | None:
    triplet = st["triplet"]

    # Forecasts: wide-merge the per-model prevruns series on (valid_date, lead).
    frames = []
    for mk in MODEL_KEYS:
        df = bf.read_station_model(mk, triplet)
        if df.empty:
            continue
        keep = ["valid_date", "lead_days"]
        ren = {}
        for c in ("snowfall_cm", "precip_mm", "tmean_c"):
            if df[c].notna().any():
                ren[c] = f"{mk}_{c}"
        if not ren:
            continue
        frames.append(df[keep + list(ren)].rename(columns=ren))
    if not frames:
        return None
    fc = frames[0]
    for f in frames[1:]:
        fc = fc.merge(f, on=["valid_date", "lead_days"], how="outer")

    # Multi-model consensus features.
    snow_cols = [f"{m}_snowfall_cm" for m in SNOW_MODELS if f"{m}_snowfall_cm" in fc.columns]
    precip_cols = [f"{m}_precip_mm" for m in MODEL_KEYS if f"{m}_precip_mm" in fc.columns]
    if snow_cols:
        fc["mm_snow_mean_cm"] = fc[snow_cols].mean(axis=1)
        fc["mm_snow_std_cm"] = fc[snow_cols].std(axis=1)
    if precip_cols:
        fc["mm_precip_mean_mm"] = fc[precip_cols].mean(axis=1)
        fc["mm_precip_std_mm"] = fc[precip_cols].std(axis=1)
        fc["mm_n"] = fc[precip_cols].notna().sum(axis=1)

    # Targets: QC'd daily snowfall + antecedent state.
    end = date.today()
    start = end - timedelta(days=hist_days)
    hist = snotel.fetch_history(triplet, start, end)
    if hist.empty:
        return None
    hist_qc = targets.qc_daily_series(hist)
    snow = targets.daily_snowfall(hist_qc)
    statics = targets.station_statics(st, hist_qc, snow)

    obs = snow.rename(columns={"snowfall_in": "obs_snowfall_in", "dswe_in": "obs_dswe_in"})
    obs["valid_date"] = obs["date"].astype(str)
    obs = obs[["valid_date", "obs_snowfall_in", "obs_dswe_in", "quality"]]

    state = pd.DataFrame({
        "valid_date": hist_qc["date"].astype(str),
        # Antecedent (previous-day) pack state — known at issue time.
        "obs_depth_prev_in": hist_qc["snwd_qc"].shift(1),
        "obs_swe_prev_in": pd.to_numeric(hist_qc["swe_in"], errors="coerce").shift(1),
    })

    out = fc.merge(obs, on="valid_date", how="inner").merge(state, on="valid_date", how="left")
    if out.empty:
        return None
    out["triplet"] = triplet
    out["doy"] = pd.to_datetime(out["valid_date"]).dt.dayofyear
    out["elevation_ft"] = statics.get("elevation_ft")
    out["lat"] = statics.get("lat")
    out["lon"] = statics.get("lon")
    out["median_slr"] = statics.get("median_slr")
    out["snow_class"] = statics.get("snow_class")
    out["nbm_version"] = out["valid_date"].map(nbm_version_for)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "training" / "pairs.csv.gz")
    args = ap.parse_args()

    stations = json.loads((ROOT / "data" / "stations.json").read_text())["stations"]
    if args.limit:
        stations = stations[: args.limit]

    frames = []
    t0 = time.time()
    for i, st in enumerate(stations, 1):
        try:
            df = build_station(st)
        except Exception as exc:
            print(f"[{i}/{len(stations)}] {st['triplet']} FAIL {exc}")
            continue
        if df is not None and not df.empty:
            frames.append(df)
        if i % 50 == 0:
            print(f"[{i}/{len(stations)}] rows so far: {sum(len(f) for f in frames)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if not frames:
        print("no pairs built — run backfill_previous_runs.py first")
        return 1
    pairs = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out, index=False, compression="gzip")
    n_evt = int((pairs["obs_snowfall_in"] > 0.5).sum())
    print(f"wrote {len(pairs)} pairs ({len(frames)} stations, {n_evt} snowfall-event rows) "
          f"-> {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
