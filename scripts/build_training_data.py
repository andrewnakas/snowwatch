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
  obs_snowfall_in, obs_dswe_in, obs_depth_issue_in, obs_swe_issue_in, quality,
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
from app.postproc import nbm_version_for  # noqa: E402

_spec = importlib.util.spec_from_file_location("bf", ROOT / "scripts" / "backfill_previous_runs.py")
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

MODEL_KEYS = ("nbm", "hrrr", "gfs", "ifs", "aifs")
SNOW_MODELS = ("nbm", "hrrr", "gfs")  # models with archived snowfall

# v1.6 Phase-3 feature trees (all keyed on (valid_date, lead_days), same
# headerless-gzip layout as the model trees; left-merged, NaN where absent):
#   gefs_ens   — GEFS 31-member spread stats + z500 (backfill_gefs_zarr.py)
#   hrrr_zarr  — HRRR NATIVE snowfall, lead 1 (backfill_hrrr_zarr.py);
#                a different quantity than hrrr_snowfall_cm (0.7 cm/mm
#                convention), hence its own column
#   gfs_phase  — GFS wet-bulb aggregates (backfill_gfs_phase_zarr.py),
#                storm-gate-masked below to mirror the live fetch gate
GEFS_ENS_COLS = ["ens_snow_mean_cm", "ens_snow_std_cm", "ens_snow_p10_cm",
                 "ens_snow_p50_cm", "ens_snow_p90_cm", "ens_prob_pos",
                 "ens_precip_mean_mm", "ens_precip_std_mm", "ens_n_members",
                 "z500_mean_m"]
GFS_PHASE_COLS = ["wb_mean_c", "wb_min_c", "hours_wb_below_0", "wind10_mean_ms"]


def _read_aux_tree(tree: str, triplet: str, cols: list[str]) -> pd.DataFrame:
    p = (Path(__file__).resolve().parents[1] / "data" / "prevruns" / tree
         / f"{triplet.replace(':', '_')}.csv.gz")
    if not p.exists():
        return pd.DataFrame(columns=["valid_date", "lead_days", *cols])
    df = pd.read_csv(p, names=["valid_date", "lead_days", *cols], compression="gzip")
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["lead_days"] = pd.to_numeric(df["lead_days"], errors="coerce").astype("Int64")
    return df.drop_duplicates(subset=["valid_date", "lead_days"], keep="last")


# SNOTEL obs history window. The Zarr GFS/HRRR/GEFS backfill reaches winter
# 2021-22, so the obs pull must too — AWDB is free, the constraint is only
# fetch time. 1900 days ≈ 5.2 winters.
HIST_DAYS = 1900


def build_station(st: dict, *, hist_days: int = HIST_DAYS) -> pd.DataFrame | None:
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

    # Phase-3 aux trees (left merges; all-NaN when the tree hasn't landed).
    ens = _read_aux_tree("gefs_ens", triplet, GEFS_ENS_COLS)
    if not ens.empty:
        fc = fc.merge(ens, on=["valid_date", "lead_days"], how="left")
    hz = _read_aux_tree("hrrr_zarr", triplet,
                        ["hrrr_native_snowfall_cm", "hrrr_native_precip_mm",
                         "hrrr_native_tmean_c"])
    if not hz.empty:
        fc = fc.merge(hz[["valid_date", "lead_days", "hrrr_native_snowfall_cm"]],
                      on=["valid_date", "lead_days"], how="left")
    ph = _read_aux_tree("gfs_phase", triplet, GFS_PHASE_COLS)
    if not ph.empty:
        fc = fc.merge(ph, on=["valid_date", "lead_days"], how="left")
        # Gate mirror: live inference only fetches phase vars when consensus
        # precip clears the storm gate — training rows below it must not
        # carry the features, or the model learns a signal it won't get.
        from app.phase_features import storm_gate
        gated_off = ~storm_gate(fc.get("mm_precip_mean_mm")).to_numpy()
        fc.loc[gated_off, GFS_PHASE_COLS] = np.nan

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

    out = fc.merge(obs, on="valid_date", how="inner")
    if out.empty:
        return None

    # Pack state as of ISSUE time (valid_date - lead), not the day before
    # valid: for lead > 1 the previous-day observation is still in the future
    # when the forecast is made — training on it would leak state the live
    # model can never have. At inference all leads share the latest obs.
    depth_by_date = dict(zip(hist_qc["date"].astype(str), hist_qc["snwd_qc"]))
    swe_by_date = dict(zip(hist_qc["date"].astype(str),
                           pd.to_numeric(hist_qc["swe_in"], errors="coerce")))
    issue = (pd.to_datetime(out["valid_date"])
             - pd.to_timedelta(out["lead_days"].astype(int), unit="D")).dt.strftime("%Y-%m-%d")
    out["obs_depth_issue_in"] = issue.map(depth_by_date)
    out["obs_swe_issue_in"] = issue.map(swe_by_date)
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

    # Coverage report: stations × winters × model — the honest picture of
    # what the pooled model actually trains on (and the Phase-2 progress bar).
    cov = pairs.copy()
    vd = pd.to_datetime(cov["valid_date"])
    cov["winter"] = np.where(vd.dt.month >= 10, vd.dt.year, vd.dt.year - 1).astype(str)
    report: dict = {"built": date.today().isoformat(), "n_pairs": int(len(pairs)),
                    "n_stations": int(cov["triplet"].nunique()),
                    "event_rows_6in": int((pairs["obs_snowfall_in"] >= 6.0).sum()),
                    "event_rows_12in": int((pairs["obs_snowfall_in"] >= 12.0).sum()),
                    "by_winter": {}, "stations_by_model": {}}
    for w, g in cov.groupby("winter"):
        report["by_winter"][w] = {"rows": int(len(g)),
                                  "stations": int(g["triplet"].nunique())}
    for mk in MODEL_KEYS:
        col = f"{mk}_precip_mm"
        if col in cov.columns:
            report["stations_by_model"][mk] = int(
                cov.loc[cov[col].notna(), "triplet"].nunique())
    cov_path = args.out.parent / "coverage.json"
    cov_path.write_text(json.dumps(report, indent=2))
    print(f"coverage: {json.dumps(report['stations_by_model'])} | "
          f"6in rows={report['event_rows_6in']} 12in rows={report['event_rows_12in']} "
          f"-> {cov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
