#!/usr/bin/env python3
"""SLR (snow-to-liquid ratio) prediction eval — win condition W5.

Veals et al. 2025 (WaF, DOI 10.1175/WAF-D-24-0233.1): random-forest SLR at
14 manual western-US sites, R² = 0.43, MAE = 2.94, vs operational baselines
R² 0.04–0.23 / MAE 4.01–9.45. This scores a LightGBM SLR head on SNOTEL
corroborated-SLR days with the SAME metrics — quoted as "same metric,
different site set": SNOTEL-derived SLR is noisier than manual boards, which
makes the target strictly harder, not easier.

Target: obs SLR = obs_snowfall_in / obs_dswe_in on days where both are
corroborated (snowfall ≥ 1 in, dSWE ≥ 0.05 in, SLR within targets' [3, 30]
sanity band — mirroring app/targets.py). Predictors: forecast-time features
only (wet-bulb trio when the gfs_phase tree is present, per-model tmean,
elevation, doy, ens spread) at lead 1 — apples-to-apples with Veals'
"forecastable SLR" setup.

Temporal split at --cutoff (default 2025-11-30: train = 4 winters, test =
core winter 2025-26). Baselines: station median_slr (climatology) and the
fixed 13:1 western mean (Baxter et al. 2005).

Usage:
    python scripts/train_slr.py [--pairs data/training/pairs.csv.gz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SLR_FEATURES = [
    "wb_mean_c", "wb_min_c", "hours_wb_below_0", "wind10_mean_ms",
    "nbm_tmean_c", "gfs_tmean_c", "hrrr_tmean_c",
    "gfs_precip_mm", "mm_precip_mean_mm",
    "ens_snow_std_cm", "ens_precip_std_mm",
    "elevation_ft", "lat", "lon", "doy_sin", "doy_cos", "median_slr",
]
SLR_MIN, SLR_MAX = 3.0, 30.0


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=ROOT / "data" / "training" / "pairs.csv.gz")
    ap.add_argument("--cutoff", default="2025-11-30")
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "models" / "metrics_slr.json")
    args = ap.parse_args()

    import lightgbm as lgb

    pairs = pd.read_csv(args.pairs, compression="gzip")
    df = pairs[pairs["lead_days"] == args.lead].copy()
    snow = pd.to_numeric(df["obs_snowfall_in"], errors="coerce")
    dswe = pd.to_numeric(df["obs_dswe_in"], errors="coerce")
    slr = snow / dswe
    ok = (snow >= 1.0) & (dswe >= 0.05) & slr.between(SLR_MIN, SLR_MAX)
    df, slr = df[ok], slr[ok]
    print(f"{len(df)} corroborated-SLR station-days at lead {args.lead} "
          f"({df['triplet'].nunique()} stations)")
    if len(df) < 2000:
        print("too few SLR days — need more pairs coverage")
        return 1

    doy = pd.to_numeric(df["doy"], errors="coerce")
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feats = [c for c in SLR_FEATURES if c in df.columns]
    have_wb = "wb_mean_c" in feats and df["wb_mean_c"].notna().any()
    print(f"features: {feats} | wet-bulb present: {have_wb}")
    X = df[feats].apply(pd.to_numeric, errors="coerce")

    tr = df["valid_date"] < args.cutoff
    te = ~tr
    if te.sum() < 500 or tr.sum() < 2000:
        print("degenerate split")
        return 1
    dset = lgb.Dataset(X[tr], label=slr[tr], free_raw_data=False)
    params = {"objective": "l2", "metric": "l2", "learning_rate": 0.05,
              "num_leaves": 63, "min_data_in_leaf": 100,
              "feature_fraction": 0.85, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "verbosity": -1}
    bst = lgb.train(params, dset, num_boost_round=500)
    pred = np.clip(bst.predict(X[te]), SLR_MIN, SLR_MAX)

    y = slr[te].to_numpy()
    out = {
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "n_stations_test": int(df.loc[te, "triplet"].nunique()),
        "lead_days": args.lead, "cutoff": args.cutoff,
        "features": feats, "wet_bulb_present": bool(have_wb),
        "model": {"r2": round(r2(y, pred), 3),
                  "mae": round(float(np.abs(y - pred).mean()), 3)},
        "baseline_station_median": {
            "r2": round(r2(y, pd.to_numeric(df.loc[te, "median_slr"],
                                            errors="coerce").fillna(13.0)), 3),
            "mae": round(float(np.abs(y - pd.to_numeric(
                df.loc[te, "median_slr"], errors="coerce").fillna(13.0)).mean()), 3)},
        "baseline_fixed_13": {
            "r2": round(r2(y, np.full_like(y, 13.0)), 3),
            "mae": round(float(np.abs(y - 13.0).mean()), 3)},
        "published_target_veals2025": {"r2": 0.43, "mae": 2.94,
                                       "note": "14 manual sites; same metric, different site set"},
        "importances": dict(sorted(zip(feats, bst.feature_importance("gain").round(0)),
                                   key=lambda kv: -kv[1])[:8]),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "features"},
                     indent=2, default=float))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
