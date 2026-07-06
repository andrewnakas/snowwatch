#!/usr/bin/env python3
"""Train/serve feature-skew gate — run before every model release and before
flipping SW_ENABLE_POSTPROC.

For a sample of stations and a recent fully-archived valid date, compute the
same feature columns through BOTH paths:

  training path: data/prevruns/<model>/ trees (Zarr- or API-backfilled),
                 exactly as scripts/build_training_data.py joins them
  live path:     app/met.py fetch_multimodel (Open-Meteo), exactly as
                 app/postproc.build_inference_features shapes them

and report per-feature mean |Δ|, correlation, and regression slope. The
model was trained on the training-path numbers; if the live path feeds it
systematically different numbers, accuracy silently degrades — this script
makes that failure loud.

Live NWP for a *past* valid date isn't fetchable (forecast API serves the
future), so the comparison uses TODAY+1..3 as valid dates against the
prevruns partial-chunk rows written in the last nightly backfill. Where the
backfill hasn't covered today yet, the station is skipped — the check needs
overlap, not completeness.

Exit 0 = within tolerance; exit 1 = skew beyond gates (block the release).

Usage:
    python scripts/check_feature_skew.py [--stations 12] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import met  # noqa: E402
from scripts.backfill_previous_runs import read_station_model  # noqa: E402

# (feature, tolerance on mean |Δ|, minimum correlation) — tolerances sized to
# feature scales: temps in °C, precip in mm, snowfall in cm.
GATES = {
    "gfs_snowfall_cm": (1.5, 0.75),
    "gfs_precip_mm": (2.5, 0.75),
    "gfs_tmean_c": (2.5, 0.90),
    "nbm_snowfall_cm": (1.0, 0.85),
    "nbm_precip_mm": (2.0, 0.85),
    "nbm_tmean_c": (2.0, 0.92),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", type=int, default=12)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    stations = json.loads((ROOT / "data" / "stations.json").read_text())["stations"]
    conus = [s for s in stations if s["triplet"].split(":")[1] != "AK"]
    sample = conus[:: max(1, len(conus) // args.stations)][: args.stations]

    today = date.today().isoformat()
    rows = []
    for st in sample:
        mm = met.fetch_multimodel(st["lat"], st["lon"], days=7)
        if mm is None or mm.empty:
            continue
        mm = mm.copy()
        mm["valid_date"] = mm["date"].astype(str)
        for mk in ("gfs", "nbm"):
            tr = read_station_model(mk, st["triplet"])
            if tr.empty:
                continue
            tr = tr[tr["valid_date"] >= today]
            if tr.empty:
                continue
            j = tr.merge(mm[["valid_date", f"{mk}_snowfall", f"{mk}_precip",
                             f"{mk}_tmean"]], on="valid_date", how="inner")
            # Live frame lacks lead alignment (one row per valid day at the
            # current issue); compare against the matching training lead.
            j["lead_live"] = (pd.to_datetime(j["valid_date"])
                              - pd.Timestamp(today)).dt.days + 1
            j = j[j["lead_days"] == j["lead_live"]]
            for feat, live_col in ((f"{mk}_snowfall_cm", f"{mk}_snowfall"),
                                   (f"{mk}_precip_mm", f"{mk}_precip"),
                                   (f"{mk}_tmean_c", f"{mk}_tmean")):
                tr_col = feat.split("_", 1)[1]
                for _, r in j.iterrows():
                    a = pd.to_numeric(pd.Series([r.get(tr_col)]), errors="coerce").iloc[0]
                    b = pd.to_numeric(pd.Series([r.get(live_col)]), errors="coerce").iloc[0]
                    if pd.notna(a) and pd.notna(b):
                        rows.append({"feature": feat, "train": float(a),
                                     "live": float(b),
                                     "triplet": st["triplet"]})
    if not rows:
        print("NO OVERLAP — nightly backfill hasn't covered today; rerun "
              "after the next backfill. Treat as a soft failure.")
        return 1
    df = pd.DataFrame(rows)
    print(f"{len(df)} feature comparisons across "
          f"{df['triplet'].nunique()} stations, valid dates >= {today}")
    failed = []
    for feat, g in df.groupby("feature"):
        mad = float((g["live"] - g["train"]).abs().mean())
        if len(g) >= 8 and g["train"].std() > 0 and g["live"].std() > 0:
            r = float(np.corrcoef(g["train"], g["live"])[0, 1])
        else:
            r = float("nan")
        tol_mad, tol_r = GATES.get(feat, (np.inf, -1))
        ok = mad <= tol_mad and (np.isnan(r) or r >= tol_r)
        status = "ok" if ok else "SKEW"
        print(f"  {feat:>18}: mean|Δ|={mad:.3f} (tol {tol_mad}) "
              f"r={r:.3f} (min {tol_r}) n={len(g)}  [{status}]")
        if not ok:
            failed.append(feat)
        if args.verbose and not ok:
            print(g.sort_values("train", ascending=False).head(8).to_string())
    if failed:
        print(f"FAILED: {failed} — do not release / do not enable postproc")
        return 1
    print("PASSED — train and live paths agree within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
