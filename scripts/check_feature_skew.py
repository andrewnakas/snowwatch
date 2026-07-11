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


def check_release3(sample: list[dict]) -> list[str]:
    """Release-3 feature parity: sources differ from the multimodel API, so
    each gets its own check. Returns failed-check names (empty = ok).

    * wind_uv — live Open-Meteo speed+direction -> u/v vs the training GFS
      Zarr components (same model, different plumbing + time sampling:
      hourly_6 live vs hourly archive), so loose gates;
    * z500 / hrrr_native — the live path IS the training store (bulk Zarr
      prefetch, scripts/fetch_live_upperair.py), so skew is structurally
      impossible; the check is freshness + coverage of the cache.
    """
    from app.phase_features import live_phase_daily
    failed = []
    # upperair cache freshness/coverage
    up_path = ROOT / "data" / "cache" / "live_upperair.json"
    try:
        up = json.loads(up_path.read_text())
        age_h = (pd.Timestamp.now("UTC")
                 - pd.Timestamp(up["created_utc"])).total_seconds() / 3600
        n_z = sum(1 for v in up["stations"].values() if v.get("z500_by_valid"))
        n_h = sum(1 for v in up["stations"].values() if v.get("hrrr_native_by_valid"))
        print(f"  upperair cache: age {age_h:.1f}h, z500 {n_z}, hrrr {n_h} stations")
        if age_h > 24 or n_z < 800:
            failed.append("z500_upperair_cache")
        if n_h < 700:   # CONUS-only (~835 in-domain)
            failed.append("hrrr_native_upperair_cache")
    except Exception as exc:
        print(f"  upperair cache unreadable ({type(exc).__name__}) — run "
              "scripts/fetch_live_upperair.py")
        failed.append("upperair_cache_missing")
    # u/v live vs training tree (gfs_wind) on today's overlap
    rows = []
    today = date.today().isoformat()
    for st in sample[:6]:
        tr = pd.DataFrame()
        p = ROOT / "data" / "prevruns" / "gfs_wind" / f"{st['triplet'].replace(':', '_')}.csv.gz"
        if p.exists():
            tr = pd.read_csv(p, names=["valid_date", "lead_days",
                                       "u10_mean_ms", "v10_mean_ms"],
                             compression="gzip")
            tr = tr[tr["valid_date"] >= today]
        if tr.empty:
            continue
        live = live_phase_daily(met.fetch_phase_hourly(st["lat"], st["lon"], days=7))
        if live.empty or live["u10_mean_ms"].isna().all():
            continue
        j = tr.merge(live[["valid_date", "u10_mean_ms", "v10_mean_ms"]],
                     on="valid_date", suffixes=("_tr", "_lv"))
        j["lead_live"] = (pd.to_datetime(j["valid_date"])
                          - pd.Timestamp(today)).dt.days + 1
        j = j[j["lead_days"] == j["lead_live"]]
        rows.append(j)
    if rows:
        j = pd.concat(rows, ignore_index=True)
        for c in ("u10_mean_ms", "v10_mean_ms"):
            a = pd.to_numeric(j[f"{c}_tr"], errors="coerce")
            b = pd.to_numeric(j[f"{c}_lv"], errors="coerce")
            ok_rows = a.notna() & b.notna()
            if ok_rows.sum() < 4:
                continue
            mad = float((a - b)[ok_rows].abs().mean())
            r = float(np.corrcoef(a[ok_rows], b[ok_rows])[0, 1]) \
                if ok_rows.sum() >= 8 else float("nan")
            ok = mad <= 2.0 and (np.isnan(r) or r >= 0.6)
            print(f"  {c:>14}: mean|Δ|={mad:.2f} (tol 2.0) r={r:.2f} "
                  f"(min 0.6) n={int(ok_rows.sum())}  [{'ok' if ok else 'SKEW'}]")
            if not ok:
                failed.append(c)
    else:
        print("  wind_uv: no gfs_wind tree overlap for today — soft (nightly "
              "zarr top-up hasn't covered today); u/v sign convention was "
              "verified against the archive 2026-07-11")
    return failed


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
    print("release-3 feature checks:")
    failed += check_release3(sample)
    if failed:
        print(f"FAILED: {failed} — do not release / do not enable postproc")
        return 1
    print("PASSED — train and live paths agree within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
