#!/usr/bin/env python3
"""Truth-agreement analysis: SNOTEL-QC snowfall (T1) vs NOHRSC observer
reports (T2) at nearby points — published on the verify page.

No instrument is truth in the mountains: SNOTEL ultrasonic depth sensors
drift and bridge; human observers measure boards at lower elevations hours
apart. Quantifying their agreement (a) bounds the verification-truth
uncertainty every scorecard number inherits, and (b) preempts the reviewer
question "you verified against your own QC — how do we know it's real?".

Matching: NOHRSC station coords come from the tree60weather master file
(12,855 stations); ids of the form "<lat>_<lon>" parse directly. Each
observer within --radius-km of a SNOTEL site (default 8 km) pairs with it;
24h-duration reports only.

Output: data/verify/truth_agreement.json + console table, stratified by
elevation delta (observers usually sit far below the SNOTEL).

Usage:
    python scripts/truth_agreement.py [--radius-km 8]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOHRSC_MASTER = Path("/Users/nakas/Documents/Treesixty/TreesixtyFirebase/"
                     "public/data/nohrsc-stations.json")
TRUTH_DIR = ROOT / "data" / "truth" / "nohrsc"
OUT_PATH = ROOT / "data" / "verify" / "truth_agreement.json"


def nohrsc_coords() -> pd.DataFrame:
    """station_id -> lat/lon/elev from the master file + parsable ids."""
    rows = []
    if NOHRSC_MASTER.exists():
        doc = json.loads(NOHRSC_MASTER.read_text())
        for s in doc["stations"] if isinstance(doc, dict) else doc:
            if s.get("lat") is not None and s.get("lon") is not None:
                rows.append({"station_id": s["id"], "lat": float(s["lat"]),
                             "lon": float(s["lon"]),
                             "elev_ft": (s.get("elev_m") or 0) * 3.28084})
    return pd.DataFrame(rows)


def parse_latlon_ids(ids: pd.Series) -> pd.DataFrame:
    """ids like '46.4530_090.1689' encode lat_lon (lon west-positive)."""
    pat = ids.str.extract(r"^(\d{1,2}\.\d+)_(\d{1,3}\.\d+)$")
    ok = pat[0].notna()
    return pd.DataFrame({
        "station_id": ids[ok],
        "lat": pd.to_numeric(pat.loc[ok, 0]),
        "lon": -pd.to_numeric(pat.loc[ok, 1]),
        "elev_ft": np.nan,
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-km", type=float, default=8.0)
    args = ap.parse_args()

    truth_files = sorted(glob.glob(str(TRUTH_DIR / "*.csv.gz")))
    if not truth_files:
        print("no NOHRSC truth files — run fetch_nohrsc.py first")
        return 1
    obs = pd.concat([pd.read_csv(p) for p in truth_files], ignore_index=True)
    obs = obs[(obs["duration_h"] == 24.0) & obs["snowfall_in"].notna()]
    print(f"{len(obs)} 24h observer reports, {obs['station_id'].nunique()} station ids")

    coords = nohrsc_coords()
    parsed = parse_latlon_ids(obs["station_id"].drop_duplicates())
    coords = (pd.concat([coords, parsed], ignore_index=True)
              .drop_duplicates(subset="station_id", keep="first"))
    print(f"coords resolved for {len(coords)} ids "
          f"({len(parsed)} parsed from lat_lon ids)")

    stations = json.loads((ROOT / "data" / "stations.json").read_text())["stations"]
    sn = pd.DataFrame([{"triplet": s["triplet"], "sn_lat": s["lat"],
                        "sn_lon": s["lon"],
                        "sn_elev_ft": s.get("elevation_ft")} for s in stations])

    # Nearest SNOTEL per observer (equirectangular — fine at 8 km scales).
    from scipy.spatial import cKDTree
    mean_lat = np.deg2rad(sn["sn_lat"].mean())
    def to_xy(lat, lon):
        return np.column_stack([np.asarray(lon) * np.cos(mean_lat) * 111.32,
                                np.asarray(lat) * 110.57])
    tree = cKDTree(to_xy(sn["sn_lat"], sn["sn_lon"]))
    d_km, idx = tree.query(to_xy(coords["lat"], coords["lon"]))
    coords = coords.assign(dist_km=d_km, triplet=sn["triplet"].to_numpy()[idx],
                           sn_elev_ft=sn["sn_elev_ft"].to_numpy()[idx])
    near = coords[coords["dist_km"] <= args.radius_km]
    print(f"{len(near)} observer sites within {args.radius_km} km of a SNOTEL")

    from app import snotel, targets
    from datetime import date
    pairs = []
    for triplet, grp in near.groupby("triplet"):
        try:
            hist = snotel.fetch_history(triplet, date(2024, 10, 1),
                                        date(2026, 5, 1), max_age_hours=999)
            if hist.empty:
                continue
            sf = targets.daily_snowfall(targets.qc_daily_series(hist))
            sf = sf[(pd.to_numeric(sf["quality"], errors="coerce")
                     .fillna(targets.QC_UNRELIABLE).astype(int)
                     & targets.QC_UNRELIABLE) == 0]
            sn_by_day = dict(zip(sf["date"].astype(str), sf["snowfall_in"]))
        except Exception:
            continue
        # Observer reports carry their own site elevation (elev_ft column in
        # the NSA table) — prefer it over the master file's, which is NaN for
        # lat_lon-parsed ids anyway.
        o = obs[obs["station_id"].isin(grp["station_id"])].merge(
            grp[["station_id", "dist_km", "sn_elev_ft"]], on="station_id")
        for _, r in o.iterrows():
            t1 = sn_by_day.get(r["valid_date"])
            if t1 is None:
                continue
            pairs.append({"triplet": triplet, "station_id": r["station_id"],
                          "valid_date": r["valid_date"],
                          "t1_snotel_in": float(t1),
                          "t2_nohrsc_in": float(r["snowfall_in"]),
                          "dist_km": float(r["dist_km"]),
                          "delev_ft": (float(r["elev_ft"] - r["sn_elev_ft"])
                                       if pd.notna(r["elev_ft"]) else None)})
    df = pd.DataFrame(pairs)
    if df.empty:
        print("no co-located day pairs found")
        return 1

    def stats(g: pd.DataFrame, a_col="t1_snotel_in", b_col="t2_nohrsc_in") -> dict:
        a, b = g[a_col], g[b_col]
        both_evt = ((a >= 1) & (b >= 1)).sum()
        either_evt = ((a >= 1) | (b >= 1)).sum()
        return {"n": int(len(g)),
                "r": round(float(np.corrcoef(a, b)[0, 1]), 3) if len(g) > 10 else None,
                "mean_t1": round(float(a.mean()), 3),
                "mean_t2": round(float(b.mean()), 3),
                "mad": round(float((a - b).abs().mean()), 3),
                "event_agreement_1in": round(float(both_evt / either_evt), 3)
                if either_evt else None}

    # Day-matched comparison is confounded by 24h-window offsets: NOHRSC
    # observers measure morning-to-morning LOCAL windows, SNOTEL targets are
    # UTC calendar days — a storm near the boundary lands on different "days"
    # in the two records. 3-day storm totals absorb the offset; report both
    # and treat the storm-total numbers as the meaningful agreement.
    df = df.sort_values("valid_date")
    storm = []
    for sid, g in df.groupby("station_id", sort=False):
        g = g.set_index(pd.to_datetime(g["valid_date"]))
        r3 = g[["t1_snotel_in", "t2_nohrsc_in"]].rolling("3D").sum()
        keep = (r3["t1_snotel_in"] >= 1) | (r3["t2_nohrsc_in"] >= 1)
        storm.append(r3[keep])
    storm_df = pd.concat(storm) if storm else pd.DataFrame(
        columns=["t1_snotel_in", "t2_nohrsc_in"])

    out = {"radius_km": args.radius_km,
           "day_matched": stats(df),
           "storm_total_3d": stats(storm_df) if not storm_df.empty else None,
           "note": ("day_matched is depressed by 24h-window offsets "
                    "(local-morning observer windows vs UTC SNOTEL days); "
                    "storm_total_3d is the meaningful agreement statistic"),
           "by_delev": {}}
    df["delev_band"] = pd.cut(df["delev_ft"],
                              [-12000, -2000, -500, 500, 12000],
                              labels=["obs ≫ lower", "obs lower",
                                      "co-elevation", "obs higher"])
    for band, g in df.groupby("delev_band", observed=True):
        if len(g) >= 30:
            out["by_delev"][str(band)] = stats(g)
    print(json.dumps(out, indent=2))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    df.to_csv(OUT_PATH.with_suffix(".csv.gz"), index=False, compression="gzip")
    print(f"wrote {OUT_PATH} (+ pairs csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
