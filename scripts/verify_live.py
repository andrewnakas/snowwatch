#!/usr/bin/env python3
"""Rolling as-issued verification — the strongest claim SnowWatch can make:
forecasts scored exactly as published, before the outcome was known.

Downloads the monthly `forecast-archive` Release assets (written by the 6h
pages build via app/archive.py), joins them to QC'd SNOTEL observations, and
emits a rolling scorecard JSON per source (postproc, model:nbm, blend depth)
with the same metrics + paired bootstrap as the offline folds.

Dedup convention: the archive holds up to 4 builds per issue day; the LAST
build of each (station, issue_date, lead, source) is scored — the same
forecast a site visitor saw at end of day.

Usage:
    python scripts/verify_live.py [--window-days 90] [--archive-dir DIR]
        [--out data/verify/live_scorecard.json]
--archive-dir skips the gh download (offline/testing).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import archive, snotel, targets, verification  # noqa: E402

CM_TO_IN = 1.0 / 2.54
RELEASE_TAG = "forecast-archive"
SOURCES = ("postproc", "model:nbm", "model:gfs", "blend", "ens")


def _months_covering(start: date, end: date) -> list[str]:
    out, cur = [], start.replace(day=1)
    while cur <= end:
        out.append(cur.strftime("%Y-%m"))
        cur = (cur + timedelta(days=32)).replace(day=1)
    return out


def download_assets(months: list[str], dest: Path) -> list[Path]:
    got = []
    for ym in months:
        asset = f"archive-{ym}.csv.gz"
        p = dest / asset
        r = subprocess.run(["gh", "release", "download", RELEASE_TAG,
                            "--pattern", asset, "--output", str(p), "--clobber"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0 and p.exists():
            got.append(p)
        else:
            print(f"  (no asset {asset})")
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=90)
    ap.add_argument("--archive-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "verify" / "live_scorecard.json")
    args = ap.parse_args()

    end = date.today() - timedelta(days=1)     # yesterday: obs complete
    start = end - timedelta(days=args.window_days)

    if args.archive_dir:
        paths = sorted(args.archive_dir.glob("archive-*.csv.gz"))
    else:
        tmp = Path(tempfile.mkdtemp(prefix="sw_archive_"))
        paths = download_assets(_months_covering(start, end), tmp)
    if not paths:
        print("no archive assets found — nothing to verify yet")
        return 1
    arc = archive.read_archive(paths)
    arc = arc[arc["source"].isin(SOURCES)]
    arc = arc[(arc["valid_date"] >= start.isoformat())
              & (arc["valid_date"] <= end.isoformat())]
    if arc.empty:
        print("archive has no rows in the window")
        return 1
    # Last build per (station, issue, lead, source).
    arc = (arc.sort_values("build_hour")
              .drop_duplicates(subset=["triplet", "issue_date", "lead_days",
                                       "valid_date", "source"], keep="last"))

    triplets = sorted(arc["triplet"].dropna().unique())
    print(f"{len(arc)} archived rows, {len(triplets)} stations, "
          f"{start} .. {end}")
    hist = snotel.fetch_history_batch(triplets, start - timedelta(days=40), end)
    obs_frames, depth_frames = [], []
    for t, h in hist.items():
        if h is None or h.empty:
            continue
        hq = targets.qc_daily_series(h)
        sf = targets.daily_snowfall(hq)
        sf = sf[(sf["quality"].astype(int) & targets.QC_UNRELIABLE) == 0]
        obs_frames.append(pd.DataFrame({
            "triplet": t, "valid_date": sf["date"].astype(str),
            "obs_snowfall_in": sf["snowfall_in"]}))
        depth_frames.append(pd.DataFrame({
            "triplet": t, "valid_date": hq["date"].astype(str),
            "obs_depth_in": hq["snwd_qc"]}))
    if not obs_frames:
        print("no usable observations")
        return 1
    obs = pd.concat(obs_frames, ignore_index=True)
    depths = pd.concat(depth_frames, ignore_index=True)

    out: dict = {"window": {"start": start.isoformat(), "end": end.isoformat()},
                 "generated": date.today().isoformat(),
                 "n_stations": len(triplets), "sources": {}}

    # Snowfall sources.
    snow = arc[arc["source"].isin(("postproc", "model:nbm", "model:gfs", "ens"))].copy()
    snow["pred_in"] = pd.to_numeric(snow["snowfall_cm"], errors="coerce") * CM_TO_IN
    snow = snow.merge(obs, on=["triplet", "valid_date"], how="inner")
    for src, g in snow.groupby("source"):
        m = verification.summarize_deterministic(g, obs_col="obs_snowfall_in",
                                                 pred_col="pred_in")
        m["by_lead"] = {int(l): verification.mae_bias(x["obs_snowfall_in"], x["pred_in"])
                        for l, x in g.groupby("lead_days")}
        out["sources"][src] = m
        print(f"{src:>10}: n={m['n']} mae={m['mae']:.3f} "
              f"csi_1in={m['csi_1in'] if m['csi_1in'] is not None else 'n/a'}")

    # Blend depth (site headline product).
    bl = arc[arc["source"] == "blend"].copy()
    bl["pred_in"] = pd.to_numeric(bl["depth_in"], errors="coerce")
    bl = bl.merge(depths, on=["triplet", "valid_date"], how="inner")
    if not bl.empty:
        out["sources"]["blend_depth"] = verification.mae_bias(
            bl["obs_depth_in"], bl["pred_in"])
        out["sources"]["blend_depth"]["by_lead"] = {
            int(l): verification.mae_bias(x["obs_depth_in"], x["pred_in"])
            for l, x in bl.groupby("lead_days")}

    # Head-to-head: postproc vs NBM on shared as-issued rows.
    pp = snow[snow["source"] == "postproc"][
        ["triplet", "valid_date", "lead_days", "pred_in", "obs_snowfall_in"]]
    nb = snow[snow["source"] == "model:nbm"][
        ["triplet", "valid_date", "lead_days", "pred_in"]]
    head = pp.merge(nb, on=["triplet", "valid_date", "lead_days"],
                    suffixes=("_pp", "_nbm"))
    if len(head) >= 200:
        head["err_pp"] = head["pred_in_pp"] - head["obs_snowfall_in"]
        head["err_nbm"] = head["pred_in_nbm"] - head["obs_snowfall_in"]
        out["vs_nbm"] = verification.paired_block_bootstrap(
            head, err_a="err_pp", err_b="err_nbm")
        bb = out["vs_nbm"]
        if bb["diff"] is not None:
            print(f"as-issued postproc vs NBM: ΔMAE={bb['diff']:+.3f} "
                  f"[{bb['ci_lo']:+.3f},{bb['ci_hi']:+.3f}] P={bb['p_a_better']:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
