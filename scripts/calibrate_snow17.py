#!/usr/bin/env python3
"""Warm the per-station SNOW-17 parameter cache.

For every station in `data/stations.json` (or the file pointed at by
`SW_STATIONS_FILE`), fetch ~2 winter seasons of paired SNOTEL + Open-Meteo
history, fit MFMAX/MFMIN/UADJ/PXTEMP via L-BFGS-B, and persist the result
to `data/snow17_params/<triplet>.json`.

The fit is cheap (each station typically <2s on warm cache) but the API
fetches are not free. Run this:

  - Once locally to seed the cache
  - On a weekly schedule (separate workflow) to refresh
  - Optionally with --limit during testing

Usage:
  python scripts/calibrate_snow17.py [--limit N] [--seasons 2]
                                     [--force] [--stations-file PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import snotel, snow17 as _snow17, snow17_calibrate as _cal, weather  # noqa: E402


def _load_stations(path: Path) -> list[dict]:
    # stations.json is {"stations": [...]}; tolerate a bare list too.
    obj = json.loads(path.read_text())
    return obj["stations"] if isinstance(obj, dict) else obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seasons", type=int, default=2,
                    help="winter seasons of history to pull (1 ≈ 400 days).")
    ap.add_argument("--force", action="store_true",
                    help="re-fit even if cache is fresh.")
    ap.add_argument("--stations-file", default=None)
    ap.add_argument("--triplet", default=None,
                    help="only calibrate one station (debug).")
    args = ap.parse_args()

    stations_path = Path(
        args.stations_file
        or os.environ.get("SW_STATIONS_FILE")
        or (ROOT / "data" / "stations.json")
    )
    stations = _load_stations(stations_path)
    if args.triplet:
        stations = [s for s in stations if s.get("triplet") == args.triplet]
    if args.limit:
        stations = stations[: args.limit]

    days_back = max(400, args.seasons * 400)
    today = date.today()
    start = today - timedelta(days=days_back)

    n_done = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()
    for i, s in enumerate(stations, 1):
        triplet = s.get("triplet")
        if not triplet:
            n_fail += 1
            continue
        if not args.force and _cal.load_cached(triplet):
            n_skip += 1
            continue
        try:
            hist = snotel.fetch_history(triplet, start, today)
            wx_hist = weather.fetch_history(s["lat"], s["lon"], start, today)
        except Exception as exc:
            print(f"[{i}/{len(stations)}] {triplet} fetch failed: {exc}")
            n_fail += 1
            continue
        base = _snow17.default_params(
            elev_m=(float(s.get("elevation_ft")) * 0.3048) if s.get("elevation_ft") else None,
            lat_deg=float(s.get("lat") or 45.0),
        )
        try:
            rec = _cal.calibrate(triplet, hist, wx_hist, base=base)
        except Exception as exc:
            print(f"[{i}/{len(stations)}] {triplet} fit error: {exc}")
            n_fail += 1
            continue
        if rec is None:
            print(f"[{i}/{len(stations)}] {triplet} insufficient history; skip")
            n_skip += 1
            continue
        delta = rec["loss_base"] - rec["loss_fit"]
        print(
            f"[{i}/{len(stations)}] {triplet} fit n={rec['n_rows']} "
            f"loss {rec['loss_base']:.3f} -> {rec['loss_fit']:.3f} (Δ {delta:+.3f})  "
            f"MFMAX={rec['params']['MFMAX']:.2f} MFMIN={rec['params']['MFMIN']:.2f} "
            f"UADJ={rec['params']['UADJ']:.3f} PXTEMP={rec['params']['PXTEMP']:.2f}"
        )
        n_done += 1

    dt = time.time() - t0
    print(f"\ndone: fit={n_done} skip={n_skip} fail={n_fail}  ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
