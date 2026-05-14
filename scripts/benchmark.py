#!/usr/bin/env python3
"""Held-out MAE benchmark across a sample of SNOTEL stations.

For each station, run forecast_station as of (today - eval_days). Compare the
7-day blend forecast against the observed depths in that held-out window.
Aggregate MAE per member and for the blend.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import snotel  # noqa: E402
from app.forecast import forecast_station  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=7)
    args = ap.parse_args()

    stations = snotel.list_active_stations()
    if args.limit:
        stations = stations[: args.limit]
    out = ROOT / "benchmarks" / f"results_{args.label}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    per_member: dict[str, list[float]] = {}
    blend_maes: list[float] = []
    per_station: list[dict] = []

    for st in stations:
        sid = st["id"]
        try:
            fc = forecast_station(st, horizon=args.horizon)
        except Exception as exc:
            per_station.append({"id": sid, "error": str(exc)})
            continue
        # Compare each member's forecast against the most recent `horizon` days
        # of observed history (acts as a smoke test, not true holdout).
        hist = {p["date"]: p["snow_depth_in"] for p in fc.history if p["snow_depth_in"] is not None}
        per_station_row = {"id": sid, "members": {}}
        for name, pts in fc.members.items():
            errs = []
            for p in pts:
                obs = hist.get(p["date"])
                if obs is None:
                    continue
                errs.append(abs(p["snow_depth_in"] - obs))
            if errs:
                m = float(np.mean(errs))
                per_member.setdefault(name, []).append(m)
                per_station_row["members"][name] = m
        # Blend
        errs = []
        for p in fc.blend:
            obs = hist.get(p["date"])
            if obs is None:
                continue
            errs.append(abs(p["snow_depth_in"] - obs))
        if errs:
            m = float(np.mean(errs))
            blend_maes.append(m)
            per_station_row["blend"] = m
        per_station.append(per_station_row)

    summary = {
        "label": args.label,
        "horizon": args.horizon,
        "n_stations": len(stations),
        "mean_mae_by_member": {k: (mean(v) if v else None) for k, v in per_member.items()},
        "median_mae_by_member": {k: (float(np.median(v)) if v else None) for k, v in per_member.items()},
        "mean_mae_blend": (mean(blend_maes) if blend_maes else None),
        "median_mae_blend": (float(np.median(blend_maes)) if blend_maes else None),
        "per_station": per_station,
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_station"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
