"""Spot-check the QC + snowfall-target layer for one or more stations.

Usage:
    python scripts/inspect_targets.py --triplet 480:MT:SNTL
    python scripts/inspect_targets.py --sample 20 --days 730
    python scripts/inspect_targets.py --triplet 480:MT:SNTL --plot out.png

Prints a per-station summary (spikes removed, stuck runs, method mix,
seasonal totals vs raw ΔSNWD⁺) and optionally writes a matplotlib plot of
raw vs QC'd depth with snowfall bars.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import snotel, targets  # noqa: E402


def inspect_one(station: dict, days: int, plot_path: str | None = None) -> dict:
    end = date.today()
    start = end - timedelta(days=days)
    hist = snotel.fetch_history(station["triplet"], start, end)
    if hist.empty:
        return {"triplet": station["triplet"], "error": "no history"}

    hist_qc = targets.qc_daily_series(hist)
    snow = targets.daily_snowfall(hist_qc)
    statics = targets.station_statics(station, hist_qc, snow, use_cache=False)

    flags = hist_qc["qc_flags"].astype(int)
    n = len(hist_qc)
    raw_dsnwd_pos = hist["snow_depth_in"].astype(float).diff().clip(lower=0).sum()
    target_total = snow["snowfall_in"].sum()

    summary = {
        "triplet": station["triplet"],
        "name": station.get("name"),
        "n_days": n,
        "spikes_removed": int((flags & targets.QC_SPIKE_REMOVED).astype(bool).sum()),
        "stuck_days": int((flags & targets.QC_STUCK).astype(bool).sum()),
        "zeroed_uncorroborated": int((snow["quality"] & targets.QC_UNCORROBORATED).astype(bool).sum()),
        "method_mix": snow["method"].value_counts().to_dict(),
        "raw_dsnwd_pos_total_in": round(float(raw_dsnwd_pos), 1),
        "target_total_in": round(float(target_total), 1),
        "target_vs_raw_ratio": round(float(target_total / raw_dsnwd_pos), 3) if raw_dsnwd_pos > 0 else None,
        "median_slr": statics["median_slr"],
        "snow_class": statics["snow_class"],
    }

    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
        dts = pd.to_datetime(hist_qc["date"])
        ax1.plot(dts, hist["snow_depth_in"], color="#bbb", lw=0.8, label="raw SNWD")
        ax1.plot(dts, hist_qc["snwd_qc"], color="#1976d2", lw=1.0, label="QC'd SNWD")
        spikes = (flags & targets.QC_SPIKE_REMOVED).astype(bool)
        if spikes.any():
            ax1.scatter(dts[spikes], hist["snow_depth_in"][spikes], color="red", s=12,
                        zorder=5, label=f"spikes ({int(spikes.sum())})")
        ax1.set_ylabel("depth (in)")
        ax1.legend(loc="upper left")
        ax1.set_title(f"{station['triplet']} {station.get('name', '')}")

        sdts = pd.to_datetime(snow["date"])
        colors = snow["method"].map({"dsnwd": "#1976d2", "dswe_slr": "#ff9800",
                                     "zeroed": "#d32f2f", "none": "#999"})
        ax2.bar(sdts, snow["snowfall_in"].fillna(0), color=colors, width=1.0)
        ax2.set_ylabel("24h snowfall (in)")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=110)
        plt.close(fig)
        summary["plot"] = plot_path
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplet", help="single station triplet, e.g. 480:MT:SNTL")
    ap.add_argument("--sample", type=int, default=0, help="inspect N evenly-spaced stations")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--plot", help="write a PNG for --triplet mode")
    args = ap.parse_args()

    stations = snotel.list_active_stations()
    if args.triplet:
        sel = [s for s in stations if s["triplet"] == args.triplet]
        if not sel:
            sel = [{"triplet": args.triplet, "name": "", "lat": None, "lon": None,
                    "elevation_ft": None, "id": args.triplet.split(":")[0]}]
    elif args.sample:
        step = max(1, len(stations) // args.sample)
        sel = stations[::step][: args.sample]
    else:
        ap.error("pass --triplet or --sample N")
        return

    ratios = []
    for st in sel:
        s = inspect_one(st, args.days, plot_path=args.plot if args.triplet else None)
        print(pd.Series(s).to_string(), "\n" + "-" * 60)
        if s.get("target_vs_raw_ratio"):
            ratios.append(s["target_vs_raw_ratio"])
    if len(ratios) > 1:
        print(f"\ntarget/raw ratio across {len(ratios)} stations: "
              f"median={np.median(ratios):.3f} min={min(ratios):.3f} max={max(ratios):.3f}")
        print("expected: slightly below 1.0 (spikes/uncorroborated jumps removed); "
              "well below 0.9 or above 1.05 deserves a look")


if __name__ == "__main__":
    main()
