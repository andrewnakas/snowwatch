#!/usr/bin/env python3
"""One-shot SNOTEL station catalogue refresh.

Writes data/stations.json with:
  { "stations": [ {id, triplet, name, state, lat, lon, elevation_ft, start_date}, ... ] }

Filters to active stations (end_date in the future) and drops any without
lat/lon. Stations that have never reported snow_depth (some are precip-only
PRCP gauges piggybacking on the SNTL network) are kept — the build script
gracefully skips stations whose SNWD history is empty.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import snotel  # noqa: E402

OUT = ROOT / "data" / "stations.json"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sites = snotel.list_active_stations(max_age_days=0)  # force a fresh pull
    sites.sort(key=lambda s: (s.get("state") or "", s.get("name") or ""))
    print(f"writing {len(sites)} active SNOTEL stations -> {OUT}")
    OUT.write_text(json.dumps({"stations": sites}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
