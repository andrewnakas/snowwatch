#!/usr/bin/env python3
"""Append this build's forecasts to the long-format archive.

Run after build_static_site.py in each shard:
    python scripts/archive_forecasts.py --shard-id N

Reads dist/forecasts/*.json and writes dist/forecast_archive_shard_N.csv.gz
(headerless gzip CSV, schema in app/archive.py). The merge job concatenates
shard files; CI byte-appends the merged file onto the monthly Release asset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import archive  # noqa: E402

DIST = ROOT / "dist"
FORECAST_DIR = DIST / "forecasts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--forecast-dir", type=Path, default=FORECAST_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    build_hour = (time.gmtime().tm_hour // 6) * 6
    out_path = args.out or (DIST / f"forecast_archive_shard_{args.shard_id}.csv.gz")

    rows: list[dict] = []
    n_files = 0
    for f in sorted(args.forecast_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        rows.extend(archive.rows_from_forecast(data, build_hour=build_hour))
        n_files += 1

    if not rows:
        print("no forecast rows to archive")
        return 0
    df = pd.DataFrame(rows)
    archive.write_archive(df, out_path)
    print(f"archived {len(df)} rows from {n_files} forecasts -> {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
