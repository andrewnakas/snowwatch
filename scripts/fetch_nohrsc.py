#!/usr/bin/env python3
"""Fetch NOHRSC National Snowfall Analysis observer point reports — the
second verification truth (T2 in benchmarks/published.json).

Source: https://www.nohrsc.noaa.gov/nsa/reports.html?var=snowfall — daily
observer-measured 24h snowfall (COOP/CoCoRaHS/spotters, the same networks
behind the gridded NSA), available for any historical date. URL pattern and
table layout reused from the tree60weather production scraper
(TreesixtyFirebase/functions/nohrsc_daily_scraper.py).

Two uses:
  1. Point truth at *observer* stations — verifies SnowWatch beyond SNOTEL
     against human-measured snowfall (the network the published CONUS SLR
     benchmark uses), and the basis for widening SnowBench past 912 stations.
  2. Truth-agreement analysis: NOHRSC observers vs SNOTEL-QC targets at
     co-located points, published on the verify page.

Output: data/truth/nohrsc/<YYYY-MM>.csv.gz with columns
  station_id, valid_date, snowfall_in, duration_h, elev_ft, description
(24h reports only are kept for verification; other durations are stored for
completeness but flagged.)

Usage:
    python scripts/fetch_nohrsc.py --start 2024-11-01 --end 2025-04-30
    python scripts/fetch_nohrsc.py --yesterday          # the weekly CI mode
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "truth" / "nohrsc"
BASE_URL = "https://www.nohrsc.noaa.gov/nsa/reports.html"
HEADERS = {"User-Agent": "snowwatch-verification/1.0 (nakas@tree60weather.com)"}

COLS = ["station_id", "valid_date", "snowfall_in", "duration_h", "elev_ft", "description"]


def fetch_day(d: date, *, retries: int = 3) -> pd.DataFrame:
    """One day's snowfall observer reports (units=e → inches)."""
    from bs4 import BeautifulSoup

    url = (f"{BASE_URL}?var=snowfall&region=National&dy={d.year}&dm={d.month}"
           f"&dd={d.day}&units=e&sort=value&filter=0")
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries:
                return pd.DataFrame(columns=COLS)
            time.sleep(5 * (attempt + 1))
    soup = BeautifulSoup(resp.text, "html.parser")
    tbody = soup.find("tbody")
    if tbody is None:
        return pd.DataFrame(columns=COLS)
    rows = []
    for tr in tbody.find_all("tr"):
        td = [c.get_text(strip=True) for c in tr.find_all("td")]
        if len(td) < 5:
            continue
        try:
            rows.append({
                "station_id": td[0],
                "valid_date": d.isoformat(),
                "snowfall_in": float(td[2]) if td[2] else None,
                "duration_h": float(td[3]) if td[3] else 24.0,
                "elev_ft": float(td[4]) if td[4] else None,
                "description": td[5] if len(td) > 5 else "",
            })
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows, columns=COLS)


def append_month(month_df: pd.DataFrame, ym: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{ym}.csv.gz"
    exists = p.exists()
    old = pd.read_csv(p, compression="gzip") if exists else pd.DataFrame(columns=COLS)
    merged = pd.concat([old, month_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["station_id", "valid_date"], keep="last")
    buf = io.BytesIO()
    merged.to_csv(buf, index=False)
    with gzip.open(p, "wb") as f:
        f.write(buf.getvalue())
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=None)
    ap.add_argument("--end", type=date.fromisoformat, default=None)
    ap.add_argument("--yesterday", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="seconds between requests (be polite to NOHRSC)")
    args = ap.parse_args()

    if args.yesterday:
        start = end = date.today() - timedelta(days=1)
    elif args.start and args.end:
        start, end = args.start, args.end
    else:
        ap.error("--start/--end or --yesterday required")

    d = start
    month_rows: list[pd.DataFrame] = []
    cur_ym = start.strftime("%Y-%m")
    n_total = 0
    t0 = time.time()
    while d <= end:
        ym = d.strftime("%Y-%m")
        if ym != cur_ym:
            if month_rows:
                p = append_month(pd.concat(month_rows, ignore_index=True), cur_ym)
                print(f"{cur_ym}: wrote {sum(len(f) for f in month_rows)} reports -> {p}",
                      flush=True)
            month_rows, cur_ym = [], ym
        day_df = fetch_day(d)
        if not day_df.empty:
            month_rows.append(day_df)
            n_total += len(day_df)
        d += timedelta(days=1)
        time.sleep(args.sleep)
    if month_rows:
        p = append_month(pd.concat(month_rows, ignore_index=True), cur_ym)
        print(f"{cur_ym}: wrote {sum(len(f) for f in month_rows)} reports -> {p}")
    print(f"done: {n_total} reports in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
