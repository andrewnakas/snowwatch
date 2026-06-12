#!/usr/bin/env python3
"""Backfill archived multi-lead forecasts from Open-Meteo's Previous Runs API.

For each (station, model) this pulls hourly `{var}_previous_dayN` series
(N = 1..7), aggregates them to daily totals/means per lead, and appends rows
to a per-station-model gzip CSV under data/prevruns/<model>/<triplet>.csv.gz:

    valid_date,lead_days,snowfall_cm,precip_mm,tmean_c

A sidecar meta JSON records which 90-day chunks are complete (or known-empty)
so the job is resumable and idempotent — rerun nightly until everything is
done. A weighted call budget stops the run cleanly before Open-Meteo's free
tier does.

Verified API facts (June 2026): hourly-only (no daily aggregation server-
side); NBM archive reaches back to ~Dec 2024, GFS/HRRR similar; ECMWF
IFS/AIFS expose temperature+precipitation but not snowfall. Chunks outside a
model's archive come back all-null and are marked "empty" once, never
refetched.

Usage:
    python scripts/backfill_previous_runs.py --shard-id 0 --total-shards 8 \
        --budget 8000 --start 2024-10-01
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
import urllib.error
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREVRUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OUT_ROOT = ROOT / "data" / "prevruns"

LEADS = list(range(1, 8))

# model key → (open-meteo id, hourly vars available, max lead)
MODELS = {
    "nbm": ("ncep_nbm_conus", ("snowfall", "precipitation", "temperature_2m"), 7),
    "hrrr": ("ncep_hrrr_conus", ("snowfall", "precipitation", "temperature_2m"), 2),
    "gfs": ("gfs_global", ("snowfall", "precipitation", "temperature_2m"), 7),
    "ifs": ("ecmwf_ifs025", ("precipitation", "temperature_2m"), 7),
    "aifs": ("ecmwf_aifs025_single", ("precipitation", "temperature_2m"), 7),
}

CHUNK_DAYS = 90
# Skip chunks that fall entirely in Jun–Sep: no snow to learn from and the
# budget is the binding constraint.
SUMMER_MONTHS = {6, 7, 8, 9}

OUT_COLS = ["valid_date", "lead_days", "snowfall_cm", "precip_mm", "tmean_c"]


def _http_json(url: str, timeout: int = 120, *, retries: int = 3) -> dict | None:
    req = Request(url, headers={"User-Agent": "snowwatch-backfill/0.2"})
    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                time.sleep(30 * (attempt + 1))
                continue
            if 400 <= exc.code < 500 and exc.code != 429:
                # Deterministic rejection (e.g. station outside the model
                # domain). Open-Meteo sends {"error":true,"reason":...} — pass
                # it through so the caller marks the chunk "empty" instead of
                # retrying it on every nightly run forever.
                try:
                    return json.loads(exc.read().decode("utf-8"))
                except Exception:
                    return None
            return None
        except Exception:
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            return None
    return None


def _chunks(start: date, end: date) -> list[tuple[date, date]]:
    out = []
    cur = start
    while cur <= end:
        ce = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        if not all(m in SUMMER_MONTHS for m in _months_between(cur, ce)):
            out.append((cur, ce))
        cur = ce + timedelta(days=1)
    return out


def _months_between(a: date, b: date) -> set[int]:
    months = set()
    cur = a.replace(day=1)
    while cur <= b:
        months.add(cur.month)
        cur = (cur + timedelta(days=32)).replace(day=1)
    return months


def _paths(model_key: str, triplet: str) -> tuple[Path, Path]:
    safe = triplet.replace(":", "_")
    d = OUT_ROOT / model_key
    return d / f"{safe}.csv.gz", d / f"{safe}.meta.json"


def _load_meta(meta_p: Path) -> dict:
    if meta_p.exists():
        try:
            return json.loads(meta_p.read_text())
        except Exception:
            pass
    return {"chunks": {}}


def _aggregate(payload: dict, vars_avail: tuple[str, ...], max_lead: int) -> pd.DataFrame:
    """Hourly previous_dayN series → daily rows per (valid_date, lead)."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time")
    if not times:
        return pd.DataFrame(columns=OUT_COLS)
    df = pd.DataFrame({"time": pd.to_datetime(times)})
    df["valid_date"] = df["time"].dt.date
    rows = []
    for lead in LEADS:
        if lead > max_lead:
            continue
        cols = {}
        for v in vars_avail:
            key = f"{v}_previous_day{lead}"
            if key in hourly:
                cols[v] = pd.to_numeric(pd.Series(hourly[key]), errors="coerce")
        if not cols:
            continue
        sub = pd.DataFrame({"valid_date": df["valid_date"], **cols})
        g = sub.groupby("valid_date")
        agg = pd.DataFrame(index=g.size().index)
        if "snowfall" in cols:
            agg["snowfall_cm"] = g["snowfall"].sum(min_count=4)
        if "precipitation" in cols:
            agg["precip_mm"] = g["precipitation"].sum(min_count=4)
        if "temperature_2m" in cols:
            agg["tmean_c"] = g["temperature_2m"].mean()
        agg["lead_days"] = lead
        agg = agg.reset_index()
        rows.append(agg)
    if not rows:
        return pd.DataFrame(columns=OUT_COLS)
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=[c for c in ("snowfall_cm", "precip_mm", "tmean_c") if c in out.columns],
                     how="all")
    out["valid_date"] = out["valid_date"].astype(str)
    return out.reindex(columns=OUT_COLS)


def _append_rows(csv_p: Path, df: pd.DataFrame) -> None:
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.to_csv(buf, header=False, index=False)
    with gzip.open(csv_p, "at") as f:
        f.write(buf.getvalue())


def read_station_model(model_key: str, triplet: str) -> pd.DataFrame:
    """Read one station-model series (used by build_training_data.py)."""
    csv_p, _ = _paths(model_key, triplet)
    if not csv_p.exists():
        return pd.DataFrame(columns=OUT_COLS)
    df = pd.read_csv(csv_p, names=OUT_COLS, compression="gzip")
    for c in ("snowfall_cm", "precip_mm", "tmean_c"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["lead_days"] = pd.to_numeric(df["lead_days"], errors="coerce").astype("Int64")
    # Chunks can be refetched (partial tail) — keep the latest occurrence.
    df = df.drop_duplicates(subset=["valid_date", "lead_days"], keep="last")
    return df


def backfill_station_model(
    triplet: str, lat: float, lon: float, model_key: str,
    start: date, end: date, budget: dict,
) -> tuple[int, int]:
    """Fetch missing chunks for one (station, model). Returns (fetched, skipped)."""
    om_id, vars_avail, max_lead = MODELS[model_key]
    csv_p, meta_p = _paths(model_key, triplet)
    meta = _load_meta(meta_p)
    fetched = skipped = 0
    today_iso = date.today().isoformat()

    for cs, ce in _chunks(start, end):
        key = f"{cs.isoformat()}_{ce.isoformat()}"
        state = meta["chunks"].get(key)
        # A chunk fetched while still in progress (covers "today") must be
        # refetched once it's fully in the past.
        if state in ("ok", "empty") and not (state == "ok" and meta.get("partial") == key):
            continue
        n_vars = sum(1 for v in vars_avail) * min(max_lead, 7)
        cost = max(1.0, n_vars * (ce - cs).days / 100.0)
        if budget["spent"] + cost > budget["total"]:
            return fetched, skipped
        if budget.get("deadline") and time.time() > budget["deadline"]:
            return fetched, skipped
        hourly_vars = [f"{v}_previous_day{lead}" for v in vars_avail
                       for lead in LEADS if lead <= max_lead]
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "hourly": ",".join(hourly_vars),
            "start_date": cs.isoformat(),
            "end_date": min(ce, date.today()).isoformat(),
            "timezone": "UTC",
            "models": om_id,
        }
        payload = _http_json(PREVRUNS_URL + "?" + urlencode(params))
        budget["spent"] += cost
        if payload is None:
            # Network/429 — retryable on the next nightly run.
            skipped += 1
            continue
        if payload.get("error"):
            # Deterministic API error (e.g. station outside the model domain):
            # mark empty so the chunk is never refetched.
            meta["chunks"][key] = "empty"
            meta_p.parent.mkdir(parents=True, exist_ok=True)
            meta_p.write_text(json.dumps(meta, separators=(",", ":")))
            skipped += 1
            continue
        rows = _aggregate(payload, vars_avail, max_lead)
        if rows.empty:
            meta["chunks"][key] = "empty"
        else:
            _append_rows(csv_p, rows)
            meta["chunks"][key] = "ok"
            if ce.isoformat() >= today_iso:
                meta["partial"] = key
            elif meta.get("partial") == key:
                meta.pop("partial", None)
            fetched += 1
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.write_text(json.dumps(meta, separators=(",", ":")))
    return fetched, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--total-shards", type=int, default=1)
    ap.add_argument("--budget", type=float, default=8000.0,
                    help="weighted Open-Meteo call budget for this run")
    ap.add_argument("--start", type=lambda s: date.fromisoformat(s),
                    default=date(2024, 10, 1))
    ap.add_argument("--end", type=lambda s: date.fromisoformat(s), default=date.today())
    ap.add_argument("--models", default="nbm,hrrr,gfs,ifs,aifs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop cleanly after this wall time (set below the CI "
                         "job timeout — a timed-out job never uploads its work)")
    args = ap.parse_args()

    stations_path = ROOT / "data" / "stations.json"
    stations = json.loads(stations_path.read_text())["stations"]
    # CONUS first: NBM/HRRR don't cover AK, and NBM is the post-processor's
    # primary input, so AK stations are the lowest-value budget spend. The
    # catalogue happens to lead with AK; don't let it eat the early runs.
    stations.sort(key=lambda s: (s["triplet"].split(":")[1] == "AK", s["triplet"]))
    if args.limit:
        stations = stations[: args.limit]
    if args.total_shards > 1:
        stations = stations[args.shard_id :: args.total_shards]
    model_keys = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]

    budget = {"spent": 0.0, "total": args.budget}
    t0 = time.time()
    if args.max_seconds > 0:
        budget["deadline"] = t0 + args.max_seconds
    total_fetched = 0
    for i, st in enumerate(stations, 1):
        for mk in model_keys:
            f, s = backfill_station_model(
                st["triplet"], st["lat"], st["lon"], mk, args.start, args.end, budget)
            total_fetched += f
        if budget.get("deadline") and time.time() > budget["deadline"]:
            print(f"deadline reached after {i}/{len(stations)} stations "
                  f"(budget {budget['spent']:.0f}/{budget['total']:.0f})")
            break
        if budget["spent"] >= budget["total"]:
            print(f"budget exhausted after {i}/{len(stations)} stations "
                  f"({budget['spent']:.0f}/{budget['total']:.0f})")
            break
        if i % 25 == 0:
            print(f"[{i}/{len(stations)}] chunks fetched={total_fetched} "
                  f"budget={budget['spent']:.0f}/{budget['total']:.0f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"done: {total_fetched} chunks fetched, budget {budget['spent']:.0f}/{budget['total']:.0f}, "
          f"{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
