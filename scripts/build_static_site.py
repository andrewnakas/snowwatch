#!/usr/bin/env python3
"""Build dist/ — the GitHub Pages target.

For each SNOTEL station, run the ensemble forecast and write
dist/forecasts/<id>.json. Also emits:
  dist/index.html
  dist/static/*
  dist/stations.json
  dist/index_summary.json   (or index_summary_shard_N.json for sharded builds)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.forecast import forecast_station  # noqa: E402

STATIONS_PATH = Path(os.environ.get("SW_STATIONS_FILE") or (ROOT / "data" / "stations.json"))
SRC_TEMPLATE = ROOT / "app" / "templates" / "index.html"
SRC_STATIC = ROOT / "app" / "static"
DIST = ROOT / "dist"
FORECAST_DIR = DIST / "forecasts"


def _clean_dist() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    FORECAST_DIR.mkdir(parents=True)
    (DIST / "static").mkdir(parents=True)


def _copy_assets() -> None:
    for item in SRC_STATIC.iterdir():
        if item.name == "app.js":
            continue  # we ship static_app.js below as the static entrypoint
        dst = DIST / "static" / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    static_app = SRC_STATIC / "static_app.js"
    if static_app.exists():
        shutil.copy2(static_app, DIST / "static" / "app.js")
    html = SRC_TEMPLATE.read_text()
    html = html.replace('href="/static/', 'href="static/').replace('src="/static/', 'src="static/')
    (DIST / "index.html").write_text(html)


def _to_jsonable(payload):
    import math

    def _scrub(v):
        if isinstance(v, float):
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(v, dict):
            return {k: _scrub(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_scrub(x) for x in v]
        return v

    return _scrub(json.loads(json.dumps(payload, default=str)))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--total-shards", type=int, default=1)
    ap.add_argument("--no-forecasts", action="store_true")
    args = ap.parse_args()

    sharded = args.total_shards > 1
    is_first = args.shard_id == 0

    if not sharded or is_first:
        _clean_dist()
        _copy_assets()
    else:
        DIST.mkdir(parents=True, exist_ok=True)
        FORECAST_DIR.mkdir(parents=True, exist_ok=True)

    payload = json.loads(STATIONS_PATH.read_text())
    all_stations = payload["stations"]
    if args.limit:
        all_stations = all_stations[: args.limit]
    if not sharded or is_first:
        (DIST / "stations.json").write_text(json.dumps({"stations": all_stations}, indent=2))

    stations = all_stations[args.shard_id :: args.total_shards] if sharded else all_stations
    if args.no_forecasts:
        print("--no-forecasts: skipping live runs")
        return 0

    successes = 0
    failures: list[dict] = []
    member_rolling: dict[str, list[float]] = {}
    blend_rolling: list[float] = []
    t0 = time.time()

    for i, st in enumerate(stations, 1):
        sid = st["id"]
        ts = time.time()
        try:
            fc = forecast_station(st, horizon=args.horizon)
            data = asdict(fc)
            data["station"] = st
            (FORECAST_DIR / f"{sid}.json").write_text(json.dumps(_to_jsonable(data), indent=2))
            successes += 1
            for name, mae in fc.rolling_mae.items():
                if mae is not None and np.isfinite(mae):
                    member_rolling.setdefault(name, []).append(float(mae))
            bm = fc.rolling_mae.get("ensemble_blend")
            if bm is not None and np.isfinite(bm):
                blend_rolling.append(float(bm))
            tag = f"shard {args.shard_id}/{args.total_shards}: " if sharded else ""
            print(f"{tag}[{i:>4}/{len(stations)}] {sid} {st.get('name', '')[:30]:30s} OK  {time.time()-ts:5.1f}s")
        except Exception as exc:
            print(f"[{i:>4}/{len(stations)}] {sid} FAIL  {exc}")
            failures.append({"id": sid, "error": str(exc)})

    summary = {
        "shard_id": args.shard_id,
        "total_shards": args.total_shards,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stations_in_shard": len(stations),
        "stations_succeeded": successes,
        "stations_failed": [f["id"] for f in failures],
        "horizon_days": args.horizon,
        "rolling_mae_mean_by_member": {
            k: (mean(v) if v else None) for k, v in member_rolling.items()
        },
        "rolling_mae_blend_mean": (mean(blend_rolling) if blend_rolling else None),
        "build_seconds": round(time.time() - t0, 1),
        "failures": failures,
    }
    if sharded:
        (DIST / f"index_summary_shard_{args.shard_id}.json").write_text(json.dumps(summary, indent=2))
    else:
        summary["stations_total"] = len(stations)
        (DIST / "index_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nShard {args.shard_id}/{args.total_shards} built in {summary['build_seconds']}s — {successes}/{len(stations)} stations")
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
