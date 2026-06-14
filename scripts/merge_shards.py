#!/usr/bin/env python3
"""Merge per-shard dist artifacts into a single dist/ directory for Pages.

Each shard uploads its own dist/ (with forecasts/<id>.json for its slice +
index_summary_shard_N.json). The CI workflow downloads them all to
shard_dists/dist-shard-*/ and runs this script with --input-root shard_dists.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, type=Path)
    ap.add_argument("--min-shards", type=int, default=0,
                    help="abort (don't deploy) if fewer shards delivered — "
                         "guards against publishing a mostly-empty site when "
                         "many runners die at once")
    args = ap.parse_args()

    shard_dirs = sorted(
        p for p in args.input_root.iterdir()
        if p.is_dir() and (p / "forecasts").exists()
    )
    print(f"merging {len(shard_dirs)} shard dirs into {DIST}")
    if args.min_shards and len(shard_dirs) < args.min_shards:
        print(f"ERROR: only {len(shard_dirs)} shards delivered forecasts "
              f"(need >= {args.min_shards}); refusing to deploy a partial site")
        return 1

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    (DIST / "forecasts").mkdir(parents=True)

    # First shard owns the assets (it always emits dist/index.html, stations.json, static/).
    asset_shard = next((d for d in shard_dirs if (d / "index.html").exists()), shard_dirs[0])
    for item in asset_shard.iterdir():
        if item.name == "forecasts" or item.name.startswith("forecast_archive_"):
            continue
        dst = DIST / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    # Merge every shard's forecasts/*.
    total_files = 0
    member_rolling: dict[str, list[float]] = {}
    blend_rolling: list[float] = []
    succeeded = 0
    failed: list[str] = []
    horizons: set[int] = set()
    build_seconds = 0.0
    failures_detail: list = []

    for d in shard_dirs:
        fcst_dir = d / "forecasts"
        if fcst_dir.exists():
            for f in fcst_dir.iterdir():
                shutil.copy2(f, DIST / "forecasts" / f.name)
                total_files += 1
        # Collect per-shard summary.
        sjs = list(d.glob("index_summary_shard_*.json"))
        for s in sjs:
            try:
                obj = json.loads(s.read_text())
            except Exception:
                continue
            succeeded += int(obj.get("stations_succeeded") or 0)
            failed += list(obj.get("stations_failed") or [])
            horizons.add(int(obj.get("horizon_days") or 7))
            build_seconds = max(build_seconds, float(obj.get("build_seconds") or 0))
            failures_detail += list(obj.get("failures") or [])
            for k, v in (obj.get("rolling_mae_mean_by_member") or {}).items():
                if v is not None:
                    member_rolling.setdefault(k, []).append(float(v))
            b = obj.get("rolling_mae_blend_mean")
            if b is not None:
                blend_rolling.append(float(b))

    merged_summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_shards": len(shard_dirs),
        "stations_succeeded": succeeded,
        "stations_total": succeeded + len(failed),
        "stations_failed": failed,
        "horizon_days": next(iter(horizons), 7),
        "rolling_mae_mean_by_member": {
            k: (mean(v) if v else None) for k, v in member_rolling.items()
        },
        "rolling_mae_blend_mean": (mean(blend_rolling) if blend_rolling else None),
        "build_seconds": round(build_seconds, 1),
        "failures": failures_detail,
        "forecast_files": total_files,
    }
    (DIST / "index_summary.json").write_text(json.dumps(merged_summary, indent=2))

    # Concatenate per-shard forecast archives (gzip streams byte-append into a
    # valid multi-stream gzip; see app/archive.py). Written OUTSIDE dist/ so it
    # rides as a workflow artifact, not onto the Pages deployment.
    archive_out = ROOT / "forecast_archive_build.csv.gz"
    n_arch = 0
    with open(archive_out, "wb") as out_f:
        for d in shard_dirs:
            for shard_arch in sorted(d.glob("forecast_archive_shard_*.csv.gz")):
                out_f.write(shard_arch.read_bytes())
                n_arch += 1
    if n_arch:
        print(f"concatenated {n_arch} shard archives -> {archive_out} "
              f"({archive_out.stat().st_size / 1024:.0f} KB)")
    else:
        archive_out.unlink(missing_ok=True)

    print(f"merged: {total_files} forecast JSONs, {succeeded} succeeded across {len(shard_dirs)} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
