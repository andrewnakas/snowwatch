#!/usr/bin/env python3
"""Release a trained post-processor to production — the gated flip.

One command, four stages, stops at the first red light:

  1. TRAIN  — production save on all data (train_postprocessor.py --cutoff,
              named --feature-groups only; groups without a skew-cleared
              live source must not ship).
  2. GATES  — read the fold-of-record metrics (metrics_folds.json must be
              from the same code/data state) and check the pre-registered
              interim gates: MAE ratio, ΔCSI@1 CI, FB bands.
  3. SKEW   — scripts/check_feature_skew.py must pass (or --skip-skew with
              a reason, recorded in the release notes).
  4. SHIP   — tar data/models -> models.tar.gz, upload to the training-data
              Release. The pages build restores it and (with
              SW_ENABLE_POSTPROC=1 in the workflow env) the member goes live
              at the next 6h build.

Usage:
    python scripts/release_model.py --feature-groups none --cutoff 2026-06-15
    python scripts/release_model.py --dry-run ...   # everything except upload
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "data" / "models"
RELEASE_TAG = "training-data"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, **kw)


def check_gates() -> tuple[bool, list[str]]:
    """Pre-registered interim gates on the fold of record."""
    p = MODELS_DIR / "metrics_folds.json"
    if not p.exists():
        return False, ["metrics_folds.json missing — run --folds first"]
    fold = json.loads(p.read_text())["folds"].get("A_core_winter")
    if not fold:
        return False, ["fold A_core_winter missing from metrics"]
    msgs, ok = [], True

    bb = fold.get("vs_nbm") or {}
    if not (bb.get("ci_hi") is not None and bb["ci_hi"] < 0):
        ok = False
        msgs.append(f"W1 FAIL: ΔMAE CI does not exclude 0 ({bb})")
    else:
        msgs.append(f"W1 ok: ΔMAE {bb['diff']:+.3f} [{bb['ci_lo']:+.3f},{bb['ci_hi']:+.3f}]")

    c1 = fold.get("vs_nbm_csi_1in") or {}
    if not (c1.get("ci_lo") is not None and c1["ci_lo"] > -0.005):
        ok = False
        msgs.append(f"CSI@1 FAIL: ΔCSI CI materially negative ({c1})")
    else:
        msgs.append(f"CSI@1 ok: {c1['stat']:+.3f} [{c1['ci_lo']:+.3f},{c1['ci_hi']:+.3f}]")

    pp = (fold.get("sources") or {}).get("postproc") or {}
    for thr, band_hi in (("1", 1.55), ("2", 1.55), ("6", 1.6)):
        fb = pp.get(f"fb_{thr}in")
        if fb is None or not (0.7 <= fb <= band_hi):
            ok = False
            msgs.append(f"FB@{thr} FAIL: realized {fb} outside sanity band")
        else:
            msgs.append(f"FB@{thr} ok: {fb:.2f}")
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-groups", required=True,
                    help="'none' or comma list — only skew-cleared groups")
    ap.add_argument("--cutoff", default=None,
                    help="production training cutoff (default: today-14d "
                         "so the eval slice is non-degenerate)")
    ap.add_argument("--skip-skew", default=None, metavar="REASON",
                    help="skip the skew gate, recording REASON")
    ap.add_argument("--skip-train", action="store_true",
                    help="package the models already in data/models")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.skip_train:
        cutoff = args.cutoff or str(date.today().replace(day=1))
        r = run([sys.executable, "scripts/train_postprocessor.py",
                 "--cutoff", cutoff, "--feature-groups", args.feature_groups])
        if r.returncode != 0:
            print("TRAIN failed")
            return 1

    ok, msgs = check_gates()
    print("\n== Release gates ==")
    for m in msgs:
        print("  " + m)
    if not ok:
        print("GATES RED — not shipping")
        return 1

    if args.skip_skew is None:
        r = run([sys.executable, "scripts/check_feature_skew.py"])
        if r.returncode != 0:
            print("SKEW gate failed — not shipping (use --skip-skew REASON "
                  "only for a documented, understood mismatch)")
            return 1
        skew_note = "check_feature_skew: PASSED"
    else:
        skew_note = f"skew gate SKIPPED: {args.skip_skew}"

    notes = {
        "released": date.today().isoformat(),
        "feature_groups": args.feature_groups,
        "gates": msgs,
        "skew": skew_note,
        "meta": json.loads((MODELS_DIR / "postproc_meta.json").read_text())
        if (MODELS_DIR / "postproc_meta.json").exists() else None,
    }
    (MODELS_DIR / "release_notes.json").write_text(json.dumps(notes, indent=2))

    run(["tar", "-czf", "models.tar.gz", "data/models"], check=True)
    size_mb = (ROOT / "models.tar.gz").stat().st_size / 1e6
    print(f"models.tar.gz: {size_mb:.1f} MB")
    if args.dry_run:
        print("DRY RUN — not uploading")
        return 0
    r = run(["gh", "release", "upload", RELEASE_TAG, "models.tar.gz", "--clobber"])
    if r.returncode != 0:
        print("upload failed")
        return 1
    print("SHIPPED — the next 6h pages build restores these models; "
          "postproc member activates if SW_ENABLE_POSTPROC=1 is set in pages.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
