#!/usr/bin/env python3
"""Train + evaluate the pooled snowfall post-processor (v1.5).

Temporal split: everything before --cutoff trains, everything on/after it
scores. Random splits are forbidden here — adjacent days share storms, so a
random split leaks the test storms into training and flatters every metric.

Scorecard compares, per lead and overall:
  postproc   — the LightGBM point model
  nbm_raw    — NBM snowfall verbatim (the baseline to beat)
  mm_mean    — multi-model consensus mean

with MAE/bias, event-conditional MAE, CSI at operational thresholds, CRPS
from the quantile members, and a paired station-week block bootstrap on
postproc-vs-NBM (the "beats NBM" arbiter).

Usage:
    python scripts/train_postprocessor.py [--pairs PATH] [--cutoff YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import postproc, verification  # noqa: E402

CM_TO_IN = 1.0 / 2.54


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=ROOT / "data" / "training" / "pairs.csv.gz")
    ap.add_argument("--cutoff", default=None,
                    help="test-period start date (default: last 20%% of dates)")
    ap.add_argument("--out", type=Path, default=postproc.MODELS_DIR)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs, compression="gzip")
    pairs = postproc.usable_rows(pairs)
    if pairs.empty:
        print("no usable pairs — build training data first")
        return 1

    dates = np.sort(pairs["valid_date"].unique())
    cutoff = args.cutoff or str(dates[int(len(dates) * 0.8)])
    train_df = pairs[pairs["valid_date"] < cutoff]
    test_df = pairs[pairs["valid_date"] >= cutoff]
    print(f"{len(pairs)} usable pairs, {pairs['triplet'].nunique()} stations | "
          f"train {len(train_df)} (< {cutoff}) / test {len(test_df)}")
    if train_df.empty or test_df.empty:
        print("degenerate split — adjust --cutoff")
        return 1

    boosters = postproc.train(train_df)
    preds = postproc.predict(boosters, test_df)

    ev = test_df.copy()
    ev["postproc"] = preds["point"]
    ev["nbm_raw"] = pd.to_numeric(ev.get("nbm_snowfall_cm"), errors="coerce") * CM_TO_IN
    ev["mm_mean"] = pd.to_numeric(ev.get("mm_snow_mean_cm"), errors="coerce") * CM_TO_IN

    metrics: dict = {"cutoff": cutoff, "n_train": len(train_df), "n_test": len(test_df),
                     "n_stations": int(pairs["triplet"].nunique()), "sources": {}}
    for src in ("postproc", "nbm_raw", "mm_mean"):
        sub = ev[ev[src].notna()]
        m = verification.summarize_deterministic(sub, obs_col="obs_snowfall_in", pred_col=src)
        by_lead = {}
        for lead, g in sub.groupby("lead_days"):
            by_lead[int(lead)] = verification.mae_bias(g["obs_snowfall_in"], g[src])
        m["by_lead"] = by_lead
        metrics["sources"][src] = m
        if m["n"] == 0:
            print(f"{src:>9}: no rows (model not in backfill coverage yet)")
            continue
        fmt = lambda v, spec="0.3f": format(v, spec) if v is not None else "n/a"  # noqa: E731
        print(f"{src:>9}: n={m['n']} mae={fmt(m['mae'])} bias={fmt(m['bias'], '+.3f')} "
              f"event_mae={fmt(m['event_mae'])} csi_1in={fmt(m['csi_1in'])}")

    qcols = sorted((c for c in preds.columns if c.startswith("q")), key=lambda c: int(c[1:]))
    crps = verification.crps_from_quantiles(
        ev["obs_snowfall_in"].to_numpy(), preds[qcols].to_numpy(),
        [int(c[1:]) / 100 for c in qcols])
    metrics["crps_postproc"] = crps
    print(f"postproc CRPS (from {qcols}): {crps:.3f}" if crps is not None else "CRPS: n/a")

    # Head-to-head on rows where both exist, blocked by station-week.
    head = ev[ev["postproc"].notna() & ev["nbm_raw"].notna()].copy()
    if not head.empty:
        head["err_pp"] = head["postproc"] - head["obs_snowfall_in"]
        head["err_nbm"] = head["nbm_raw"] - head["obs_snowfall_in"]
        bb = verification.paired_block_bootstrap(head, err_a="err_pp", err_b="err_nbm")
        metrics["vs_nbm"] = bb
        if bb["diff"] is not None:
            print(f"postproc vs NBM: ΔMAE={bb['diff']:+.3f} in "
                  f"[{bb['ci_lo']:+.3f}, {bb['ci_hi']:+.3f}] "
                  f"P(better)={bb['p_a_better']:.2f} over {bb['n_blocks']} station-weeks")

    if not args.no_save:
        meta = {"features": postproc.FEATURES, "quantiles": list(postproc.QUANTILES),
                "trained_through": cutoff, "metrics": metrics}
        postproc.save(boosters, models_dir=args.out, meta=meta)
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        print(f"saved models + metrics -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
