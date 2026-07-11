#!/usr/bin/env python3
"""Out-of-fold (OOF) prediction cache for rare-threshold calibration.

The production calibration slice (last CAL_TAIL_DAYS of train = Oct-Nov)
has ~zero >=12in events, so isotonic + gates at rare thresholds are fit
blind: fit_isotonic returns None (exc_12 runs uncalibrated) and tune_gate
finds no admissible 12in gate outside lead bucket 3-4 — the cascade
physically cannot call 12in events at most leads (fold A POD@12 0.106 vs
NBM 0.333). The prior core winters hold ~25K usable >=12in rows, but they
sit inside the training region where the boosters have memorized them.

Fix: cross-fitting. For each core winter W inside the fold's training
region, train the rare-layer heads on inner-train MINUS W and predict W.
Pooled, that yields leakage-free season-matched predictions with real rare
events to fit isotonic + gates on — WITHOUT removing any winter from the
final model's training set (season-matched cal windows that shrink
training lost twice; see the FOLDS comment in train_postprocessor.py).

Outputs under data/training/oof/<fset>/<fold>/ (parquet, checkpointed —
reruns skip jobs whose outputs all exist):
  oof_<winter>.parquet          OOF-model predictions on winter W
  caltail_oof_<winter>.parquet  same OOF model on the 60d cal tail
  final_caltail.parquet         final model (all winters) on the cal tail
  final_test.parquet            final model on the fold TEST slice — EVAL
                                ONLY, never an input to tuning

The caltail files measure OOF-model vs final-model score shift on a slice
neither trained on: rare isotonic is fit on OOF-model probs but applied to
final-model probs, and that transfer is only safe if the two models score
alike out of sample.

Schema (all files): triplet, valid_date, lead_days, obs_snowfall_in,
nbm_snowfall_cm, amount, p1/p2/p6/p12, q80/q90/q95. The p-columns are the
monotone UNcalibrated probabilities postproc.predict(calib=None) emits —
exactly the inputs fit_calibration sees, so curves fit here apply cleanly
at inference.

Usage:
  nohup .venv/bin/python scripts/build_oof_preds.py \
      --feature-groups ens,phase > oof_build.log 2>&1 &
  # fold C confirmation set (only winters inside ITS train region):
  ... --fold C_prior_winter
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app import postproc  # noqa: E402
from train_postprocessor import CAL_TAIL_DAYS, FOLDS  # noqa: E402

# Only the heads the rare decision layer reads. exc_1 rides along so the
# lab can also test season-matched common-threshold gates; the other 6
# quantiles and point_l1 would double training time for nothing.
HEADS = {"amount", "exc_1", "exc_2", "exc_6", "exc_12", "q80", "q90", "q95"}


def dump_preds(boosters: dict, df: pd.DataFrame, out_path: Path) -> None:
    out = postproc.dump_predictions(boosters, df, out_path)
    ev6 = int((pd.to_numeric(out["obs_snowfall_in"], errors="coerce") >= 6).sum())
    ev12 = int((pd.to_numeric(out["obs_snowfall_in"], errors="coerce") >= 12).sum())
    print(f"  wrote {out_path.name}: {len(out)} rows, >=6in {ev6}, >=12in {ev12}",
          flush=True)


def core_winters(train_df: pd.DataFrame, train_end: str) -> list[tuple[str, str, str]]:
    """(label, start, end) for each complete Dec-Feb winter before train_end.
    ISO date strings compare lexicographically; the -02-29 upper bound is
    inclusive and simply matches nothing in non-leap years."""
    first_year = int(str(train_df["valid_date"].min())[:4])
    out = []
    for y in range(first_year, 2100):
        start, end = f"{y}-12-01", f"{y + 1}-02-29"
        if end >= train_end:
            break
        if ((train_df["valid_date"] >= start) & (train_df["valid_date"] <= end)).any():
            out.append((f"{y}-{(y + 1) % 100:02d}", start, end))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=ROOT / "data" / "training" / "pairs.csv.gz")
    ap.add_argument("--fold", default="A_core_winter",
                    choices=sorted(FOLDS) + ["production"])
    ap.add_argument("--train-end", default=None,
                    help="required with --fold production: the production "
                         "training cutoff. No test slice is dumped — the "
                         "caches exist to tune rare_calib.json for the "
                         "release build (rare_tail_lab.py --recipe-v2)")
    ap.add_argument("--feature-groups", default="ens,phase",
                    help="comma-separated Phase-3 groups ('none' = base only)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "training" / "oof")
    args = ap.parse_args()

    groups = (tuple() if args.feature_groups == "none"
              else tuple(g.strip() for g in args.feature_groups.split(",")))
    fset = "+".join(groups) if groups else "base"
    out_dir = args.out / fset / args.fold
    if args.fold == "production":
        if not args.train_end:
            ap.error("--fold production requires --train-end")
        fold = {"train_end": args.train_end,
                "test_start": args.train_end, "test_end": args.train_end}
    else:
        fold = FOLDS[args.fold]

    # Memory diet: the production job (6.4M-row inner-train) got OOM-killed
    # twice with the full frame. Read only the columns training/dumping
    # reads and hold numerics as float32 (LightGBM bins to float32 anyway);
    # ~halves resident memory.
    wanted = set(postproc.FEATURES + ["doy", "triplet", "valid_date", "lead_days",
                                      "obs_snowfall_in", "nbm_snowfall_cm",
                                      "quality"])
    pairs = pd.read_csv(args.pairs, compression="gzip",
                        usecols=lambda c: c in wanted)
    pairs = postproc.usable_rows(pairs)
    for c in pairs.columns:
        if c in postproc.NUMERIC_FEATURES or c in (
                "doy", "lead_days", "obs_snowfall_in", "nbm_snowfall_cm"):
            pairs[c] = pd.to_numeric(pairs[c], errors="coerce").astype("float32")
    gc.collect()
    print(f"{len(pairs)} usable pairs, {pairs['triplet'].nunique()} stations",
          flush=True)

    # Mirror evaluate_split exactly: train < train_end, last CAL_TAIL_DAYS
    # held out of every booster fit here too, so the tail stays a shared
    # holdout for the OOF-vs-final shift check.
    train_df = pairs[pairs["valid_date"] < fold["train_end"]]
    cutoff = (pd.to_datetime(train_df["valid_date"]).max()
              - pd.Timedelta(days=CAL_TAIL_DAYS)).strftime("%Y-%m-%d")
    cal_df = train_df[train_df["valid_date"] >= cutoff]
    inner_df = train_df[train_df["valid_date"] < cutoff]
    test_df = pairs[pairs["valid_date"] >= fold["test_start"]]
    if fold["test_end"]:
        test_df = test_df[test_df["valid_date"] <= fold["test_end"]]

    winters = core_winters(inner_df, fold["train_end"])
    print(f"fold {args.fold}: inner-train {len(inner_df)} rows (< {cutoff}), "
          f"cal tail {len(cal_df)}, test {len(test_df)}; "
          f"OOF winters: {[w for w, _, _ in winters]}", flush=True)

    jobs: list[tuple[str, pd.DataFrame, list[tuple[pd.DataFrame, Path]]]] = []
    for label, start, end in winters:
        in_w = (inner_df["valid_date"] >= start) & (inner_df["valid_date"] <= end)
        jobs.append((f"oof {label}", inner_df[~in_w], [
            (inner_df[in_w], out_dir / f"oof_{label}.parquet"),
            (cal_df, out_dir / f"caltail_oof_{label}.parquet"),
        ]))
    final_targets = [(cal_df, out_dir / "final_caltail.parquet")]
    if not test_df.empty:
        final_targets.append((test_df, out_dir / "final_test.parquet"))
    jobs.append(("final", inner_df, final_targets))

    for name, fit_df, targets in jobs:
        if all(p.exists() for _, p in targets):
            print(f"[skip] {name}: outputs exist", flush=True)
            continue
        t0 = time.time()
        print(f"[train] {name}: {len(fit_df)} rows, heads={sorted(HEADS)}",
              flush=True)
        boosters = postproc.train(fit_df, feature_groups=groups, heads=HEADS)
        for target_df, path in targets:
            dump_preds(boosters, target_df, path)
        del boosters
        gc.collect()
        print(f"[done] {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"all jobs complete -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
