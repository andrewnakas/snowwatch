#!/usr/bin/env python3
"""Train + evaluate the pooled snowfall post-processor (v1.5).

Temporal split: everything before --cutoff trains, everything on/after it
scores. Random splits are forbidden here — adjacent days share storms, so a
random split leaks the test storms into training and flatters every metric.

Scorecard compares, per lead and overall:
  postproc   — the LightGBM point model
  nbm_raw    — NBM snowfall verbatim (the baseline to beat)
  mm_mean    — multi-model consensus mean

with MAE/bias, event-conditional MAE, CSI/POD/FAR/frequency-bias at
operational thresholds, CRPS from the quantile members (+ CRPSS vs a
per-station-month climatology), and paired station-week block bootstraps
on postproc-vs-NBM (the "beats NBM" arbiter).

Modes:
  default   — single temporal cutoff; trains, scores, saves models+metrics.
  --folds   — named temporal folds, evaluation only (no models saved):
                A_core_winter  train < 2025-11-30, test Dec 2025–Feb 2026
                B_spring       train < 2026-02-28, test ≥ 2026-02-28
              Fold A is the fold of record: a melt-season-only test window
              (the old default) starves event stats of positives and lets a
              low-bias model look better than it is.

Usage:
    python scripts/train_postprocessor.py [--pairs PATH] [--cutoff YYYY-MM-DD]
    python scripts/train_postprocessor.py --folds
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

from app import calibration, postproc, verification  # noqa: E402

CM_TO_IN = 1.0 / 2.54

# Days carved off the END of the training period for isotonic calibration +
# gate tuning. The final boosters are trained WITHOUT these days (standard
# split-calibration); the tail must never touch the test period.
CAL_TAIL_DAYS = 60

# Temporal folds for --folds mode. Fold A (core winter) is the fold of
# record for every "beats NBM" gate; Fold B preserves continuity with the
# original 2026-02-28 cutoff (test = melt season, kept as the cautionary
# comparison, not as evidence).
FOLDS = {
    # Calibration slice: last CAL_TAIL_DAYS of training (default). The
    # season-matched prior-winter window was tried TWICE and lost twice:
    # at 98 stns × 1.3 winters (MAE 0.714→0.809) AND at 912 stns × 5
    # winters (ΔCSI@1 vs NBM +0.020→−0.007, ΔCSI@6 −0.083→−0.097) — losing
    # the most recent core winter from training always cost more than
    # better-seasoned calibration gained. The tail's real defect (no valid
    # 6/12in gates: coarse candidate grid jumping the FB band at long
    # leads) is fixed in calibration.tune_gate instead (0.01 grid +
    # lead-pooled fallback). Do not re-try season-matched cal windows
    # without new evidence.
    "A_core_winter": {"train_end": "2025-11-30",
                      "test_start": "2025-12-01", "test_end": "2026-02-28"},
    "B_spring": {"train_end": "2026-02-28",
                 "test_start": "2026-02-28", "test_end": None},
}


def _fmt(v, spec="0.3f"):
    return format(v, spec) if v is not None else "n/a"


def fit_calibration(boosters: dict, cal_df: pd.DataFrame) -> dict:
    """Isotonic curves + tuned gates from the calibration tail.

    Curves are fit on the monotone UNcalibrated probabilities that predict()
    emits without a calib payload — the same transform inference applies
    before the curves, so fit and apply see identical inputs.
    """
    raw_preds = postproc.predict(boosters, cal_df, calib=None)
    y = pd.to_numeric(cal_df["obs_snowfall_in"], errors="coerce").to_numpy()
    leads = pd.to_numeric(cal_df["lead_days"], errors="coerce").to_numpy()
    amount = raw_preds["amount"].to_numpy()   # cascade's other event source
    iso: dict = {}
    gates: dict = {}
    for thr in postproc.EVENT_THRESHOLDS:
        col = f"p{thr:g}"
        if col not in raw_preds.columns:
            continue
        key = f"{thr:g}"
        p_raw = raw_preds[col].to_numpy()
        curve = calibration.fit_isotonic(p_raw, y >= thr)
        iso[key] = curve
        p_cal = calibration.apply_isotonic(p_raw, curve)
        gates[key] = calibration.tune_gate(p_cal, y, leads, threshold_in=thr,
                                           amount_pred=amount)
    return {"iso": iso, "gates": gates,
            "cal_rows": int(len(cal_df)),
            "cal_start": str(cal_df["valid_date"].min()),
            "cal_end": str(cal_df["valid_date"].max())}


def evaluate_split(train_df: pd.DataFrame, test_df: pd.DataFrame, *,
                   label: str, n_boot: int = 1000,
                   cal_window: tuple[str, str] | None = None,
                   feature_groups: tuple[str, ...] | None = None) -> tuple[dict, dict, dict | None]:
    """Train on train_df minus a held-out calibration tail, fit isotonic +
    FB-target gates on the tail, score on test_df. Returns (metrics, boosters,
    calib). cal_window overrides the default tail slice when given."""
    print(f"\n=== {label}: train {len(train_df)} rows "
          f"(<= {train_df['valid_date'].max()}) / test {len(test_df)} rows "
          f"({test_df['valid_date'].min()} .. {test_df['valid_date'].max()}) ===")
    if cal_window:
        cal_start, cal_end = cal_window
        in_cal = (train_df["valid_date"] >= cal_start) & (train_df["valid_date"] <= cal_end)
    else:
        # Last CAL_TAIL_DAYS of training, held out of the booster fit. The
        # slice's own base rate no longer drives realized FB because gates are
        # tuned to FB≈1 on calibrated probs (calibration.tune_gate), which
        # transfers across base rates — so the simplest out-of-sample slice is
        # the right one. (Season-matched slices were tried and lost: in-sample
        # inflated probs; out-of-sample-but-higher-base-rate under-fired.)
        cutoff = (pd.to_datetime(train_df["valid_date"]).max()
                  - pd.Timedelta(days=CAL_TAIL_DAYS)).strftime("%Y-%m-%d")
        in_cal = train_df["valid_date"] >= cutoff
    inner_df = train_df[~in_cal]
    cal_start = str(train_df.loc[in_cal, "valid_date"].min()) if in_cal.any() else None
    cal_df = train_df[in_cal]
    if inner_df.empty or len(cal_df) < 500:
        inner_df, cal_df = train_df, train_df.iloc[0:0]
    boosters = postproc.train(inner_df, feature_groups=feature_groups)
    calib = None
    if not cal_df.empty and any(k.startswith("exc_") for k in boosters):
        calib = fit_calibration(boosters, cal_df)
        for thr_key, by_bucket in calib["gates"].items():
            desc = " ".join(
                f"{b}:{(g['gate'] if g['gate'] is not None else '-')}"
                for b, g in by_bucket.items())
            print(f"  gates @{thr_key}in  {desc}")
    preds = postproc.predict(boosters, test_df, calib=calib)

    ev = test_df.copy()
    ev["postproc"] = preds["point"]               # Tweedie+hurdle (production)
    if "point_l1" in preds.columns:
        ev["postproc_l1"] = preds["point_l1"]     # diagnostic MAE-optimal head
    ev["nbm_raw"] = pd.to_numeric(ev.get("nbm_snowfall_cm"), errors="coerce") * CM_TO_IN
    ev["mm_mean"] = pd.to_numeric(ev.get("mm_snow_mean_cm"), errors="coerce") * CM_TO_IN

    obs = ev["obs_snowfall_in"]
    event_rate_1in = float((pd.to_numeric(obs, errors="coerce") >= 1.0).mean())
    metrics: dict = {
        "label": label,
        "n_train": len(train_df), "n_test": len(test_df),
        "n_inner_train": len(inner_df), "n_cal": len(cal_df),
        "cal_start": cal_start if not cal_df.empty else None,
        "n_stations": int(pd.concat([train_df, test_df])["triplet"].nunique()),
        "test_event_rate_1in": event_rate_1in,
        "sources": {},
    }
    sources = ["postproc"] + (["postproc_l1"] if "postproc_l1" in ev.columns else []) \
        + ["nbm_raw", "mm_mean"]
    for src in sources:
        sub = ev[ev[src].notna()]
        m = verification.summarize_deterministic(sub, obs_col="obs_snowfall_in", pred_col=src)
        m["by_lead"] = {int(lead): verification.mae_bias(g["obs_snowfall_in"], g[src])
                        for lead, g in sub.groupby("lead_days")}
        m["by_month"] = {}
        for mon, g in sub.groupby(pd.to_datetime(sub["valid_date"]).dt.strftime("%Y-%m")):
            mm = verification.mae_bias(g["obs_snowfall_in"], g[src])
            mm.update({k: v for k, v in verification.csi_pod_far(
                g["obs_snowfall_in"], g[src], threshold_in=1.0).items()
                if k in ("csi", "pod", "freq_bias")})
            m["by_month"][mon] = mm
        metrics["sources"][src] = m
        if m["n"] == 0:
            print(f"{src:>12}: no rows (model not in backfill coverage yet)")
            continue
        print(f"{src:>12}: n={m['n']} mae={_fmt(m['mae'])} bias={_fmt(m['bias'], '+.3f')} "
              f"event_mae={_fmt(m['event_mae'])} "
              f"csi_1in={_fmt(m['csi_1in'])} pod_1in={_fmt(m['pod_1in'])} "
              f"fb_1in={_fmt(m['fb_1in'], '0.2f')} "
              f"csi_6in={_fmt(m['csi_6in'])} pod_6in={_fmt(m['pod_6in'])}")

    # Probabilistic block: CRPS from the quantile members + CRPSS vs a
    # per-(station, month) climatology built from the training period only.
    qcols = sorted((c for c in preds.columns if c.startswith("q")), key=lambda c: int(c[1:]))
    q_levels = [int(c[1:]) / 100 for c in qcols]
    crps = verification.crps_from_quantiles(
        ev["obs_snowfall_in"].to_numpy(), preds[qcols].to_numpy(), q_levels)
    metrics["crps_postproc"] = crps
    clim_q = verification.climatology_quantiles(train_df, ev, q_levels=q_levels)
    crps_clim = verification.crps_from_quantiles(
        ev["obs_snowfall_in"].to_numpy(), clim_q, q_levels)
    metrics["crps_climatology"] = crps_clim
    metrics["crpss_vs_climatology"] = verification.crpss(crps, crps_clim)
    print(f"postproc CRPS={_fmt(crps)}  clim CRPS={_fmt(crps_clim)}  "
          f"CRPSS={_fmt(metrics['crpss_vs_climatology'])}")

    # Probability heads: Brier/BSS + reliability per threshold (W4 needs all
    # of them, not just 1in). Falls back to the legacy psnow column on
    # pre-v1.6 models where p2/p6/p12 don't exist.
    prob_cols = [(thr, f"p{thr:g}") for thr in postproc.EVENT_THRESHOLDS
                 if f"p{thr:g}" in preds.columns]
    if not prob_cols and "psnow" in preds.columns:
        prob_cols = [(postproc.EVENT_THRESHOLD_IN, "psnow")]
    for thr, col in prob_cols:
        bs = verification.brier_skill(ev["obs_snowfall_in"], preds[col],
                                      threshold_in=thr)
        rel = verification.reliability_curve(ev["obs_snowfall_in"], preds[col],
                                             threshold_in=thr)
        metrics[f"prob_{thr:g}in"] = {**bs, "reliability": rel}
        print(f"P(>= {thr:g}in): Brier={_fmt(bs['brier'], '0.4f')} "
              f"BSS={_fmt(bs['bss'])}  reliability(pred->obs): "
              + " ".join(f"{r['pred']}->{r['obs']}(n{r['n']})" for r in rel))
    if prob_cols and prob_cols[0][1] in ("psnow", "p1"):
        metrics["psnow"] = metrics[f"prob_{prob_cols[0][0]:g}in"]  # legacy key

    # nbm_tuned — the fair-fight baseline: NBM's own amounts with per-
    # threshold, per-lead-bucket decision thresholds CSI-optimized on the
    # SAME calibration window the postproc gates got. Beating raw NBM's
    # fixed thresholds alone would be a straw man. Event stats only: its
    # amounts (and hence MAE) are nbm_raw's.
    if not cal_df.empty and "nbm_snowfall_cm" in cal_df.columns:
        nbm_cal = pd.to_numeric(cal_df["nbm_snowfall_cm"], errors="coerce") * CM_TO_IN
        cal_ok = nbm_cal.notna()
        nbm_test = pd.to_numeric(ev.get("nbm_snowfall_cm"), errors="coerce") * CM_TO_IN
        test_ok = nbm_test.notna()
        if cal_ok.sum() > 500 and test_ok.sum() > 0:
            tuned: dict = {}
            test_leads = pd.to_numeric(ev.loc[test_ok, "lead_days"]).astype(int)
            buckets_test = np.array([calibration.lead_bucket(ld) for ld in test_leads])
            for thr in postproc.EVENT_THRESHOLDS:
                grid = np.round(np.arange(0.2, 2.01, 0.1) * thr, 2)
                g_by_bucket = calibration.tune_gate(
                    nbm_cal[cal_ok].to_numpy(),
                    pd.to_numeric(cal_df.loc[cal_ok, "obs_snowfall_in"]).to_numpy(),
                    pd.to_numeric(cal_df.loc[cal_ok, "lead_days"]).astype(int).to_numpy(),
                    threshold_in=thr, candidates=grid)
                pred = np.zeros(int(test_ok.sum()))
                for b, info in g_by_bucket.items():
                    g = (info or {}).get("gate")
                    if g is None:
                        continue
                    hit = (buckets_test == b) & (nbm_test[test_ok].to_numpy() >= g)
                    pred[hit] = thr
                c = verification.csi_pod_far(
                    ev.loc[test_ok, "obs_snowfall_in"], pred, threshold_in=thr)
                tuned[f"csi_{thr:g}in"] = c["csi"]
                tuned[f"pod_{thr:g}in"] = c["pod"]
                tuned[f"far_{thr:g}in"] = c["far"]
                tuned[f"fb_{thr:g}in"] = c["freq_bias"]
            tuned["n"] = int(test_ok.sum())
            tuned["note"] = "event stats only; amounts (MAE) are nbm_raw's"
            metrics["sources"]["nbm_tuned"] = tuned
            print(f"   nbm_tuned: csi_1in={_fmt(tuned.get('csi_1in'))} "
                  f"csi_6in={_fmt(tuned.get('csi_6in'))} "
                  f"csi_12in={_fmt(tuned.get('csi_12in'))} "
                  f"fb_6in={_fmt(tuned.get('fb_6in'), '0.2f')}")

    # Head-to-head vs NBM on shared rows, blocked by station-week: MAE delta
    # (paired bootstrap) plus CSI deltas at 1/6 in via the generic block
    # bootstrap — the event-detection claims need CIs too.
    head = ev[ev["postproc"].notna() & ev["nbm_raw"].notna()].copy()
    if not head.empty:
        head["err_pp"] = head["postproc"] - head["obs_snowfall_in"]
        head["err_nbm"] = head["nbm_raw"] - head["obs_snowfall_in"]
        bb = verification.paired_block_bootstrap(
            head, err_a="err_pp", err_b="err_nbm", n_boot=n_boot)
        metrics["vs_nbm"] = bb
        if bb["diff"] is not None:
            print(f"postproc vs NBM: ΔMAE={bb['diff']:+.3f} in "
                  f"[{bb['ci_lo']:+.3f}, {bb['ci_hi']:+.3f}] "
                  f"P(better)={bb['p_a_better']:.2f} over {bb['n_blocks']} station-weeks")
        for thr in (1.0, 6.0):
            def csi_delta(d, thr=thr):
                a = verification.csi_pod_far(d["obs_snowfall_in"], d["postproc"],
                                             threshold_in=thr)["csi"]
                b = verification.csi_pod_far(d["obs_snowfall_in"], d["nbm_raw"],
                                             threshold_in=thr)["csi"]
                return np.nan if a is None or b is None else a - b
            bs = verification.block_bootstrap_stat(
                head, csi_delta, n_boot=min(n_boot, 500))
            metrics[f"vs_nbm_csi_{thr:g}in"] = bs
            if bs["stat"] is not None and bs["ci_lo"] is not None:
                print(f"postproc vs NBM ΔCSI@{thr:g}in: {bs['stat']:+.3f} "
                      f"[{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}] "
                      f"P(better)={bs['p_gt_0']:.2f}")

    return metrics, boosters, calib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=ROOT / "data" / "training" / "pairs.csv.gz")
    ap.add_argument("--cutoff", default=None,
                    help="test-period start date (default: last 20%% of dates)")
    ap.add_argument("--folds", action="store_true",
                    help="evaluate the named temporal folds (no models saved)")
    ap.add_argument("--ablate", action="store_true",
                    help="fold-A ablation over Phase-3 feature groups "
                         "(base / +ens / +phase / +hrrr_native / all)")
    ap.add_argument("--feature-groups", default=None,
                    help="comma-separated Phase-3 groups to include "
                         "(default: all; 'none' = base only). Production "
                         "saves should name only groups whose LIVE source "
                         "exists and passed check_feature_skew")
    ap.add_argument("--out", type=Path, default=postproc.MODELS_DIR)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs, compression="gzip")
    pairs = postproc.usable_rows(pairs)
    if pairs.empty:
        print("no usable pairs — build training data first")
        return 1
    print(f"{len(pairs)} usable pairs, {pairs['triplet'].nunique()} stations")

    groups: tuple[str, ...] | None = None
    if args.feature_groups is not None:
        groups = (tuple() if args.feature_groups == "none"
                  else tuple(g.strip() for g in args.feature_groups.split(",")))

    if args.ablate:
        f = FOLDS["A_core_winter"]
        train_df = pairs[pairs["valid_date"] < f["train_end"]]
        test_df = pairs[(pairs["valid_date"] >= f["test_start"])
                        & (pairs["valid_date"] <= f["test_end"])]
        out_path = args.out / "metrics_ablation.json"
        args.out.mkdir(parents=True, exist_ok=True)
        # Resume-safe: reload any variants already computed (machine crashes
        # mid-ablation have been frequent this session) and skip them.
        results = {}
        if out_path.exists():
            try:
                results = json.loads(out_path.read_text())
                if results:
                    print(f"resuming ablation; already have: {sorted(results)}")
            except json.JSONDecodeError:
                results = {}
        variants: list[tuple[str, tuple[str, ...] | None]] = [
            ("base", tuple()),
            *[(f"+{g}", (g,)) for g in postproc.FEATURE_GROUPS],
            ("all", None),
        ]
        for name, gset in variants:
            if name in results:
                continue
            m, _, _ = evaluate_split(train_df, test_df,
                                     label=f"ablate:{name}", n_boot=300,
                                     feature_groups=gset)
            results[name] = {k: m["sources"]["postproc"].get(k) for k in
                             ("mae", "bias", "csi_1in", "csi_2in", "csi_6in",
                              "csi_12in", "pod_6in", "fb_6in")}
            results[name]["crps"] = m.get("crps_postproc")
            # Checkpoint after each variant so a crash loses at most one.
            out_path.write_text(json.dumps(results, indent=2, default=float))
            print(f"  checkpointed {name} -> {out_path}")
        print(f"\nwrote ablation table -> {out_path}")
        return 0

    if args.folds:
        results = {}
        for name, f in FOLDS.items():
            train_df = pairs[pairs["valid_date"] < f["train_end"]]
            test_df = pairs[pairs["valid_date"] >= f["test_start"]]
            if f["test_end"]:
                test_df = test_df[test_df["valid_date"] <= f["test_end"]]
            if train_df.empty or test_df.empty:
                print(f"fold {name}: degenerate split, skipped")
                continue
            cal_w = (f["cal_start"], f["cal_end"]) \
                if f.get("cal_start") and f.get("cal_end") else None
            m, _, _ = evaluate_split(train_df, test_df, label=name,
                                     cal_window=cal_w, feature_groups=groups)
            m["fold"] = f
            results[name] = m
        out_path = args.out / "metrics_folds.json"
        args.out.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"folds": results}, indent=2, default=float))
        print(f"\nwrote fold metrics -> {out_path}")
        return 0

    dates = np.sort(pairs["valid_date"].unique())
    cutoff = args.cutoff or str(dates[int(len(dates) * 0.8)])
    train_df = pairs[pairs["valid_date"] < cutoff]
    test_df = pairs[pairs["valid_date"] >= cutoff]
    if train_df.empty or test_df.empty:
        print("degenerate split — adjust --cutoff")
        return 1
    metrics, boosters, calib = evaluate_split(train_df, test_df,
                                              label=f"cutoff_{cutoff}",
                                              feature_groups=groups)
    metrics["cutoff"] = cutoff

    if not args.no_save:
        meta = {"features": postproc.feature_list(groups),
                "feature_groups": (sorted(groups) if groups is not None else "all"),
                "quantiles": list(postproc.QUANTILES),
                "event_thresholds": list(postproc.EVENT_THRESHOLDS),
                "trained_through": cutoff, "metrics": metrics}
        postproc.save(boosters, models_dir=args.out, meta=meta)
        if calib is not None:
            postproc.save_calib(calib, models_dir=args.out)
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        print(f"saved models + metrics -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
