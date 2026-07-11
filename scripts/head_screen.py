#!/usr/bin/env python3
"""Screen model-head variants for the 6in ceiling (Stage 3).

The decision layer is maxed at 6in: with test-knowledge oracles, no
in-band rule on the CURRENT heads exceeds shared CSI@6 ~0.214-0.216 vs
NBM 0.207 (nbm_tuned 0.213) — composition (2-quantile AND-rules, OR
gates) moves the ceiling by <=0.002. Winning 6in decisively means
raising the ceiling with better heads.

For each candidate this trains ONLY the affected booster(s) on the
fold-A inner-train split (identical to evaluate_split's), predicts the
fold-A test slice, and reports ceiling diagnostics on shared-NBM rows:
  exc head   — PR-AUC + best in-band (FB in [0.8,1.5]) oracle CSI@6 of
               amount OR score>=cut
  quantiles  — best in-band q-rule oracle CSI@6 per level
These are TEST-KNOWLEDGE CEILINGS for triage only (same category as the
Stage-0.5 oracles): a candidate whose ceiling doesn't clear the current
one is dead regardless of tuning; winners graduate to the full OOF
pipeline (build_oof_preds heads= retrain -> rare_tail_lab) and a
one-shot honest confirmation.

Checkpointed to data/training/oof/head_screen_results.json — rerun
skips finished candidates. Runs nice(15) so a concurrent release build
keeps priority.

Usage:
  nohup nice -n 15 .venv/bin/python scripts/head_screen.py > head_screen.log 2>&1 &
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import lightgbm as lgb  # noqa: E402

from app import postproc, verification  # noqa: E402
from train_postprocessor import CAL_TAIL_DAYS, FOLDS  # noqa: E402

OUT = ROOT / "data" / "training" / "oof" / "head_screen_results.json"
GROUPS = ("ens", "phase")
THR = 6.0


def load_split():
    pairs = pd.read_csv(ROOT / "data" / "training" / "pairs.csv.gz", compression="gzip")
    pairs = postproc.usable_rows(pairs)
    f = FOLDS["A_core_winter"]
    train_df = pairs[pairs["valid_date"] < f["train_end"]]
    cutoff = (pd.to_datetime(train_df["valid_date"]).max()
              - pd.Timedelta(days=CAL_TAIL_DAYS)).strftime("%Y-%m-%d")
    inner = train_df[train_df["valid_date"] < cutoff]
    test = pairs[(pairs["valid_date"] >= f["test_start"])
                 & (pairs["valid_date"] <= f["test_end"])]
    return inner, test


def _dset(df, feats):
    X = postproc.build_features(df, feats)
    y = pd.to_numeric(df["obs_snowfall_in"], errors="coerce")
    return X, y


def train_binary(inner, feats, *, thr=THR, rows_mask=None, num_rounds=600,
                 **param_over):
    X, y = _dset(inner if rows_mask is None else inner[rows_mask], feats)
    lab = (y.to_numpy() >= thr).astype(int)
    pos = lab.sum()
    spw = param_over.pop("scale_pos_weight",
                         min((lab.size - pos) / pos, postproc.SPW_CAP))
    params = dict(postproc.LGB_PARAMS, objective="binary",
                  metric="binary_logloss", scale_pos_weight=spw, **param_over)
    params.pop("alpha", None)
    d = lgb.Dataset(X, label=lab, categorical_feature=postproc.CATEGORICAL_FEATURES,
                    free_raw_data=False)
    return lgb.train(params, d, num_boost_round=num_rounds)


def train_quantiles(inner, feats, levels=(0.8, 0.9, 0.95), *, num_rounds=600,
                    **param_over):
    X, y = _dset(inner, feats)
    d = lgb.Dataset(X, label=y, categorical_feature=postproc.CATEGORICAL_FEATURES,
                    free_raw_data=False)
    out = {}
    for q in levels:
        params = dict(postproc.LGB_PARAMS, alpha=q, **param_over)
        out[f"q{int(q * 100)}"] = lgb.train(params, d, num_boost_round=num_rounds)
    return out


def oracle_exc(score, test, shared, amount, y) -> dict:
    """Best in-band oracle CSI@6 of amount OR score>=cut, shared rows."""
    from sklearn.metrics import average_precision_score
    ev = y >= THR
    ap = float(average_precision_score(ev[shared].astype(int), score[shared]))
    cuts = np.unique(np.quantile(score, np.linspace(0.85, 0.9999, 120)))
    best = {"csi": None}
    for cut in cuts:
        pt = np.where((amount >= THR) | (score >= cut), THR, 0.0)
        s = verification.csi_pod_far(y[shared], pt[shared], threshold_in=THR)
        if s["freq_bias"] and 0.8 <= s["freq_bias"] <= 1.5:
            if best["csi"] is None or (s["csi"] or 0) > best["csi"]:
                best = {"csi": s["csi"], "pod": s["pod"], "fb": s["freq_bias"]}
    return {"pr_auc": ap, "oracle": best}


def oracle_q(qpreds: dict, test, shared, amount, y) -> dict:
    out = {}
    for name, q in qpreds.items():
        best = {"csi": None}
        for cut in np.arange(THR * 0.5, THR * 2.5, THR * 0.05):
            pt = np.where((amount >= THR) | (q >= cut), THR, 0.0)
            s = verification.csi_pod_far(y[shared], pt[shared], threshold_in=THR)
            if s["freq_bias"] and 0.8 <= s["freq_bias"] <= 1.5:
                if best["csi"] is None or (s["csi"] or 0) > best["csi"]:
                    best = {"csi": s["csi"], "cut": float(cut), "fb": s["freq_bias"]}
        out[name] = best
    return out


def main() -> int:
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    inner, test = load_split()
    feats = postproc.feature_list(GROUPS)
    Xt, yt = _dset(test, feats)
    y = yt.to_numpy()
    nbm = pd.to_numeric(test["nbm_snowfall_cm"], errors="coerce").to_numpy() / 2.54
    shared = np.isfinite(nbm)
    print(f"inner {len(inner)}, test {len(test)}, shared {shared.sum()}", flush=True)

    # amount reference comes from the existing fold-A cache (same split)
    cache = pd.read_parquet(ROOT / "data" / "training" / "oof" / "ens+phase"
                            / "A_core_winter" / "final_test.parquet")
    assert len(cache) == len(test)
    amount = cache["amount"].to_numpy()

    nb = verification.csi_pod_far(y[shared], nbm[shared], threshold_in=THR)
    print(f"bars: nbm_raw csi6={nb['csi']:.3f}; current-head ceiling ~0.214-0.216",
          flush=True)

    # mm_precip>0.5mm keeps 97% of >=6in rows but drops ~2/3 of the no-snow
    # majority — the head spends its capacity discriminating within storms.
    precip = pd.to_numeric(inner["mm_precip_mean_mm"], errors="coerce").fillna(0)

    def combo():
        """Screen winners together: storm-conditioned exc_6 OR upgraded
        tail quantile rules. Persists test preds so follow-up analysis
        never retrains these."""
        exc = train_binary(inner, feats, rows_mask=(precip > 0.5).to_numpy())
        qs = train_quantiles(inner, feats, num_leaves=127, min_data_in_leaf=50,
                             learning_rate=0.03, num_rounds=1500)
        preds = pd.DataFrame({
            "exc6_precip": exc.predict(Xt),
            **{f"{n}_cap": np.maximum(0.0, b.predict(Xt)) for n, b in qs.items()},
        })
        preds.to_parquet(ROOT / "data" / "training" / "oof" / "head_screen_preds.parquet",
                         index=False)
        best = {"csi": None}
        gate_cuts = np.unique(np.quantile(preds["exc6_precip"],
                                          np.linspace(0.9, 0.9999, 40)))
        for q95c in np.arange(7.0, 12.01, 0.5):
            qhit = preds["q95_cap"].to_numpy() >= q95c
            for g in gate_cuts:
                hit = qhit | (preds["exc6_precip"].to_numpy() >= g)
                pt = np.where((amount >= THR) | hit, THR, 0.0)
                s = verification.csi_pod_far(y[shared], pt[shared], threshold_in=THR)
                if s["freq_bias"] and 0.8 <= s["freq_bias"] <= 1.5:
                    if best["csi"] is None or (s["csi"] or 0) > best["csi"]:
                        best = {"csi": s["csi"], "pod": s["pod"], "fb": s["freq_bias"],
                                "q95_cut": float(q95c), "gate_cut": float(g)}
        return "raw", {"combined_oracle": best}

    candidates = {
        "exc6_spw10":   lambda: ("exc", train_binary(inner, feats, scale_pos_weight=10)),
        "exc6_spw60":   lambda: ("exc", train_binary(inner, feats, scale_pos_weight=60)),
        "exc6_spw100":  lambda: ("exc", train_binary(inner, feats, scale_pos_weight=100)),
        "exc6_cap":     lambda: ("exc", train_binary(
            inner, feats, num_leaves=127, learning_rate=0.03, num_rounds=1500)),
        "exc6_precip":  lambda: ("exc", train_binary(
            inner, feats, rows_mask=(precip > 0.5).to_numpy())),
        "q_cap":        lambda: ("q", train_quantiles(
            inner, feats, num_leaves=127, min_data_in_leaf=50,
            learning_rate=0.03, num_rounds=1500)),
        "combo_v1":     combo,
    }

    for name, make in candidates.items():
        if name in results:
            print(f"[skip] {name}", flush=True)
            continue
        t0 = time.time()
        kind, model = make()
        if kind == "exc":
            score = model.predict(Xt)
            res = oracle_exc(score, test, shared, amount, y)
        elif kind == "raw":
            res = model
        else:
            qpreds = {n: np.maximum(0.0, b.predict(Xt)) for n, b in model.items()}
            res = oracle_q(qpreds, test, shared, amount, y)
        res["minutes"] = round((time.time() - t0) / 60, 1)
        results[name] = res
        OUT.write_text(json.dumps(results, indent=2, default=float))
        print(f"[done] {name} ({res['minutes']} min): {json.dumps(res, default=float)}",
              flush=True)
    print("screen complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
