#!/usr/bin/env python3
"""Rare-tail decision-layer lab: tune 6/12in calibration on OOF winters.

Works entirely from the prediction caches written by build_oof_preds.py —
no retraining — so a decision-layer experiment costs seconds. The heads
are frozen; everything here is isotonic curves, gates, and floor rules.

Protocol (pre-registered, see plan):
  * All tuning/selection happens on the pooled OOF prior winters.
  * The fold TEST cache is scored only for --check (reproduce the
    production baseline), --oracle (ceiling diagnostics, explicitly
    labeled upper bounds), and --score of <=3 OOF-chosen finalists.
  * --finalize writes rare_calib.json for train_postprocessor.py
    --rare-calib to consume in the full-folds confirmation run.

Modes:
  --check      harness validation: tail-recipe cascade on the test cache
               must reproduce metrics_folds.json fold-A postproc numbers;
               OOF-model vs final-model score-shift table on the shared
               cal tail.
  --oracle     ceilings on the test cache: best achievable cascade
               CSI@6/12 with test-knowledge gates / quantile rules.
  --variants   tune the variant ladder on OOF, write lab_results.json.
  --score a,b  score named finalists on the test cache with guardrails
               (vs-NBM shared-row bootstraps, FB bands, reliability).
  --finalize a write rare_calib.json for the named variant.

Usage:
  .venv/bin/python scripts/rare_tail_lab.py \
      --oof-dir data/training/oof/ens+phase/A_core_winter --check --oracle
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import calibration, verification  # noqa: E402

CM_TO_IN = 1.0 / 2.54
THRESHOLDS = (1.0, 2.0, 6.0, 12.0)
RARE = (6.0, 12.0)
QLEVELS = ("q80", "q90", "q95")


# ---------------------------------------------------------------- loading

def load_caches(oof_dir: Path) -> dict:
    oof_files = sorted(oof_dir.glob("oof_*.parquet"))
    if not oof_files:
        raise SystemExit(f"no oof_*.parquet under {oof_dir} — run build_oof_preds.py")
    oof = pd.concat([pd.read_parquet(p) for p in oof_files], ignore_index=True)
    d = pd.to_datetime(oof["valid_date"])
    oof["winter"] = np.where(d.dt.month <= 6,
                             (d.dt.year - 1).astype(str) + "-" + (d.dt.year % 100).map("{:02d}".format),
                             d.dt.year.astype(str) + "-" + ((d.dt.year + 1) % 100).map("{:02d}".format))
    out = {
        "oof": oof,
        "winters": [p.stem.replace("oof_", "") for p in oof_files],
        "test": pd.read_parquet(oof_dir / "final_test.parquet"),
        "caltail_final": pd.read_parquet(oof_dir / "final_caltail.parquet"),
        "caltail_oof": {p.stem.replace("caltail_oof_", ""): pd.read_parquet(p)
                        for p in sorted(oof_dir.glob("caltail_oof_*.parquet"))},
    }
    for name in ("oof", "test", "caltail_final"):
        df = out[name]
        df["obs_snowfall_in"] = pd.to_numeric(df["obs_snowfall_in"], errors="coerce")
        df["lead_days"] = pd.to_numeric(df["lead_days"], errors="coerce").astype(int)
        df["nbm_in"] = pd.to_numeric(df["nbm_snowfall_cm"], errors="coerce") * CM_TO_IN
        y = df["obs_snowfall_in"].to_numpy()
        print(f"{name}: {len(df)} rows, >=6in {(y >= 6).sum()}, >=12in {(y >= 12).sum()}")
    return out


# ------------------------------------------------- cascade from the cache

def calibrated_probs(df: pd.DataFrame, iso: dict) -> dict[float, np.ndarray]:
    """Replicates predict(): per-threshold isotonic on the cached monotone
    raw probs, then re-enforce monotonicity across thresholds."""
    cal, running = {}, None
    for thr in THRESHOLDS:
        p = calibration.apply_isotonic(df[f"p{thr:g}"].to_numpy(), iso.get(f"{thr:g}"))
        running = p if running is None else np.minimum(running, p)
        cal[thr] = running
    return cal


def cascade_point(df: pd.DataFrame, calib: dict) -> np.ndarray:
    """point = max(amount, gated floors, qfloor) — mirrors postproc.predict."""
    cal = calibrated_probs(df, calib.get("iso", {}))
    amount = df["amount"].to_numpy()
    floor = np.zeros(len(df))
    buckets = np.array([calibration.lead_bucket(int(ld)) for ld in df["lead_days"]])
    for thr in THRESHOLDS:
        for b, info in calib.get("gates", {}).get(f"{thr:g}", {}).items():
            g = (info or {}).get("gate")
            if g is None:
                continue
            hit = (buckets == b) & (cal[thr] >= g)
            floor = np.where(hit, np.maximum(floor, thr), floor)
    for thr_key, by_bucket in (calib.get("qfloor") or {}).items():
        thr = float(thr_key)
        for b, (qname, qcut) in (by_bucket or {}).items():
            if qname in df.columns:
                hit = (buckets == b) & (df[qname].to_numpy() >= float(qcut))
                floor = np.where(hit, np.maximum(floor, thr), floor)
    return np.maximum(amount, floor)


def score_events(y, point) -> dict:
    out = {"mae": verification.mae_bias(y, point)["mae"]}
    for thr in THRESHOLDS:
        c = verification.csi_pod_far(y, point, threshold_in=thr)
        for k in ("csi", "pod", "far"):
            out[f"{k}_{thr:g}"] = c[k]
        out[f"fb_{thr:g}"] = c["freq_bias"]
    return out


def _fmt_scores(m: dict) -> str:
    def f(v, spec=".3f"):
        return format(v, spec) if v is not None else "  n/a"
    return (f"mae={f(m['mae'])} | "
            + " ".join(f"csi{t:g}={f(m[f'csi_{t:g}'])}" for t in THRESHOLDS) + " | "
            + " ".join(f"fb{t:g}={f(m[f'fb_{t:g}'], '.2f')}" for t in THRESHOLDS)
            + f" | pod6={f(m['pod_6'])} pod12={f(m['pod_12'])}"
            + f" far6={f(m['far_6'])} far12={f(m['far_12'])}")


# ----------------------------------------------------------- calibrations

def fit_tail_calib(caltail: pd.DataFrame) -> dict:
    """The production recipe, replayed from cache: isotonic + residual
    expected-count gates on the 60d tail (fit_calibration equivalent)."""
    y = caltail["obs_snowfall_in"].to_numpy()
    leads = caltail["lead_days"].to_numpy()
    amount = caltail["amount"].to_numpy()
    iso, gates = {}, {}
    for thr in THRESHOLDS:
        key = f"{thr:g}"
        p_raw = caltail[f"p{thr:g}"].to_numpy()
        iso[key] = calibration.fit_isotonic(p_raw, y >= thr)
        p_cal = calibration.apply_isotonic(p_raw, iso[key])
        gates[key] = calibration.tune_gate(p_cal, y, leads, threshold_in=thr,
                                           amount_pred=amount)
    return {"iso": iso, "gates": gates}


def qfloor_hits(df: pd.DataFrame, by_bucket: dict, buckets: np.ndarray) -> np.ndarray:
    """Row mask of a per-bucket (head, cut) rule's calls."""
    hit = np.zeros(len(df), dtype=bool)
    for b, (qname, qcut) in (by_bucket or {}).items():
        hit |= (buckets == b) & (df[qname].to_numpy() >= float(qcut))
    return hit


def tune_qfloor(oof: pd.DataFrame, thr: float,
                fb_band: tuple[float, float] | None = None,
                already=None) -> dict[str, list] | None:
    """Best per-lead-bucket (quantile head, cut) event rule for `thr` on
    the OOF winters: CSI-max over head x cut subject to the union (amount
    OR already OR q>=cut) landing inside the FB band WITHIN THE BUCKET.
    Per bucket because quantile heads sharpen at short leads — one global
    cut over-fires there (measured: FB@6 1.87 at leads 1-2 vs 1.41 at
    5-7). The cut is a free parameter — the oracle showed the optimum sits
    well above the threshold itself (e.g. q95>=~15.6 for 12in: quantile
    heads regress toward the mean too, just far less than amount).
    `already` = calls from HIGHER thresholds' floors, which count as
    events here too. Buckets with no admissible rule are omitted.
    """
    if fb_band is None:
        fb_band = calibration.fb_band_for(thr)
    y = oof["obs_snowfall_in"].to_numpy()
    amount = oof["amount"].to_numpy()
    base_call = (amount >= thr)
    if already is not None:
        base_call = base_call | np.asarray(already, dtype=bool)
    buckets = np.array([calibration.lead_bucket(int(ld)) for ld in oof["lead_days"]])
    out: dict[str, list] = {}
    for lo, hi in calibration.LEAD_BUCKETS:
        b = f"{lo}-{hi}"
        in_b = buckets == b
        y_b, base_b = y[in_b], base_call[in_b]
        best = None
        for qname in QLEVELS:
            q = oof[qname].to_numpy()[in_b]
            for qcut in np.round(np.arange(thr * 0.5, thr * 2.5, thr * 0.025), 2):
                pt = np.where(base_b | (q >= qcut), thr, 0.0)
                s = verification.csi_pod_far(y_b, pt, threshold_in=thr)
                if (s["freq_bias"] is None
                        or not (fb_band[0] <= s["freq_bias"] <= fb_band[1])):
                    continue
                if best is None or (s["csi"] or 0) > best[0]:
                    best = ((s["csi"] or 0), qname, float(qcut))
        if best is not None:
            out[b] = [best[1], best[2]]
    return out or None


def fit_oof_overrides(oof: pd.DataFrame, *, thresholds=RARE,
                      fb_target: dict[float, float] | None = None,
                      pooled: set[float] = frozenset(),
                      qfloor: dict[str, dict] | None = None,
                      no_gate: set[float] = frozenset()) -> dict:
    """Rare-threshold iso + gates from the pooled OOF winters.

    qfloor entries are per-bucket {bucket: [head, cut]} rules; their rows
    count as already-called events during gate tuning (the gate supplies
    only the residual), exactly like the amount head does — otherwise the
    union over-fires. no_gate thresholds get the qfloor/amount union only.

    Thresholds are tuned DESCENDING and each tuned floor's calls feed the
    next (lower) threshold's residual: a 12in floor call is also a >=6in
    call, so tuning 6in blind to it stacks the two above the 6in target
    (observed: fb6 1.61 with a 1.4 target). Same union arithmetic as the
    amount head, one level up.
    """
    y = oof["obs_snowfall_in"].to_numpy()
    leads = oof["lead_days"].to_numpy()
    amount = oof["amount"].to_numpy()
    iso, gates = {}, {}
    # calls already made by HIGHER thresholds' floors (their floor value
    # exceeds every lower threshold, so they count as events there too)
    upper_calls = np.zeros(len(oof), dtype=bool)
    buckets = np.array([calibration.lead_bucket(int(ld)) for ld in leads])
    for thr in sorted(thresholds, reverse=True):
        key = f"{thr:g}"
        p_raw = oof[f"p{thr:g}"].to_numpy()
        iso[key] = calibration.fit_isotonic(p_raw, y >= thr)
        p_cal = calibration.apply_isotonic(p_raw, iso[key])
        amount_eff = np.maximum(amount, np.where(upper_calls, thr, 0.0))
        qspec = (qfloor or {}).get(key)
        qhit = np.zeros(len(oof), dtype=bool)
        if qspec:
            qhit = qfloor_hits(oof, qspec, buckets)
            amount_eff = np.maximum(amount_eff, np.where(qhit, thr, 0.0))
        if thr in no_gate:
            gates[key] = {f"{lo}-{hi}": {"gate": None}
                          for lo, hi in calibration.LEAD_BUCKETS}
        else:
            lead_arg = np.ones_like(leads) if thr in pooled else leads
            g = calibration.tune_gate(p_cal, y, lead_arg, threshold_in=thr,
                                      amount_pred=amount_eff,
                                      fb_target=(fb_target or {}).get(thr))
            if thr in pooled:
                # tuned on a single synthetic bucket; replicate everywhere
                one = g[calibration.lead_bucket(1)]
                g = {f"{lo}-{hi}": dict(one) for lo, hi in calibration.LEAD_BUCKETS}
            gates[key] = g
            for b, info in g.items():
                gv = (info or {}).get("gate")
                if gv is not None:
                    upper_calls |= (buckets == b) & (p_cal >= gv)
        upper_calls |= qhit
    out = {"iso": iso, "gates": gates}
    if qfloor:
        out["qfloor"] = {k: dict(v) for k, v in qfloor.items()}
    return out


def tuning_pool(oof: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The OOF slice a config's rules are tuned on. drop_winters excludes
    winters whose model-feature coverage is unrepresentative of live
    (2021-22: NBM/HRRR at ~75 stations vs 484 now) — a rule part-tuned on
    muted-quantile rows sets cuts too low and over-fires on rich-coverage
    seasons (measured LOWO FB drift x2.28 with it, x1.4 worst without)."""
    drop = set(cfg.get("drop_winters", []))
    if not drop:
        return oof
    return oof[~oof["winter"].isin(drop)].reset_index(drop=True)


def overrides_from_cfg(cfg: dict, oof: pd.DataFrame) -> dict:
    """fit_oof_overrides for a JSON-safe variant config. qfloor cuts are
    baked into the config by the ladder (tuned once on OOF), so --score
    and --finalize rebuild the exact same payload."""
    pool = tuning_pool(oof, cfg)
    thresholds = tuple(sorted(set(RARE) | ({2.0} if cfg.get("two_in") else set())))
    return fit_oof_overrides(
        pool, thresholds=thresholds,
        fb_target={float(k): v for k, v in (cfg.get("fb_target") or {}).items()},
        pooled={12.0} if cfg.get("pooled12") else frozenset(),
        qfloor=cfg.get("qfloor"),
        no_gate={float(k) for k in cfg.get("no_gate", [])})


def variant_calib(base: dict, cfg: dict, oof: pd.DataFrame) -> dict:
    """Assemble a full calib payload: tail recipe for common thresholds,
    OOF overrides per the variant config."""
    ov = overrides_from_cfg(cfg, oof)
    calib = {"iso": dict(base["iso"]), "gates": dict(base["gates"])}
    calib["iso"].update(ov["iso"])
    calib["gates"].update(ov["gates"])
    if ov.get("qfloor"):
        calib["qfloor"] = ov["qfloor"]
    return calib


# ------------------------------------------------------------ diagnostics

def check(c: dict, oof_dir: Path) -> None:
    print("\n=== harness check: tail recipe replayed on the test cache ===")
    base = fit_tail_calib(c["caltail_final"])
    for thr_key, by_bucket in base["gates"].items():
        desc = " ".join(f"{b}:{(g['gate'] if g['gate'] is not None else '-')}"
                        for b, g in by_bucket.items())
        print(f"  tail gates @{thr_key}in  {desc}")
    test = c["test"]
    m = score_events(test["obs_snowfall_in"].to_numpy(), cascade_point(test, base))
    print("  lab cascade :", _fmt_scores(m))
    ref_path = ROOT / "data" / "models" / "metrics_folds.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())["folds"]
        fold = oof_dir.name if oof_dir.name in ref else "A_core_winter"
        r = ref[fold]["sources"]["postproc"]
        print(f"  {fold} ref :", " ".join(
            f"{k}={r.get(k):.3f}" for k in
            ("mae", "csi_1in", "csi_6in", "csi_12in", "fb_6in", "fb_12in")
            if r.get(k) is not None))
        nb = ref[fold]["sources"].get("nbm_raw", {})
        print("  nbm_raw ref :", " ".join(
            f"{k}={nb.get(k):.3f}" for k in
            ("mae", "csi_1in", "csi_6in", "csi_12in", "fb_6in", "fb_12in")
            if nb.get(k) is not None))

    print("\n=== OOF-model vs final-model score shift (shared cal tail) ===")
    fin = c["caltail_final"].set_index(["triplet", "valid_date", "lead_days"])
    for w, df in c["caltail_oof"].items():
        o = df.set_index(["triplet", "valid_date", "lead_days"])
        join = fin.join(o, rsuffix="_oof", how="inner")
        row = [f"oof {w} (n={len(join)}):"]
        for col in ("p6", "p12", "amount"):
            a, b = join[col].to_numpy(), join[f"{col}_oof"].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            r = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 10 else np.nan
            row.append(f"{col}: r={r:.3f} mean {a[ok].mean():.4f}/{b[ok].mean():.4f}"
                       f" p99 {np.percentile(a[ok], 99):.3f}/{np.percentile(b[ok], 99):.3f}")
        print("  " + "  |  ".join(row))


def oracle(c: dict) -> None:
    """Ceilings with test knowledge — NOT tunable results. If even these
    trail NBM, the heads (not the decision layer) are the bottleneck."""
    test, oof = c["test"], c["oof"]
    y = test["obs_snowfall_in"].to_numpy()
    amount = test["amount"].to_numpy()
    # OOF isotonic (leakage-free) so oracle gates live on a realistic scale;
    # a monotone remap can't change the achievable-ceiling anyway.
    ov = fit_oof_overrides(oof)
    cal = calibrated_probs(test, ov["iso"])
    nbm = test["nbm_in"].to_numpy()
    shared = np.isfinite(nbm)
    print(f"\n=== oracle ceilings on the test cache (n={len(test)}, "
          f"shared-NBM n={int(shared.sum())}) ===")
    for thr in RARE:
        nb = verification.csi_pod_far(y[shared], nbm[shared], threshold_in=thr)
        print(f"@{thr:g}in nbm_raw shared rows: csi={nb['csi']:.3f} pod={nb['pod']:.3f} "
              f"fb={nb['freq_bias']:.2f}")
        p = cal[thr]
        rows = []
        for g in np.round(np.arange(0.01, 0.97, 0.01), 2):
            pt = np.where((amount >= thr) | (p >= g), thr, 0.0)
            s = verification.csi_pod_far(y, pt, threshold_in=thr)
            sh = verification.csi_pod_far(y[shared], pt[shared], threshold_in=thr)
            rows.append((s["csi"] or 0, g, s, sh))
        rows.sort(reverse=True, key=lambda r: r[0])
        for csi, g, s, sh in rows[:3]:
            print(f"  gate-oracle g={g:.2f}: all csi={s['csi']:.3f} pod={s['pod']:.3f} "
                  f"far={s['far']:.3f} fb={s['freq_bias']:.2f} | shared csi={sh['csi']:.3f}")
        for qname in QLEVELS:
            q = test[qname].to_numpy()
            pt = np.where((amount >= thr) | (q >= thr), thr, 0.0)
            s = verification.csi_pod_far(y, pt, threshold_in=thr)
            called = int((q >= thr).sum())
            print(f"  q-rule {qname}>= {thr:g}: calls={called} csi={s['csi']:.3f} "
                  f"pod={s['pod']:.3f} fb={s['freq_bias']:.2f}")
        # combined oracle: best gate OR best quantile rule
        bq = max(QLEVELS, key=lambda qn: verification.csi_pod_far(
            y, np.where((amount >= thr) | (test[qn].to_numpy() >= thr), thr, 0.0),
            threshold_in=thr)["csi"] or 0)
        qhit = test[bq].to_numpy() >= thr
        rows = []
        for g in np.round(np.arange(0.01, 0.97, 0.01), 2):
            pt = np.where((amount >= thr) | (p >= g) | qhit, thr, 0.0)
            s = verification.csi_pod_far(y, pt, threshold_in=thr)
            rows.append((s["csi"] or 0, g, s))
        csi, g, s = max(rows, key=lambda r: r[0])
        print(f"  OR-oracle ({bq}, g={g:.2f}): csi={s['csi']:.3f} pod={s['pod']:.3f} "
              f"far={s['far']:.3f} fb={s['freq_bias']:.2f}")
        # head discrimination + amount behavior on true events
        ev = y >= thr
        if ev.sum():
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(ev.astype(int), p)
            print(f"  exc head PR-AUC={ap:.3f} (base rate {ev.mean():.4f}); "
                  f"amount on true >={thr:g}in rows: "
                  f"median={np.median(amount[ev]):.2f} p90={np.percentile(amount[ev], 90):.2f} "
                  f"frac>=thr={np.mean(amount[ev] >= thr):.3f}")


# ------------------------------------------------------------- variants

def run_variants(c: dict, oof_dir: Path) -> None:
    """Tune the v2 ladder on the OOF pool; selection metric = pool cascade
    CSI@6/@12 with conservative FB placement. Writes lab_results.json.

    v2 lessons baked in (from look #1 + leakage-free LOWO drift analysis):
    * drop 2021-22 from tuning (thin feature coverage -> muted quantiles
      -> cuts too low -> x2+ FB drift on rich-coverage seasons);
    * tune FB LOW (~<=1.05 on the pool): held-out-winter drift is x0.7-1.5,
      and the test-CSI plateau is flat on the low-FB side but collapses
      when over-firing — err under, land in band;
    * no 12in gates (the p12 head adds only false alarms over the q-rule);
    * no p-AND-q hybrid (measured: same drift, less CSI).
    """
    oof = c["oof"]
    base = fit_tail_calib(c["caltail_final"])
    drop = ["2021-22"]
    pool = oof[~oof["winter"].isin(drop)].reset_index(drop=True)
    y = pool["obs_snowfall_in"].to_numpy()
    print(f"tuning pool: {len(pool)} rows (dropped {drop}), "
          f">=6in {(y >= 6).sum()}, >=12in {(y >= 12).sum()}")
    results: dict[str, dict] = {}

    def evaluate(name: str, cfg: dict) -> dict:
        t0 = time.time()
        calib = variant_calib(base, cfg, oof)
        m = score_events(y, cascade_point(pool, calib))
        results[name] = {"cfg": cfg, "oof": m}
        print(f"[{name:14s}] {_fmt_scores(m)}  ({time.time() - t0:.0f}s)")
        return m

    print("\n=== v2 ladder on the OOF pool (selection set) ===")
    evaluate("gates_cons", {"drop_winters": drop, "fb_target": {"6": 1.0, "12": 1.0}})
    buckets_pool = np.array([calibration.lead_bucket(int(ld))
                             for ld in pool["lead_days"]])
    qf12 = tune_qfloor(pool, 12.0, fb_band=(0.5, 1.05))
    q12hit = qfloor_hits(pool, qf12, buckets_pool) if qf12 else None
    qf6 = tune_qfloor(pool, 6.0, fb_band=(0.5, 1.05), already=q12hit)
    qf6_hot = tune_qfloor(pool, 6.0, fb_band=(0.5, 1.2), already=q12hit)
    print(f"tuned qfloor rules (cap 1.05): 6in={qf6} 12in={qf12}")
    print(f"tuned qfloor rule  (cap 1.2) : 6in={qf6_hot}")
    if qf6 and qf12:
        evaluate("v2_primary", {
            "drop_winters": drop, "fb_target": {"2": 1.0, "6": 1.0},
            "qfloor": {"6": qf6, "12": qf12}, "no_gate": ["12"],
            "two_in": True})
        evaluate("v2_alt6hot", {
            "drop_winters": drop, "fb_target": {"2": 1.0, "6": 1.15},
            "qfloor": {"6": qf6_hot, "12": qf12}, "no_gate": ["12"],
            "two_in": True})
        evaluate("v2_qonly", {
            "drop_winters": drop, "fb_target": {"2": 1.0},
            "qfloor": {"6": qf6, "12": qf12}, "no_gate": ["6", "12"],
            "two_in": True})

    out_path = oof_dir / "lab_results.json"
    out_path.write_text(json.dumps(
        {"winters": c["winters"], "results": results}, indent=2, default=float))
    print(f"\nwrote {out_path}")
    print("NOTE: pick <=3 finalists on these OOF numbers, then --score them "
          "on the test cache ONCE. Do not iterate against the test scores.")


# ------------------------------------------------------------- test score

def nbm_tuned_oof(c: dict) -> dict:
    """Fair-fight NBM: per-threshold/lead-bucket decision thresholds tuned
    on the same OOF winters the postproc rare layer gets."""
    oof, test = c["oof"], c["test"]
    ok = oof["nbm_in"].notna()
    t_ok = test["nbm_in"].notna()
    nbm_t = test.loc[t_ok, "nbm_in"].to_numpy()
    y_t = test.loc[t_ok, "obs_snowfall_in"].to_numpy()
    buckets = np.array([calibration.lead_bucket(int(ld))
                        for ld in test.loc[t_ok, "lead_days"]])
    out = {"n": int(t_ok.sum())}
    for thr in THRESHOLDS:
        grid = np.round(np.arange(0.2, 2.01, 0.1) * thr, 2)
        g_by = calibration.tune_gate(
            oof.loc[ok, "nbm_in"].to_numpy(),
            oof.loc[ok, "obs_snowfall_in"].to_numpy(),
            oof.loc[ok, "lead_days"].to_numpy(),
            threshold_in=thr, candidates=grid)
        pred = np.zeros(len(nbm_t))
        for b, info in g_by.items():
            g = (info or {}).get("gate")
            if g is not None:
                pred[(buckets == b) & (nbm_t >= g)] = thr
        s = verification.csi_pod_far(y_t, pred, threshold_in=thr)
        out[f"csi_{thr:g}"] = s["csi"]
        out[f"pod_{thr:g}"] = s["pod"]
        out[f"fb_{thr:g}"] = s["freq_bias"]
    return out


def score_finalists(c: dict, names: list[str], oof_dir: Path, n_boot: int) -> None:
    lab = json.loads((oof_dir / "lab_results.json").read_text())["results"]
    base = fit_tail_calib(c["caltail_final"])
    test = c["test"]
    y = test["obs_snowfall_in"].to_numpy()
    nbm = test["nbm_in"]
    shared_mask = nbm.notna().to_numpy()

    print("\n=== references on the test cache ===")
    m_nbm = score_events(y[shared_mask], nbm.to_numpy()[shared_mask])
    print("nbm_raw (shared rows):", _fmt_scores(m_nbm))
    tuned = nbm_tuned_oof(c)
    print("nbm_tuned_oof        :", " ".join(
        f"csi{t:g}={tuned[f'csi_{t:g}']:.3f}" for t in THRESHOLDS), "|", " ".join(
        f"fb{t:g}={tuned[f'fb_{t:g}']:.2f}" for t in THRESHOLDS))
    m_tail = score_events(y, cascade_point(test, base))
    print("tail baseline (all)  :", _fmt_scores(m_tail))

    all_out = {"nbm_raw_shared": m_nbm, "nbm_tuned_oof": tuned, "tail_baseline": m_tail}
    for name in names:
        if name not in lab:
            print(f"!! {name} not in lab_results.json, skipping")
            continue
        cfg = lab[name]["cfg"]
        calib = variant_calib(base, cfg, c["oof"])
        point = cascade_point(test, calib)
        m = score_events(y, point)
        print(f"\n=== finalist {name} cfg={cfg} ===")
        print("all rows   :", _fmt_scores(m))
        msh = score_events(y[shared_mask], point[shared_mask])
        print("shared rows:", _fmt_scores(msh))
        head = pd.DataFrame({
            "triplet": test["triplet"], "valid_date": test["valid_date"],
            "obs": y, "pp": point, "nbm": nbm.to_numpy(),
        })[shared_mask]
        head["err_pp"] = head["pp"] - head["obs"]
        head["err_nbm"] = head["nbm"] - head["obs"]
        bb = verification.paired_block_bootstrap(
            head, err_a="err_pp", err_b="err_nbm", n_boot=n_boot)
        vs = {"mae": bb}
        print(f"ΔMAE vs NBM {bb['diff']:+.3f} [{bb['ci_lo']:+.3f},{bb['ci_hi']:+.3f}] "
              f"P={bb['p_a_better']:.2f}")
        for thr in THRESHOLDS:
            def csi_delta(d, thr=thr):
                a = verification.csi_pod_far(d["obs"], d["pp"], threshold_in=thr)["csi"]
                b = verification.csi_pod_far(d["obs"], d["nbm"], threshold_in=thr)["csi"]
                return np.nan if a is None or b is None else a - b
            bs = verification.block_bootstrap_stat(head, csi_delta,
                                                   n_boot=min(n_boot, 500))
            vs[f"csi_{thr:g}"] = bs
            if bs["stat"] is not None and bs["ci_lo"] is not None:
                print(f"ΔCSI@{thr:g} vs NBM {bs['stat']:+.3f} "
                      f"[{bs['ci_lo']:+.3f},{bs['ci_hi']:+.3f}] P={bs['p_gt_0']:.2f}")
        # reliability of the OOF-calibrated rare probs on test (guardrail)
        cal = calibrated_probs(test, calib["iso"])
        rel = {f"{thr:g}": verification.reliability_curve(y, cal[thr], threshold_in=thr)
               for thr in RARE}
        for thr_key, rows in rel.items():
            print(f"reliability p{thr_key}:",
                  " ".join(f"{r['pred']}->{r['obs']}(n{r['n']})" for r in rows))
        all_out[name] = {"cfg": cfg, "all": m, "shared": msh, "vs_nbm": vs,
                         "reliability": rel}
    out_path = oof_dir / "finalists.json"
    out_path.write_text(json.dumps(all_out, indent=2, default=float))
    print(f"\nwrote {out_path}")


def write_rare_calib(c: dict, cfg: dict, name: str, oof_dir: Path) -> None:
    oof = c["oof"]
    thresholds = tuple(sorted(set(RARE) | ({2.0} if cfg.get("two_in") else set())))
    ov = overrides_from_cfg(cfg, oof)
    y = tuning_pool(oof, cfg)["obs_snowfall_in"].to_numpy()
    ov["provenance"] = {
        "variant": name, "cfg": cfg, "winters": c["winters"],
        "oof_dir": str(oof_dir),
        "n_events": {f"{t:g}": int((y >= t).sum()) for t in thresholds},
        "created": pd.Timestamp.now("UTC").isoformat(),
    }
    out_path = oof_dir / "rare_calib.json"
    out_path.write_text(json.dumps(ov, indent=2, default=float))
    print(f"wrote {out_path} (variant {name}, cfg {cfg})")


def finalize(c: dict, name: str, oof_dir: Path) -> None:
    lab = json.loads((oof_dir / "lab_results.json").read_text())["results"]
    write_rare_calib(c, lab[name]["cfg"], name, oof_dir)


def make_recipe_v2(c: dict, oof_dir: Path) -> None:
    """The pre-registered v2 recipe applied to THIS fold's OOF pool, with
    no variant ladder and no test peeking — for confirmation folds (C).
    Structure fixed by the fold-A study: conservative per-bucket q-rules
    (FB cap 1.05 — held-out-winter drift runs x0.7-1.5, and the CSI
    plateau is flat on the low-FB side), fb_target-1.0 residual gate at
    6in, no 12in gate, thin-coverage winters dropped. NO 2in override:
    the winter-tuned 2in isotonic+gates collapse on a spring test (fold B
    csi_2in 0.298 -> 0.231, fb 0.43) while the tail 2in path keeps the
    +0.021 win over NBM — 6/12in floors are winter phenomena, 2in is
    year-round and stays season-generic."""
    oof = c["oof"]
    have = set(oof["winter"])
    drop = [w for w in ("2021-22",) if w in have]
    pool = oof[~oof["winter"].isin(drop)].reset_index(drop=True)
    buckets_pool = np.array([calibration.lead_bucket(int(ld))
                             for ld in pool["lead_days"]])
    qf12 = tune_qfloor(pool, 12.0, fb_band=(0.5, 1.05))
    q12hit = qfloor_hits(pool, qf12, buckets_pool) if qf12 else None
    qf6 = tune_qfloor(pool, 6.0, fb_band=(0.5, 1.05), already=q12hit)
    cfg = {"drop_winters": drop, "fb_target": {"6": 1.0},
           "qfloor": {"6": qf6 or {}, "12": qf12 or {}},
           "no_gate": ["12"]}
    print(f"recipe-v2 on {sorted(have - set(drop))}: qf6={qf6} qf12={qf12}")
    write_rare_calib(c, cfg, "recipe_v2", oof_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--score", default=None, help="comma-separated finalist names")
    ap.add_argument("--finalize", default=None, help="variant name -> rare_calib.json")
    ap.add_argument("--recipe-v2", action="store_true",
                    help="apply the pre-registered v2 recipe to this fold's "
                         "OOF pool -> rare_calib.json (no ladder, no test)")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    c = load_caches(args.oof_dir)
    if args.check:
        check(c, args.oof_dir)
    if args.oracle:
        oracle(c)
    if args.variants:
        run_variants(c, args.oof_dir)
    if args.recipe_v2:
        make_recipe_v2(c, args.oof_dir)
    if args.score:
        score_finalists(c, [s.strip() for s in args.score.split(",")],
                        args.oof_dir, args.n_boot)
    if args.finalize:
        finalize(c, args.finalize, args.oof_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
