"""Forecast verification metrics — the arbiter of every "beats X" claim.

Pure pandas/numpy, no I/O. Conventions:
  - Deterministic snowfall metrics operate on inches.
  - Event thresholds follow operational practice: 1/2/6/12 in for contingency
    stats, with event-conditional MAE alongside (no-snow days dominate any
    raw MAE).
  - paired_block_bootstrap resamples station-week blocks: daily errors are
    strongly correlated within a storm, so naive bootstrap CIs are far too
    tight.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

EVENT_THRESHOLDS_IN = (1.0, 2.0, 6.0, 12.0)


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(t) & np.isfinite(p)
    return t[ok], p[ok]


def mae_bias(y_true, y_pred) -> dict:
    t, p = _clean(y_true, y_pred)
    if t.size == 0:
        return {"mae": None, "bias": None, "rmse": None, "n": 0}
    err = p - t
    return {
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "n": int(t.size),
    }


def event_conditional_mae(y_true, y_pred, *, threshold_in: float = 0.5) -> dict:
    """MAE restricted to days where snow actually fell (obs ≥ threshold)."""
    t, p = _clean(y_true, y_pred)
    mask = t >= threshold_in
    if mask.sum() == 0:
        return {"mae": None, "bias": None, "rmse": None, "n": 0}
    return mae_bias(t[mask], p[mask])


def csi_pod_far(y_true, y_pred, *, threshold_in: float) -> dict:
    """Contingency stats for the event 'snowfall >= threshold'."""
    t, p = _clean(y_true, y_pred)
    if t.size == 0:
        return {"csi": None, "pod": None, "far": None, "freq_bias": None,
                "hits": 0, "misses": 0, "false_alarms": 0, "n": 0}
    obs = t >= threshold_in
    fcst = p >= threshold_in
    hits = int(np.sum(obs & fcst))
    misses = int(np.sum(obs & ~fcst))
    fas = int(np.sum(~obs & fcst))
    csi = hits / (hits + misses + fas) if (hits + misses + fas) else None
    pod = hits / (hits + misses) if (hits + misses) else None
    far = fas / (hits + fas) if (hits + fas) else None
    fbias = (hits + fas) / (hits + misses) if (hits + misses) else None
    return {"csi": csi, "pod": pod, "far": far, "freq_bias": fbias,
            "hits": hits, "misses": misses, "false_alarms": fas, "n": int(t.size)}


def crps_from_quantiles(y_true, q_preds: np.ndarray, q_levels: Iterable[float]) -> Optional[float]:
    """Mean CRPS approximated as the average pinball loss over quantile levels
    (×2, which makes it converge to CRPS as levels densify).

    q_preds: (n_samples, n_quantiles), columns ordered like q_levels.
    """
    t = np.asarray(y_true, dtype=float)
    q = np.asarray(q_preds, dtype=float)
    levels = np.asarray(list(q_levels), dtype=float)
    if t.size == 0 or q.shape[0] != t.size or q.shape[1] != levels.size:
        return None
    ok = np.isfinite(t) & np.isfinite(q).all(axis=1)
    t, q = t[ok], q[ok]
    if t.size == 0:
        return None
    diff = t[:, None] - q
    pinball = np.maximum(levels[None, :] * diff, (levels[None, :] - 1.0) * diff)
    return float(2.0 * np.mean(pinball))


def crps_gaussian(y_true, mu, sigma) -> Optional[float]:
    """Closed-form CRPS for a Gaussian forecast (for ens-stat baselines)."""
    from math import erf, exp, pi, sqrt
    t = np.asarray(y_true, dtype=float)
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(s) & (s > 0)
    if ok.sum() == 0:
        return None
    t, m, s = t[ok], m[ok], s[ok]
    z = (t - m) / s
    phi = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + np.vectorize(erf)(z / np.sqrt(2)))
    crps = s * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def brier_skill(y_true, prob_fcst, *, threshold_in: float,
                clim_freq: Optional[float] = None) -> dict:
    """Brier score (and skill vs climatological frequency) for the event
    'snowfall >= threshold'."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(prob_fcst, dtype=float)
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if t.size == 0:
        return {"brier": None, "bss": None, "n": 0}
    obs = (t >= threshold_in).astype(float)
    bs = float(np.mean((p - obs) ** 2))
    base = clim_freq if clim_freq is not None else float(np.mean(obs))
    bs_clim = float(np.mean((base - obs) ** 2))
    bss = (1.0 - bs / bs_clim) if bs_clim > 0 else None
    return {"brier": bs, "bss": bss, "n": int(t.size)}


def paired_block_bootstrap(
    df: pd.DataFrame, *, err_a: str, err_b: str,
    station_col: str = "triplet", date_col: str = "valid_date",
    n_boot: int = 1000, seed: int = 0,
) -> dict:
    """Is mean(|err_a|) < mean(|err_b|), resampling station-weeks?

    df holds one row per forecast with signed errors in err_a/err_b.
    Returns the MAE difference (a − b; negative = a better), a 95% CI, and
    the fraction of resamples where a beats b (~one-sided p-value).
    """
    d = df[[station_col, date_col, err_a, err_b]].dropna().copy()
    if d.empty:
        return {"diff": None, "ci_lo": None, "ci_hi": None, "p_a_better": None, "n": 0}
    week = pd.to_datetime(d[date_col]).dt.strftime("%G-%V")
    d["_block"] = d[station_col].astype(str) + "_" + week
    blocks = d.groupby("_block")
    block_stats = blocks.agg(
        a_abs=(err_a, lambda x: np.abs(x).sum()),
        b_abs=(err_b, lambda x: np.abs(x).sum()),
        cnt=(err_a, "size"),
    ).reset_index(drop=True)
    n_blocks = len(block_stats)
    rng = np.random.default_rng(seed)
    a = block_stats["a_abs"].to_numpy()
    b = block_stats["b_abs"].to_numpy()
    c = block_stats["cnt"].to_numpy(dtype=float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_blocks, n_blocks)
        tot = c[idx].sum()
        diffs[i] = (a[idx].sum() - b[idx].sum()) / tot if tot else np.nan
    diffs = diffs[np.isfinite(diffs)]
    point = float((np.abs(d[err_a]).mean() - np.abs(d[err_b]).mean()))
    return {
        "diff": point,
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "p_a_better": float(np.mean(diffs < 0)),
        "n": int(len(d)),
        "n_blocks": int(n_blocks),
    }


def summarize_deterministic(
    df: pd.DataFrame, *, obs_col: str, pred_col: str,
) -> dict:
    """Standard per-(source, lead) scorecard block."""
    out = mae_bias(df[obs_col], df[pred_col])
    out["event_mae"] = event_conditional_mae(df[obs_col], df[pred_col])["mae"]
    for thr in EVENT_THRESHOLDS_IN:
        c = csi_pod_far(df[obs_col], df[pred_col], threshold_in=thr)
        out[f"csi_{thr:g}in"] = c["csi"]
        out[f"pod_{thr:g}in"] = c["pod"]
        out[f"far_{thr:g}in"] = c["far"]
        out[f"fb_{thr:g}in"] = c["freq_bias"]
    return out


def reliability_curve(y_true, prob_fcst, *, threshold_in: float,
                      n_bins: int = 5) -> list[dict]:
    """Reliability table for P(snowfall >= threshold): per probability bin,
    mean forecast probability vs observed event frequency. A calibrated
    forecast has pred ≈ obs in every bin (slope ~1 on the diagram)."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(prob_fcst, dtype=float)
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    occ = (t >= threshold_in).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                         "pred": round(float(p[m].mean()), 3),
                         "obs": round(float(occ[m].mean()), 3)})
    return rows


def performance_diagram_points(
    df: pd.DataFrame, *, obs_col: str, pred_cols: Iterable[str],
    thresholds: Iterable[float] = EVENT_THRESHOLDS_IN,
) -> list[dict]:
    """POD vs success ratio (1−FAR) per source and threshold — the points of
    a Roebber performance diagram. CSI isopleths belong to the plot layer."""
    pts = []
    for src in pred_cols:
        sub = df[df[src].notna()]
        for thr in thresholds:
            c = csi_pod_far(sub[obs_col], sub[src], threshold_in=thr)
            sr = None if c["far"] is None else 1.0 - c["far"]
            pts.append({"source": src, "threshold_in": thr, "pod": c["pod"],
                        "sr": sr, "csi": c["csi"], "freq_bias": c["freq_bias"],
                        "n": c["n"]})
    return pts


def block_bootstrap_stat(
    df: pd.DataFrame, stat_fn, *,
    station_col: str = "triplet", date_col: str = "valid_date",
    n_boot: int = 1000, seed: int = 0,
) -> dict:
    """Block bootstrap of an arbitrary statistic, resampling station-weeks.

    stat_fn(sub_df) -> float is recomputed on each resample (rows repeated
    per block draw), so it works for non-linear statistics — CSI/POD deltas,
    BSS, CRPS differences — where paired_block_bootstrap's mean-|err| algebra
    doesn't apply. Returns the full-sample point estimate, a 95% percentile
    CI, and P(stat < 0) / P(stat > 0) over resamples.
    """
    d = df.copy()
    if d.empty:
        return {"stat": None, "ci_lo": None, "ci_hi": None,
                "p_lt_0": None, "p_gt_0": None, "n": 0, "n_blocks": 0}
    week = pd.to_datetime(d[date_col]).dt.strftime("%G-%V")
    codes, _ = pd.factorize(d[station_col].astype(str) + "_" + week)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.flatnonzero(np.r_[True, np.diff(sorted_codes) != 0])
    bounds = np.r_[starts, len(sorted_codes)]
    idx_by_block = [order[bounds[i]:bounds[i + 1]] for i in range(len(starts))]
    n_blocks = len(idx_by_block)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_blocks, n_blocks)
        rows = np.concatenate([idx_by_block[j] for j in pick])
        stats[i] = stat_fn(d.iloc[rows])
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return {"stat": float(stat_fn(d)), "ci_lo": None, "ci_hi": None,
                "p_lt_0": None, "p_gt_0": None, "n": int(len(d)),
                "n_blocks": int(n_blocks)}
    return {
        "stat": float(stat_fn(d)),
        "ci_lo": float(np.percentile(stats, 2.5)),
        "ci_hi": float(np.percentile(stats, 97.5)),
        "p_lt_0": float(np.mean(stats < 0)),
        "p_gt_0": float(np.mean(stats > 0)),
        "n": int(len(d)),
        "n_blocks": int(n_blocks),
    }


def crpss(crps_fcst: Optional[float], crps_ref: Optional[float]) -> Optional[float]:
    """CRPS skill score vs a reference (climatology): 1 − CRPS/CRPS_ref."""
    if crps_fcst is None or crps_ref is None or crps_ref <= 0:
        return None
    return float(1.0 - crps_fcst / crps_ref)


def climatology_quantiles(
    train_df: pd.DataFrame, test_df: pd.DataFrame, *,
    obs_col: str = "obs_snowfall_in", station_col: str = "triplet",
    date_col: str = "valid_date", q_levels: Iterable[float] = (0.1, 0.5, 0.9),
) -> np.ndarray:
    """Per-(station, month) empirical quantiles of training-period obs,
    aligned to test rows — the climatology reference forecaster for CRPSS.

    Month buckets, not day-of-year: with a couple of winters per station,
    DOY cells are too thin to hold a quantile. Fallback: station-all-months,
    then global. Rows in the returned (n_test, n_levels) matrix line up with
    test_df row order.
    """
    levels = list(q_levels)

    def month_of(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s).dt.month

    tr = train_df[[station_col, date_col, obs_col]].dropna().copy()
    tr["_m"] = month_of(tr[date_col])
    by_sm = tr.groupby([station_col, "_m"])[obs_col].apply(
        lambda x: np.quantile(x, levels) if len(x) >= 20 else None)
    by_s = tr.groupby(station_col)[obs_col].apply(
        lambda x: np.quantile(x, levels) if len(x) >= 20 else None)
    global_q = np.quantile(tr[obs_col], levels) if len(tr) else np.zeros(len(levels))

    te_station = test_df[station_col].to_numpy()
    te_month = month_of(test_df[date_col]).to_numpy()
    out = np.empty((len(test_df), len(levels)))
    for i, (s, m) in enumerate(zip(te_station, te_month)):
        q = by_sm.get((s, m))
        if q is None:
            q = by_s.get(s)
        out[i] = q if q is not None else global_q
    return out
