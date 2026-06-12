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
    return out
