"""Probability calibration + decision-gate tuning for the event heads.

Isotonic regression maps each exceedance head's raw score to a calibrated
probability; serialized as plain breakpoint arrays in JSON so inference
needs neither sklearn nor pickle (CI images stay lean, and pickle across
sklearn versions is a footgun).

Gate tuning picks, per (threshold, lead bucket), the calibrated-probability
cutoff that maximizes CSI subject to a frequency-bias band. The FB band is
the guard rail against the degenerate CSI win: spamming events raises POD
(and often CSI at rare thresholds) while wrecking the false-alarm ratio a
forecaster actually lives with.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .verification import csi_pod_far

# Lead buckets for gate tuning: skill (and the right amount of caution)
# decays with lead, but per-lead-day tuning would overfit thin calibration
# slices at rare thresholds.
LEAD_BUCKETS = ((1, 2), (3, 4), (5, 7))
FB_BAND = (0.8, 1.3)
# Rare thresholds trade differently: at 6in+ the CSI optimum sits at higher
# frequency bias (NBM itself runs FB ~1.3 at 1in), and a miss costs far more
# than a false alarm operationally. Band widened, not removed — the guard
# against CSI-by-spam stays.
FB_BAND_RARE = (0.8, 1.5)
RARE_THRESHOLD_IN = 6.0

# Gates are TUNED against a band shrunk toward center: FB drifts between the
# calibration slice and the verification season (observed 2026-07-06: gates
# tuned to FB<=1.3 on the Oct-Nov tail realized FB 1.55 in Dec-Feb). Aiming
# at the center keeps realized FB inside the pre-registered reporting band
# [0.8, 1.3] ([0.8, 1.5] rare), which is what W2 is judged against.
FB_TUNE_BAND = (0.85, 1.15)
FB_TUNE_BAND_RARE = (0.85, 1.3)


def fb_band_for(threshold_in: float) -> tuple[float, float]:
    """Tuning band (aim-center). Reporting band stays FB_BAND/FB_BAND_RARE."""
    return FB_TUNE_BAND_RARE if threshold_in >= RARE_THRESHOLD_IN else FB_TUNE_BAND


def lead_bucket(lead_days: int) -> str:
    for lo, hi in LEAD_BUCKETS:
        if lo <= lead_days <= hi:
            return f"{lo}-{hi}"
    return f"{LEAD_BUCKETS[-1][0]}-{LEAD_BUCKETS[-1][1]}"


def fit_isotonic(raw_prob, event_occurred) -> Optional[dict]:
    """Fit isotonic regression raw score -> P(event); return breakpoints.

    Serialized form: {"x": [...], "y": [...]} for np.interp at apply time.
    Returns None when the slice is degenerate (no positives or negatives —
    nothing to calibrate against).
    """
    from sklearn.isotonic import IsotonicRegression

    p = np.asarray(raw_prob, dtype=float)
    o = np.asarray(event_occurred, dtype=float)
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = p[ok], o[ok]
    if p.size < 50 or o.sum() == 0 or o.sum() == o.size:
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, o)
    x = np.asarray(iso.X_thresholds_, dtype=float)
    y = np.asarray(iso.y_thresholds_, dtype=float)
    return {"x": x.tolist(), "y": y.tolist()}


def apply_isotonic(raw_prob, curve: Optional[dict]) -> np.ndarray:
    """Map raw scores through fitted breakpoints; identity if no curve."""
    p = np.asarray(raw_prob, dtype=float)
    if not curve:
        return p
    return np.interp(p, np.asarray(curve["x"]), np.asarray(curve["y"]))


def tune_gate(prob_cal, y_true, lead_days, *, threshold_in: float,
              fb_band: Optional[tuple[float, float]] = None,
              candidates: Optional[np.ndarray] = None) -> dict:
    """Per-lead-bucket probability cutoffs maximizing CSI within the FB band.

    Returns {bucket: {"gate": g, "csi": ..., "pod": ..., "far": ...,
    "freq_bias": ..., "n_events": ...}}. A bucket with no admissible gate
    (every candidate breaks the FB band, or no events to tune on) gets
    gate=None — the cascade must then skip flooring at this threshold for
    that bucket rather than fall back to an uncalibrated guess.
    """
    if fb_band is None:
        fb_band = fb_band_for(threshold_in)
    if candidates is None:
        # 0.01 steps: at rare thresholds/long leads the FB band is crossed
        # in a narrow probability window — a 0.05 grid jumps straight over
        # it (observed 2026-07-06: no valid 6in gate at leads 3-7, CSI@6
        # 0.197→0.104) — and a missing gate costs far more than a slightly
        # off-optimum one.
        candidates = np.round(np.arange(0.02, 0.97, 0.01), 2)
    df = pd.DataFrame({
        "p": np.asarray(prob_cal, dtype=float),
        "y": np.asarray(y_true, dtype=float),
        "bucket": [lead_bucket(int(ld)) for ld in np.asarray(lead_days)],
    }).dropna()
    def _best_gate(sub: pd.DataFrame) -> dict:
        n_events = int((sub["y"] >= threshold_in).sum())
        best = {"gate": None, "csi": None, "pod": None, "far": None,
                "freq_bias": None, "n_events": n_events}
        if n_events == 0:
            return best
        for g in candidates:
            pred = np.where(sub["p"].to_numpy() >= g, threshold_in, 0.0)
            c = csi_pod_far(sub["y"], pred, threshold_in=threshold_in)
            if c["csi"] is None or c["freq_bias"] is None:
                continue
            if not (fb_band[0] <= c["freq_bias"] <= fb_band[1]):
                continue
            if best["csi"] is None or c["csi"] > best["csi"]:
                best = {"gate": float(g), "csi": c["csi"], "pod": c["pod"],
                        "far": c["far"], "freq_bias": c["freq_bias"],
                        "n_events": n_events}
        return best

    out = {}
    pooled = None
    for (lo, hi) in LEAD_BUCKETS:
        b = f"{lo}-{hi}"
        best = _best_gate(df[df["bucket"] == b])
        if best["gate"] is None:
            # Lead-pooled fallback: 3x the events, averages lead-dependent
            # bias. A slightly mistuned gate beats no gate — without one the
            # cascade simply cannot call events at this threshold/lead.
            if pooled is None:
                pooled = _best_gate(df)
            if pooled["gate"] is not None:
                best = dict(pooled, pooled_fallback=True,
                            n_events=best["n_events"])
        out[b] = best
    return out
