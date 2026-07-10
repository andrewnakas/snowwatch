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

# Tuning band = the pre-registered reporting band. tune_gate targets FB≈1
# WITHIN it (not the edge), so realized FB lands near 1 and comfortably
# inside the band regardless of the calibration slice's base rate — no inward
# margin needed once the objective is FB-target rather than CSI-max.
def fb_band_for(threshold_in: float) -> tuple[float, float]:
    return FB_BAND_RARE if threshold_in >= RARE_THRESHOLD_IN else FB_BAND


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
              amount_pred=None,
              fb_band: Optional[tuple[float, float]] = None,
              fb_target: Optional[float] = None,
              candidates: Optional[np.ndarray] = None) -> dict:
    """Per-lead-bucket probability cutoffs for the cascade floor.

    amount_pred: the Tweedie regressor's per-row output. The cascade predicts
    an event when EITHER amount>=threshold OR the gate fires, so the gate is
    tuned on the RESIDUAL — the realized contingency scored is
    max(amount, gate_floor) >= threshold, not the gate alone. Without this,
    the amount regressor's own event calls stack on top of a gate tuned to
    FB≈1 and inflate realized FB (verified 2026-07-07: amount-alone FB 0.83@1
    / 0.10@6, cascade 1.28 / 1.37). Passing amount_pred=None recovers the
    gate-only behavior (legacy).

    Returns {bucket: {"gate": g, "csi": ..., ...}}; gate=None where no
    admissible gate exists (cascade then skips flooring for that bucket).
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
    amount = (np.asarray(amount_pred, dtype=float) if amount_pred is not None
              else np.zeros(len(np.asarray(prob_cal))))
    df = pd.DataFrame({
        "p": np.asarray(prob_cal, dtype=float),
        "y": np.asarray(y_true, dtype=float),
        "amount": amount,
        "bucket": [lead_bucket(int(ld)) for ld in np.asarray(lead_days)],
    }).dropna()
    def _cascade_pred(sub, gate):
        """The production quantity: event iff amount>=thr OR p>=gate."""
        return np.where((sub["amount"].to_numpy() >= threshold_in)
                        | (sub["p"].to_numpy() >= gate), threshold_in, 0.0)

    def _best_gate(sub: pd.DataFrame) -> dict:
        n_events = int((sub["y"] >= threshold_in).sum())
        best = {"gate": None, "csi": None, "pod": None, "far": None,
                "freq_bias": None, "n_events": n_events}
        if n_events == 0:
            return best
        p = sub["p"].to_numpy()
        amt = sub["amount"].to_numpy()
        # RESIDUAL expected-count gate. The cascade already calls an event
        # wherever amount>=threshold; the gate must supply only the REMAINING
        # events. With calibrated p, E[#events]=Σp, so among rows the amount
        # regressor does NOT already flag, threshold the top-(target − already)
        # by probability. Using only p (not slice outcomes) keeps it base-rate
        # invariant; subtracting the amount calls keeps the cascade UNION at
        # FB≈fb_target instead of stacking above it (verified 2026-07-07:
        # gate-alone tuning let amount's own calls inflate cascade FB to
        # 1.28@1 / 1.37@6). fb_target runs rare thresholds slightly hot
        # (miss ≫ false alarm operationally); callers with a season-matched
        # calibration set (rare_tail_lab) sweep it instead.
        target = (fb_target if fb_target is not None
                  else 1.0 if threshold_in < RARE_THRESHOLD_IN else 1.15)
        already = amt >= threshold_in
        target_total = float(np.sum(p)) * target
        residual_target = max(0.0, target_total - float(already.sum()))
        # Rank only the not-already-flagged rows by probability.
        p_resid = np.where(already, -np.inf, p)
        order = np.sort(p_resid)[::-1]
        order = order[np.isfinite(order)]
        gate = 1.0
        if residual_target >= 1 and len(order):
            csum = np.cumsum(order)
            k = min(int(np.searchsorted(csum, residual_target)), len(order) - 1)
            gate = float(order[k])
        gate = float(min(candidates, key=lambda c: abs(c - gate)))
        c = csi_pod_far(sub["y"], _cascade_pred(sub, gate), threshold_in=threshold_in)
        # Guard: if realized cascade FB lands outside the band, scan for the
        # candidate whose cascade FB is closest to the target within band.
        if c["freq_bias"] is None or not (fb_band[0] <= c["freq_bias"] <= fb_band[1]):
            cand = []
            for g in candidates:
                cc = csi_pod_far(sub["y"], _cascade_pred(sub, g), threshold_in=threshold_in)
                if cc["freq_bias"] is not None and fb_band[0] <= cc["freq_bias"] <= fb_band[1]:
                    cand.append((abs(cc["freq_bias"] - target), float(g), cc))
            if cand:
                cand.sort()
                gate, c = cand[0][1], cand[0][2]
            else:
                return best
        best = {"gate": gate, "csi": c["csi"], "pod": c["pod"],
                "far": c["far"], "freq_bias": c["freq_bias"], "n_events": n_events}
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
