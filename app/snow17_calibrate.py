"""Per-station SNOW-17 parameter calibration.

The default `Snow17Params` are seeded from elevation + latitude and work
reasonably anywhere in CONUS, but every SNOTEL site has its own quirks —
sheltered vs exposed, north-facing vs south-facing, persistent vs maritime.
Anderson's recommended tuning fits MFMAX / MFMIN / UADJ / PXTEMP against an
observed SWE record. We do exactly that with L-BFGS-B over a tight box.

The objective is a continuous (non-anchored) walk through 1–2 winter seasons:
seed the state from the first observation in the calibration window and let
the model drift through the period without resets. Loss is a weighted average
of |W_e - SWE_obs| and |depth_model - depth_obs|.

We cache fitted params to `data/snow17_params/<triplet>.json`. `forecast.py`
loads them on every run; if the cache is missing/stale, the elev+lat seed is
used. `scripts/calibrate_snow17.py` warms the cache offline; CI doesn't need
to refit on every run since SNOTEL sites change parameters seasonally not
hourly.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import snow17 as _snow17

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover — scipy missing -> fall back to grid scan
    _HAVE_SCIPY = False

ROOT = Path(__file__).resolve().parents[1]
PARAMS_DIR = ROOT / "data" / "snow17_params"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# Box bounds. These cover the literature range generously and keep the
# physical interpretation intact — UADJ has to stay positive, MFMAX > MFMIN.
_BOUNDS = {
    "MFMAX":  (0.50, 2.20),
    "MFMIN":  (0.05, 1.00),
    "UADJ":   (0.01, 0.20),
    "PXTEMP": (-1.0,  3.0),
}

# Calibration loss weights — depth and SWE are both in inches, so weighting
# them 1:1 makes sense, but SWE is the "physical" quantity and depth includes
# all the density noise we're trying not to overfit. Weight SWE higher.
_W_SWE = 1.0
_W_DEPTH = 0.5

_DEFAULT_MAX_FIT_DAYS = 700  # ~2 winter seasons


def _params_path(triplet: str) -> Path:
    return PARAMS_DIR / f"{triplet.replace(':', '_')}.json"


def load_cached(triplet: str, *, max_age_days: int = 60) -> Optional[dict]:
    p = _params_path(triplet)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text())
    except Exception:
        return None
    fitted_at = rec.get("fitted_at", 0)
    if time.time() - float(fitted_at) > max_age_days * 86400:
        return None
    return rec


def apply_cached(triplet: str, base: _snow17.Snow17Params, *, max_age_days: int = 60) -> _snow17.Snow17Params:
    """Merge cached params into `base`. Returns `base` unchanged if no cache."""
    rec = load_cached(triplet, max_age_days=max_age_days)
    if not rec:
        return base
    fitted = rec.get("params") or {}
    return replace(
        base,
        MFMAX=float(fitted.get("MFMAX", base.MFMAX)),
        MFMIN=float(fitted.get("MFMIN", base.MFMIN)),
        UADJ=float(fitted.get("UADJ", base.UADJ)),
        PXTEMP=float(fitted.get("PXTEMP", base.PXTEMP)),
    )


def _pair_history(hist: pd.DataFrame, wx_hist: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Inner-join SNOTEL (date, snow_depth_in, swe_in) with Open-Meteo daily.
    Returns None if the overlap is too thin to fit."""
    if hist is None or hist.empty or wx_hist is None or wx_hist.empty:
        return None
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"]).dt.date
    w = wx_hist.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.date
    df = h.merge(w, on="date", how="inner")
    df = df.dropna(subset=["snow_depth_in", "swe_in"], how="all")
    if len(df) < 60:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _run_continuous(
    df: pd.DataFrame, params: _snow17.Snow17Params,
) -> Tuple[np.ndarray, np.ndarray]:
    """Walk SNOW-17 through df without re-anchoring. Returns (swe_model, depth_model)."""
    # Seed from the first row with finite SWE+depth; if SWE missing, estimate
    # from depth × 0.30.
    seed_idx = 0
    for i in range(len(df)):
        if pd.notna(df.iloc[i].get("snow_depth_in")) and pd.notna(df.iloc[i].get("swe_in")):
            seed_idx = i
            break
    seed = df.iloc[seed_idx]
    swe0 = float(seed["swe_in"]) if pd.notna(seed["swe_in"]) else float(seed["snow_depth_in"]) * 0.30
    depth0 = float(seed["snow_depth_in"]) if pd.notna(seed["snow_depth_in"]) else swe0 / 0.30
    density0 = (swe0 / max(depth0, 0.01)) if depth0 > 0 else 0.30
    state = _snow17.Snow17State(
        W_e=max(0.0, swe0), W_q=0.0, ATI=-2.0,
        depth_in=max(0.0, depth0),
        pack_density=max(0.10, min(0.55, density0)),
    )

    swe_pred = np.full(len(df), np.nan)
    depth_pred = np.full(len(df), np.nan)
    for i in range(seed_idx, len(df)):
        row = df.iloc[i]
        precip_in = float(row.get("precipitation_sum") or 0.0) / 25.4
        tmean = row.get("temperature_2m_mean")
        tmax = row.get("temperature_2m_max")
        rad = row.get("shortwave_radiation_sum")
        wind_kmh = row.get("windspeed_10m_max")
        wind_mph = (float(wind_kmh) / 1.609) if wind_kmh is not None and pd.notna(wind_kmh) else None
        doy = pd.Timestamp(row["date"]).dayofyear
        state, _depth = _snow17._daily_step(
            state, precip_in=precip_in,
            tmean_c=float(tmean) if pd.notna(tmean) else 0.0,
            tmax_c=float(tmax) if (tmax is not None and pd.notna(tmax)) else None,
            rad_mj=float(rad) if (rad is not None and pd.notna(rad)) else None,
            wind_mph=wind_mph, doy=int(doy), params=params,
        )
        swe_pred[i] = state.W_e
        depth_pred[i] = state.depth_in
    return swe_pred, depth_pred


def _loss(x: np.ndarray, df: pd.DataFrame, base: _snow17.Snow17Params) -> float:
    p = replace(base, MFMAX=float(x[0]), MFMIN=float(x[1]),
                UADJ=float(x[2]), PXTEMP=float(x[3]))
    # Enforce MFMAX > MFMIN softly with a penalty (L-BFGS-B doesn't support
    # inequality constraints directly).
    penalty = 0.0
    if p.MFMAX <= p.MFMIN + 0.05:
        penalty += 10.0 * (p.MFMIN + 0.05 - p.MFMAX)

    swe_pred, depth_pred = _run_continuous(df, p)
    swe_obs = df["swe_in"].to_numpy(dtype=float)
    depth_obs = df["snow_depth_in"].to_numpy(dtype=float)

    swe_err = np.abs(swe_pred - swe_obs)
    depth_err = np.abs(depth_pred - depth_obs)
    swe_mae = float(np.nanmean(swe_err)) if np.isfinite(swe_err).any() else 10.0
    depth_mae = float(np.nanmean(depth_err)) if np.isfinite(depth_err).any() else 10.0
    return _W_SWE * swe_mae + _W_DEPTH * depth_mae + penalty


def calibrate(
    triplet: str,
    hist: pd.DataFrame, wx_hist: pd.DataFrame, *,
    base: _snow17.Snow17Params,
    max_days: int = _DEFAULT_MAX_FIT_DAYS,
    save: bool = True,
) -> Optional[dict]:
    """Fit MFMAX/MFMIN/UADJ/PXTEMP for one station against its SNOTEL +
    Open-Meteo history. Returns the fitted record dict, or None if there
    aren't enough rows to fit."""
    paired = _pair_history(hist, wx_hist)
    if paired is None:
        return None
    if len(paired) > max_days:
        paired = paired.tail(max_days).reset_index(drop=True)

    if not _HAVE_SCIPY:
        # Coarse grid search fallback. Cheap because each eval is O(N) walk.
        best_loss = float("inf")
        best_x = np.array([base.MFMAX, base.MFMIN, base.UADJ, base.PXTEMP])
        for mfmax in np.linspace(0.6, 1.8, 5):
            for mfmin in np.linspace(0.10, 0.80, 5):
                if mfmin >= mfmax - 0.05:
                    continue
                for uadj in np.linspace(0.02, 0.12, 4):
                    for pxt in (0.0, 1.0, 2.0):
                        x = np.array([mfmax, mfmin, uadj, pxt])
                        l = _loss(x, paired, base)
                        if l < best_loss:
                            best_loss = l
                            best_x = x
        fitted = best_x
        final_loss = best_loss
    else:
        x0 = np.array([base.MFMAX, base.MFMIN, base.UADJ, base.PXTEMP], dtype=float)
        bounds = [_BOUNDS["MFMAX"], _BOUNDS["MFMIN"],
                  _BOUNDS["UADJ"], _BOUNDS["PXTEMP"]]
        res = _scipy_minimize(
            _loss, x0, args=(paired, base), method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 60, "ftol": 1e-4, "eps": 1e-3},
        )
        fitted = res.x
        final_loss = float(res.fun)

    base_loss = _loss(np.array([base.MFMAX, base.MFMIN, base.UADJ, base.PXTEMP]), paired, base)

    rec = {
        "triplet": triplet,
        "fitted_at": time.time(),
        "n_rows": int(len(paired)),
        "first_date": str(paired["date"].iloc[0]),
        "last_date": str(paired["date"].iloc[-1]),
        "base_params": {
            "MFMAX": base.MFMAX, "MFMIN": base.MFMIN,
            "UADJ": base.UADJ, "PXTEMP": base.PXTEMP,
        },
        "params": {
            "MFMAX": float(fitted[0]), "MFMIN": float(fitted[1]),
            "UADJ":  float(fitted[2]), "PXTEMP": float(fitted[3]),
        },
        "loss_base": float(base_loss),
        "loss_fit": float(final_loss),
    }
    if save:
        _params_path(triplet).write_text(json.dumps(rec, separators=(",", ":")))
    return rec
