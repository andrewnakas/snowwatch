"""Verification-grade observation QC + 24h snowfall target construction.

SNOTEL ultrasonic snow-depth sensors are noisy: ±5 cm accuracy with
non-physical spikes during storms (false echoes off falling snow), diurnal
temperature-compensation artifacts, and occasional multi-day stuck readings.
Daily depth *change* also conflates new snow with settlement and melt.
Everything downstream — model training, member anchoring, and the
"beats NBM" verification claim — needs a target built with those failure
modes handled explicitly, so this module is the single source of truth for:

  - `qc_daily_series(hist)`        despiked/flagged daily SNWD series
  - `daily_snowfall(hist_qc)`      24h new-snowfall estimate per day, with a
                                   per-row quality bitmask and method tag
  - `station_statics(...)`         pooled-model static attributes per station
  - `last_reliable_depth(hist_qc)` QC'd anchor for the live ensemble

Snowfall logic: ΔSNWD⁺ (positive part of daily depth change) is the primary
estimate. Large depth jumps must be corroborated by the snow pillow (ΔWTEQ⁺)
or the precip gauge; the implied snow-to-liquid ratio ΔSNWD⁺/ΔWTEQ⁺ must land
in the physical band [3, 30], otherwise we fall back to ΔWTEQ⁺ × station
climatological SLR. Uncorroborated jumps are zeroed (wind drift / sensor).
"""
from __future__ import annotations

import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Quality bitmask. OK rows are 0; anything nonzero says how the value was
# altered or why it is unreliable. Training and verification drop rows with
# UNRELIABLE bits; display keeps everything.
QC_OK = 0
QC_SPIKE_REMOVED = 1     # SNWD value replaced with NaN by the despiker
QC_SLR_OOB = 2           # implied SLR outside [SLR_MIN, SLR_MAX]
QC_WTEQ_FALLBACK = 4     # snowfall came from ΔWTEQ⁺ × climatological SLR
QC_STUCK = 8             # inside a zero-variance run during snow season
QC_MISSING = 16          # no usable estimate; snowfall_in is NaN
QC_UNCORROBORATED = 32   # large ΔSNWD⁺ with no pillow/gauge support; zeroed

# Bits that make a row unusable as a training/verification target.
QC_UNRELIABLE = QC_STUCK | QC_MISSING

SLR_MIN, SLR_MAX = 3.0, 30.0
DEFAULT_SLR = 13.0            # Baxter et al. 2005 western-US mean (not 10:1)
SPIKE_RESID_IN = 6.0          # rolling-median residual that triggers the despiker
SPIKE_SUPPORT_SWE_IN = 0.3    # |ΔWTEQ| that legitimizes a big depth move
SPIKE_SUPPORT_PRECIP_IN = 0.3
CORROBORATION_DEPTH_IN = 2.0  # ΔSNWD⁺ above this requires pillow/gauge support
CORROB_SWE_IN = 0.05
CORROB_PRECIP_IN = 0.05
STUCK_RUN_DAYS = 10
SNOW_SEASON_MONTHS = {11, 12, 1, 2, 3, 4, 5}

STATIC_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "station_static"


def qc_daily_series(hist: pd.DataFrame) -> pd.DataFrame:
    """Return `hist` (date, swe_in, snow_depth_in, precip_in, tavg_f) sorted by
    date with two added columns:

      snwd_qc   — snow_depth_in after clamping negatives, despiking, and
                  NaN-ing stuck runs (use this everywhere downstream)
      qc_flags  — per-row bitmask of QC_* flags

    The despiker compares each value to a centered 5-day rolling median and
    rejects residuals > SPIKE_RESID_IN that have no support from the pillow or
    the precip gauge (real dumps move WTEQ/PRCPSA; false echoes don't).
    """
    cols = ["date", "swe_in", "snow_depth_in", "precip_in", "tavg_f"]
    if hist is None or hist.empty:
        out = pd.DataFrame(columns=cols + ["snwd_qc", "qc_flags"])
        return out

    df = hist.copy()
    for c in ("swe_in", "snow_depth_in", "precip_in", "tavg_f"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    df = df.sort_values("date").reset_index(drop=True)

    snwd = df["snow_depth_in"].astype(float).copy()
    flags = np.zeros(len(df), dtype=np.uint8)

    # Negative depths are sensor artifacts; clamp.
    snwd[snwd < 0] = 0.0

    # Despike. A spike jumps away from BOTH neighbors and reverts — a real
    # storm is a step change that persists into the next day, so comparing
    # against a rolling median would wrongly flag legitimate points beside
    # genuine steps. Pillow/gauge movement legitimizes any excursion.
    dswe = df["swe_in"].diff().abs()
    precip = df["precip_in"].fillna(0.0)
    supported = (dswe.fillna(0.0) >= SPIKE_SUPPORT_SWE_IN) | (precip >= SPIKE_SUPPORT_PRECIP_IN)
    d_prev = snwd - snwd.shift(1)
    d_next = snwd - snwd.shift(-1)
    half = SPIKE_RESID_IN / 2.0
    spike = (
        ((d_prev > half) & (d_next > half)) | ((d_prev < -half) & (d_next < -half))
    ) & ~supported
    # The newest observation has no next-day neighbor, but it is the live
    # anchor — treat a big unsupported final jump as a spike too.
    if len(snwd) >= 2:
        last = snwd.index[-1]
        if pd.notna(d_prev.loc[last]) and abs(d_prev.loc[last]) > SPIKE_RESID_IN and not supported.loc[last]:
            spike.loc[last] = True
    flags[spike.to_numpy()] |= QC_SPIKE_REMOVED
    snwd[spike] = np.nan

    # Stuck sensor: identical nonzero depth for >= STUCK_RUN_DAYS during the
    # snow season. Genuine snowpack settles continuously; a flat line that
    # long is a dead sensor. Flag (don't NaN — the level may still be roughly
    # right) so training/verification can exclude these rows.
    months = pd.to_datetime(df["date"]).dt.month
    in_season = months.isin(SNOW_SEASON_MONTHS).to_numpy()
    vals = snwd.to_numpy()
    run_start = 0
    for i in range(1, len(vals) + 1):
        if i < len(vals) and vals[i] == vals[run_start] and np.isfinite(vals[i]) and vals[i] > 0:
            continue
        run_len = i - run_start
        if run_len >= STUCK_RUN_DAYS and in_season[run_start:i].any():
            flags[run_start:i] |= QC_STUCK
        run_start = i

    df["snwd_qc"] = snwd
    df["qc_flags"] = flags
    return df


def station_median_slr(hist_qc: pd.DataFrame) -> float:
    """Median implied SLR (ΔSNWD⁺/ΔWTEQ⁺) over well-corroborated snowfall days.
    Falls back to DEFAULT_SLR when the station lacks usable pairs."""
    if hist_qc is None or hist_qc.empty or "snwd_qc" not in hist_qc.columns:
        return DEFAULT_SLR
    dsnwd = hist_qc["snwd_qc"].diff()
    dswe = hist_qc["swe_in"].diff()
    mask = (dsnwd >= 1.0) & (dswe >= 0.1)
    slr = (dsnwd[mask] / dswe[mask]).replace([np.inf, -np.inf], np.nan).dropna()
    slr = slr[(slr >= SLR_MIN) & (slr <= SLR_MAX)]
    if len(slr) < 10:
        return DEFAULT_SLR
    return float(slr.median())


def daily_snowfall(hist_qc: pd.DataFrame, *, station_slr: Optional[float] = None) -> pd.DataFrame:
    """24h new-snowfall target series from a QC'd history.

    Returns columns: date, snowfall_in, dswe_in, method, quality.

      method ∈ {dsnwd, dswe_slr, zeroed, none}
      quality — QC_* bitmask (row flags from qc_daily_series OR'd with
                target-construction flags)

    Rules, per day t (diffs are t minus t-1):
      1. ΔSNWD⁺ is the primary estimate.
      2. ΔSNWD⁺ ≥ CORROBORATION_DEPTH_IN needs pillow (ΔWTEQ⁺) or gauge
         support; unsupported jumps are zeroed and flagged.
      3. With a solid pillow signal, the implied SLR must be in
         [SLR_MIN, SLR_MAX]; otherwise snowfall = ΔWTEQ⁺ × station SLR.
      4. If the depth diff is unavailable but the pillow moved, use
         ΔWTEQ⁺ × station SLR (flagged WTEQ_FALLBACK).
      5. Nothing usable → NaN + MISSING.
    """
    out_cols = ["date", "snowfall_in", "dswe_in", "method", "quality"]
    if hist_qc is None or hist_qc.empty or "snwd_qc" not in hist_qc.columns:
        return pd.DataFrame(columns=out_cols)

    slr_clim = float(station_slr) if station_slr else station_median_slr(hist_qc)
    slr_clim = min(max(slr_clim, SLR_MIN), SLR_MAX)

    snwd = hist_qc["snwd_qc"].astype(float)
    swe = pd.to_numeric(hist_qc["swe_in"], errors="coerce").astype(float)
    precip = pd.to_numeric(hist_qc["precip_in"], errors="coerce").fillna(0.0)
    base_flags = hist_qc["qc_flags"].astype(int)

    dsnwd = snwd.diff()
    dswe = swe.diff()

    rows = []
    for i in range(len(hist_qc)):
        d = hist_qc["date"].iloc[i]
        q = int(base_flags.iloc[i])
        if i > 0:
            q |= int(base_flags.iloc[i - 1])  # a flagged prior day taints the diff
        ds = dsnwd.iloc[i]
        dw = dswe.iloc[i]
        dw_pos = max(0.0, dw) if pd.notna(dw) else None
        p = float(precip.iloc[i])

        snowfall: Optional[float]
        if i == 0 or pd.isna(ds):
            # No depth diff. Pillow-only route if the pillow clearly moved.
            if dw_pos is not None and dw_pos >= CORROB_SWE_IN and i > 0:
                snowfall = dw_pos * slr_clim
                method = "dswe_slr"
                q |= QC_WTEQ_FALLBACK
            else:
                snowfall = None
                method = "none"
                q |= QC_MISSING
        else:
            ds_pos = max(0.0, float(ds))
            if ds_pos < CORROBORATION_DEPTH_IN:
                # Small changes ride through as-is: light powder often doesn't
                # register on a 0.1-in-resolution pillow, and zeroing it would
                # bias the target low.
                snowfall = ds_pos
                method = "dsnwd"
            else:
                corroborated = (dw_pos is not None and dw_pos >= CORROB_SWE_IN) or p >= CORROB_PRECIP_IN
                if not corroborated:
                    snowfall = 0.0
                    method = "zeroed"
                    q |= QC_UNCORROBORATED
                elif dw_pos is not None and dw_pos >= 0.1:
                    slr = ds_pos / dw_pos
                    if SLR_MIN <= slr <= SLR_MAX:
                        snowfall = ds_pos
                        method = "dsnwd"
                    else:
                        snowfall = dw_pos * slr_clim
                        method = "dswe_slr"
                        q |= QC_SLR_OOB | QC_WTEQ_FALLBACK
                else:
                    # Gauge-supported but pillow quiet (powder below pillow
                    # resolution): trust the depth sensor.
                    snowfall = ds_pos
                    method = "dsnwd"

        rows.append({
            "date": d,
            "snowfall_in": round(snowfall, 3) if snowfall is not None else np.nan,
            "dswe_in": round(max(0.0, float(dw)), 3) if pd.notna(dw) else np.nan,
            "method": method,
            "quality": q,
        })
    return pd.DataFrame(rows, columns=out_cols)


def last_reliable_depth(hist_qc: pd.DataFrame) -> tuple[Optional[float], Optional[date], Optional[float]]:
    """(depth_in, date, swe_in) of the most recent QC-clean depth observation.
    Skips despiked rows so the ensemble never anchors to a sensor spike."""
    if hist_qc is None or hist_qc.empty or "snwd_qc" not in hist_qc.columns:
        return None, None, None
    ok = hist_qc[hist_qc["snwd_qc"].notna()]
    if ok.empty:
        return None, None, None
    row = ok.iloc[-1]
    d = row["date"]
    if hasattr(d, "date") and not isinstance(d, date):
        d = d.date()
    swe = row.get("swe_in")
    swe_f = float(swe) if pd.notna(swe) else None
    return float(row["snwd_qc"]), d, swe_f


# ---------- station static attributes --------------------------------------

def _snow_class(djf_tavg_f: Optional[float], djf_precip_in: Optional[float]) -> str:
    """Coarse Sturm-style snow climate class from winter temperature and
    precipitation. Three buckets are plenty for a pooled-model categorical."""
    if djf_tavg_f is None or djf_precip_in is None:
        return "unknown"
    if djf_tavg_f >= 27.0 and djf_precip_in >= 8.0:
        return "maritime"
    if djf_tavg_f < 18.0:
        return "continental"
    return "intermountain"


def station_statics(
    station: dict,
    hist_qc: pd.DataFrame,
    snowfall_df: Optional[pd.DataFrame] = None,
    *,
    use_cache: bool = True,
    max_age_days: int = 30,
) -> dict:
    """Static per-station attributes for the pooled post-processor.

    Derived from the station's own record (no PRISM dependency):
      elevation_ft, lat, lon, djf_tavg_f, djf_precip_in, median_slr,
      mean_max_swe_in, mean_max_depth_in, snow_class, n_years
    """
    triplet = station.get("triplet") or ""
    safe = triplet.replace(":", "_") or "unknown"
    cache_p = STATIC_CACHE_DIR / f"{safe}.json"
    if use_cache and cache_p.exists() and (time.time() - cache_p.stat().st_mtime) < max_age_days * 86400:
        try:
            return json.loads(cache_p.read_text())
        except Exception:
            pass

    out = {
        "triplet": triplet,
        "elevation_ft": float(station["elevation_ft"]) if station.get("elevation_ft") else None,
        "lat": float(station["lat"]) if station.get("lat") is not None else None,
        "lon": float(station["lon"]) if station.get("lon") is not None else None,
        "djf_tavg_f": None,
        "djf_precip_in": None,
        "median_slr": DEFAULT_SLR,
        "mean_max_swe_in": None,
        "mean_max_depth_in": None,
        "snow_class": "unknown",
        "n_years": 0,
    }
    if hist_qc is not None and not hist_qc.empty and "snwd_qc" in hist_qc.columns:
        df = hist_qc.copy()
        dts = pd.to_datetime(df["date"])
        df["_month"] = dts.dt.month
        # Water year: Oct-Sep, labeled by the calendar year it ends in.
        df["_wy"] = dts.dt.year + (dts.dt.month >= 10).astype(int)

        djf = df[df["_month"].isin([12, 1, 2])]
        if len(djf) >= 90:
            tavg = pd.to_numeric(djf["tavg_f"], errors="coerce").dropna()
            if len(tavg) >= 90:
                out["djf_tavg_f"] = round(float(tavg.mean()), 2)
            psum = pd.to_numeric(djf["precip_in"], errors="coerce").dropna()
            n_djf_seasons = djf["_wy"].nunique()
            if len(psum) >= 90 and n_djf_seasons:
                out["djf_precip_in"] = round(float(psum.sum()) / n_djf_seasons, 2)

        by_wy = df.groupby("_wy")
        max_swe = by_wy["swe_in"].max().dropna()
        max_depth = by_wy["snwd_qc"].max().dropna()
        # Only count water years with enough coverage to trust the max.
        coverage = by_wy.size()
        full = coverage[coverage >= 200].index
        max_swe = max_swe[max_swe.index.isin(full)]
        max_depth = max_depth[max_depth.index.isin(full)]
        if len(max_swe):
            out["mean_max_swe_in"] = round(float(max_swe.mean()), 2)
        if len(max_depth):
            out["mean_max_depth_in"] = round(float(max_depth.mean()), 2)
        out["n_years"] = int(len(full))
        out["median_slr"] = round(station_median_slr(hist_qc), 2)
        out["snow_class"] = _snow_class(out["djf_tavg_f"], out["djf_precip_in"])

    if use_cache:
        try:
            STATIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_p.write_text(json.dumps(out, separators=(",", ":")))
        except Exception:
            pass
    return out
