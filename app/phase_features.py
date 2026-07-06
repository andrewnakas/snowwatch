"""Hourly → daily aggregation for precipitation-phase features (v1.6 Tier 1).

ONE implementation imported by BOTH the training backfill
(scripts/backfill_previous_runs.py aux tree, scripts/backfill_gfs_zarr.py)
and the live path (app/met.py fetch_phase_hourly → app/forecast.py).
Train/serve consistency by shared code: the postproc feature docstring
already flags layout drift as the silent killer — never fork this logic.

Features (all per (station, valid_date, lead), from the forecast model's own
hourly values at that lead — no analysis-time data, ever):
  wb_mean_c        daily mean wet-bulb (Stull 2011, from coincident t/rh)
  wb_min_c         daily min wet-bulb
  hours_wb_below_0 count of hours with wet-bulb <= 0°C (phase-hours proxy)
  freezing_level_m daily mean freezing-level height (if the source has it)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .snowphysics import wet_bulb_c

PHASE_COLS = ["wb_mean_c", "wb_min_c", "hours_wb_below_0", "freezing_level_m"]

# Storm gate for phase features, shared by BOTH sides: live (fetch
# fetch_phase_hourly only when consensus precip clears it) and training
# (build_training_data NaN-masks phase columns on rows below it, mirroring
# what inference will actually see). One constant, two call sites — a gate
# that drifts apart is train/serve skew.
PHASE_GATE_PRECIP_MM = 1.0


def storm_gate(mm_precip_mean_mm) -> "pd.Series":
    p = pd.to_numeric(pd.Series(mm_precip_mean_mm), errors="coerce")
    return p >= PHASE_GATE_PRECIP_MM


def daily_phase_aggregates(
    t_c, rh_pct, freezing_level_m=None,
) -> dict[str, Optional[float]]:
    """Aggregates for ONE day's hourly series. Wet-bulb is computed hour by
    hour from coincident t/rh — a wet-bulb of daily means is biased warm on
    exactly the marginal-phase days these features exist to catch."""
    t = pd.to_numeric(pd.Series(t_c), errors="coerce")
    rh = pd.to_numeric(pd.Series(rh_pct), errors="coerce")
    ok = t.notna() & rh.notna()
    out: dict[str, Optional[float]] = dict.fromkeys(PHASE_COLS)
    if ok.sum() >= 4:   # enough hours for a meaningful daily statistic
        wb = np.array([wet_bulb_c(ti, ri) for ti, ri in zip(t[ok], rh[ok])],
                      dtype=float)
        wb = wb[np.isfinite(wb)]
        if wb.size:
            out["wb_mean_c"] = float(np.mean(wb))
            out["wb_min_c"] = float(np.min(wb))
            # Scale hour-count to a 24h-equivalent day so 6-hourly sources
            # (hourly_6) and hourly sources produce the same feature.
            out["hours_wb_below_0"] = float(np.sum(wb <= 0.0) * 24.0 / wb.size)
    if freezing_level_m is not None:
        fl = pd.to_numeric(pd.Series(freezing_level_m), errors="coerce")
        if fl.notna().sum() >= 2:
            out["freezing_level_m"] = float(fl.mean())
    return out


def live_phase_daily(phase_hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily GFS wet-bulb aggregates from met.fetch_phase_hourly output.

    Open-Meteo multi-model responses suffix variables per model; training
    phase features come from the GFS archive, so ONLY the gfs025 columns
    feed the model (NBM columns are fetched for the UI/future use). Output:
    valid_date + wb_mean_c/wb_min_c/hours_wb_below_0 — merge-ready for
    postproc.build_inference_features(phase_daily=...).
    """
    if phase_hourly is None or phase_hourly.empty:
        return pd.DataFrame(columns=["valid_date", "wb_mean_c", "wb_min_c",
                                     "hours_wb_below_0"])
    t_col = next((c for c in ("temperature_2m_gfs025", "temperature_2m")
                  if c in phase_hourly.columns), None)
    rh_col = next((c for c in ("relative_humidity_2m_gfs025", "relative_humidity_2m")
                   if c in phase_hourly.columns), None)
    if t_col is None or rh_col is None:
        return pd.DataFrame(columns=["valid_date", "wb_mean_c", "wb_min_c",
                                     "hours_wb_below_0", "wind10_mean_ms"])
    out = phase_frame_from_hourly(phase_hourly, time_col="datetime",
                                  t_col=t_col, rh_col=rh_col, fl_col=None)
    # wind10: Open-Meteo serves speed directly (km/h) — convert to m/s to
    # match the training tree (|u,v| from the GFS archive).
    w_col = next((c for c in ("wind_speed_10m_gfs025", "wind_speed_10m")
                  if c in phase_hourly.columns), None)
    if w_col is not None:
        w = phase_hourly[["datetime", w_col]].copy()
        w["valid_date"] = pd.to_datetime(w["datetime"]).dt.strftime("%Y-%m-%d")
        wind = (pd.to_numeric(w[w_col], errors="coerce") / 3.6).groupby(
            w["valid_date"]).mean().rename("wind10_mean_ms").reset_index()
        out = out.merge(wind, on="valid_date", how="left")
    else:
        out["wind10_mean_ms"] = np.nan
    return out[["valid_date", "wb_mean_c", "wb_min_c", "hours_wb_below_0",
                "wind10_mean_ms"]]


def phase_frame_from_hourly(
    df: pd.DataFrame, *, time_col: str = "time", t_col: str = "t_c",
    rh_col: str = "rh_pct", fl_col: Optional[str] = "freezing_level_m",
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Daily phase features from an hourly frame.

    Groups by UTC calendar day (+ any extra group_cols, e.g. lead_days) and
    applies daily_phase_aggregates. Returns one row per group with
    valid_date + group_cols + PHASE_COLS.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["valid_date", *group_cols, *PHASE_COLS])
    d = df.copy()
    d["valid_date"] = pd.to_datetime(d[time_col]).dt.strftime("%Y-%m-%d")
    rows = []
    for keys, g in d.groupby(["valid_date", *group_cols]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        fl = g[fl_col] if fl_col and fl_col in g.columns else None
        agg = daily_phase_aggregates(g[t_col], g[rh_col], fl)
        rows.append({"valid_date": keys[0],
                     **dict(zip(group_cols, keys[1:])), **agg})
    return pd.DataFrame(rows)
