"""SnowWatch ensemble forecast engine.

Members
-------
1. persistence_lag1  yhat[t+h] = snow_depth[t]  (also the anchor target).
2. climatology       per-DOY mean snow depth from the station's record.
3. snow17            Full Anderson 1976 SNOW-17 conceptual model — seasonal
                     temperature-index melt, ATI-driven refreeze, rain-on-snow
                     turbulent energy balance, Hedstrom-Pomeroy fresh-snow
                     density, Anderson compaction. See app/snow17.py.
4. nbm_snowfall      cumulative NBM snowfall above last observed depth,
                     decayed by a positive-degree-day melt term.
5. ridge_snow        LightGBM (falls back to Ridge) on lagged depth, DOY,
                     and rolling precip / tmean / SWE / snowfall covariates.
                     Direct multi-step (one model per horizon day).
6. chronos_bolt      Amazon Chronos-Bolt zero-shot foundation model on the
                     depth series. Optional; gracefully skipped if missing.

Each member's horizon=1 prediction is anchored to the last observed snow
depth with a linearly decaying correction:
    yhat'[h] = yhat[h] - (yhat[1] - obs) * max(0, 1 - (h-1)/decay)
Decay horizons follow RiverWatch2's pattern — longer for members with
persistent initial-condition bias, shorter for fast-mean-reverting members.

Blend
-----
Each member is rolling-validated on its training window (held-out h-step
forecasts vs observed). Member weight = 1 / (rolling_mae + eps), normalized.
Members that fail validation (insufficient history, NaN MAE) get weight 0.
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import met as _met, nbm as _nbm, nws as _nws, snotel, snow17 as _snow17, snow17_calibrate as _snow17_cal, targets as _targets, weather

try:
    import lightgbm as _lgb
    _LGB_OK = True
except Exception:
    _LGB_OK = False

try:
    from chronos import BaseChronosPipeline
    _CHRONOS_OK = True
except Exception:
    _CHRONOS_OK = False


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HORIZON_DAYS = 7
TRAIN_LOOKBACK_DAYS = 365 * 10           # 10 years of SNOTEL is plenty for ridge/clim
HIST_FLOOR = date(1980, 1, 1)
LAGS = [1, 2, 3, 5, 7, 14, 30]
PRECIP_WINDOWS = [1, 3, 7, 14]
TEMP_WINDOWS = [3, 7, 14]
PRECIP_LAGS = [1, 2, 3, 5, 7]
SNOW_DENSITY_DEFAULT = 0.30              # SWE/depth ratio when no obs pair available
# NBM short-range member uses these constants directly (it's a simple
# add-snowfall-with-melt rule, not a full snowpack model — when both NBM and
# snow17 agree the blend gets a robust point; when they diverge the stacker
# can route through the more skillful one for the regime).
MELT_FACTOR_IN_PER_DEGC_DAY = 0.10
FREEZE_THRESHOLD_C = 1.0

_CHRONOS_PIPE = None


def _f_to_c(t_f):
    if t_f is None:
        return None
    return (t_f - 32.0) * 5.0 / 9.0


def _anchor_to_observed(pred: List[float], obs: float, *, decay_h: int) -> List[float]:
    """Linearly fade the h=1 bias toward zero over `decay_h` steps."""
    if not pred or obs is None or not np.isfinite(obs):
        return pred
    delta = float(pred[0]) - float(obs)
    if not np.isfinite(delta) or delta == 0.0:
        return pred
    out = []
    for h, v in enumerate(pred, start=1):
        weight = max(0.0, 1.0 - (h - 1) / max(1, decay_h))
        out.append(max(0.0, float(v) - delta * weight))
    return out


def _doy_climatology(series: pd.Series, dates: pd.DatetimeIndex) -> Optional[pd.Series]:
    """Per-DOY mean of `series`, smoothed with a wrap-around 31-day window."""
    if len(series) < 540:
        return None
    df = pd.DataFrame({"v": np.asarray(series, dtype=float), "doy": dates.dayofyear}).dropna()
    if len(df) < 540:
        return None
    base = df.groupby("doy")["v"].mean()
    full = base.reindex(range(1, 367))
    pad_l = full.iloc[-15:].copy(); pad_l.index = range(-14, 1)
    pad_r = full.iloc[:15].copy(); pad_r.index = range(367, 382)
    padded = pd.concat([pad_l, full, pad_r]).sort_index()
    smooth = padded.rolling(window=31, min_periods=5, center=True).mean()
    out = smooth.reindex(range(1, 367))
    if out.notna().sum() < 200:
        return None
    return out.fillna(float(np.nanmean(out.values)))


# ---------- ensemble members ----------------------------------------------

def _member_persistence(last_obs: float, horizon: int) -> List[float]:
    return [float(last_obs)] * horizon


def _member_climatology(clim: Optional[pd.Series], anchor_date: date, horizon: int,
                        last_obs: Optional[float]) -> List[float]:
    if clim is None:
        return []
    out = []
    for h in range(1, horizon + 1):
        d = anchor_date + timedelta(days=h)
        doy = d.timetuple().tm_yday
        v = float(clim.get(doy, np.nan))
        if not np.isfinite(v):
            v = float(last_obs) if last_obs is not None else 0.0
        out.append(max(0.0, v))
    return out


def _member_snow17(
    last_depth: float, last_swe: Optional[float], wx_forecast: pd.DataFrame,
    *, params: _snow17.Snow17Params,
) -> List[float]:
    """Anderson 1976 SNOW-17 conceptual model. See app/snow17.py."""
    return _snow17.predict(
        last_depth_in=last_depth, last_swe_in=last_swe,
        wx_forecast=wx_forecast, params=params,
    )


def _build_nbm_wx_for_snow17(
    nbm_df: pd.DataFrame, wx_fcst: pd.DataFrame, horizon: int,
) -> pd.DataFrame:
    """Construct a SNOW-17-ready forecast DataFrame from NBM precip + temp,
    augmented with Open-Meteo radiation/wind where NBM doesn't provide them.

    Lets us drive the SNOW-17 physics with the NWS official blended forecast
    (which has independent skill on precip / temp) rather than Open-Meteo's
    own blend. NBM caps at 11 days; days beyond fall back to Open-Meteo.

    With no NBM at all (Alaska stations, NBM outage) returns empty so the
    member is skipped — falling back to wx_fcst here would duplicate the
    `snow17` member and double SNOW-17's weight in the blend.
    """
    if nbm_df is None or nbm_df.empty:
        return pd.DataFrame()
    wx_by_date: Dict[str, dict] = {}
    if wx_fcst is not None and not wx_fcst.empty:
        for _, r in wx_fcst.iterrows():
            d = r["date"]
            d = d.isoformat() if hasattr(d, "isoformat") else str(d)
            wx_by_date[d] = r.to_dict()

    rows: List[dict] = []
    for _, nb in nbm_df.head(horizon).iterrows():
        d = nb["date"]
        d_iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
        wx = wx_by_date.get(d_iso, {})
        snowfall_cm = nb.get("nbm_snowfall_sum")
        precip_mm = nb.get("nbm_precip_sum")
        # If NBM precip is missing but snowfall is present, back-fill precip
        # using a 10:1 ratio so SNOW-17 sees realistic accumulation.
        if (precip_mm is None or not pd.notna(precip_mm)) and snowfall_cm is not None and pd.notna(snowfall_cm):
            precip_mm = float(snowfall_cm) * 10.0 / 10.0  # cm → mm (1cm snow @ 10:1 = 1mm SWE)
        rows.append({
            "date": d,
            "precipitation_sum": precip_mm,
            "temperature_2m_mean": nb.get("nbm_tmean"),
            "temperature_2m_max": nb.get("nbm_tmax"),
            "temperature_2m_min": nb.get("nbm_tmin"),
            "snowfall_sum": snowfall_cm,
            "shortwave_radiation_sum": wx.get("shortwave_radiation_sum"),
            "windspeed_10m_max": wx.get("windspeed_10m_max"),
        })

    # If horizon > NBM length, append Open-Meteo rows for the tail.
    nbm_dates = {r["date"].isoformat() if hasattr(r["date"], "isoformat") else str(r["date"]) for r in rows}
    if wx_fcst is not None and not wx_fcst.empty and len(rows) < horizon:
        for _, wr in wx_fcst.iterrows():
            d_iso = wr["date"].isoformat() if hasattr(wr["date"], "isoformat") else str(wr["date"])
            if d_iso in nbm_dates:
                continue
            if len(rows) >= horizon:
                break
            rows.append(wr.to_dict())
    return pd.DataFrame(rows)


def _member_nbm_snow17(
    last_depth: float, last_swe: Optional[float], nbm_df: pd.DataFrame,
    wx_fcst: pd.DataFrame, *, params: _snow17.Snow17Params, horizon: int,
) -> List[float]:
    """SNOW-17 driven by NBM (NWS official) precip + temperature.

    Combines the physical realism of SNOW-17 with the calibrated forecast
    skill of the NWS national blend. Distinct from `snow17` which uses
    Open-Meteo blend weather.
    """
    wx_like = _build_nbm_wx_for_snow17(nbm_df, wx_fcst, horizon)
    if wx_like is None or wx_like.empty:
        return []
    return _snow17.predict(
        last_depth_in=last_depth, last_swe_in=last_swe,
        wx_forecast=wx_like, params=params,
    )


def _member_postproc_snowfall(
    last_depth: float, last_swe: Optional[float], mm_fcst: pd.DataFrame,
    station: dict, hist_qc: pd.DataFrame, *, today: date, horizon: int,
    ens_fcst: Optional[pd.DataFrame] = None,
) -> tuple[List[float], List[dict]]:
    """Pooled LightGBM post-processed snowfall accumulated onto the pack with
    the same degree-day melt as nbm_snowfall (v1.5); v1.6 adds the calibrated
    multi-threshold cascade + a daily snowfall detail list for the archive.

    Off unless SW_ENABLE_POSTPROC=1 AND trained models sit in data/models —
    this member must never ship on smoke-trained weights by accident.
    """
    if os.environ.get("SW_ENABLE_POSTPROC") != "1":
        return [], []
    if mm_fcst is None or mm_fcst.empty:
        return [], []
    from . import postproc as _postproc
    boosters = _postproc.load()
    if not boosters or ("point" not in boosters and "amount" not in boosters):
        return [], []
    statics = _targets.station_statics(station, hist_qc)

    # Phase features (v1.6 Tier 1), storm-gated: fetch the hourly t/rh only
    # when some forecast day clears the shared gate — per-day masking to the
    # same gate happens inside build_inference_features, mirroring training.
    model_feats = set(next(iter(boosters.values())).feature_name())
    phase_daily = None
    if any(c in model_feats for c in (_postproc.FEATURE_GROUPS["phase"]
                                      + _postproc.FEATURE_GROUPS["wind_uv"])):
        from . import met as _met
        from .phase_features import live_phase_daily, storm_gate
        precip_cols = [c for c in mm_fcst.columns if c.endswith("_precip")]
        if precip_cols and storm_gate(mm_fcst[precip_cols].mean(axis=1)).any():
            phase_daily = live_phase_daily(
                _met.fetch_phase_hourly(station["lat"], station["lon"], days=horizon))

    # Upperair features (release-3): the per-build bulk Zarr prefetch
    # (scripts/fetch_live_upperair.py, run at shard start) — read this
    # station's slice; absent file/entry degrades to NaN columns.
    upperair = None
    if any(c in model_feats for c in ("z500_mean_m", "hrrr_native_snowfall_cm")):
        try:
            cache = json.loads(
                (Path(__file__).resolve().parents[1] / "data" / "cache"
                 / "live_upperair.json").read_text())
            upperair = (cache.get("stations") or {}).get(station.get("triplet"))
        except Exception:
            upperair = None

    feats = _postproc.build_inference_features(
        mm_fcst, statics=statics, last_depth=last_depth, last_swe=last_swe,
        issue_date=today, horizon=horizon, ens_fcst=ens_fcst,
        phase_daily=phase_daily, upperair=upperair)
    if feats.empty:
        return [], []
    preds = _postproc.predict(boosters, feats, calib=_postproc.load_calib())
    daily: List[dict] = []
    for i, (_, frow) in enumerate(feats.iterrows()):
        d = {"date": frow["valid_date"], "snowfall_in": round(float(preds["point"].iloc[i]), 3)}
        for c in ("p1", "p2", "p6", "p12", "q10", "q50", "q90"):
            if c in preds.columns and np.isfinite(preds[c].iloc[i]):
                d[c] = round(float(preds[c].iloc[i]), 3)
        daily.append(d)
    snow_by_lead = dict(zip(feats["lead_days"], preds["point"]))
    tmean_by_lead = dict(zip(feats["lead_days"],
                             pd.to_numeric(feats.get("nbm_tmean_c"), errors="coerce")))
    out: List[float] = []
    depth = float(last_depth)
    for h in range(1, horizon + 1):
        snow_in = float(snow_by_lead.get(h) or 0.0)
        t = tmean_by_lead.get(h)
        melt_in = (max(0.0, float(t)) * MELT_FACTOR_IN_PER_DEGC_DAY
                   if t is not None and np.isfinite(t) and float(t) > FREEZE_THRESHOLD_C else 0.0)
        depth = max(0.0, depth + snow_in - melt_in)
        out.append(depth)
    return out, daily


def _member_nbm_snowfall(last_depth: float, nbm_df: pd.DataFrame, horizon: int) -> List[float]:
    """Cumulative NBM snowfall added to last observed depth, with degree-day melt."""
    if nbm_df is None or nbm_df.empty:
        return []
    depth = float(last_depth)
    out: List[float] = []
    for i in range(horizon):
        if i >= len(nbm_df):
            # NBM caps at 11 days; beyond that hold the last value.
            out.append(out[-1] if out else depth)
            continue
        row = nbm_df.iloc[i]
        snowfall_cm = row.get("nbm_snowfall_sum") or 0.0
        tmean_c = row.get("nbm_tmean")
        snow_in = snowfall_cm / 2.54
        if tmean_c is None or not np.isfinite(tmean_c):
            tmean_c = 0.0
        melt_in = max(0.0, tmean_c) * MELT_FACTOR_IN_PER_DEGC_DAY if tmean_c > FREEZE_THRESHOLD_C else 0.0
        depth = max(0.0, depth + snow_in - melt_in)
        out.append(round(depth, 3))
    return out


def _build_ridge_features(
    hist: pd.DataFrame, wx_hist: pd.DataFrame, *, horizon: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a daily feature panel keyed on the SNOTEL history date.

    Columns:
      depth_lag_{L}     L in LAGS
      swe_lag_{L}
      doy_sin / doy_cos
      precip_sum_w{W}, tmean_w{W}, snowfall_sum_w{W}
      precip_lag_{L}
    """
    df = hist.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["tmean_c_obs"] = df["tavg_f"].apply(_f_to_c)

    feats = pd.DataFrame({"date": df["date"]})
    feats["depth"] = df["snow_depth_in"]
    feats["swe"] = df["swe_in"]
    feats["precip_in"] = df["precip_in"]
    feats["tmean_c_obs"] = df["tmean_c_obs"]

    for L in LAGS:
        feats[f"depth_lag_{L}"] = df["snow_depth_in"].shift(L)
        feats[f"swe_lag_{L}"] = df["swe_in"].shift(L)

    feats["doy"] = df["date"].dt.dayofyear
    feats["doy_sin"] = np.sin(2 * math.pi * feats["doy"] / 366.0)
    feats["doy_cos"] = np.cos(2 * math.pi * feats["doy"] / 366.0)

    if wx_hist is not None and not wx_hist.empty:
        wx = wx_hist.copy()
        wx["date"] = pd.to_datetime(wx["date"])
        feats = feats.merge(wx, on="date", how="left")
    for w in PRECIP_WINDOWS:
        feats[f"precip_w{w}"] = feats.get("precipitation_sum", pd.Series([np.nan] * len(feats))).rolling(w, min_periods=1).sum()
        feats[f"snowfall_w{w}"] = feats.get("snowfall_sum", pd.Series([np.nan] * len(feats))).rolling(w, min_periods=1).sum()
    for w in TEMP_WINDOWS:
        feats[f"tmean_w{w}"] = feats.get("temperature_2m_mean", pd.Series([np.nan] * len(feats))).rolling(w, min_periods=1).mean()
    for L in PRECIP_LAGS:
        feats[f"precip_lag_{L}"] = feats.get("precipitation_sum", pd.Series([np.nan] * len(feats))).shift(L)
        feats[f"snowfall_lag_{L}"] = feats.get("snowfall_sum", pd.Series([np.nan] * len(feats))).shift(L)

    feat_cols = [c for c in feats.columns if c not in ("date", "depth")]
    return feats, feat_cols


def _fit_regressor(Xtr: np.ndarray, ytr: np.ndarray, *, alpha: float = 1.0):
    if _LGB_OK and len(Xtr) >= 200:
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": max(20, len(Xtr) // 40),
            "feature_fraction": 0.7,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "num_threads": 1,
        }
        try:
            ds = _lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
            booster = _lgb.train(params, ds, num_boost_round=120)
            class _LGBWrap:
                def __init__(self, b): self._b = b
                def predict(self, X): return self._b.predict(X)
            return _LGBWrap(booster)
        except Exception:
            pass
    sc = StandardScaler().fit(Xtr)
    rg = Ridge(alpha=alpha, random_state=0).fit(sc.transform(Xtr), ytr)
    class _RidgeWrap:
        def __init__(self, sc, m): self._sc = sc; self._m = m
        def predict(self, X): return self._m.predict(self._sc.transform(X))
    return _RidgeWrap(sc, rg)


def _member_ridge_snow(
    feats: pd.DataFrame, feat_cols: List[str], *, horizon: int,
) -> Tuple[List[float], Dict[int, float]]:
    """Direct multi-step ridge: a separate model fits each horizon h in [1..H].

    Returns (yhat_list_len_H, rolling_mae_by_h).
    """
    df = feats.dropna(subset=["depth"]).copy()
    if len(df) < 120:
        return [], {}
    # Coerce features to numeric. SNOTEL/Open-Meteo occasionally returns string
    # cells for variables they don't report; pandas keeps them as object dtype
    # and `np.isnan` on object arrays raises ValueError. `errors="coerce"`
    # promotes them to NaN, which the lag/window/imputation logic already
    # tolerates downstream.
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X_full = df[feat_cols].to_numpy(dtype=float, copy=True)
    y_full = pd.to_numeric(df["depth"], errors="coerce").to_numpy(dtype=float)

    # Holdout for rolling MAE: last 30 days, predicting each horizon.
    n = len(df)
    holdout = min(30, max(7, n // 20))
    yhat_list: List[float] = []
    mae_by_h: Dict[int, float] = {}

    for h in range(1, horizon + 1):
        # Shift y by h so X[i] predicts depth at i+h.
        y_shift = df["depth"].shift(-h).values
        valid = ~np.isnan(y_shift) & ~np.isnan(X_full).any(axis=1)
        Xv = X_full[valid]
        yv = y_shift[valid]
        if len(Xv) < 60:
            continue
        cutoff = max(40, len(Xv) - holdout)
        Xtr, ytr = Xv[:cutoff], yv[:cutoff]
        Xho, yho = Xv[cutoff:], yv[cutoff:]
        try:
            model = _fit_regressor(Xtr, ytr)
            yhat_ho = model.predict(Xho)
            if len(yho) > 0:
                mae_by_h[h] = float(np.mean(np.abs(yhat_ho - yho)))
            # Issue: use the very last fully-observed feature row.
            last_row = X_full[-1]
            if np.isnan(last_row).any():
                continue
            yhat_list.append(max(0.0, float(model.predict(last_row.reshape(1, -1))[0])))
        except Exception:
            continue
    return yhat_list, mae_by_h


def _member_chronos(depth_series: pd.Series, horizon: int) -> List[float]:
    if not _CHRONOS_OK:
        return []
    global _CHRONOS_PIPE
    if _CHRONOS_PIPE is None:
        try:
            import torch
            _CHRONOS_PIPE = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32,
            )
        except Exception:
            return []
    series = depth_series.dropna().astype(float).values[-512:]
    if len(series) < 60:
        return []
    try:
        import torch
        with torch.no_grad():
            quantiles, mean = _CHRONOS_PIPE.predict_quantiles(
                context=torch.tensor(series, dtype=torch.float32),
                prediction_length=horizon,
                quantile_levels=[0.5],
            )
        out = mean[0].cpu().numpy().tolist() if mean.ndim > 1 else mean.cpu().numpy().tolist()
        return [max(0.0, float(v)) for v in out]
    except Exception:
        return []


# ---------- top-level forecast --------------------------------------------

@dataclass
class StationForecast:
    station_id: str
    triplet: str
    issued_at: str
    horizon_days: int
    history: List[dict]
    members: Dict[str, List[dict]]
    blend: List[dict]
    weights: Dict[str, float]
    rolling_mae: Dict[str, float]
    last_observed: Optional[float]
    last_observed_date: Optional[str]
    last_swe: Optional[float]
    elevation_ft: Optional[float]
    daily_forecast: List[dict] = field(default_factory=list)
    nws_divergence: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    # Raw multi-model + ensemble-spread rows (one dict per forecast day),
    # archived by scripts/archive_forecasts.py for training & verification.
    multimodel: List[dict] = field(default_factory=list)
    ensemble_stats: List[dict] = field(default_factory=list)
    # Post-processor daily snowfall detail (point/quantiles/exceedance
    # probs), archived as source="postproc" — the as-issued rows the live
    # scorecard verifies. Depth members alone can't verify daily snowfall.
    postproc_daily: List[dict] = field(default_factory=list)


_DECAY_BY_MEMBER = {
    "persistence_lag1": 0,         # IS the anchor — no correction needed
    "climatology": 5,
    "snow17": 5,
    "nbm_snow17": 5,
    "nbm_snowfall": 6,
    "postproc_snowfall": 6,
    "ridge_snow": 5,
    "chronos_bolt": 4,
}

# Walk-forward member MAEs are structural skill numbers — they drift on week
# scales, not build scales — but computing them (snow17/chronos backtests) is
# the heaviest CPU in the 6h build. Cache per (station, horizon) with a TTL.
MAE_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "member_mae"


def _mae_cache_ttl_days() -> float:
    try:
        return float(os.environ.get("SW_MAE_CACHE_DAYS", "7"))
    except ValueError:
        return 7.0


def _load_mae_cache(triplet: str, horizon: int) -> Optional[Dict[str, float]]:
    ttl = _mae_cache_ttl_days()
    if ttl <= 0:
        return None
    p = MAE_CACHE_DIR / f"{triplet.replace(':', '_')}_h{horizon}.json"
    if not p.exists() or (time.time() - p.stat().st_mtime) > ttl * 86400:
        return None
    try:
        return {str(k): float(v) for k, v in json.loads(p.read_text()).items()}
    except Exception:
        return None


def _save_mae_cache(triplet: str, horizon: int, rolling_mae: Dict[str, float]) -> None:
    if _mae_cache_ttl_days() <= 0:
        return
    try:
        MAE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = MAE_CACHE_DIR / f"{triplet.replace(':', '_')}_h{horizon}.json"
        p.write_text(json.dumps(
            {k: float(v) for k, v in rolling_mae.items() if v is not None and np.isfinite(v)},
            separators=(",", ":")))
    except Exception:
        pass


def _to_point_list(start: date, values: List[float]) -> List[dict]:
    return [{"date": (start + timedelta(days=h)).isoformat(), "snow_depth_in": float(v)}
            for h, v in enumerate(values, start=1)]


def _c_to_f(t_c):
    if t_c is None or not np.isfinite(t_c):
        return None
    return t_c * 9.0 / 5.0 + 32.0


def _build_daily_forecast(
    start: date, horizon: int, wx_fcst: pd.DataFrame, nbm_fcst: pd.DataFrame,
) -> List[dict]:
    """One row per forecast day with the inputs the chart panel renders as
    bars (snowfall) and a temperature line (tmin/tmean/tmax in F).

    Source priority: NBM (calibrated CONUS) → Open-Meteo blend → null. NBM caps
    at 11 days, so days 12+ fall through to Open-Meteo (which carries the full
    horizon anyway).
    """
    nbm_by_date: Dict[str, dict] = {}
    if nbm_fcst is not None and not nbm_fcst.empty:
        for _, r in nbm_fcst.iterrows():
            d = r["date"]
            d = d.isoformat() if hasattr(d, "isoformat") else str(d)
            nbm_by_date[d] = r.to_dict()
    wx_by_date: Dict[str, dict] = {}
    if wx_fcst is not None and not wx_fcst.empty:
        for _, r in wx_fcst.iterrows():
            d = r["date"]
            d = d.isoformat() if hasattr(d, "isoformat") else str(d)
            wx_by_date[d] = r.to_dict()

    out: List[dict] = []
    for h in range(1, horizon + 1):
        d = (start + timedelta(days=h)).isoformat()
        nb = nbm_by_date.get(d, {})
        wx = wx_by_date.get(d, {})

        def _pick(*candidates):
            for c in candidates:
                if c is None:
                    continue
                try:
                    f = float(c)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(f):
                    return f
            return None

        snowfall_cm = _pick(nb.get("nbm_snowfall_sum"), wx.get("snowfall_sum"))
        precip_mm = _pick(nb.get("nbm_precip_sum"), wx.get("precipitation_sum"))
        tmean_c = _pick(nb.get("nbm_tmean"), wx.get("temperature_2m_mean"))
        tmax_c = _pick(nb.get("nbm_tmax"), wx.get("temperature_2m_max"))
        tmin_c = _pick(nb.get("nbm_tmin"), wx.get("temperature_2m_min"))
        pop = _pick(nb.get("nbm_pop_mean"))
        source = "nbm" if d in nbm_by_date else ("openmeteo" if d in wx_by_date else "none")

        out.append({
            "date": d,
            "snowfall_in": (snowfall_cm / 2.54) if snowfall_cm is not None else None,
            "precip_in": (precip_mm / 25.4) if precip_mm is not None else None,
            "tmean_f": _c_to_f(tmean_c),
            "tmax_f": _c_to_f(tmax_c),
            "tmin_f": _c_to_f(tmin_c),
            "pop_pct": pop,
            "source": source,
        })
    return out


def _build_nws_divergence(
    start: date, horizon: int,
    nws_df: pd.DataFrame, nbm_df: pd.DataFrame,
    blend_vals: List[float], members_anchored: Dict[str, List[float]],
    last_depth: float, station_elev_ft: Optional[float],
) -> dict:
    """Quantify why SnowWatch's blend differs from the NWS public forecast.

    For each forecast day d in 1..horizon we compute:
      nws_implied_depth   = last_depth + Σ NWS snowfall - Σ degree-day melt
      sw_blend_depth      = ensemble blend (already anchored to last_depth)
      Δ_in                = sw_blend_depth - nws_implied_depth

    Then attribute the gap. The biggest drivers are usually:
      - NWS vs blended snowfall (precip volume disagreement)
      - NWS vs blended melt (temperature/snow-level disagreement)
      - Member voting (other members override the NBM/NWS-driven members)

    Returns a dict with `daily` (per-day attribution rows), `summary`
    (over-horizon totals), and `available` (False if NWS is offline).
    """
    if nws_df is None or nws_df.empty:
        return {"available": False, "reason": "no NWS forecast available"}

    by_date_nws: Dict[str, dict] = {}
    for _, r in nws_df.iterrows():
        d = r["date"]
        d_iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
        by_date_nws[d_iso] = r.to_dict()
    by_date_nbm: Dict[str, dict] = {}
    if nbm_df is not None and not nbm_df.empty:
        for _, r in nbm_df.iterrows():
            d = r["date"]
            d_iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
            by_date_nbm[d_iso] = r.to_dict()

    def _f_to_c(t_f):
        return (float(t_f) - 32.0) * 5.0 / 9.0 if t_f is not None and np.isfinite(t_f) else None

    nws_implied = float(last_depth)
    daily: List[dict] = []
    cum_sw_snow_in = 0.0
    cum_nws_snow_in = 0.0
    cum_sw_melt_in = 0.0
    cum_nws_melt_in = 0.0

    for h in range(1, horizon + 1):
        d_iso = (start + timedelta(days=h)).isoformat()
        nws = by_date_nws.get(d_iso)
        nbm = by_date_nbm.get(d_iso, {})

        nws_snow_in = nws.get("nws_snowfall_in") if nws else None
        nws_tmean_c = _f_to_c(nws.get("nws_tmean_f")) if nws else None
        nws_snow_lvl_ft = nws.get("nws_snow_level_ft") if nws else None

        # Phase check: NWS publishes snowfall at the grid-point elevation, but
        # if the snowLevel rises above the station the NWS forecast will show
        # 0 snow even though precip falls. We surface that explicitly.
        above_snow_line = (
            station_elev_ft is not None and nws_snow_lvl_ft is not None
            and station_elev_ft >= nws_snow_lvl_ft
        )

        # NWS-implied depth evolution: same simple add+melt as nbm_snowfall,
        # so comparison is to a like-for-like "if you trusted NWS verbatim"
        # baseline. Real ensemble blend uses richer physics; that's the whole
        # point of why we differ.
        nws_melt_in = max(0.0, (nws_tmean_c or 0.0)) * MELT_FACTOR_IN_PER_DEGC_DAY \
            if (nws_tmean_c is not None and nws_tmean_c > FREEZE_THRESHOLD_C) else 0.0
        nws_implied = max(0.0, nws_implied + (nws_snow_in or 0.0) - nws_melt_in)

        sw_depth = float(blend_vals[h - 1]) if h - 1 < len(blend_vals) else None
        delta = (sw_depth - nws_implied) if sw_depth is not None else None

        # Blended snowfall — what SnowWatch *implicitly* used: average across
        # members that consume snowfall (snow17, nbm_snow17, nbm_snowfall)
        # weighted equally for simplicity (the actual MAE-weighted blend is
        # already in `sw_depth`; this is just the input-side picture).
        sw_snow_in = (nbm.get("nbm_snowfall_sum") or 0.0) / 2.54
        sw_tmean_c = nbm.get("nbm_tmean")
        sw_melt_in = max(0.0, (sw_tmean_c or 0.0)) * MELT_FACTOR_IN_PER_DEGC_DAY \
            if (sw_tmean_c is not None and sw_tmean_c > FREEZE_THRESHOLD_C) else 0.0

        cum_sw_snow_in += sw_snow_in
        cum_nws_snow_in += nws_snow_in or 0.0
        cum_sw_melt_in += sw_melt_in
        cum_nws_melt_in += nws_melt_in

        # Per-day driver attribution: snowfall vs melt vs other.
        snow_diff = sw_snow_in - (nws_snow_in or 0.0)
        melt_diff = nws_melt_in - sw_melt_in   # NWS more melt = lower NWS depth = positive Δ
        explained = snow_diff + melt_diff
        unexplained = (delta - explained) if delta is not None else None

        daily.append({
            "date": d_iso,
            "sw_depth_in": sw_depth,
            "nws_implied_depth_in": round(nws_implied, 2),
            "delta_in": round(delta, 2) if delta is not None else None,
            "nws_snowfall_in": nws_snow_in,
            "sw_snowfall_in": round(sw_snow_in, 2),
            "snow_diff_in": round(snow_diff, 2),
            "nws_tmean_f": nws.get("nws_tmean_f") if nws else None,
            "nws_snow_level_ft": nws_snow_lvl_ft,
            "above_nws_snow_line": above_snow_line,
            "melt_diff_in": round(melt_diff, 2),
            "unexplained_in": round(unexplained, 2) if unexplained is not None else None,
        })

    final_delta = (blend_vals[-1] - nws_implied) if blend_vals else None
    summary = {
        "horizon_days": horizon,
        "final_sw_depth_in": round(float(blend_vals[-1]), 2) if blend_vals else None,
        "final_nws_implied_depth_in": round(nws_implied, 2),
        "final_delta_in": round(final_delta, 2) if final_delta is not None else None,
        "cum_nws_snowfall_in": round(cum_nws_snow_in, 2),
        "cum_sw_snowfall_in": round(cum_sw_snow_in, 2),
        "cum_snow_diff_in": round(cum_sw_snow_in - cum_nws_snow_in, 2),
        "cum_nws_melt_in": round(cum_nws_melt_in, 2),
        "cum_sw_melt_in": round(cum_sw_melt_in, 2),
        "cum_melt_diff_in": round(cum_nws_melt_in - cum_sw_melt_in, 2),
        "office": str(nws_df["nws_office"].iloc[0]) if "nws_office" in nws_df.columns and len(nws_df) else None,
    }
    # Headline reason — the dominant driver of the gap.
    # The NWS-implied baseline uses naive add-snowfall + degree-day-melt, so
    # most of the gap typically comes from physics our model includes that the
    # NWS public forecast doesn't render: density compaction, refreeze, ROS
    # energy balance, snow-level variability across the day. Tease apart:
    #   (a) input-side disagreement: snowfall and melt
    #   (b) physics-side disagreement: everything else
    if final_delta is None:
        reason = "no comparable forecast"
    else:
        snow_diff = summary["cum_snow_diff_in"]
        melt_diff = summary["cum_melt_diff_in"]
        physics_residual = final_delta - snow_diff - melt_diff
        snow_share = abs(snow_diff)
        melt_share = abs(melt_diff)
        physics_share = abs(physics_residual)
        biggest = max(snow_share, melt_share, physics_share)
        bits = []
        if snow_share > 0.2 and snow_share >= biggest * 0.5:
            sign = "more" if snow_diff > 0 else "less"
            bits.append(f"SnowWatch sees {sign} snowfall ({summary['cum_sw_snowfall_in']:.1f}″ vs NWS {summary['cum_nws_snowfall_in']:.1f}″)")
        if melt_share > 0.2 and melt_share >= biggest * 0.5:
            sign = "more" if melt_diff > 0 else "less"
            bits.append(f"NWS expects {sign} melt ({summary['cum_nws_melt_in']:.1f}″ vs {summary['cum_sw_melt_in']:.1f}″)")
        if physics_share > 0.5 and physics_share >= biggest * 0.5:
            direction = "down" if physics_residual < 0 else "up"
            bits.append(
                f"physics adjustments push {direction} {abs(physics_residual):.1f}″ "
                f"(density compaction, refreeze, snow-level/elevation effects)"
            )
        if not bits:
            reason = "near-identical to NWS"
        else:
            reason = "; ".join(bits)
    summary["headline"] = reason
    summary["physics_residual_in"] = round(
        final_delta - summary["cum_snow_diff_in"] - summary["cum_melt_diff_in"], 2,
    ) if final_delta is not None else None
    return {"available": True, "summary": summary, "daily": daily}


def _walkforward_snow17_mae(
    hist: pd.DataFrame, wx_hist: pd.DataFrame, *,
    horizon: int, params: _snow17.Snow17Params,
    n_issues: int = 8, stride: int = 7,
) -> Optional[float]:
    """True walk-forward MAE for SNOW-17 using observed historical weather.

    Issue the model at `n_issues` evenly-spaced dates in the last
    ~`n_issues * stride` days; at each issue point we seed from the
    SNOTEL-observed (depth, SWE) on the issue date, run snow17 forward for
    `horizon` days using Open-Meteo *historical* weather, and compare to
    observed snow_depth. This is the right proxy for snow17 skill because
    in production we use NBM/blend forecast weather, not perfect-foresight.
    The historical-weather run separates the model's structural skill
    (which is what the MAE-blend weights against) from the forecast input
    quality (which is dealt with by the anchor + decay).
    """
    if hist is None or hist.empty or wx_hist is None or wx_hist.empty:
        return None
    h = hist[["date", "snow_depth_in", "swe_in"]].copy()
    h["date"] = pd.to_datetime(h["date"]).dt.date
    w = wx_hist.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.date
    df = h.merge(w, on="date", how="inner").dropna(subset=["snow_depth_in"], how="any")
    if len(df) < horizon + n_issues * stride + 5:
        return None
    df = df.sort_values("date").reset_index(drop=True)

    end_idx = len(df) - horizon - 1
    issues = [end_idx - i * stride for i in range(n_issues) if end_idx - i * stride > horizon]
    if not issues:
        return None

    errors: List[float] = []
    for issue in issues:
        seed = df.iloc[issue]
        d0 = float(seed["snow_depth_in"])
        swe0 = float(seed["swe_in"]) if pd.notna(seed["swe_in"]) else None
        wx_window = df.iloc[issue + 1 : issue + 1 + horizon].copy()
        if len(wx_window) < horizon:
            continue
        try:
            pred = _snow17.predict(
                last_depth_in=d0, last_swe_in=swe0,
                wx_forecast=wx_window, params=params,
            )
        except Exception:
            continue
        if not pred or len(pred) < horizon:
            continue
        truth = df.iloc[issue + 1 : issue + 1 + horizon]["snow_depth_in"].to_numpy(dtype=float)
        if not np.isfinite(truth).all():
            continue
        errors.append(float(np.mean(np.abs(np.array(pred[:horizon]) - truth))))
    return float(np.mean(errors)) if errors else None


def _walkforward_nbm_proxy_mae(
    hist: pd.DataFrame, wx_hist: pd.DataFrame, *,
    horizon: int, n_issues: int = 8, stride: int = 7,
) -> Optional[float]:
    """Backtest the NBM member with archive snowfall as a stand-in for NBM.

    NBM doesn't expose a public archive, but `_member_nbm_snowfall` is just
    "add forecast snowfall to last observed depth, decay with degree-days".
    Replaying it against Open-Meteo archive snowfall + tmean gives a realistic
    upper bound on NBM skill — actual NBM does slightly better day-1 and is
    similar by day-7, so this is conservative.
    """
    if hist is None or hist.empty or wx_hist is None or wx_hist.empty:
        return None
    h = hist[["date", "snow_depth_in"]].copy()
    h["date"] = pd.to_datetime(h["date"]).dt.date
    w = wx_hist.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.date
    df = h.merge(w, on="date", how="inner").dropna(subset=["snow_depth_in"], how="any")
    if len(df) < horizon + n_issues * stride + 5:
        return None
    df = df.sort_values("date").reset_index(drop=True)

    end_idx = len(df) - horizon - 1
    issues = [end_idx - i * stride for i in range(n_issues) if end_idx - i * stride > horizon]
    if not issues:
        return None

    errors: List[float] = []
    for issue in issues:
        d0 = float(df.iloc[issue]["snow_depth_in"])
        wx_window = df.iloc[issue + 1 : issue + 1 + horizon]
        # Map Open-Meteo columns onto the structure _member_nbm_snowfall expects.
        proxy = pd.DataFrame({
            "date": wx_window["date"].values,
            "nbm_snowfall_sum": wx_window["snowfall_sum"].values,   # already cm
            "nbm_tmean": wx_window["temperature_2m_mean"].values,
        })
        try:
            pred = _member_nbm_snowfall(d0, proxy, horizon)
        except Exception:
            continue
        if not pred or len(pred) < horizon:
            continue
        truth = wx_window["snow_depth_in"].to_numpy(dtype=float) if "snow_depth_in" in wx_window.columns \
            else df.iloc[issue + 1 : issue + 1 + horizon]["snow_depth_in"].to_numpy(dtype=float)
        if not np.isfinite(truth).all():
            continue
        errors.append(float(np.mean(np.abs(np.array(pred[:horizon]) - truth))))
    return float(np.mean(errors)) if errors else None


def _walkforward_chronos_mae(
    series: pd.Series, *, horizon: int, n_issues: int = 6, stride: int = 7,
) -> Optional[float]:
    """Walk-forward MAE for Chronos-Bolt. Chronos is the priciest member
    (foundation-model forward pass) so we keep n_issues small."""
    s = series.dropna()
    if len(s) < horizon + n_issues * stride + 30:
        return None
    end_idx = len(s) - horizon - 1
    issues = [end_idx - i * stride for i in range(n_issues) if end_idx - i * stride > horizon]
    if not issues:
        return None
    errors: List[float] = []
    for issue in issues:
        train = s.iloc[: issue + 1]
        truth = s.iloc[issue + 1 : issue + 1 + horizon].to_numpy(dtype=float)
        if len(truth) < horizon or not np.isfinite(truth).all():
            continue
        try:
            pred = _member_chronos(train, horizon)
        except Exception:
            continue
        if not pred or len(pred) < horizon:
            continue
        errors.append(float(np.mean(np.abs(np.array(pred[:horizon]) - truth))))
    return float(np.mean(errors)) if errors else None


def _rolling_validation_mae(
    series: pd.Series, member_fn, *, horizon: int, holdout: int = 14,
) -> Optional[float]:
    """Walk-forward MAE for a stateless member callable (uses last-N days to issue)."""
    s = series.dropna()
    if len(s) < holdout + horizon + 10:
        return None
    errors: List[float] = []
    for end in range(len(s) - horizon - holdout, len(s) - horizon):
        train = s.iloc[: end + 1]
        truth = s.iloc[end + 1 : end + 1 + horizon].values
        if len(truth) < horizon:
            break
        try:
            pred = member_fn(train)
            if not pred or len(pred) < horizon:
                continue
            errors.append(float(np.mean(np.abs(np.array(pred[:horizon]) - truth))))
        except Exception:
            continue
    return float(np.mean(errors)) if errors else None


def forecast_station(
    station: dict, *, horizon: int = HORIZON_DAYS,
) -> StationForecast:
    """Run the ensemble for one SNOTEL station and return a StationForecast.

    Expects `station` with at least: id, triplet, name, state, lat, lon,
    elevation_ft. Pulls SNOTEL history + Open-Meteo archive/forecast + NBM
    forecast, runs every member, MAE-weighted blends, anchors.
    """
    today = date.today()
    hist_start = max(HIST_FLOOR, today - timedelta(days=TRAIN_LOOKBACK_DAYS))
    hist_end = today

    hist = snotel.fetch_history(station["triplet"], hist_start, hist_end)
    hist = hist.sort_values("date").reset_index(drop=True) if not hist.empty else hist

    wx_hist = weather.fetch_history(station["lat"], station["lon"], hist_start, hist_end)
    wx_fcst = weather.fetch_forecast(station["lat"], station["lon"], days=horizon)
    # Multi-model fetch (NBM+HRRR+GFS+IFS+AIFS in one call). The NBM columns
    # feed the existing members via the compat shim; the full frame plus
    # ensemble spread stats are archived for training/verification and feed
    # the pooled post-processor. Falls back to the legacy single-model NBM
    # fetcher if the multi-model call is degraded/empty.
    mm_fcst = _met.fetch_multimodel(station["lat"], station["lon"], days=horizon)
    ens_fcst = _met.fetch_ensemble(station["lat"], station["lon"], days=horizon)
    nbm_fcst = _met.nbm_compat(mm_fcst)
    if nbm_fcst.empty:
        nbm_fcst = _nbm.fetch_forecast(station["lat"], station["lon"], days=min(horizon, _nbm.NBM_MAX_DAYS))
    nws_fcst = _nws.fetch_forecast(station["lat"], station["lon"], days=horizon)

    notes: List[str] = []
    if hist.empty:
        raise RuntimeError(f"no SNOTEL history for {station.get('id')}")

    # QC the history before anything touches it: the despiked series is what
    # members train/validate on, and the anchor must never be a sensor spike
    # (a false echo on the last day would otherwise contaminate every member
    # via the anchoring correction).
    hist_qc = _targets.qc_daily_series(hist)
    n_spikes = int((hist_qc["qc_flags"].astype(int) & _targets.QC_SPIKE_REMOVED).astype(bool).sum())
    if n_spikes:
        notes.append(f"qc removed {n_spikes} depth spikes")
    hist = hist_qc.drop(columns=["snow_depth_in"]).rename(columns={"snwd_qc": "snow_depth_in"})

    depth_series = pd.Series(hist["snow_depth_in"].values, index=pd.to_datetime(hist["date"]))
    last_depth, last_depth_date, last_swe = _targets.last_reliable_depth(hist_qc)
    if last_depth is None:
        raise RuntimeError(f"no finite snow_depth observations for {station.get('id')}")

    # Members ----------------------------------------------------------------
    members_raw: Dict[str, List[float]] = {}

    members_raw["persistence_lag1"] = _member_persistence(last_depth, horizon)

    clim = _doy_climatology(depth_series, pd.DatetimeIndex(depth_series.index))
    clim_pred = _member_climatology(clim, today, horizon, last_depth)
    if clim_pred:
        members_raw["climatology"] = clim_pred

    snow17_params = _snow17.default_params(
        elev_m=(float(station.get("elevation_ft")) * 0.3048) if station.get("elevation_ft") else None,
        lat_deg=float(station.get("lat") or 45.0),
    )
    triplet = station.get("triplet")
    if triplet:
        snow17_params = _snow17_cal.apply_cached(triplet, snow17_params)
    s17 = _member_snow17(last_depth, last_swe, wx_fcst.head(horizon), params=snow17_params)
    if s17:
        members_raw["snow17"] = s17

    # SNOW-17 driven by NBM (NWS) precip + temp instead of Open-Meteo blend.
    # When NBM and Open-Meteo disagree on storm timing this member tracks the
    # NWS view, and the MAE-blend lets the ensemble pick the winner.
    s17_nbm = _member_nbm_snow17(
        last_depth, last_swe, nbm_fcst, wx_fcst,
        params=snow17_params, horizon=horizon,
    )
    if s17_nbm:
        members_raw["nbm_snow17"] = s17_nbm

    nbm_pred = _member_nbm_snowfall(last_depth, nbm_fcst, horizon)
    if nbm_pred:
        members_raw["nbm_snowfall"] = nbm_pred

    pp_pred, pp_daily = _member_postproc_snowfall(
        last_depth, last_swe, mm_fcst, station, hist_qc, today=today,
        horizon=horizon, ens_fcst=ens_fcst)
    if pp_pred:
        members_raw["postproc_snowfall"] = pp_pred

    feats, feat_cols = _build_ridge_features(hist, wx_hist, horizon=horizon)
    ridge_pred, ridge_mae_by_h = _member_ridge_snow(feats, feat_cols, horizon=horizon)
    if ridge_pred and len(ridge_pred) >= horizon:
        members_raw["ridge_snow"] = ridge_pred[:horizon]

    chronos_pred = _member_chronos(depth_series, horizon)
    if chronos_pred:
        members_raw["chronos_bolt"] = chronos_pred[:horizon]

    # Rolling validation for the simple members (persistence/climatology/snow17/
    # nbm/chronos use the same depth-only walk-forward; ridge already returned
    # its own per-h MAE). For snow17/nbm we can't easily replay historical
    # weather forecasts, so we approximate with depth-only persistence-of-state.
    rolling_mae: Dict[str, float] = {}
    cached_mae = _load_mae_cache(triplet, horizon) if triplet else None
    if cached_mae is not None and all(k in cached_mae for k in members_raw):
        rolling_mae = {k: float(cached_mae[k]) for k in members_raw}
    else:
        def _pers_fn(train):
            v = float(train.iloc[-1])
            return [v] * horizon
        rolling_mae["persistence_lag1"] = _rolling_validation_mae(depth_series, _pers_fn, horizon=horizon) or 0.0

        if "climatology" in members_raw:
            # Backtest climatology with leave-future-out by reusing the same clim curve.
            def _clim_fn(train):
                anchor = pd.Timestamp(train.index[-1]).date()
                return _member_climatology(clim, anchor, horizon, float(train.iloc[-1]))
            rolling_mae["climatology"] = _rolling_validation_mae(depth_series, _clim_fn, horizon=horizon) or 0.0

        if ridge_mae_by_h:
            rolling_mae["ridge_snow"] = float(np.mean(list(ridge_mae_by_h.values())))

        # snow17 / nbm / chronos: replace the old persistence-scaled proxies with
        # real walk-forward MAEs. snow17 uses observed Open-Meteo weather as a
        # perfect-foresight stand-in for forecast input; nbm uses Open-Meteo
        # snowfall as a stand-in for the NBM snowfall feed; chronos walks
        # forward on the raw depth series. Each falls back to the persistence
        # proxy if it doesn't have enough usable history.
        if "snow17" in members_raw:
            s17_mae = _walkforward_snow17_mae(hist, wx_hist, horizon=horizon, params=snow17_params)
            rolling_mae["snow17"] = s17_mae if s17_mae is not None else rolling_mae["persistence_lag1"]
        if "nbm_snow17" in members_raw:
            # NBM has no public historical archive; the structural snow17-skill
            # number transfers since both members run identical physics. NBM
            # inputs typically beat Open-Meteo on day-1 by a small margin, so we
            # nudge the proxy 5% better to reflect that real-world bias.
            nbm17_proxy = rolling_mae.get("snow17") or rolling_mae["persistence_lag1"]
            rolling_mae["nbm_snow17"] = nbm17_proxy * 0.95
        if "nbm_snowfall" in members_raw:
            nbm_mae = _walkforward_nbm_proxy_mae(hist, wx_hist, horizon=horizon)
            rolling_mae["nbm_snowfall"] = nbm_mae if nbm_mae is not None else rolling_mae["persistence_lag1"] * 1.1
        if "postproc_snowfall" in members_raw:
            # No live backtest until the forecast archive accumulates; seed
            # slightly better than nbm_snowfall (the pooled model exists to
            # beat it) and let continuous verification correct the weight.
            pp_proxy = rolling_mae.get("nbm_snowfall") or rolling_mae["persistence_lag1"]
            rolling_mae["postproc_snowfall"] = pp_proxy * 0.9
        if "chronos_bolt" in members_raw:
            c_mae = _walkforward_chronos_mae(depth_series, horizon=horizon)
            rolling_mae["chronos_bolt"] = c_mae if c_mae is not None else rolling_mae["persistence_lag1"] * 0.95
        if triplet:
            _save_mae_cache(triplet, horizon, rolling_mae)

    # Anchor every member to last observed depth.
    members_anchored: Dict[str, List[float]] = {}
    for name, pred in members_raw.items():
        if name == "persistence_lag1":
            members_anchored[name] = pred
        else:
            decay = _DECAY_BY_MEMBER.get(name, 5)
            members_anchored[name] = _anchor_to_observed(pred, last_depth, decay_h=decay)

    # MAE-weighted blend.
    eps = 0.01
    weights: Dict[str, float] = {}
    total_w = 0.0
    for name in members_anchored:
        mae = rolling_mae.get(name)
        if mae is None or not np.isfinite(mae):
            continue
        w = 1.0 / (mae + eps)
        weights[name] = w
        total_w += w
    for k in list(weights.keys()):
        weights[k] = weights[k] / total_w if total_w > 0 else 1.0 / max(1, len(weights))

    blend_vals = [0.0] * horizon
    if weights:
        for name, w in weights.items():
            for h, v in enumerate(members_anchored[name][:horizon]):
                blend_vals[h] += w * float(v)
    else:
        blend_vals = members_anchored["persistence_lag1"][:horizon]

    # Compute a notional blend rolling MAE for display = weighted avg of member MAEs.
    blend_mae = sum(weights.get(k, 0.0) * rolling_mae.get(k, 0.0) for k in weights) if weights else None
    if blend_mae is not None:
        rolling_mae["ensemble_blend"] = float(blend_mae)

    issued_at = today.isoformat()
    history_payload = [
        {"date": (d.isoformat() if hasattr(d, "isoformat") else str(d)),
         "snow_depth_in": (float(v) if v is not None and np.isfinite(v) else None),
         "swe_in": (float(s) if s is not None and np.isfinite(s) else None)}
        for d, v, s in zip(hist["date"], hist["snow_depth_in"], hist["swe_in"])
    ]

    members_payload: Dict[str, List[dict]] = {}
    for name, vals in members_anchored.items():
        members_payload[name] = _to_point_list(today, vals[:horizon])
    blend_payload = _to_point_list(today, blend_vals)

    daily_forecast = _build_daily_forecast(today, horizon, wx_fcst, nbm_fcst)
    nws_divergence = _build_nws_divergence(
        today, horizon, nws_fcst, nbm_fcst,
        blend_vals, members_anchored,
        last_depth=last_depth,
        station_elev_ft=station.get("elevation_ft"),
    )

    def _records(df: pd.DataFrame) -> List[dict]:
        if df is None or df.empty:
            return []
        out = df.copy()
        out["date"] = out["date"].map(lambda d: d.isoformat() if hasattr(d, "isoformat") else str(d))
        return [
            {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in r.items()}
            for r in out.to_dict(orient="records")
        ]

    return StationForecast(
        station_id=str(station["id"]),
        triplet=station["triplet"],
        issued_at=issued_at,
        horizon_days=horizon,
        history=history_payload,
        members=members_payload,
        blend=blend_payload,
        weights=weights,
        rolling_mae=rolling_mae,
        daily_forecast=daily_forecast,
        nws_divergence=nws_divergence,
        last_observed=last_depth,
        last_observed_date=last_depth_date.isoformat(),
        last_swe=last_swe,
        elevation_ft=station.get("elevation_ft"),
        notes=notes,
        multimodel=_records(mm_fcst),
        ensemble_stats=_records(ens_fcst),
        postproc_daily=pp_daily,
    )
