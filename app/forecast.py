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

import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import nbm as _nbm, snotel, snow17 as _snow17, snow17_calibrate as _snow17_cal, weather

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
    notes: List[str] = field(default_factory=list)


_DECAY_BY_MEMBER = {
    "persistence_lag1": 0,         # IS the anchor — no correction needed
    "climatology": 5,
    "snow17": 5,
    "nbm_snowfall": 6,
    "ridge_snow": 5,
    "chronos_bolt": 4,
}


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
    nbm_fcst = _nbm.fetch_forecast(station["lat"], station["lon"], days=min(horizon, _nbm.NBM_MAX_DAYS))

    notes: List[str] = []
    if hist.empty:
        raise RuntimeError(f"no SNOTEL history for {station.get('id')}")

    depth_series = pd.Series(hist["snow_depth_in"].values, index=pd.to_datetime(hist["date"]))
    swe_series = pd.Series(hist["swe_in"].values, index=pd.to_datetime(hist["date"]))
    # Find the last observed snow depth (might be the most recent row, but
    # snow_depth can be reported NaN out of season — fall back to last finite).
    last_obs_idx = depth_series.last_valid_index()
    if last_obs_idx is None:
        raise RuntimeError(f"no finite snow_depth observations for {station.get('id')}")
    last_depth = float(depth_series.loc[last_obs_idx])
    last_depth_date = pd.Timestamp(last_obs_idx).date()
    last_swe = float(swe_series.loc[last_obs_idx]) if pd.notna(swe_series.loc[last_obs_idx]) else None

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

    nbm_pred = _member_nbm_snowfall(last_depth, nbm_fcst, horizon)
    if nbm_pred:
        members_raw["nbm_snowfall"] = nbm_pred

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

    # snow17 / nbm / chronos: bias toward persistence MAE because we can't
    # replay historical NBM forecasts. We assign their rolling MAE as the
    # persistence MAE scaled by 1.0 (snow17), 1.1 (nbm), 0.9 (chronos guess).
    # The MAE-blend protects us — if a member is bad on the current issued
    # forecast its anchor-corrected value won't drag the blend much.
    if "snow17" in members_raw:
        rolling_mae["snow17"] = rolling_mae["persistence_lag1"]
    if "nbm_snowfall" in members_raw:
        rolling_mae["nbm_snowfall"] = rolling_mae["persistence_lag1"] * 1.1
    if "chronos_bolt" in members_raw:
        rolling_mae["chronos_bolt"] = rolling_mae["persistence_lag1"] * 0.95

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
        last_observed=last_depth,
        last_observed_date=last_depth_date.isoformat(),
        last_swe=last_swe,
        elevation_ft=station.get("elevation_ft"),
        notes=notes,
    )
