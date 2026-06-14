"""Pooled LightGBM post-processor for 24h snowfall (v1.5).

One model pooled across all stations, trained on the Previous-Runs backfill
pairs (`scripts/build_training_data.py`): raw multi-model NWP features plus
station statics and antecedent pack state, target = QC'd observed 24h
snowfall. Pooling is the point — a single station has at most a couple of
winters of (forecast, obs) pairs, but the failure modes the model must learn
(NBM under-forecasting upslope events, SLR drift with temperature, lead-time
skill decay) are shared physics, visible only across hundreds of stations.

Quantile members (p10/p50/p90) give calibrated spread; the point model
(L1 objective) feeds deterministic verification against the raw-NBM baseline.

LightGBM handles NaN features natively, which matters here: HRRR exists only
for leads 1-2, IFS/AIFS carry no snowfall, and AK stations lack NBM entirely.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - exercised only without the wheel
    lgb = None

from .targets import QC_UNRELIABLE

MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "models"

QUANTILES = (0.1, 0.5, 0.9)

NUMERIC_FEATURES = [
    "nbm_snowfall_cm", "nbm_precip_mm", "nbm_tmean_c",
    "hrrr_snowfall_cm", "hrrr_precip_mm", "hrrr_tmean_c",
    "gfs_snowfall_cm", "gfs_precip_mm", "gfs_tmean_c",
    "ifs_precip_mm", "ifs_tmean_c",
    "aifs_precip_mm", "aifs_tmean_c",
    "mm_snow_mean_cm", "mm_snow_std_cm",
    "mm_precip_mean_mm", "mm_precip_std_mm", "mm_n",
    "lead_days", "doy_sin", "doy_cos",
    "elevation_ft", "lat", "lon", "median_slr",
    "obs_depth_issue_in", "obs_swe_issue_in",
]
CATEGORICAL_FEATURES = ["snow_class", "nbm_version"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LGB_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,     # pooled data is plentiful; keep leaves smooth
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
}
NUM_ROUNDS = 600


def nbm_version_for(d: str) -> str:
    """NBM version epoch by valid date (training feature / stratifier)."""
    if d >= "2026-05-05":
        return "v5.0"
    if d >= "2025-05-28":
        return "v4.3"
    return "v4.2"


def build_inference_features(
    mm_fcst: pd.DataFrame, *, statics: dict, last_depth: Optional[float],
    last_swe: Optional[float], issue_date, horizon: int,
) -> pd.DataFrame:
    """Pairs-shaped frame (one row per forecast day) from the live multimodel
    frame, mirroring scripts/build_training_data.py exactly — any drift
    between the two layouts silently degrades the model.

    mm_fcst columns are `{model}_{snowfall|precip|tmean}` (app/met.py);
    training columns are `{model}_{snowfall_cm|precip_mm|tmean_c}`.
    """
    rows = []
    for _, r in (mm_fcst if mm_fcst is not None else pd.DataFrame()).iterrows():
        d = r.get("date")
        d_iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
        lead = (pd.Timestamp(d_iso).date() - issue_date).days
        if lead < 1 or lead > horizon:
            continue
        row = {"valid_date": d_iso, "lead_days": lead}
        for mk in ("nbm", "hrrr", "gfs", "ifs", "aifs"):
            row[f"{mk}_snowfall_cm"] = r.get(f"{mk}_snowfall")
            row[f"{mk}_precip_mm"] = r.get(f"{mk}_precip")
            row[f"{mk}_tmean_c"] = r.get(f"{mk}_tmean")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    snow_cols = [f"{m}_snowfall_cm" for m in ("nbm", "hrrr", "gfs")]
    precip_cols = [f"{m}_precip_mm" for m in ("nbm", "hrrr", "gfs", "ifs", "aifs")]
    for c in snow_cols + precip_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mm_snow_mean_cm"] = df[snow_cols].mean(axis=1)
    df["mm_snow_std_cm"] = df[snow_cols].std(axis=1)
    df["mm_precip_mean_mm"] = df[precip_cols].mean(axis=1)
    df["mm_precip_std_mm"] = df[precip_cols].std(axis=1)
    df["mm_n"] = df[precip_cols].notna().sum(axis=1)
    df["doy"] = pd.to_datetime(df["valid_date"]).dt.dayofyear
    df["elevation_ft"] = statics.get("elevation_ft")
    df["lat"] = statics.get("lat")
    df["lon"] = statics.get("lon")
    df["median_slr"] = statics.get("median_slr")
    df["snow_class"] = statics.get("snow_class")
    df["nbm_version"] = df["valid_date"].map(nbm_version_for)
    df["obs_depth_issue_in"] = last_depth
    df["obs_swe_issue_in"] = last_swe
    return df


def usable_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows whose target is trustworthy: QC-reliable and non-NaN."""
    q = pd.to_numeric(pairs["quality"], errors="coerce").fillna(QC_UNRELIABLE).astype(int)
    ok = (q & QC_UNRELIABLE) == 0
    ok &= pd.to_numeric(pairs["obs_snowfall_in"], errors="coerce").notna()
    return pairs[ok]


def build_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Feature frame in FEATURES order. Missing columns become all-NaN so a
    model trained on full coverage still scores thin early datasets."""
    df = pairs.copy()
    doy = pd.to_numeric(df.get("doy"), errors="coerce")
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    for c in CATEGORICAL_FEATURES:
        df[c] = df.get(c, pd.Series(index=df.index, dtype=object)).astype("category")
    return df[FEATURES]


# Snowfall is a zero-inflated continuous target: >90% of station-days are
# zero, with a heavy positive tail. An L1/L2 point objective collapses to
# predicting ~0 everywhere — it minimizes mean error by winning the no-snow
# majority while being useless on the event days that actually matter (the
# 24-station pilot showed MAE 0.12 but CSI@1in ~0.03). Tweedie is the
# textbook objective for this regime (compound Poisson-Gamma), so it is the
# production point model. We still fit an L1 head for an apples-to-MAE
# comparison in the scorecard, but the blend should consume `point`.
POINT_OBJECTIVE = "tweedie"
TWEEDIE_VARIANCE_POWER = 1.3   # 1->Poisson, 2->Gamma; ~1.3 fits precip-like tails


def train(pairs: pd.DataFrame, *, quantiles=QUANTILES, num_rounds: int = NUM_ROUNDS) -> dict:
    """Train quantile boosters + a Tweedie point booster (+ an L1 head for
    comparison). Returns {name: Booster} with `point` = the production model.

    `pairs` must already be filtered to usable rows and to the training
    period — temporal splitting is the caller's job (see
    scripts/train_postprocessor.py), because honest evaluation needs the
    split to respect time, not random rows.
    """
    if lgb is None:
        raise RuntimeError("lightgbm is required to train the post-processor")
    X = build_features(pairs)
    y = pd.to_numeric(pairs["obs_snowfall_in"], errors="coerce")
    dset = lgb.Dataset(X, label=y, categorical_feature=CATEGORICAL_FEATURES,
                       free_raw_data=False)
    out = {}
    for q in quantiles:
        params = dict(LGB_PARAMS, alpha=q)
        out[f"q{int(q * 100)}"] = lgb.train(params, dset, num_boost_round=num_rounds)

    # Production point model: Tweedie (handles the zero-inflated tail).
    tw_params = dict(LGB_PARAMS, objective="tweedie", metric="tweedie",
                     tweedie_variance_power=TWEEDIE_VARIANCE_POWER)
    tw_params.pop("alpha", None)
    out["point"] = lgb.train(tw_params, dset, num_boost_round=num_rounds)

    # Diagnostic L1 head — kept so the scorecard can show the MAE-optimal
    # model's CSI collapse next to Tweedie's. Not used in production.
    l1_params = dict(LGB_PARAMS, objective="l1", metric="l1")
    l1_params.pop("alpha", None)
    out["point_l1"] = lgb.train(l1_params, dset, num_boost_round=num_rounds)
    return out


def predict(boosters: dict, pairs: pd.DataFrame) -> pd.DataFrame:
    """Predictions per row: point + one column per quantile, snowfall clamped
    at zero and quantiles re-sorted (quantile GBMs can cross)."""
    X = build_features(pairs)
    out = pd.DataFrame(index=pairs.index)
    for name, bst in boosters.items():
        out[name] = np.maximum(0.0, bst.predict(X))
    qcols = sorted((c for c in out.columns if c.startswith("q")),
                   key=lambda c: int(c[1:]))
    if len(qcols) > 1:
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
    return out


def save(boosters: dict, *, models_dir: Path = MODELS_DIR, meta: Optional[dict] = None) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, bst in boosters.items():
        bst.save_model(str(models_dir / f"postproc_{name}.txt"))
    if meta is not None:
        (models_dir / "postproc_meta.json").write_text(json.dumps(meta, indent=2))


def load(*, models_dir: Path = MODELS_DIR) -> Optional[dict]:
    """Load saved boosters; None if missing or lightgbm unavailable."""
    if lgb is None:
        return None
    paths = sorted(models_dir.glob("postproc_*.txt"))
    if not paths:
        return None
    return {p.stem.replace("postproc_", ""): lgb.Booster(model_file=str(p)) for p in paths}
