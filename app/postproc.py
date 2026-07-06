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
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - exercised only without the wheel
    lgb = None

from .calibration import apply_isotonic, lead_bucket
from .targets import QC_UNRELIABLE

MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "models"

QUANTILES = (0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95)

BASE_FEATURES = [
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
# Phase-3 groups, ablatable via train(feature_groups=...). z500_mean_m and
# ens_n_members are deliberately EXCLUDED until the live source matches the
# training source (GEFS-only reduction; z500 has no live feed yet) and
# check_feature_skew clears them — see plan, train/serve skew rule.
FEATURE_GROUPS = {
    "ens": ["ens_snow_mean_cm", "ens_snow_std_cm", "ens_snow_p10_cm",
            "ens_snow_p50_cm", "ens_snow_p90_cm", "ens_prob_pos",
            "ens_precip_mean_mm", "ens_precip_std_mm"],
    "phase": ["wb_mean_c", "wb_min_c", "hours_wb_below_0", "wind10_mean_ms"],
    "hrrr_native": ["hrrr_native_snowfall_cm"],
}
NUMERIC_FEATURES = BASE_FEATURES + [c for g in FEATURE_GROUPS.values() for c in g]
CATEGORICAL_FEATURES = ["snow_class", "nbm_version"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def feature_list(groups: Optional[Iterable[str]] = None) -> list[str]:
    """FEATURES restricted to base + the named groups (None = all groups)."""
    if groups is None:
        return FEATURES
    cols = list(BASE_FEATURES)
    for g in groups:
        cols += FEATURE_GROUPS[g]
    return cols + CATEGORICAL_FEATURES

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


# Live ensemble frame (app/met.py) → the training column names from the
# gefs_ens tree. The gefs_-prefixed columns are the GEFS-only reduction —
# the member population training used; the mixed GEFS+AIFS stats exist for
# the site UI and are NOT fed to the model. Old caches without the gefs_
# columns simply leave the features NaN.
ENS_LIVE_TO_TRAIN = {
    "gefs_ens_snow_mean": "ens_snow_mean_cm", "gefs_ens_snow_std": "ens_snow_std_cm",
    "gefs_ens_snow_p10": "ens_snow_p10_cm", "gefs_ens_snow_p50": "ens_snow_p50_cm",
    "gefs_ens_snow_p90": "ens_snow_p90_cm", "gefs_ens_snow_prob_pos": "ens_prob_pos",
    "gefs_ens_precip_mean": "ens_precip_mean_mm",
    "gefs_ens_precip_std": "ens_precip_std_mm",
}


def build_inference_features(
    mm_fcst: pd.DataFrame, *, statics: dict, last_depth: Optional[float],
    last_swe: Optional[float], issue_date, horizon: int,
    ens_fcst: Optional[pd.DataFrame] = None,
    phase_daily: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Pairs-shaped frame (one row per forecast day) from the live multimodel
    frame, mirroring scripts/build_training_data.py exactly — any drift
    between the two layouts silently degrades the model.

    mm_fcst columns are `{model}_{snowfall|precip|tmean}` (app/met.py);
    training columns are `{model}_{snowfall_cm|precip_mm|tmean_c}`.
    ens_fcst: live ensemble stats (met.fetch_ensemble), mapped via
    ENS_LIVE_TO_TRAIN. phase_daily: daily wet-bulb aggregates keyed by
    valid_date (met.fetch_phase_hourly → phase_features). Absent sources
    leave their columns NaN — LightGBM routes them natively.
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

    # Phase-3 group columns — populated from their live frames when passed,
    # NaN otherwise (hrrr_native has no live source yet; see FEATURE_GROUPS).
    for cols in FEATURE_GROUPS.values():
        for c in cols:
            df[c] = np.nan
    if ens_fcst is not None and not ens_fcst.empty:
        e = ens_fcst.copy()
        e["valid_date"] = e["date"].astype(str)
        e = e.rename(columns=ENS_LIVE_TO_TRAIN)
        keep = ["valid_date"] + [c for c in ENS_LIVE_TO_TRAIN.values() if c in e.columns]
        df = df.drop(columns=[c for c in keep if c != "valid_date"]).merge(
            e[keep], on="valid_date", how="left")
    if phase_daily is not None and not phase_daily.empty:
        p = phase_daily.copy()
        pcols = [c for c in FEATURE_GROUPS["phase"] if c in p.columns]
        df = df.drop(columns=pcols).merge(
            p[["valid_date"] + pcols], on="valid_date", how="left")
        # Gate mirror of build_training_data.py: phase features exist only
        # on rows whose consensus precip clears the shared storm gate.
        from .phase_features import storm_gate
        gated_off = ~storm_gate(df.get("mm_precip_mean_mm")).to_numpy()
        df.loc[gated_off, pcols] = np.nan
    return df


def usable_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows whose target is trustworthy: QC-reliable and non-NaN."""
    q = pd.to_numeric(pairs["quality"], errors="coerce").fillna(QC_UNRELIABLE).astype(int)
    ok = (q & QC_UNRELIABLE) == 0
    ok &= pd.to_numeric(pairs["obs_snowfall_in"], errors="coerce").notna()
    return pairs[ok]


def build_features(pairs: pd.DataFrame, features: Optional[list[str]] = None) -> pd.DataFrame:
    """Feature frame in `features` order (default FEATURES). Missing columns
    become all-NaN so a model trained on full coverage still scores thin
    early datasets."""
    features = features or FEATURES
    df = pairs.copy()
    doy = pd.to_numeric(df.get("doy"), errors="coerce")
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for c in features:
        if c in CATEGORICAL_FEATURES:
            df[c] = df.get(c, pd.Series(index=df.index, dtype=object)).astype("category")
        else:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df[features]


# Snowfall is a zero-inflated continuous target: >90% of station-days are
# zero, with a heavy positive tail. An L1/L2 point objective collapses to
# predicting ~0 everywhere — it minimizes mean error by winning the no-snow
# majority while being useless on the event days that actually matter.
#
# HURDLE (two-part) model: a single regressor — even Tweedie — trades event
# detection for MAE (raising tweedie_variance_power lowers MAE but DROPS POD;
# verified by sweep 2026-06-14). The fix is to separate the two questions:
#   psnow   — binary classifier: P(snowfall >= EVENT_THRESHOLD_IN)
#   amount  — Tweedie regressor: how much snow, given the column
# The production `point` gates the amount by psnow: when an event is likely
# but the regressor under-calls it, floor the call at the event threshold.
# This recovers POD (0.25 -> 0.40 @ p>=0.3) and CSI (0.22 -> 0.28) for a
# negligible MAE cost — the right tradeoff for snow (miss < false alarm).
TWEEDIE_VARIANCE_POWER = 1.3   # 1->Poisson, 2->Gamma; ~1.3 fits precip-like tails
EVENT_THRESHOLD_IN = 1.0       # what counts as a "snow event" for the classifier
HURDLE_GATE = 0.3              # psnow above this floors the call at EVENT_THRESHOLD_IN

# v1.6 multi-threshold decision layer: one exceedance head per operational
# threshold, calibrated (isotonic) and gated per lead bucket. The single-gate
# hurdle above is the legacy fallback for models saved before v1.6.
EVENT_THRESHOLDS = (1.0, 2.0, 6.0, 12.0)
SPW_CAP = 300.0                # scale_pos_weight ceiling: 12in positives are
                               # ~0.1% of rows; uncapped neg/pos destabilizes


def _exc_key(thr: float) -> str:
    return f"exc_{thr:g}".replace(".", "p")


def train(pairs: pd.DataFrame, *, quantiles=QUANTILES, num_rounds: int = NUM_ROUNDS,
          feature_groups: Optional[Iterable[str]] = None) -> dict:
    """Train quantile boosters + a Tweedie point booster (+ an L1 head for
    comparison). Returns {name: Booster} with `point` = the production model.

    `pairs` must already be filtered to usable rows and to the training
    period — temporal splitting is the caller's job (see
    scripts/train_postprocessor.py), because honest evaluation needs the
    split to respect time, not random rows.
    """
    if lgb is None:
        raise RuntimeError("lightgbm is required to train the post-processor")
    X = build_features(pairs, feature_list(feature_groups) if feature_groups is not None else None)
    y = pd.to_numeric(pairs["obs_snowfall_in"], errors="coerce")
    dset = lgb.Dataset(X, label=y, categorical_feature=CATEGORICAL_FEATURES,
                       free_raw_data=False)
    out = {}
    for q in quantiles:
        params = dict(LGB_PARAMS, alpha=q)
        out[f"q{int(q * 100)}"] = lgb.train(params, dset, num_boost_round=num_rounds)

    # Amount head: Tweedie regressor (handles the zero-inflated tail).
    tw_params = dict(LGB_PARAMS, objective="tweedie", metric="tweedie",
                     tweedie_variance_power=TWEEDIE_VARIANCE_POWER)
    tw_params.pop("alpha", None)
    out["amount"] = lgb.train(tw_params, dset, num_boost_round=num_rounds)

    # Exceedance heads: P(snowfall >= T) per operational threshold. Each is
    # isotonic-calibrated downstream (app/calibration.py), so what matters
    # here is ranking quality on the rare positives — hence scale_pos_weight.
    # The 1in head doubles as the legacy `psnow` occurrence probability.
    for thr in EVENT_THRESHOLDS:
        clf_label = (y.to_numpy() >= thr).astype(int)
        pos = clf_label.sum()
        if pos == 0 or pos == clf_label.size:
            continue                      # nothing to learn at this threshold
        spw = min((clf_label.size - pos) / pos, SPW_CAP)
        cdset = lgb.Dataset(X, label=clf_label,
                            categorical_feature=CATEGORICAL_FEATURES,
                            free_raw_data=False)
        clf_params = dict(LGB_PARAMS, objective="binary",
                          metric="binary_logloss", scale_pos_weight=spw)
        clf_params.pop("alpha", None)
        out[_exc_key(thr)] = lgb.train(clf_params, cdset, num_boost_round=num_rounds)

    # Diagnostic L1 head — kept so the scorecard can show the MAE-optimal
    # model's CSI collapse next to the hurdle model. Not used in production.
    l1_params = dict(LGB_PARAMS, objective="l1", metric="l1")
    l1_params.pop("alpha", None)
    out["point_l1"] = lgb.train(l1_params, dset, num_boost_round=num_rounds)
    return out


def predict(boosters: dict, pairs: pd.DataFrame, *, calib: Optional[dict] = None,
            gate: float = HURDLE_GATE) -> pd.DataFrame:
    """Predictions per row. Columns:
      amount   — raw Tweedie regressor output (>=0)
      p1/p2/p6/p12 — calibrated exceedance probabilities (monotone in T)
      psnow    — alias of p1 (legacy UI name)
      point    — production prediction: cascade of amount + gated floors
      q5..q95  — quantile members (clamped >=0, re-sorted; GBMs can cross)

    The cascade generalizes the single hurdle: point = max(amount, largest T
    whose calibrated exceedance probability clears its per-lead-bucket gate).
    `calib` is the postproc_calib.json payload ({"iso": ..., "gates": ...});
    without it (or with pre-v1.6 saved models) the legacy 1in hurdle applies,
    so old model dirs keep producing exactly what they used to.
    """
    # Feature list comes from the boosters themselves (LightGBM stores it):
    # a model trained on an ablated / older feature set predicts correctly
    # regardless of what this module's FEATURES currently says.
    try:
        feats = next(iter(boosters.values())).feature_name()
    except AttributeError:   # test fakes / stubs without feature_name()
        feats = None
    X = build_features(pairs, feats if feats else None)
    raw = {name: bst.predict(X) for name, bst in boosters.items()}
    out = pd.DataFrame(index=pairs.index)

    amount = np.maximum(0.0, raw.get("amount", raw.get("point", np.zeros(len(pairs)))))
    out["amount"] = amount

    present = [t for t in EVENT_THRESHOLDS if _exc_key(t) in raw]
    if present:
        # Monotone raw scores (P(>=T) can't rise with T), calibrate each,
        # then re-enforce monotonicity — separate isotonic curves can cross.
        iso = (calib or {}).get("iso", {})
        cal: dict[float, np.ndarray] = {}
        running = None
        for thr in present:
            p = raw[_exc_key(thr)]
            running = p if running is None else np.minimum(running, p)
            cal[thr] = apply_isotonic(running, iso.get(f"{thr:g}"))
        running = None
        for thr in present:
            running = cal[thr] if running is None else np.minimum(running, cal[thr])
            cal[thr] = running
            out[f"p{thr:g}"] = cal[thr]
        out["psnow"] = cal[present[0]]

        gates = (calib or {}).get("gates", {})
        floor = np.zeros(len(pairs))
        if gates:
            buckets = np.array([lead_bucket(int(ld))
                                for ld in pd.to_numeric(pairs["lead_days"])])
            for thr in present:
                for b, info in gates.get(f"{thr:g}", {}).items():
                    g = (info or {}).get("gate")
                    if g is None:
                        continue
                    hit = (buckets == b) & (cal[thr] >= g)
                    floor = np.where(hit, np.maximum(floor, thr), floor)
        else:
            # No tuned gates yet (evaluation during calibration fitting, or a
            # calib-less deploy): fall back to the legacy 1in hurdle.
            floor = np.where(cal[present[0]] >= gate, EVENT_THRESHOLD_IN, 0.0)
        out["point"] = np.maximum(amount, floor)
    elif "psnow" in raw:
        psnow = raw["psnow"]
        out["psnow"] = psnow
        point = np.where((psnow >= gate) & (amount < EVENT_THRESHOLD_IN),
                         EVENT_THRESHOLD_IN, amount)
        out["point"] = np.maximum(0.0, point)
    else:
        out["point"] = amount

    for name in raw:
        if name.startswith("q") and name[1:].isdigit():
            out[name] = np.maximum(0.0, raw[name])
    qcols = sorted((c for c in out.columns if c.startswith("q") and c[1:].isdigit()),
                   key=lambda c: int(c[1:]))
    if len(qcols) > 1:
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)

    if "point_l1" in raw:
        out["point_l1"] = np.maximum(0.0, raw["point_l1"])
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


def save_calib(calib: dict, *, models_dir: Path = MODELS_DIR) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "postproc_calib.json").write_text(json.dumps(calib, indent=2))


def load_calib(*, models_dir: Path = MODELS_DIR) -> Optional[dict]:
    """Isotonic curves + tuned gates (postproc_calib.json); None pre-v1.6."""
    p = models_dir / "postproc_calib.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
