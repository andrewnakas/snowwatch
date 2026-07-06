"""Unit tests for the pooled post-processor feature layer (app/postproc.py)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import postproc


def _mm_fcst(n, start):
    rows = []
    for i in range(n):
        r = {"date": start + timedelta(days=i)}
        for mk in ("nbm", "hrrr", "gfs", "ifs", "aifs"):
            r[f"{mk}_snowfall"] = 5.0 if mk in ("nbm", "hrrr", "gfs") else None
            r[f"{mk}_precip"] = 6.0
            r[f"{mk}_tmean"] = -2.0
        rows.append(r)
    return pd.DataFrame(rows)


STATICS = {"elevation_ft": 9000.0, "lat": 39.5, "lon": -106.0,
           "median_slr": 14.0, "snow_class": "continental"}


class TestBuildInferenceFeatures:
    def test_layout_matches_training(self):
        issue = date(2026, 1, 10)
        feats = postproc.build_inference_features(
            _mm_fcst(8, issue + timedelta(days=1)), statics=STATICS,
            last_depth=30.0, last_swe=8.0, issue_date=issue, horizon=7)
        assert list(feats["lead_days"]) == [1, 2, 3, 4, 5, 6, 7]
        # Every model feature column the trainer uses must exist.
        missing = [c for c in postproc.FEATURES
                   if c not in feats.columns and c not in ("doy_sin", "doy_cos")]
        assert missing == []
        # build_features must accept the frame without error and keep order.
        X = postproc.build_features(feats)
        assert list(X.columns) == postproc.FEATURES

    def test_issue_state_constant_across_leads(self):
        issue = date(2026, 1, 10)
        feats = postproc.build_inference_features(
            _mm_fcst(7, issue + timedelta(days=1)), statics=STATICS,
            last_depth=30.0, last_swe=8.0, issue_date=issue, horizon=7)
        assert (feats["obs_depth_issue_in"] == 30.0).all()
        assert (feats["obs_swe_issue_in"] == 8.0).all()

    def test_consensus_excludes_no_snow_models(self):
        issue = date(2026, 1, 10)
        feats = postproc.build_inference_features(
            _mm_fcst(3, issue + timedelta(days=1)), statics=STATICS,
            last_depth=10.0, last_swe=2.0, issue_date=issue, horizon=3)
        # snowfall consensus over nbm/hrrr/gfs only -> mean 5.0, std 0
        assert np.allclose(feats["mm_snow_mean_cm"], 5.0)
        # precip consensus over all five
        assert (feats["mm_n"] == 5).all()

    def test_rows_outside_horizon_dropped(self):
        issue = date(2026, 1, 10)
        mm = _mm_fcst(10, issue)  # includes lead-0 (issue day) and beyond horizon
        feats = postproc.build_inference_features(
            mm, statics=STATICS, last_depth=10.0, last_swe=2.0,
            issue_date=issue, horizon=7)
        assert feats["lead_days"].min() == 1
        assert feats["lead_days"].max() <= 7

    def test_empty_input(self):
        feats = postproc.build_inference_features(
            pd.DataFrame(), statics=STATICS, last_depth=10.0, last_swe=2.0,
            issue_date=date(2026, 1, 10), horizon=7)
        assert feats.empty


class TestNbmVersion:
    def test_epochs(self):
        assert postproc.nbm_version_for("2025-05-01") == "v4.2"
        assert postproc.nbm_version_for("2025-05-28") == "v4.3"
        assert postproc.nbm_version_for("2026-05-05") == "v5.0"


class _ConstBooster:
    """Minimal stand-in for a lgb.Booster: returns a fixed per-row value."""
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class TestHurdleGating:
    """The production `point` is the hurdle combination of amount + psnow.
    Verified deterministically with constant fake boosters — no training."""

    def _pairs(self, n=3):
        # build_features only needs the columns it coerces; an empty-ish frame
        # with the right length is enough for the const boosters.
        return pd.DataFrame({"doy": [1] * n, "obs_snowfall_in": [0.0] * n})

    def test_likely_event_underforecast_is_floored(self):
        # amount below threshold but psnow above gate -> floored to threshold.
        b = {"amount": _ConstBooster(0.2),
             "psnow": _ConstBooster(postproc.HURDLE_GATE + 0.1)}
        out = postproc.predict(b, self._pairs())
        assert np.allclose(out["point"], postproc.EVENT_THRESHOLD_IN)
        assert np.allclose(out["amount"], 0.2)  # raw amount preserved

    def test_unlikely_event_not_floored(self):
        # psnow below gate -> amount passes through untouched.
        b = {"amount": _ConstBooster(0.2),
             "psnow": _ConstBooster(postproc.HURDLE_GATE - 0.1)}
        out = postproc.predict(b, self._pairs())
        assert np.allclose(out["point"], 0.2)

    def test_confident_amount_not_lowered(self):
        # amount already above threshold -> gate never lowers it.
        b = {"amount": _ConstBooster(5.0),
             "psnow": _ConstBooster(0.9)}
        out = postproc.predict(b, self._pairs())
        assert np.allclose(out["point"], 5.0)

    def test_no_psnow_falls_back_to_amount(self):
        # Older saved models without the occurrence head still work.
        b = {"amount": _ConstBooster(0.2)}
        out = postproc.predict(b, self._pairs())
        assert "psnow" not in out.columns
        assert np.allclose(out["point"], 0.2)

    def test_gate_override(self):
        b = {"amount": _ConstBooster(0.2), "psnow": _ConstBooster(0.5)}
        # gate above psnow -> no floor
        assert np.allclose(postproc.predict(b, self._pairs(), gate=0.6)["point"], 0.2)
        # gate below psnow -> floor
        assert np.allclose(postproc.predict(b, self._pairs(), gate=0.4)["point"],
                           postproc.EVENT_THRESHOLD_IN)


class TestMultiThresholdCascade:
    """v1.6 decision layer: per-threshold exceedance heads, monotone
    calibrated probabilities, per-lead-bucket gated floors."""

    def _pairs(self, n=3, lead=1):
        return pd.DataFrame({"doy": [1] * n, "lead_days": [lead] * n,
                             "obs_snowfall_in": [0.0] * n})

    def _heads(self, p1, p2, p6, p12, amount=0.2):
        return {"amount": _ConstBooster(amount),
                "exc_1": _ConstBooster(p1), "exc_2": _ConstBooster(p2),
                "exc_6": _ConstBooster(p6), "exc_12": _ConstBooster(p12)}

    @staticmethod
    def _gates(**per_thr):
        # per_thr like t6=0.5 -> gates for all buckets at that threshold
        g = {}
        for k, v in per_thr.items():
            thr = k[1:]
            g[thr] = {b: {"gate": v} for b in ("1-2", "3-4", "5-7")}
        return g

    def test_probabilities_monotone_even_when_raw_heads_cross(self):
        # Raw heads deliberately non-monotone: p6 raw > p1 raw.
        b = self._heads(p1=0.3, p2=0.2, p6=0.8, p12=0.1)
        out = postproc.predict(b, self._pairs())
        assert (out["p2"] <= out["p1"] + 1e-12).all()
        assert (out["p6"] <= out["p2"] + 1e-12).all()
        assert (out["p12"] <= out["p6"] + 1e-12).all()

    def test_cascade_floors_at_highest_cleared_threshold(self):
        # amount tiny, p6 confident: point must be floored at 6in.
        b = self._heads(p1=0.95, p2=0.95, p6=0.9, p12=0.01)
        calib = {"iso": {}, "gates": self._gates(t1=0.3, t2=0.3, t6=0.5, t12=0.9)}
        out = postproc.predict(b, self._pairs(lead=2), calib=calib)
        assert np.allclose(out["point"], 6.0)

    def test_cascade_respects_lead_bucket(self):
        # Gate exists only for bucket 1-2; a lead-5 row must not be floored.
        b = self._heads(p1=0.95, p2=0.95, p6=0.9, p12=0.01)
        calib = {"iso": {}, "gates": {"6": {"1-2": {"gate": 0.5},
                                            "3-4": {"gate": None},
                                            "5-7": {"gate": None}}}}
        near = postproc.predict(b, self._pairs(lead=1), calib=calib)
        far = postproc.predict(b, self._pairs(lead=5), calib=calib)
        assert np.allclose(near["point"], 6.0)
        assert np.allclose(far["point"], 0.2)

    def test_amount_above_floor_wins(self):
        b = self._heads(p1=0.95, p2=0.95, p6=0.9, p12=0.01, amount=8.0)
        calib = {"iso": {}, "gates": self._gates(t6=0.5)}
        out = postproc.predict(b, self._pairs(lead=1), calib=calib)
        assert np.allclose(out["point"], 8.0)

    def test_no_calib_falls_back_to_legacy_hurdle(self):
        b = self._heads(p1=0.5, p2=0.4, p6=0.01, p12=0.0)
        out = postproc.predict(b, self._pairs())
        # p1 0.5 >= HURDLE_GATE 0.3 -> floored at the 1in event threshold
        assert np.allclose(out["point"], postproc.EVENT_THRESHOLD_IN)
        assert "psnow" in out.columns  # legacy alias preserved

    def test_isotonic_applied_from_calib(self):
        b = self._heads(p1=0.6, p2=0.5, p6=0.4, p12=0.0)
        # Curve that maps everything to ~0 kills the floor at 1in.
        crush = {"x": [0.0, 1.0], "y": [0.0, 0.0]}
        calib = {"iso": {"1": crush, "2": crush, "6": crush, "12": crush},
                 "gates": self._gates(t1=0.3)}
        out = postproc.predict(b, self._pairs(lead=1), calib=calib)
        assert np.allclose(out["point"], 0.2)
        assert np.allclose(out["p1"], 0.0)
