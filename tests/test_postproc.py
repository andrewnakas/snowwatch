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
