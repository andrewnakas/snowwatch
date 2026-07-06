"""Unit tests for app/verification.py against hand-computed cases."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import verification as V


class TestMaeBias:
    def test_hand_computed(self):
        out = V.mae_bias([1.0, 2.0, 3.0], [2.0, 2.0, 1.0])
        # errors: +1, 0, -2 → MAE 1.0, bias -1/3, RMSE sqrt(5/3)
        assert out["mae"] == pytest.approx(1.0)
        assert out["bias"] == pytest.approx(-1 / 3)
        assert out["rmse"] == pytest.approx(np.sqrt(5 / 3))
        assert out["n"] == 3

    def test_nan_filtered(self):
        out = V.mae_bias([1.0, np.nan, 3.0], [2.0, 2.0, np.nan])
        assert out["n"] == 1
        assert out["mae"] == pytest.approx(1.0)

    def test_empty(self):
        assert V.mae_bias([], [])["mae"] is None


class TestEventConditional:
    def test_only_event_days_counted(self):
        # obs: [0, 0, 5], pred: [3, 0, 7] → only day 3 counts → MAE 2
        out = V.event_conditional_mae([0, 0, 5], [3, 0, 7], threshold_in=0.5)
        assert out["mae"] == pytest.approx(2.0)
        assert out["n"] == 1


class TestCsiPodFar:
    def test_hand_computed(self):
        # threshold 1: obs events at idx 0,1,2; fcst events at idx 0,3
        # hits=1 (idx0), misses=2 (idx1,2), false_alarms=1 (idx3)
        obs = [2.0, 1.5, 1.0, 0.0, 0.0]
        fcst = [3.0, 0.0, 0.5, 2.0, 0.0]
        out = V.csi_pod_far(obs, fcst, threshold_in=1.0)
        assert out["hits"] == 1 and out["misses"] == 2 and out["false_alarms"] == 1
        assert out["csi"] == pytest.approx(1 / 4)
        assert out["pod"] == pytest.approx(1 / 3)
        assert out["far"] == pytest.approx(1 / 2)
        assert out["freq_bias"] == pytest.approx(2 / 3)


class TestCrps:
    def test_perfect_median_lower_than_biased(self):
        rng = np.random.default_rng(0)
        y = rng.gamma(2, 2, 500)
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        # "Good" quantiles from the true distribution vs shifted ones.
        good = np.quantile(rng.gamma(2, 2, 100000), levels)
        bad = good + 3.0
        crps_good = V.crps_from_quantiles(y, np.tile(good, (500, 1)), levels)
        crps_bad = V.crps_from_quantiles(y, np.tile(bad, (500, 1)), levels)
        assert crps_good < crps_bad

    def test_pinball_identity_single_median(self):
        # With only the median, CRPS≈2×pinball(0.5)=mean absolute error.
        y = [1.0, 3.0]
        q = np.array([[2.0], [2.0]])
        out = V.crps_from_quantiles(y, q, [0.5])
        assert out == pytest.approx(1.0)

    def test_gaussian_crps_sharp_beats_flat(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 1000)
        sharp = V.crps_gaussian(y, np.zeros(1000), np.ones(1000))
        flat = V.crps_gaussian(y, np.zeros(1000), 5 * np.ones(1000))
        # Known closed form: CRPS of N(0,1) vs N(0,1) draws ≈ 0.5642*σ*(...)
        assert sharp < flat
        assert sharp == pytest.approx(0.5637, abs=0.05)


class TestBrier:
    def test_hand_computed(self):
        # obs events: [1, 0]; probs [0.8, 0.4] → BS = (0.04 + 0.16)/2 = 0.1
        out = V.brier_skill([2.0, 0.0], [0.8, 0.4], threshold_in=1.0)
        assert out["brier"] == pytest.approx(0.1)
        # clim freq 0.5 → BS_clim = 0.25 → BSS = 0.6
        assert out["bss"] == pytest.approx(0.6)


class TestBootstrap:
    def test_clear_winner_significant(self):
        rng = np.random.default_rng(2)
        n = 600
        df = pd.DataFrame({
            "triplet": [f"s{i % 20}" for i in range(n)],
            "valid_date": pd.date_range("2025-01-01", periods=n // 20).tolist() * 20,
            "err_a": rng.normal(0, 1, n),        # MAE ~0.8
            "err_b": rng.normal(0, 3, n),        # MAE ~2.4
        })
        out = V.paired_block_bootstrap(df, err_a="err_a", err_b="err_b", n_boot=300)
        assert out["diff"] < 0
        assert out["p_a_better"] > 0.99
        assert out["ci_hi"] < 0

    def test_identical_not_significant(self):
        rng = np.random.default_rng(3)
        n = 400
        e = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "triplet": [f"s{i % 10}" for i in range(n)],
            "valid_date": pd.date_range("2025-01-01", periods=n // 10).tolist() * 10,
            "err_a": e,
            "err_b": e + rng.normal(0, 0.01, n),
        })
        out = V.paired_block_bootstrap(df, err_a="err_a", err_b="err_b", n_boot=300)
        assert abs(out["diff"]) < 0.05
        assert out["ci_lo"] < 0 < out["ci_hi"]


class TestSummaryFreqBias:
    def test_fb_surfaced(self):
        df = pd.DataFrame({"obs": [2.0, 1.5, 1.0, 0.0, 0.0],
                           "pred": [3.0, 0.0, 0.5, 2.0, 0.0]})
        out = V.summarize_deterministic(df, obs_col="obs", pred_col="pred")
        assert out["fb_1in"] == pytest.approx(2 / 3)


class TestReliabilityCurve:
    def test_calibrated_bins(self):
        # probs 0.1 (never happens) and 0.9 (always happens): pred≈obs per bin
        y = [0.0] * 50 + [2.0] * 50
        p = [0.1] * 50 + [0.9] * 50
        rows = V.reliability_curve(y, p, threshold_in=1.0, n_bins=5)
        assert len(rows) == 2
        assert rows[0]["pred"] == pytest.approx(0.1)
        assert rows[0]["obs"] == pytest.approx(0.0)
        assert rows[1]["pred"] == pytest.approx(0.9)
        assert rows[1]["obs"] == pytest.approx(1.0)


class TestPerformanceDiagram:
    def test_points_match_contingency(self):
        df = pd.DataFrame({"obs": [2.0, 1.5, 1.0, 0.0, 0.0],
                           "a": [3.0, 0.0, 0.5, 2.0, 0.0]})
        pts = V.performance_diagram_points(df, obs_col="obs", pred_cols=["a"],
                                           thresholds=[1.0])
        (p,) = pts
        assert p["pod"] == pytest.approx(1 / 3)
        assert p["sr"] == pytest.approx(1 / 2)   # 1 − FAR
        assert p["csi"] == pytest.approx(1 / 4)


class TestBlockBootstrapStat:
    def test_recovers_mean_difference(self):
        rng = np.random.default_rng(4)
        n = 600
        df = pd.DataFrame({
            "triplet": [f"s{i % 20}" for i in range(n)],
            "valid_date": pd.date_range("2025-01-01", periods=n // 20).tolist() * 20,
            "x": rng.normal(1.0, 0.5, n),
        })
        out = V.block_bootstrap_stat(df, lambda d: d["x"].mean(), n_boot=300)
        assert out["stat"] == pytest.approx(df["x"].mean())
        assert out["ci_lo"] < 1.0 < out["ci_hi"]
        assert out["p_gt_0"] > 0.99

    def test_null_stat_straddles_zero(self):
        rng = np.random.default_rng(5)
        n = 400
        df = pd.DataFrame({
            "triplet": [f"s{i % 10}" for i in range(n)],
            "valid_date": pd.date_range("2025-01-01", periods=n // 10).tolist() * 10,
            "x": rng.normal(0.0, 1.0, n),
        })
        out = V.block_bootstrap_stat(df, lambda d: d["x"].mean(), n_boot=300)
        assert out["ci_lo"] < 0 < out["ci_hi"]


class TestCrpss:
    def test_skill_directions(self):
        assert V.crpss(0.5, 1.0) == pytest.approx(0.5)
        assert V.crpss(1.0, 1.0) == pytest.approx(0.0)
        assert V.crpss(2.0, 1.0) == pytest.approx(-1.0)
        assert V.crpss(None, 1.0) is None
        assert V.crpss(0.5, 0.0) is None


class TestClimatologyQuantiles:
    def test_station_month_lookup_and_fallback(self):
        rng = np.random.default_rng(6)
        # Station s1: 60 Jan days of heavy snow, 60 Jun days of zero.
        train = pd.DataFrame({
            "triplet": ["s1"] * 120,
            "valid_date": (["2025-01-%02d" % (i % 28 + 1) for i in range(60)]
                           + ["2025-06-%02d" % (i % 28 + 1) for i in range(60)]),
            "obs_snowfall_in": np.r_[rng.gamma(4, 2, 60), np.zeros(60)],
        })
        test = pd.DataFrame({
            "triplet": ["s1", "s1", "s_unknown"],
            "valid_date": ["2026-01-15", "2026-06-15", "2026-01-15"],
        })
        q = V.climatology_quantiles(train, test, q_levels=[0.5])
        assert q.shape == (3, 1)
        assert q[0, 0] > 1.0          # January median: snowy
        assert q[1, 0] == pytest.approx(0.0)   # June median: zero
        assert np.isfinite(q[2, 0])   # unknown station falls back, stays finite
