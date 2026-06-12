"""Unit tests for the QC + snowfall-target layer (app/targets.py)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import targets


def _hist(depths, swes=None, precips=None, tavgs=None, start=date(2025, 1, 1)):
    n = len(depths)
    return pd.DataFrame({
        "date": [start + timedelta(days=i) for i in range(n)],
        "snow_depth_in": depths,
        "swe_in": swes if swes is not None else [np.nan] * n,
        "precip_in": precips if precips is not None else [0.0] * n,
        "tavg_f": tavgs if tavgs is not None else [25.0] * n,
    })


class TestQcDailySeries:
    def test_clean_series_untouched(self):
        depths = [10, 11, 12, 12, 13, 14, 14]
        qc = targets.qc_daily_series(_hist(depths))
        assert (qc["qc_flags"] == 0).all()
        assert qc["snwd_qc"].tolist() == [float(d) for d in depths]

    def test_unsupported_spike_removed(self):
        # 20-inch one-day jump and return, no pillow/gauge support: false echo.
        depths = [10, 10, 30, 10, 10, 10, 10]
        qc = targets.qc_daily_series(_hist(depths))
        assert np.isnan(qc["snwd_qc"].iloc[2])
        assert qc["qc_flags"].iloc[2] & targets.QC_SPIKE_REMOVED

    def test_supported_jump_kept(self):
        # Same jump but the pillow gained 2 inches SWE: a real dump.
        depths = [10, 10, 30, 28, 27, 27, 26]
        swes = [3.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        qc = targets.qc_daily_series(_hist(depths, swes=swes))
        assert qc["snwd_qc"].iloc[2] == 30.0
        assert not (qc["qc_flags"].iloc[2] & targets.QC_SPIKE_REMOVED)

    def test_negative_depth_clamped(self):
        # Slightly-negative bare-ground readings clamp to 0 without tripping
        # the despiker (the excursion is small and gradual).
        qc = targets.qc_daily_series(_hist([0.5, -2, -1, -0.5, 0]))
        assert qc["snwd_qc"].iloc[1] == 0.0
        assert not (qc["qc_flags"] & targets.QC_SPIKE_REMOVED).any()

    def test_dropout_dip_despiked(self):
        # A one-day collapse to a negative reading and back is a sensor
        # dropout: clamped then removed as a down-spike.
        qc = targets.qc_daily_series(_hist([5, -2, 4, 4, 4]))
        assert np.isnan(qc["snwd_qc"].iloc[1])
        assert qc["qc_flags"].iloc[1] & targets.QC_SPIKE_REMOVED

    def test_stuck_run_flagged_in_season(self):
        depths = [20.0] * 15
        qc = targets.qc_daily_series(_hist(depths, start=date(2025, 1, 1)))
        assert (qc["qc_flags"] & targets.QC_STUCK).all()

    def test_zero_depth_summer_not_stuck(self):
        depths = [0.0] * 15
        qc = targets.qc_daily_series(_hist(depths, start=date(2025, 7, 1)))
        assert not (qc["qc_flags"] & targets.QC_STUCK).any()

    def test_empty(self):
        qc = targets.qc_daily_series(pd.DataFrame())
        assert qc.empty and "snwd_qc" in qc.columns


class TestDailySnowfall:
    def test_small_increase_passes_through(self):
        depths = [10, 11.5, 11.5, 11.0]
        snow = targets.daily_snowfall(targets.qc_daily_series(_hist(depths)))
        assert snow["snowfall_in"].iloc[1] == pytest.approx(1.5)
        assert snow["method"].iloc[1] == "dsnwd"
        assert snow["snowfall_in"].iloc[3] == 0.0  # settlement is not snowfall

    def test_big_jump_with_pillow_support_and_sane_slr(self):
        # +10in depth on +1.0in SWE → SLR 10, in band → take ΔSNWD.
        depths = [10, 20, 20]
        swes = [3.0, 4.0, 4.0]
        snow = targets.daily_snowfall(targets.qc_daily_series(_hist(depths, swes=swes)))
        assert snow["snowfall_in"].iloc[1] == pytest.approx(10.0)
        assert snow["method"].iloc[1] == "dsnwd"

    def test_big_jump_slr_out_of_band_uses_pillow(self):
        # +10in depth on +0.2in SWE → SLR 50, absurd → ΔWTEQ × station SLR.
        depths = [10, 20, 20]
        swes = [3.0, 3.2, 3.2]
        snow = targets.daily_snowfall(
            targets.qc_daily_series(_hist(depths, swes=swes)), station_slr=13.0)
        assert snow["snowfall_in"].iloc[1] == pytest.approx(0.2 * 13.0)
        assert snow["method"].iloc[1] == "dswe_slr"
        assert snow["quality"].iloc[1] & targets.QC_SLR_OOB

    def test_uncorroborated_jump_zeroed(self):
        # +5in depth, pillow flat, gauge dry → wind drift/sensor; zero it.
        # Jump must survive the despiker (gradual enough), so ramp the median.
        depths = [10, 10, 15, 15, 15]
        snow = targets.daily_snowfall(targets.qc_daily_series(_hist(depths)))
        assert snow["snowfall_in"].iloc[2] == 0.0
        assert snow["method"].iloc[2] == "zeroed"
        assert snow["quality"].iloc[2] & targets.QC_UNCORROBORATED

    def test_gauge_support_without_pillow(self):
        # Depth jump with gauge precip but quiet pillow → trust depth.
        depths = [10, 10, 15, 15, 15]
        precips = [0, 0, 0.5, 0, 0]
        snow = targets.daily_snowfall(targets.qc_daily_series(_hist(depths, precips=precips)))
        assert snow["snowfall_in"].iloc[2] == pytest.approx(5.0)
        assert snow["method"].iloc[2] == "dsnwd"

    def test_missing_depth_falls_back_to_pillow(self):
        depths = [10, np.nan, 20, 20]
        swes = [3.0, 3.5, 4.0, 4.0]
        snow = targets.daily_snowfall(
            targets.qc_daily_series(_hist(depths, swes=swes)), station_slr=10.0)
        assert snow["method"].iloc[1] == "dswe_slr"
        assert snow["snowfall_in"].iloc[1] == pytest.approx(0.5 * 10.0)
        assert snow["quality"].iloc[1] & targets.QC_WTEQ_FALLBACK

    def test_first_row_is_missing(self):
        snow = targets.daily_snowfall(targets.qc_daily_series(_hist([10, 11])))
        assert snow["method"].iloc[0] == "none"
        assert np.isnan(snow["snowfall_in"].iloc[0])


class TestStationStaticsAndAnchor:
    def test_median_slr_default_when_sparse(self):
        assert targets.station_median_slr(targets.qc_daily_series(_hist([1, 2, 3]))) == targets.DEFAULT_SLR

    def test_median_slr_computed(self):
        # 30 alternating snowfall days at SLR exactly 12.
        depths, swes = [10.0], [1.0]
        for i in range(60):
            if i % 2 == 0:
                depths.append(depths[-1] + 1.2)
                swes.append(swes[-1] + 0.1)
            else:
                depths.append(depths[-1])
                swes.append(swes[-1])
        qc = targets.qc_daily_series(_hist(depths, swes=swes, start=date(2025, 1, 1)))
        assert targets.station_median_slr(qc) == pytest.approx(12.0, rel=0.01)

    def test_last_reliable_depth_skips_trailing_spike(self):
        depths = [10, 10, 10, 10, 30]  # final-day false echo
        qc = targets.qc_daily_series(_hist(depths))
        d, dt, swe = targets.last_reliable_depth(qc)
        assert d == 10.0
        assert dt == date(2025, 1, 4)

    def test_statics_shape(self):
        n = 800
        start = date(2022, 10, 1)
        rng = np.random.default_rng(0)
        depths = np.clip(np.cumsum(rng.normal(0, 1, n)) + 20, 0, None)
        st = {"triplet": "999:XX:SNTL", "lat": 45.0, "lon": -110.0, "elevation_ft": 8000}
        qc = targets.qc_daily_series(_hist(list(depths), start=start))
        out = targets.station_statics(st, qc, use_cache=False)
        assert out["elevation_ft"] == 8000
        assert out["snow_class"] in {"maritime", "continental", "intermountain", "unknown"}
        assert out["n_years"] >= 1
