"""Unit tests for physical snowfall features (app/snowphysics.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import snowphysics as sp

LEVELS = (1000, 925, 850, 700, 600, 500)


def _cold_saturated_column():
    # A column with a saturated dendritic growth zone (-12..-18C) at 700/600.
    return {
        "temperature_1000hPa": -2.0, "relative_humidity_1000hPa": 90.0,
        "temperature_925hPa": -6.0, "relative_humidity_925hPa": 90.0,
        "temperature_850hPa": -10.0, "relative_humidity_850hPa": 92.0,
        "temperature_700hPa": -14.0, "relative_humidity_700hPa": 95.0,
        "temperature_600hPa": -17.0, "relative_humidity_600hPa": 93.0,
        "temperature_500hPa": -25.0, "relative_humidity_500hPa": 70.0,
        "geopotential_height_1000hPa": 100.0, "geopotential_height_500hPa": 5350.0,
    }


class TestWetBulb:
    def test_saturated_equals_drybulb(self):
        # At 100% RH wet-bulb ~ dry-bulb.
        assert abs(sp.wet_bulb_c(5.0, 100.0) - 5.0) < 1.0

    def test_dry_air_depresses(self):
        # Snow can fall above 0C dry-bulb if wet-bulb < 0: 3C @ 20% RH.
        tw = sp.wet_bulb_c(3.0, 20.0)
        assert tw < 0.0

    def test_vectorized(self):
        out = sp.wet_bulb_c([0.0, 10.0], [80.0, 50.0])
        assert out.shape == (2,)
        assert out[0] < 0.5  # near freezing, humid


class TestDGZ:
    def test_saturated_dgz_detected(self):
        f = sp.dgz_features(_cold_saturated_column(), LEVELS)
        assert f["dgz_present"] == 1.0
        assert f["dgz_n_levels"] >= 2
        assert f["dgz_mean_rh"] >= 90.0
        assert f["dgz_saturated"] == 1.0

    def test_warm_column_no_dgz(self):
        warm = {f"temperature_{l}hPa": 5.0 for l in LEVELS}
        warm.update({f"relative_humidity_{l}hPa": 90.0 for l in LEVELS})
        f = sp.dgz_features(warm, LEVELS)
        assert f["dgz_present"] == 0.0
        assert np.isnan(f["dgz_mean_rh"])


class TestThickness:
    def test_snow_thickness(self):
        col = _cold_saturated_column()
        thk = sp.thickness_m(col)
        assert thk == 5250.0
        assert thk < sp.THICKNESS_SNOW_M  # below 5400 -> snow

    def test_missing_levels(self):
        assert sp.thickness_m({"temperature_850hPa": -5.0}) is None


class TestProfileSLR:
    def test_saturated_dgz_high_slr(self):
        slr = sp.profile_slr_estimate(_cold_saturated_column(), LEVELS, surface_t_c=-2.0)
        assert slr > 13.0  # saturated DGZ + cold -> fluffy

    def test_warm_surface_low_slr(self):
        col = _cold_saturated_column()
        cold = sp.profile_slr_estimate(col, LEVELS, surface_t_c=-5.0)
        warm = sp.profile_slr_estimate(col, LEVELS, surface_t_c=2.0)
        assert warm < cold  # warm surface compacts/melts -> lower SLR

    def test_band_clamped(self):
        col = {f"temperature_{l}hPa": -40.0 for l in LEVELS}
        col.update({f"relative_humidity_{l}hPa": 99.0 for l in LEVELS})
        slr = sp.profile_slr_estimate(col, LEVELS, surface_t_c=-30.0)
        assert 3.0 <= slr <= 30.0


class TestDailyAggregation:
    def test_empty_returns_nans(self):
        f = sp.daily_profile_features([], LEVELS)
        assert np.isnan(f["thickness_1000_500_m"])
        assert np.isnan(f["profile_slr_estimate"])

    def test_aggregates_snapshots(self):
        cols = [_cold_saturated_column(), _cold_saturated_column()]
        f = sp.daily_profile_features(cols, LEVELS, surface_t_c=-3.0)
        assert f["dgz_present"] == 1.0
        assert f["thickness_1000_500_m"] == 5250.0
        assert 3.0 <= f["profile_slr_estimate"] <= 30.0
