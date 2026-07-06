"""Unit tests for app/phase_features.py (shared train/live aggregation)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import phase_features as PF


class TestDailyPhaseAggregates:
    def test_cold_saturated_day(self):
        # -5°C at 90% RH all day: wet-bulb below zero every hour.
        out = PF.daily_phase_aggregates([-5.0] * 24, [90.0] * 24)
        assert out["wb_mean_c"] < 0
        assert out["wb_min_c"] <= out["wb_mean_c"]
        assert out["hours_wb_below_0"] == pytest.approx(24.0)

    def test_warm_day(self):
        out = PF.daily_phase_aggregates([10.0] * 24, [50.0] * 24)
        assert out["wb_mean_c"] > 0
        assert out["hours_wb_below_0"] == pytest.approx(0.0)

    def test_six_hourly_scaled_to_24h(self):
        # 4 samples (hourly_6), all below zero → 24h-equivalent count.
        out = PF.daily_phase_aggregates([-3.0] * 4, [85.0] * 4)
        assert out["hours_wb_below_0"] == pytest.approx(24.0)

    def test_too_few_hours_gives_none(self):
        out = PF.daily_phase_aggregates([-3.0] * 2, [85.0] * 2)
        assert out["wb_mean_c"] is None

    def test_freezing_level(self):
        out = PF.daily_phase_aggregates([-1.0] * 6, [80.0] * 6,
                                        freezing_level_m=[2500, 2700, 2600] * 2)
        assert out["freezing_level_m"] == pytest.approx(2600.0)


class TestPhaseFrame:
    def test_groups_by_day_and_lead(self):
        times = (pd.date_range("2026-01-01", periods=24, freq="h").tolist()
                 + pd.date_range("2026-01-02", periods=24, freq="h").tolist())
        df = pd.DataFrame({
            "time": times * 1,
            "t_c": [-4.0] * 24 + [5.0] * 24,
            "rh_pct": [90.0] * 48,
            "lead_days": [1] * 48,
        })
        out = PF.phase_frame_from_hourly(df, group_cols=("lead_days",))
        assert len(out) == 2
        cold = out[out["valid_date"] == "2026-01-01"].iloc[0]
        warm = out[out["valid_date"] == "2026-01-02"].iloc[0]
        assert cold["wb_mean_c"] < 0 < warm["wb_mean_c"]

    def test_empty(self):
        out = PF.phase_frame_from_hourly(pd.DataFrame())
        assert out.empty
