"""Physical snowfall features from thermodynamic + pressure-level profiles.

The hard part of snow-depth forecasting is not QPF (the models give decent
liquid-equivalent) — it is the snow-to-liquid ratio (SLR) and precip phase,
which together convert QPF into snow depth. Both are governed by the vertical
temperature/humidity structure of the atmosphere, not surface conditions
alone. This module turns a profile (temperature, relative humidity, wind,
geopotential at standard pressure levels) plus surface temp/humidity into
the features that physically determine SLR and phase:

  wet_bulb_c(t, rh)            phase discriminator — snow can fall at air
                              temps well above 0C if the air is dry
  dgz_*                       dendritic growth zone (-12C..-18C) presence,
                              depth, and RH — fluffy high-SLR crystals grow
                              here; warm/dry growth gives dense low-SLR snow
  thickness_1000_500_m        classic 5400 m rain/snow thickness line
  profile_slr_estimate        Cobb-style physical SLR from the column
  melting_level_*             height/temperature of the 0C wet-bulb crossing

Everything is pure numpy/pandas, no I/O. The fetch lives in app/met.py
(fetch_profile); the pooled post-processor consumes these as features.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Dendritic (planar/branched crystal) growth zone. Crystals grown here are
# large and low-density -> high SLR. Outside it, growth favors columns/plates
# or riming -> dense snow.
DGZ_WARM_C = -12.0
DGZ_COLD_C = -18.0

THICKNESS_SNOW_M = 5400.0  # 1000-500 hPa thickness below which precip is snow


def wet_bulb_c(t_c, rh_pct):
    """Stull (2011) wet-bulb approximation, valid for typical surface ranges.
    t_c: dry-bulb temperature (C); rh_pct: relative humidity (%). Vectorized."""
    t = np.asarray(t_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    tw = (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return tw


def _level_temp(profile: dict, level: int) -> Optional[float]:
    v = profile.get(f"temperature_{level}hPa")
    return float(v) if v is not None and np.isfinite(v) else None


def _level_rh(profile: dict, level: int) -> Optional[float]:
    v = profile.get(f"relative_humidity_{level}hPa")
    return float(v) if v is not None and np.isfinite(v) else None


def dgz_features(profile: dict, levels: Sequence[int]) -> dict:
    """Dendritic-growth-zone diagnostics from a single-time column.

    profile: {temperature_<lev>hPa, relative_humidity_<lev>hPa, ...}.
    Returns presence flag, how many sampled levels sit in the DGZ, and the
    mean RH there (saturated DGZ = efficient dendrite production = high SLR).
    """
    in_dgz_rh = []
    n_in = 0
    for lev in levels:
        t = _level_temp(profile, lev)
        if t is None:
            continue
        if DGZ_COLD_C <= t <= DGZ_WARM_C:
            n_in += 1
            rh = _level_rh(profile, lev)
            if rh is not None:
                in_dgz_rh.append(rh)
    return {
        "dgz_present": 1.0 if n_in else 0.0,
        "dgz_n_levels": float(n_in),
        "dgz_mean_rh": float(np.mean(in_dgz_rh)) if in_dgz_rh else np.nan,
        # Saturated DGZ is the high-SLR signature: in the zone AND humid.
        "dgz_saturated": 1.0 if (in_dgz_rh and np.mean(in_dgz_rh) >= 80.0) else 0.0,
    }


def thickness_m(profile: dict) -> Optional[float]:
    """1000-500 hPa geopotential thickness (m): the operational rain/snow
    discriminator (~5400 m line). Needs geopotential at both levels."""
    z1000 = profile.get("geopotential_height_1000hPa")
    z500 = profile.get("geopotential_height_500hPa")
    if z1000 is None or z500 is None or not (np.isfinite(z1000) and np.isfinite(z500)):
        return None
    return float(z500) - float(z1000)


def profile_slr_estimate(profile: dict, levels: Sequence[int], *, surface_t_c: Optional[float] = None) -> float:
    """Physical first-guess SLR from the column, in the spirit of Cobb &
    Waldstreicher (2005): start from a temperature-based baseline, boost it
    when a saturated dendritic growth zone is present, damp it for warm
    low-levels (riming/melting) and for dry DGZ.

    Returns an SLR in a physical band; the learned model refines it, but this
    is a sane physical prior and a standalone fallback when no model is
    trained yet.
    """
    dgz = dgz_features(profile, levels)
    # Coldest sampled level temp drives the crystal-habit baseline.
    temps = [_level_temp(profile, lev) for lev in levels]
    temps = [t for t in temps if t is not None]
    coldest = min(temps) if temps else (surface_t_c if surface_t_c is not None else -10.0)

    # Baseline: colder columns -> higher SLR, plateauing. Roughly the Roebber
    # central tendency: ~8 near 0C climbing toward ~18-20 by -20C.
    base = float(np.clip(8.0 + (-coldest) * 0.6, 3.0, 22.0))

    if dgz["dgz_saturated"]:
        base *= 1.25          # efficient dendrite factory
    elif dgz["dgz_present"] and not np.isnan(dgz["dgz_mean_rh"]) and dgz["dgz_mean_rh"] < 60.0:
        base *= 0.85          # in the zone but too dry to grow dendrites

    if surface_t_c is not None and surface_t_c > 0.5:
        # Warm surface: partial melt / wet snow / riming -> compact, low SLR.
        base *= float(np.clip(1.0 - (surface_t_c) * 0.15, 0.4, 1.0))

    return float(np.clip(base, 3.0, 30.0))


def daily_profile_features(
    hourly_profiles: list[dict], levels: Sequence[int], *, surface_t_c: Optional[float] = None,
) -> dict:
    """Reduce a day's worth of hourly column snapshots to per-day features.

    Aggregates over the snapshots so the post-processor sees the day's
    dominant thermodynamic regime: any DGZ presence, mean DGZ RH, mean
    thickness, and the physical SLR prior.
    """
    if not hourly_profiles:
        return {
            "dgz_present": np.nan, "dgz_n_levels": np.nan, "dgz_mean_rh": np.nan,
            "dgz_saturated": np.nan, "thickness_1000_500_m": np.nan,
            "profile_slr_estimate": np.nan,
        }
    dgzs = [dgz_features(p, levels) for p in hourly_profiles]
    thicks = [t for t in (thickness_m(p) for p in hourly_profiles) if t is not None]
    slrs = [profile_slr_estimate(p, levels, surface_t_c=surface_t_c) for p in hourly_profiles]
    return {
        "dgz_present": float(np.nanmax([d["dgz_present"] for d in dgzs])),
        "dgz_n_levels": float(np.nanmean([d["dgz_n_levels"] for d in dgzs])),
        "dgz_mean_rh": float(np.nanmean([d["dgz_mean_rh"] for d in dgzs])),
        "dgz_saturated": float(np.nanmax([d["dgz_saturated"] for d in dgzs])),
        "thickness_1000_500_m": float(np.mean(thicks)) if thicks else np.nan,
        "profile_slr_estimate": float(np.nanmean(slrs)) if slrs else np.nan,
    }
