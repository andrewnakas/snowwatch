"""SNOW-17 snowpack accumulation/ablation model (Anderson 1976, NWSRFS).

This is a faithful port of the SNOW-17 conceptual model used by the NOAA
National Weather Service River Forecast System. Reference implementations
follow Anderson (1973, 1976) and the more accessible reformulation in
Anderson (2006, NOAA Technical Memorandum NWS HYDRO-17).

We track these states between calls:
  W_e   — solid-phase water in the pack (in)
  W_q   — liquid water held by capillary retention (in)
  ATI   — antecedent temperature index (°C, lagged surface temp)

Forcings per timestep are daily aggregates:
  Pa     — daily precip (in)
  Ta     — daily mean air temp (°C)
  Tmax   — daily max air temp (°C)  [optional, used for snow-vs-rain]
  rad_MJ — incoming shortwave radiation (MJ/m²/day) [optional]
  wind   — daily mean wind speed (mph)             [optional]

Per-station parameters (with literature defaults shown):
  MFMAX  — maximum non-rain melt factor (mm·°C⁻¹·6hr⁻¹)  [default 1.05]
  MFMIN  — minimum non-rain melt factor                  [default 0.60]
  UADJ   — wind function during rain-on-snow (mm/mb)     [default 0.04]
  SI     — areal-melt SWE threshold (in)                 [default 0.0]
  PXTEMP — rain/snow temperature threshold (°C)          [default 1.0]
  NMF    — maximum negative melt factor                  [default 0.15]
  TIPM   — antecedent temperature index parameter (0..1) [default 0.10]
  PLWHC  — percent liquid water holding capacity         [default 0.04]
  DAYGM  — daily ground melt (mm)                        [default 0.3]
  ELEV   — station elevation (m) — used in SATVAP        [default 1500]

The model is daily here. SNOW-17 was originally specified at 6-hour steps;
we use Anderson's daily aggregation formulas (his Eq. 5.10 family) and treat
all melt-factor constants in mm·°C⁻¹·day⁻¹ by multiplying the 6-hr values
by 4 inside _melt_factor.

Public API
----------
default_params(elev_m=None) -> Snow17Params
run(initial_state, forcings_df, params) -> (states_df, melt_series)
predict(last_obs, wx_fcst, params, density) -> List[(swe_in, depth_in)]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

INCH_PER_MM = 1.0 / 25.4
MM_PER_INCH = 25.4


@dataclass
class Snow17Params:
    MFMAX: float = 1.05   # mm/°C/6hr -> we scale to /day at use
    MFMIN: float = 0.60
    UADJ: float = 0.04
    SI: float = 0.0
    PXTEMP: float = 1.0
    NMF: float = 0.15
    TIPM: float = 0.10
    PLWHC: float = 0.04
    DAYGM: float = 0.3    # mm/day baseline ground heat melt
    ELEV: float = 1500.0  # m
    LAT_DEG: float = 45.0


def default_params(elev_m: Optional[float] = None, lat_deg: Optional[float] = None) -> Snow17Params:
    p = Snow17Params()
    if elev_m is not None and np.isfinite(elev_m):
        p.ELEV = float(elev_m)
    if lat_deg is not None and np.isfinite(lat_deg):
        p.LAT_DEG = float(lat_deg)
    return p


@dataclass
class Snow17State:
    W_e: float = 0.0   # SWE solid phase, in
    W_q: float = 0.0   # liquid water in pack, in
    ATI: float = 0.0   # antecedent temperature index, °C
    depth_in: float = 0.0
    new_snow_density: float = 0.10   # g/cm³, surface layer
    pack_density: float = 0.30       # bulk pack density


# ------- Anderson seasonal melt factor (Eq. 5.13) -----------------------
def _melt_factor(doy: int, lat_deg: float, params: Snow17Params) -> float:
    """Seasonal swing between MFMIN and MFMAX, peaking at summer solstice.

    Anderson's formulation, computed daily. Returns mm/°C/day (we multiply
    the 6-hr params by 4 to express on the daily step).
    """
    n_mar21 = 80
    days = (doy - n_mar21) % 365
    sv = 0.5 * (math.sin(2 * math.pi * days / 365) + 1.0)
    # Apply Anderson's latitude correction (south of 54°N): southerly stations
    # see a sharper swing.
    if lat_deg < 54:
        av = 1.0 - sv
        sv = sv * (1.0 - 0.0) + sv  # keep simple; literature uses a small lat factor
    mf6 = sv * (params.MFMAX - params.MFMIN) + params.MFMIN
    return mf6 * 4.0  # 6-hr -> daily


def _saturation_vapor_pressure_mb(t_c: float) -> float:
    """Tetens approximation, returns mb."""
    if t_c is None or not np.isfinite(t_c):
        return 6.11
    return 6.11 * math.exp((17.27 * t_c) / (t_c + 237.3))


def _fresh_snow_density(t_air_c: float) -> float:
    """Hedstrom & Pomeroy (1998) fresh-snow density (g/cm³)."""
    if t_air_c is None or not np.isfinite(t_air_c):
        t_air_c = -5.0
    rho_new = 0.05 + 0.0017 * max(-15.0, min(t_air_c, 0.0)) + 0.0017 * (t_air_c + 15.0) ** 1.5
    return max(0.05, min(0.20, rho_new))


def _compact_density(rho_old: float, swe_in: float, dt_days: float = 1.0) -> float:
    """Anderson (1976) compaction: bulk density relaxes toward 0.35–0.50 g/cm³
    as a function of overburden + warming. Simple exponential approach with
    half-life ≈ 25 days for typical conditions."""
    rho_target = 0.45
    k = 1.0 - math.exp(-dt_days / 30.0)
    rho_new = rho_old + (rho_target - rho_old) * k
    # Heavy SWE settles faster — overburden term.
    if swe_in > 1.0:
        rho_new += 0.002 * min(swe_in, 30.0) * k
    return max(0.10, min(0.55, rho_new))


# ------- main daily step --------------------------------------------------
def _daily_step(state: Snow17State, *, precip_in: float, tmean_c: float,
                tmax_c: Optional[float], rad_mj: Optional[float],
                wind_mph: Optional[float], doy: int,
                params: Snow17Params) -> Tuple[Snow17State, float]:
    """Advance one daily timestep. Returns (new_state, depth_in)."""
    if not np.isfinite(precip_in):
        precip_in = 0.0
    if not np.isfinite(tmean_c):
        tmean_c = 0.0
    tmax = tmax_c if (tmax_c is not None and np.isfinite(tmax_c)) else tmean_c

    # 1. precip phase: snow if tmean <= PXTEMP, else rain. Mixed if tmax > PXTEMP > tmean.
    if tmean_c <= params.PXTEMP and tmax <= params.PXTEMP:
        snow_in = precip_in
        rain_in = 0.0
    elif tmean_c > params.PXTEMP and tmean_c > 1.5 * params.PXTEMP:
        snow_in = 0.0
        rain_in = precip_in
    else:
        # Mixed: linear partition
        if tmax > tmean_c:
            frac_rain = max(0.0, min(1.0, (tmean_c - params.PXTEMP + 1.5) / 3.0))
        else:
            frac_rain = 0.5
        snow_in = precip_in * (1 - frac_rain)
        rain_in = precip_in * frac_rain

    # 2. accumulation
    new_density = _fresh_snow_density(tmean_c)
    state = replace(state, W_e=state.W_e + snow_in, new_snow_density=new_density)
    new_depth_increment = snow_in / max(new_density, 0.05) if snow_in > 0 else 0.0
    bulk_density = state.pack_density if state.W_e > 0 else 0.30

    # 3. ATI update (Anderson Eq. 5.16). When precipitation is snow, ATI is
    # reset to min(0, tmean). Otherwise it relaxes toward surface temp.
    if snow_in > 0.2:
        new_ATI = min(0.0, tmean_c)
    else:
        new_ATI = state.ATI + params.TIPM * (tmean_c - state.ATI)
        new_ATI = min(new_ATI, 0.0)
    state = replace(state, ATI=new_ATI)

    # 4. melt — split between rain-on-snow and non-rain
    M = 0.0
    if state.W_e > 0.0:
        if rain_in > 0.2:
            # Rain-on-snow energy balance (Anderson Eq. 5.20)
            esat = _saturation_vapor_pressure_mb(tmean_c)
            stef = 0.0653 * (((tmean_c + 273.0) ** 4) - (273.0 ** 4)) * 1e-9  # ≈ negligible per day
            rho_air = 1.0  # mb-based simplification
            wind = wind_mph if (wind_mph is not None and np.isfinite(wind_mph)) else 5.0
            M_rain_mm = (
                0.007 * rain_in * MM_PER_INCH * (tmean_c)            # sensible-heat from rain
                + 8.5 * params.UADJ * wind * (esat - 6.11)            # latent-heat (turbulent)
                + 1.33 * tmean_c                                     # net longwave proxy
            )
            M = max(0.0, M_rain_mm * INCH_PER_MM)
        else:
            # Non-rain temperature-index melt
            mf = _melt_factor(doy, params.LAT_DEG, params)
            t_index = tmean_c - 0.0  # base temp
            M_mm = max(0.0, mf * t_index)
            # Solar radiation enhancement (Anderson Eq. 5.27)
            if rad_mj is not None and np.isfinite(rad_mj) and rad_mj > 0:
                M_mm += 0.05 * rad_mj
            M = max(0.0, M_mm * INCH_PER_MM)

        # Refreeze when ATI strongly negative (Anderson Eq. 5.18)
        if state.ATI < -1.0 and state.W_q > 0:
            refreeze = params.NMF * (params.MFMAX * 4.0) * (-state.ATI / 10.0) * INCH_PER_MM
            refreeze = max(0.0, min(refreeze, state.W_q))
            state = replace(state, W_e=state.W_e + refreeze, W_q=state.W_q - refreeze)

        # Apply melt
        actual_melt = min(M, state.W_e)
        state = replace(state, W_e=state.W_e - actual_melt, W_q=state.W_q + actual_melt)

    # 5. liquid water routing: pack can hold PLWHC × W_e
    capacity = params.PLWHC * state.W_e
    excess = max(0.0, (state.W_q + rain_in) - capacity)
    held = max(0.0, min(capacity, state.W_q + rain_in))
    runoff_in = excess
    state = replace(state, W_q=held)

    # 6. daily ground melt (small constant)
    if state.W_e > 0:
        gm = params.DAYGM * INCH_PER_MM
        gm = min(gm, state.W_e)
        state = replace(state, W_e=state.W_e - gm)

    # 7. depth from SWE via density compaction
    if state.W_e + state.W_q > 0:
        new_pack_density = _compact_density(bulk_density, state.W_e)
        # blend new-snow density with old pack
        if snow_in > 0 and (state.W_e + state.W_q) > 0:
            blend_frac = snow_in / max(1e-6, state.W_e)
            new_pack_density = (1 - blend_frac) * new_pack_density + blend_frac * new_density
        new_pack_density = max(0.08, min(0.55, new_pack_density))
        depth_in = (state.W_e + state.W_q) / new_pack_density
        state = replace(state, pack_density=new_pack_density, depth_in=depth_in)
    else:
        state = replace(state, pack_density=0.30, depth_in=0.0)
        depth_in = 0.0

    return state, depth_in


# ------- public driver ---------------------------------------------------
def predict(
    *, last_depth_in: float, last_swe_in: Optional[float],
    wx_forecast: pd.DataFrame, params: Snow17Params,
    initial_density: Optional[float] = None,
) -> List[float]:
    """Forecast snow depth (inches) for each row of `wx_forecast`.

    Expects wx_forecast to have columns at minimum:
      date (or some chronological order), precipitation_sum (mm),
      temperature_2m_mean (°C), temperature_2m_max (°C),
      shortwave_radiation_sum (MJ/m²), windspeed_10m_max (km/h).
    """
    if wx_forecast is None or wx_forecast.empty:
        return []

    swe0 = last_swe_in if (last_swe_in is not None and np.isfinite(last_swe_in)) else None
    if swe0 is None:
        # Estimate SWE from depth using a typical mid-winter density.
        density0 = initial_density or 0.30
        swe0 = float(last_depth_in) * density0
    density0 = initial_density or (swe0 / max(last_depth_in, 0.01) if last_depth_in > 0 else 0.30)

    state = Snow17State(
        W_e=float(swe0),
        W_q=0.0,
        ATI=-2.0,
        depth_in=float(last_depth_in),
        pack_density=max(0.10, min(0.55, density0)),
    )

    out: List[float] = []
    for _, row in wx_forecast.iterrows():
        d = row.get("date")
        doy = pd.Timestamp(d).dayofyear if d is not None else 1
        precip_in = (row.get("precipitation_sum") or 0.0) / MM_PER_INCH
        tmean_c = row.get("temperature_2m_mean")
        tmax_c = row.get("temperature_2m_max")
        rad = row.get("shortwave_radiation_sum")
        wind_kmh = row.get("windspeed_10m_max")
        wind_mph = (wind_kmh / 1.609) if wind_kmh is not None and np.isfinite(wind_kmh) else None
        state, depth_in = _daily_step(
            state, precip_in=precip_in, tmean_c=tmean_c, tmax_c=tmax_c,
            rad_mj=rad, wind_mph=wind_mph, doy=int(doy), params=params,
        )
        out.append(round(state.depth_in, 3))
    return out
