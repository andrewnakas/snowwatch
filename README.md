# SnowWatch

Live 7-day ensemble snow-depth forecasts for SNOTEL stations across the
western US — the same blueprint as RiverWatch2 but applied to point snowpack
instead of river discharge.

For every active SNOTEL station, the pipeline runs a multi-member ensemble
against the NRCS daily SNWD/WTEQ record and Open-Meteo's NBM/blend forecast,
then ships per-station JSON forecasts to a static map UI on GitHub Pages.

Members:

- `persistence_lag1` — naive baseline (yhat = last observed snow depth).
- `climatology` — per-DOY mean snow depth from the station's full record.
- `snow17` — full Anderson 1976 SNOW-17 conceptual snowpack model. Daily
  step with seasonal temperature-index melt (MFMAX/MFMIN swing), antecedent
  temperature index, negative-degree-day refreeze, rain-on-snow turbulent
  energy balance, Hedstrom-Pomeroy fresh-snow density, Anderson compaction.
  Per-station parameters seeded from elevation + latitude.
- `nbm_snowfall` — NOAA NBM daily snowfall added to last observed depth and
  decayed by a degree-day melt term.
- `ridge_snow` — LightGBM (falls back to Ridge) on lagged depth + DOY +
  rolling precip/temp/SWE/snowfall covariates. Direct multi-step.
- `chronos_bolt` — Amazon Chronos-Bolt zero-shot foundation model on the
  depth series. Optional; skipped silently if the wheel is unavailable.

Each member is anchored to the last observed snow depth with a linearly
decaying correction (decay horizon depends on the member's typical drift
character), then combined into a rolling-MAE-weighted blend.

## Quickstart

```bash
cd snowwatch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# build the SNOTEL station catalogue (one-time; ~1 min)
python scripts/fetch_stations.py

# build the static site for one station (smoke test)
python scripts/build_static_site.py --limit 5 --horizon 7

# serve the map UI locally
python -m app.server --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

## Live demo

The Pages site at **https://andrewnakas.github.io/snowwatch** rebuilds every 6
hours and on every push to `main`. The frontend reads the per-station JSON
files directly — no backend.

## Project layout

```
app/
  server.py        Flask app: /, /api/stations, /api/forecast/<id>
  forecast.py      Ensemble members + MAE-weighted blend + anchoring
  snotel.py        NRCS AWDB SNOTEL fetcher (SNWD + WTEQ + PRCP + TAVG)
  weather.py       Open-Meteo historical + blend forecast
  nbm.py           Open-Meteo NBM CONUS short-range snowfall forecast
  templates/       index.html
  static/          app.js + styles.css
data/
  stations.json               SNOTEL station catalogue (active sites)
  cache/                      On-disk JSON cache (SNOTEL records, weather)
scripts/
  fetch_stations.py           Build/refresh stations.json from NRCS AWDB
  build_static_site.py        Builds dist/ for GitHub Pages
  merge_shards.py             CI: merge per-shard dist artifacts
  benchmark.py                Held-out MAE evaluation
```

## Environment variables

| Var | Effect |
|---|---|
| `SW_NO_FETCH=1` | Skip network calls; serve only from cache. CI uses this on the build shards. |
| `SW_STATIONS_FILE` | Override the default `data/stations.json` path. |
| `SW_ENABLE_CHRONOS=1` | Enable the Chronos-Bolt member (otherwise skipped). |

## Roadmap

- [x] v1: persistence + climatology + snow17_lite + nbm_snowfall + ridge_snow + chronos
- [x] v1.1: replace snow17_lite with full SNOW-17 (refreeze, rain-on-snow, density compaction)
- [x] v1.2: per-station SNOW-17 parameter calibration (L-BFGS-B over MFMAX/MFMIN/UADJ/PXTEMP, weekly workflow)
- [ ] v1.3: walk-forward backtest of every member to replace proxy MAEs
- [ ] v1.4: elevation lapse-rate + dewpoint + SRAD covariates
- [ ] v1.5: pooled LightGBM stacker meta-learner
- [ ] v1.6: full snow-water year history overlay (Oct 1 baseline + multi-year climatology band)
