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
- `nbm_snow17` — the same SNOW-17 physics driven by NBM precip/temperature
  instead of Open-Meteo blend weather (skipped where NBM has no coverage,
  e.g. Alaska).
- `nbm_snowfall` — NOAA NBM daily snowfall added to last observed depth and
  decayed by a degree-day melt term.
- `ridge_snow` — LightGBM (falls back to Ridge) on lagged depth + DOY +
  rolling precip/temp/SWE/snowfall covariates. Direct multi-step.
- `chronos_bolt` — Amazon Chronos-Bolt zero-shot foundation model on the
  depth series. Optional; skipped silently if the wheel is unavailable.

Each member is anchored to the last observed snow depth with a linearly
decaying correction (decay horizon depends on the member's typical drift
character), then combined into a rolling-MAE-weighted blend. The anchor and
all training/verification targets come from a QC layer (`app/targets.py`)
that despikes the ultrasonic depth sensor, flags stuck runs, and corroborates
big depth jumps against the snow pillow and precip gauge.

Every 6h build also archives what each member, the blend, and each raw NWP
model (NBM/HRRR/GFS/IFS/AIFS) predicted, as long-format rows appended to a
monthly GitHub Release asset — the foundation for training-pair construction
and continuous verification (`app/archive.py`, `app/verification.py`). Each
forecast additionally ships a per-day attribution of why the blend differs
from the official NWS public forecast (`app/nws.py` + `nws_divergence`).

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
  targets.py       Depth QC + 24h snowfall target construction (bitmask flags)
  weather.py       Open-Meteo historical + blend forecast
  met.py           Open-Meteo multi-model (NBM/HRRR/GFS/IFS/AIFS) + ensemble
  nbm.py           Open-Meteo NBM CONUS short-range snowfall forecast (legacy fallback)
  nws.py           api.weather.gov official forecast (snowfall, QPF, snow level)
  archive.py       Long-format forecast archive schema + gzip CSV IO
  verification.py  MAE/CSI/CRPS/Brier + paired block bootstrap
  templates/       index.html
  static/          app.js + styles.css
data/
  stations.json               SNOTEL station catalogue (active sites)
  cache/                      On-disk JSON cache (SNOTEL records, weather)
  prevruns/                   Backfilled Previous-Runs forecasts per model
scripts/
  fetch_stations.py           Build/refresh stations.json from NRCS AWDB
  build_static_site.py        Builds dist/ for GitHub Pages
  archive_forecasts.py        Shard step: dist JSON -> archive rows
  merge_shards.py             CI: merge per-shard dist artifacts
  backfill_previous_runs.py   Open-Meteo Previous-Runs API backfill (resumable;
                              also runs nightly in CI, state on the
                              training-data Release)
  build_training_data.py      Join backfill + QC targets -> training pairs
  train_postprocessor.py      Train/evaluate the pooled LightGBM post-processor
  benchmark.py                Held-out MAE evaluation
tests/                        pytest unit tests (targets, verification, members)
```

## Environment variables

| Var | Effect |
|---|---|
| `SW_NO_FETCH=1` | Skip network calls; serve only from cache. CI uses this on the build shards. |
| `SW_STATIONS_FILE` | Override the default `data/stations.json` path. |
| `SW_ENABLE_CHRONOS=1` | Enable the Chronos-Bolt member (otherwise skipped). |
| `SW_ENSEMBLE_BUILD=1` | Pull fresh ensemble-member spread stats (00Z/12Z builds); otherwise reuse ≤11h cache. |
| `SW_NWS_OFF=1` | Skip api.weather.gov calls (no NWS divergence panel). |
| `SW_MET_BUDGET` | Per-process weighted Open-Meteo call budget for `app/met.py` (degrades to cache when spent). |

## Roadmap

- [x] v1: persistence + climatology + snow17_lite + nbm_snowfall + ridge_snow + chronos
- [x] v1.1: replace snow17_lite with full SNOW-17 (refreeze, rain-on-snow, density compaction)
- [x] v1.2: per-station SNOW-17 parameter calibration (L-BFGS-B over MFMAX/MFMIN/UADJ/PXTEMP, weekly workflow)
- [x] v1.3: walk-forward backtest of every member to replace proxy MAEs
- [x] v1.4: verification foundation — depth QC + snowfall targets, forecast
      archive to Release assets, multi-model fetch (NBM/HRRR/GFS/IFS/AIFS),
      `nbm_snow17` member, NWS divergence attribution, Previous-Runs backfill
- [ ] v1.5: pooled LightGBM post-processor trained on the backfill pairs
      (training/eval pipeline in place — `app/postproc.py` +
      `scripts/train_postprocessor.py`; awaiting CONUS backfill coverage,
      then ensemble-member integration)
- [ ] v1.6: full snow-water year history overlay (Oct 1 baseline + multi-year climatology band)
