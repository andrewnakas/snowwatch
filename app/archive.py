"""Forecast archive schema + IO.

Every 6h build appends one long-format row per (station, issue, lead, source)
capturing what each member, the blend, and each raw NWP model said — the
foundation for training-pair construction, analog search, and the continuous
"beats NBM" verification.

Files are headerless gzip CSV with the fixed column order below. Headerless
because gzip streams are byte-appendable: the CI workflow grows the monthly
Release asset with a plain byte concat, no parse-merge-rewrite cycle. The
schema is versioned here in code; bump ARCHIVE_SCHEMA and add a migration in
read_archive() if columns ever change.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ARCHIVE_SCHEMA = 1

# source values:
#   blend, member:<name>          — depth_in forecasts (snow depth, inches)
#   model:<nbm|hrrr|gfs|ifs|aifs> — raw model daily snowfall/precip/temp
#   ens                           — ensemble spread stats (snowfall_cm=p50)
#   postproc                      — post-processor daily snowfall: point in
#                                   snowfall_cm (converted from inches),
#                                   q10/q90 reuse ens_snow_p10/p90 (cm),
#                                   P(>=1in) reuses ens_prob_pos. Reusing the
#                                   generic columns keeps the byte-append
#                                   stream schema-stable (no migration).
ARCHIVE_COLUMNS = [
    "schema",        # int, ARCHIVE_SCHEMA
    "triplet",       # str  station triplet
    "issue_date",    # ISO date the forecast was issued (UTC)
    "build_hour",    # int  UTC hour of the build (0/6/12/18)
    "lead_days",     # int  1..7
    "valid_date",    # ISO date the forecast is for
    "source",        # str  see above
    "snowfall_cm",   # float daily snowfall (raw models / ens p50)
    "precip_mm",     # float daily precip
    "tmean_c",       # float daily mean temp
    "depth_in",      # float snow-depth forecast (members/blend)
    "ens_snow_std",  # float ens-only spread stats
    "ens_snow_p10",
    "ens_snow_p90",
    "ens_prob_pos",
]


def write_archive(df: pd.DataFrame, path: Path) -> None:
    """Write rows as headerless gzip CSV in fixed column order."""
    out = df.reindex(columns=ARCHIVE_COLUMNS)
    out["schema"] = ARCHIVE_SCHEMA
    buf = io.StringIO()
    out.to_csv(buf, header=False, index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(buf.getvalue())


def read_archive(paths: Iterable[Path]) -> pd.DataFrame:
    """Read one or more (possibly multi-stream) archive files into one frame."""
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, names=ARCHIVE_COLUMNS, compression="gzip")
        except Exception:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["schema"] == ARCHIVE_SCHEMA]
    for c in ("snowfall_cm", "precip_mm", "tmean_c", "depth_in",
              "ens_snow_std", "ens_snow_p10", "ens_snow_p90", "ens_prob_pos"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["lead_days"] = pd.to_numeric(out["lead_days"], errors="coerce").astype("Int64")
    return out


def rows_from_forecast(fc_data: dict, *, build_hour: int) -> list[dict]:
    """Long-format archive rows from one station's forecast JSON payload
    (the asdict(StationForecast) structure written to dist/forecasts/)."""
    triplet = fc_data.get("triplet")
    issue = fc_data.get("issued_at")
    if not triplet or not issue:
        return []
    rows: list[dict] = []

    def _lead(valid_iso: str) -> int | None:
        try:
            from datetime import date
            return (date.fromisoformat(valid_iso) - date.fromisoformat(issue)).days
        except Exception:
            return None

    # Blend + members: depth forecasts.
    named = [("blend", fc_data.get("blend") or [])]
    for m, pts in (fc_data.get("members") or {}).items():
        named.append((f"member:{m}", pts or []))
    for source, pts in named:
        for pt in pts:
            lead = _lead(pt.get("date", ""))
            if not lead or lead < 1:
                continue
            rows.append({
                "triplet": triplet, "issue_date": issue, "build_hour": build_hour,
                "lead_days": lead, "valid_date": pt["date"], "source": source,
                "depth_in": pt.get("snow_depth_in"),
            })

    # Raw models from the multimodel frame.
    for r in fc_data.get("multimodel") or []:
        lead = _lead(r.get("date", ""))
        if not lead or lead < 1:
            continue
        for model in ("nbm", "hrrr", "gfs", "ifs", "aifs"):
            snow = r.get(f"{model}_snowfall")
            precip = r.get(f"{model}_precip")
            tmean = r.get(f"{model}_tmean")
            if snow is None and precip is None and tmean is None:
                continue
            rows.append({
                "triplet": triplet, "issue_date": issue, "build_hour": build_hour,
                "lead_days": lead, "valid_date": r["date"], "source": f"model:{model}",
                "snowfall_cm": snow, "precip_mm": precip, "tmean_c": tmean,
            })

    # Post-processor daily snowfall (as-issued; verified by the live
    # scorecard). Values arrive in inches, archived in cm like model rows.
    in_to_cm = 2.54
    for r in fc_data.get("postproc_daily") or []:
        lead = _lead(r.get("date", ""))
        if not lead or lead < 1:
            continue
        snow_in = r.get("snowfall_in")
        rows.append({
            "triplet": triplet, "issue_date": issue, "build_hour": build_hour,
            "lead_days": lead, "valid_date": r["date"], "source": "postproc",
            "snowfall_cm": None if snow_in is None else snow_in * in_to_cm,
            "ens_snow_p10": None if r.get("q10") is None else r["q10"] * in_to_cm,
            "ens_snow_p90": None if r.get("q90") is None else r["q90"] * in_to_cm,
            "ens_prob_pos": r.get("p1"),
        })

    # Ensemble spread stats.
    for r in fc_data.get("ensemble_stats") or []:
        lead = _lead(r.get("date", ""))
        if not lead or lead < 1:
            continue
        rows.append({
            "triplet": triplet, "issue_date": issue, "build_hour": build_hour,
            "lead_days": lead, "valid_date": r["date"], "source": "ens",
            "snowfall_cm": r.get("ens_snow_p50"),
            "precip_mm": r.get("ens_precip_mean"),
            "ens_snow_std": r.get("ens_snow_std"),
            "ens_snow_p10": r.get("ens_snow_p10"),
            "ens_snow_p90": r.get("ens_snow_p90"),
            "ens_prob_pos": r.get("ens_snow_prob_pos"),
        })
    return rows
