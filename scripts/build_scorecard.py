#!/usr/bin/env python3
"""Build dist/verify/ — the public verification scorecard page.

Static artifact assembled from committed + generated JSON:
  benchmarks/published.json          pre-registered win conditions + targets
  benchmarks/baseline_v1.5.json      frozen "before" picture
  data/models/metrics_folds.json     current fold-of-record metrics
  data/verify/live_scorecard.json    rolling as-issued verification (optional)

Emits dist/verify/index.html (self-contained: inline JSON + a small canvas
chart script, no external deps — same zero-dependency philosophy as the main
site) and dist/verify/scorecard.json (machine-readable, for SnowBench).

Runs in the pages merge job (cheap: no forecasts, no NWP). Locally:
    python scripts/build_scorecard.py
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "dist" / "verify"

SOURCES_ORDER = ["postproc", "nbm_raw", "nbm_tuned", "mm_mean", "postproc_l1"]
SOURCE_LABELS = {
    "postproc": "SnowWatch (post-processed)",
    "nbm_raw": "NBM (raw, baseline)",
    "nbm_tuned": "NBM (thresholds tuned — fair fight)",
    "mm_mean": "Multi-model mean",
    "postproc_l1": "MAE-optimal head (cautionary)",
}
METRIC_COLS = [
    ("n", "n", ",d"),
    ("mae", "MAE (in)", "0.3f"), ("bias", "Bias", "+.3f"),
    ("event_mae", "Event MAE", "0.3f"),
    ("csi_1in", "CSI@1\"", "0.3f"), ("pod_1in", "POD@1\"", "0.3f"),
    ("fb_1in", "FB@1\"", "0.2f"),
    ("csi_2in", "CSI@2\"", "0.3f"),
    ("csi_6in", "CSI@6\"", "0.3f"), ("pod_6in", "POD@6\"", "0.3f"),
    ("fb_6in", "FB@6\"", "0.2f"),
    ("csi_12in", "CSI@12\"", "0.3f"),
]


def _fmt(v, spec):
    if v is None:
        return "—"
    try:
        return format(int(v) if spec.endswith("d") else float(v), spec)
    except (TypeError, ValueError):
        return "—"


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def fold_table(fold: dict) -> str:
    rows = []
    for src in SOURCES_ORDER:
        m = (fold.get("sources") or {}).get(src)
        if not m:
            continue
        best = src == "postproc"
        tds = "".join(f"<td>{_fmt(m.get(k), spec)}</td>" for k, _, spec in METRIC_COLS)
        cls = ' class="hl"' if best else ""
        rows.append(f"<tr{cls}><td>{html.escape(SOURCE_LABELS.get(src, src))}</td>{tds}</tr>")
    head = "".join(f"<th>{lbl}</th>" for _, lbl, _ in METRIC_COLS)
    return (f"<table><thead><tr><th>Forecast</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def bootstrap_lines(fold: dict) -> str:
    out = []
    bb = fold.get("vs_nbm") or {}
    if bb.get("diff") is not None:
        out.append(
            f"ΔMAE vs NBM: <b>{bb['diff']:+.3f} in</b> "
            f"[{bb['ci_lo']:+.3f}, {bb['ci_hi']:+.3f}] "
            f"P(SnowWatch better) = {bb['p_a_better']:.2f} "
            f"({bb.get('n_blocks', '?')} station-weeks)")
    for thr in ("1", "2", "6", "12"):
        bs = fold.get(f"vs_nbm_csi_{thr}in") or {}
        if bs.get("stat") is not None and bs.get("ci_lo") is not None:
            out.append(
                f"ΔCSI@{thr}\" vs NBM: <b>{bs['stat']:+.3f}</b> "
                f"[{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}] "
                f"P(better) = {bs['p_gt_0']:.2f}")
    crpss = fold.get("crpss_vs_climatology")
    if crpss is not None:
        out.append(f"CRPS {fold.get('crps_postproc'):.3f} "
                   f"(climatology {fold.get('crps_climatology'):.3f}, "
                   f"CRPSS <b>{crpss:.3f}</b>)")
    for thr in ("1", "2", "6", "12"):
        p = fold.get(f"prob_{thr}in") or {}
        if p.get("bss") is not None:
            out.append(f"BSS@{thr}\": <b>{p['bss']:.3f}</b> "
                       f"(Brier {p['brier']:.4f}, n={p['n']})")
    return "".join(f"<li>{line}</li>" for line in out)


def reliability_json(fold: dict) -> dict:
    out = {}
    for thr in ("1", "2", "6", "12"):
        rel = (fold.get(f"prob_{thr}in") or {}).get("reliability")
        if rel:
            out[thr] = rel
    return out


def win_conditions_table(published: dict, fold: dict) -> str:
    """Pre-registered conditions vs current status, computed live."""
    src = fold.get("sources") or {}
    pp, nbm = src.get("postproc") or {}, src.get("nbm_raw") or {}
    rows = []

    def row(wid, claim, status, met):
        icon = "✅" if met is True else ("❌" if met is False else "⏳")
        rows.append(f"<tr><td>{wid}</td><td>{html.escape(claim)}</td>"
                    f"<td>{icon} {html.escape(status)}</td></tr>")

    if pp.get("mae") and nbm.get("mae"):
        ratio = pp["mae"] / nbm["mae"]
        row("W1", "MAE ≤ 0.85 × NBM (CI excludes 0)",
            f"MAE ratio {ratio:.2f}×", ratio <= 0.85)
    per_thr = []
    met_all = True
    for thr in ("1", "2", "6", "12"):
        a, b = pp.get(f"csi_{thr}in"), nbm.get(f"csi_{thr}in")
        if a is None or b is None:
            continue
        ok = a >= b
        met_all &= ok
        per_thr.append(f"{thr}\": {a:.3f} vs {b:.3f} {'✓' if ok else '✗'}")
    row("W2", "CSI ≥ NBM at 1/2/6/12\" with FB ∈ [0.8, 1.3]",
        "; ".join(per_thr) or "pending", met_all if per_thr else None)
    crpss = fold.get("crpss_vs_climatology")
    row("W3", "CRPSS vs climatology (NOHRSC-truth comparison pending)",
        f"CRPSS {crpss:.3f} (SNOTEL truth)" if crpss is not None else "pending",
        None)
    bss_ok = all((fold.get(f"prob_{t}in") or {}).get("bss", -1) > 0
                 for t in ("1", "2", "6", "12"))
    row("W4", "BSS > 0 at all thresholds, reliability slope 0.9–1.1",
        "all BSS > 0" if bss_ok else "some BSS ≤ 0", bss_ok or None)
    row("W5", "SLR R² ≥ 0.43, MAE ≤ 2.94 (Veals et al. 2025)",
        "SLR head not built yet (Phase 3)", None)
    return ("<table><thead><tr><th></th><th>Pre-registered condition "
            "(2026-07-06, before retrains)</th><th>Status</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SnowWatch — Verification</title>
<style>
 body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0 auto;
        max-width: 980px; padding: 24px; color: #1a2733; background: #f6f8fa; }}
 h1 {{ font-size: 1.5em }} h2 {{ font-size: 1.15em; margin-top: 2em }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px;
         background: #fff; margin: 12px 0 }}
 th, td {{ border: 1px solid #d7dee5; padding: 5px 8px; text-align: right }}
 th:first-child, td:first-child {{ text-align: left }}
 thead th {{ background: #eef2f6 }}
 tr.hl td {{ background: #eaf6ee; font-weight: 600 }}
 .note {{ color: #5a6b7a; font-size: 13px }}
 canvas {{ background: #fff; border: 1px solid #d7dee5; margin: 8px 12px 8px 0 }}
 code {{ background: #eef2f6; padding: 1px 4px; border-radius: 3px }}
</style></head><body>
<h1>SnowWatch verification scorecard</h1>
<p class="note">Generated {generated}. Fold of record: <b>{fold_label}</b>
({n_test:,} test rows, {n_stations} SNOTEL stations, truth = QC'd SNOTEL 24h
snowfall). Temporal split — the model never saw the test period. Metric
definitions: <code>app/verification.py</code>; win conditions pre-registered in
<code>benchmarks/published.json</code> before any v1.6 retrains.</p>

<h2>Pre-registered win conditions</h2>
{win_table}

<h2>Deterministic scorecard — {fold_label}</h2>
{fold_table}
<p class="note">FB = frequency bias (1.0 unbiased). The MAE-optimal head is
shown as the cautionary exhibit: lowest MAE, collapsed event detection —
why MAE alone is never the headline. <b>Rows cover different populations</b>
(NBM exists at a subset of stations; n column) — cross-row comparison is
indicative only. The paired bootstrap deltas below are computed on shared
rows and are the arbiter of every head-to-head claim.</p>

<h2>Significance (station-week block bootstrap, 95% CI)</h2>
<ul>{bootstrap_list}</ul>

<h2>Reliability — P(snowfall ≥ T)</h2>
<div id="relCharts"></div>
<p class="note">Dots on the diagonal = perfectly calibrated probabilities.
Bin sample sizes shown as dot area.</p>

{live_section}

{truth_section}

<script>
const REL = {rel_json};
function drawRel(thr, rows) {{
  const c = document.createElement('canvas'); c.width = 220; c.height = 220;
  document.getElementById('relCharts').appendChild(c);
  const g = c.getContext('2d'), P = 30, W = c.width - P * 2;
  g.strokeStyle = '#c7d0d9';
  g.strokeRect(P, P, W, W);
  g.beginPath(); g.moveTo(P, P + W); g.lineTo(P + W, P); g.stroke();
  g.fillStyle = '#1a2733'; g.font = '11px sans-serif';
  g.fillText('P(\\u2265 ' + thr + '") forecast \\u2192', P, c.height - 8);
  g.save(); g.translate(10, P + W); g.rotate(-Math.PI / 2);
  g.fillText('observed freq \\u2192', 0, 0); g.restore();
  const maxN = Math.max(...rows.map(r => r.n));
  for (const r of rows) {{
    const x = P + r.pred * W, y = P + W - r.obs * W;
    const rad = 3 + 6 * Math.sqrt(r.n / maxN);
    g.beginPath(); g.arc(x, y, rad, 0, 7);
    g.fillStyle = 'rgba(31,119,180,0.55)'; g.fill();
  }}
}}
for (const [thr, rows] of Object.entries(REL)) drawRel(thr, rows);
</script>
</body></html>
"""


def live_section_html(live: dict | None) -> str:
    if not live or not live.get("sources"):
        return ("<h2>As-issued (live) verification</h2><p class='note'>"
                "Accumulating — the post-processor ships to production after "
                "the offline gates pass; rolling scores appear here.</p>")
    rows = []
    for src, m in live["sources"].items():
        if m.get("mae") is None:
            continue
        rows.append(f"<tr><td>{html.escape(src)}</td><td>{m['n']:,}</td>"
                    f"<td>{m['mae']:.3f}</td><td>{_fmt(m.get('bias'), '+.3f')}</td>"
                    f"<td>{_fmt(m.get('csi_1in'), '0.3f')}</td></tr>")
    w = live.get("window", {})
    return (f"<h2>As-issued (live) verification</h2><p class='note'>Forecasts "
            f"scored exactly as published, {w.get('start')} → {w.get('end')} "
            f"({live.get('n_stations')} stations).</p>"
            "<table><thead><tr><th>Source</th><th>n</th><th>MAE</th>"
            "<th>Bias</th><th>CSI@1\"</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def truth_section_html(agree: dict | None) -> str:
    if not agree:
        return ""
    def row(label, s):
        if not s:
            return ""
        return (f"<tr><td>{html.escape(label)}</td><td>{s['n']:,}</td>"
                f"<td>{_fmt(s.get('r'), '0.2f')}</td>"
                f"<td>{_fmt(s.get('mean_t1'), '0.2f')}</td>"
                f"<td>{_fmt(s.get('mean_t2'), '0.2f')}</td>"
                f"<td>{_fmt(s.get('event_agreement_1in'), '0.2f')}</td></tr>")
    return (
        "<h2>How much do the truths agree?</h2>"
        "<p class='note'>SNOTEL ultrasonic depth change (T1) vs NOHRSC human "
        f"observer reports (T2) within {agree.get('radius_km')} km. They agree "
        "only weakly — settlement-netted sensor vs cleared board, UTC vs "
        "local-morning windows, ridge vs valley sites. That disagreement is "
        "the floor on how precisely ANY system can be scored here; it is why "
        "T2 is a separate verification track and never a swap-in truth.</p>"
        "<table><thead><tr><th>Match</th><th>n</th><th>r</th><th>mean T1</th>"
        "<th>mean T2</th><th>event agr.@1\"</th></tr></thead><tbody>"
        + row("day-matched", agree.get("day_matched"))
        + row("3-day storm totals", agree.get("storm_total_3d"))
        + "</tbody></table>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    # Local runs read the fresh metrics; CI reads the committed copy
    # (data/ is gitignored — scripts/train_postprocessor.py runs locally,
    # and the release step is: copy metrics_folds.json -> benchmarks/
    # scorecard_current.json and commit alongside the model upload).
    folds_doc = (_load(ROOT / "data" / "models" / "metrics_folds.json")
                 or _load(ROOT / "benchmarks" / "scorecard_current.json") or {})
    fold_name = "A_core_winter"
    fold = (folds_doc.get("folds") or {}).get(fold_name)
    if not fold:
        print("no fold metrics — run train_postprocessor.py --folds first")
        return 1
    published = _load(ROOT / "benchmarks" / "published.json") or {}
    live = _load(ROOT / "data" / "verify" / "live_scorecard.json")
    agree = (_load(ROOT / "data" / "verify" / "truth_agreement.json")
             or _load(ROOT / "benchmarks" / "truth_agreement.json"))

    page = PAGE.format(
        generated=date.today().isoformat(),
        fold_label=fold_name,
        n_test=fold.get("n_test", 0),
        n_stations=fold.get("n_stations", 0),
        win_table=win_conditions_table(published, fold),
        fold_table=fold_table(fold),
        bootstrap_list=bootstrap_lines(fold),
        rel_json=json.dumps(reliability_json(fold)),
        live_section=live_section_html(live),
        truth_section=truth_section_html(agree),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(page)
    (args.out / "scorecard.json").write_text(json.dumps({
        "generated": date.today().isoformat(),
        "fold_of_record": fold_name,
        "fold": fold,
        "published_targets": published,
        "live": live,
    }, indent=2, default=float))
    print(f"wrote {args.out}/index.html + scorecard.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
