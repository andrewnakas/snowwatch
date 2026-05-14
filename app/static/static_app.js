"use strict";

// SnowWatch static frontend. Loads stations.json + per-station forecast JSON
// directly (no backend). Draws a Leaflet map and an ensemble snow-depth chart
// with optional min/max envelope across members.

const map = L.map("map", { worldCopyJump: true }).setView([42, -113], 5);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18, attribution: "&copy; OpenStreetMap",
}).addTo(map);

const cluster = L.markerClusterGroup({ maxClusterRadius: 40 });
map.addLayer(cluster);
const markersById = new Map();
let currentStation = null;

function colorForElev(ft) {
  if (ft == null) return "#8ec5ff";
  if (ft >= 9000) return "#ffffff";
  if (ft >= 7000) return "#cfe2ff";
  return "#8ec5ff";
}

function makeIcon(elev) {
  const color = colorForElev(elev);
  return L.divIcon({
    className: "sw-marker",
    html: `<div style="background:${color}; width:12px; height:12px; border-radius:50%; border:2px solid #0b1424; box-shadow: 0 0 6px rgba(0,0,0,0.6);"></div>`,
    iconSize: [12, 12], iconAnchor: [6, 6],
  });
}

async function loadStations() {
  const r = await fetch("stations.json");
  const j = await r.json();
  const bounds = [];
  for (const s of j.stations) {
    if (s.lat == null || s.lon == null) continue;
    const m = L.marker([s.lat, s.lon], { icon: makeIcon(Number(s.elevation_ft)) });
    m.bindTooltip(`${s.id} — ${s.name} (${s.state})`);
    m.on("click", () => selectStation(s));
    cluster.addLayer(m);
    markersById.set(String(s.id), m);
    bounds.push([s.lat, s.lon]);
  }
  if (bounds.length) map.fitBounds(bounds, { padding: [40, 40] });

  try {
    const sr = await fetch("index_summary.json");
    if (sr.ok) {
      const sj = await sr.json();
      const note = document.createElement("p");
      note.style.fontSize = "12px";
      note.style.color = "#aab7d4";
      note.style.margin = "4px 0 0";
      const total = sj.stations_total ?? (sj.stations_succeeded + (sj.stations_failed?.length ?? 0));
      note.innerHTML = `Built ${sj.generated_at} · ${sj.stations_succeeded}/${total} stations · ${sj.horizon_days}-day forecast`;
      document.querySelector(".topbar").appendChild(note);
    }
  } catch (_) {}
}

function fmtNumber(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function renderMeta(station, payload) {
  document.getElementById("station-title").textContent = `${station.name} (${station.state})`;
  const el = document.getElementById("station-meta");
  el.innerHTML = `
    <b>${station.id}</b> · triplet <code>${station.triplet}</code><br/>
    <span style="opacity:0.7">elev ${fmtNumber(station.elevation_ft, 0)} ft · last obs ${payload.last_observed_date || "—"} · depth ${fmtNumber(payload.last_observed, 1)} in · SWE ${fmtNumber(payload.last_swe, 1)} in</span>
  `;
}

const MS_PER_DAY = 86400000;

function pickDateTicks(xmin, xmax, target = 8) {
  const span = Math.max(1, (xmax - xmin) / MS_PER_DAY);
  const steps = [1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90, 180, 365];
  let step = steps[0];
  for (const s of steps) {
    if (span / s <= target) { step = s; break; }
    step = s;
  }
  const ticks = [];
  const startDay = Math.ceil(xmin / MS_PER_DAY);
  for (let d = startDay; d * MS_PER_DAY <= xmax; d += step) ticks.push(d * MS_PER_DAY);
  return { ticks, step };
}

let _chartState = {
  canvas: null, history: [], members: {}, blend: [], showMembers: false, showBand: true,
};

function drawChart(canvas, history, members, blend, lastObsDate) {
  _chartState.canvas = canvas;
  _chartState.history = history;
  _chartState.members = members;
  _chartState.blend = blend;
  _chartState.lastObsDate = lastObsDate;
  _renderChart();
}

function _renderChart() {
  const c = _chartState.canvas; if (!c) return;
  const ctx = c.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = c.clientWidth || 640;
  const cssH = 320;
  c.width = Math.round(cssW * dpr); c.height = Math.round(cssH * dpr);
  c.style.height = cssH + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  // Trim history to the last 90 days for legibility.
  const histAll = _chartState.history.filter(p => p.snow_depth_in != null);
  const cutoff = Date.now() - 90 * MS_PER_DAY;
  const hist = histAll.filter(p => Date.parse(p.date) >= cutoff);
  const blend = _chartState.blend;
  const memberKeys = Object.keys(_chartState.members);
  if (!hist.length && !blend.length) {
    ctx.fillStyle = "#aab7d4";
    ctx.font = "13px Inter, sans-serif";
    ctx.fillText("no chartable data", 12, 24);
    return;
  }

  const all = [...hist, ...blend, ...memberKeys.flatMap(k => _chartState.members[k])];
  const xs = all.map(p => Date.parse(p.date));
  const ys = all.map(p => Number(p.snow_depth_in)).filter(v => Number.isFinite(v));
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(0, ...ys), ymax = Math.max(...ys);
  if (ymax - ymin < 4) ymax = ymin + 4;
  ymax = ymax * 1.08;

  const padL = 44, padR = 12, padT = 14, padB = 28;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;

  const xAt = t => padL + ((t - xmin) / (xmax - xmin || 1)) * plotW;
  const yAt = v => padT + (1 - (v - ymin) / (ymax - ymin || 1)) * plotH;

  // Axes / grid
  ctx.strokeStyle = "#1f2a44"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padT + (i / 4) * plotH;
    ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#7c87a8";
  ctx.font = "11px Inter, sans-serif";
  for (let i = 0; i <= 4; i++) {
    const v = ymax - (i / 4) * (ymax - ymin);
    ctx.fillText(v.toFixed(0) + " in", 4, padT + (i / 4) * plotH + 4);
  }
  const { ticks } = pickDateTicks(xmin, xmax, 7);
  let lastYear = null;
  for (const t of ticks) {
    const x = xAt(t);
    ctx.strokeStyle = "#1f2a44"; ctx.beginPath();
    ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    const d = new Date(t);
    const mo = d.toLocaleString(undefined, { month: "short", timeZone: "UTC" });
    const day = d.getUTCDate();
    const yr = d.getUTCFullYear();
    const label = lastYear !== yr ? `${mo} ${day} '${yr.toString().slice(2)}` : `${mo} ${day}`;
    lastYear = yr;
    ctx.fillStyle = "#7c87a8";
    ctx.fillText(label, x + 2, cssH - 8);
  }

  // Forecast region shading (everything after last observed date)
  if (_chartState.lastObsDate) {
    const tCut = Date.parse(_chartState.lastObsDate);
    const xCut = xAt(tCut);
    ctx.fillStyle = "rgba(140, 197, 255, 0.05)";
    ctx.fillRect(xCut, padT, padL + plotW - xCut, plotH);
    ctx.strokeStyle = "rgba(140, 197, 255, 0.35)";
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(xCut, padT); ctx.lineTo(xCut, padT + plotH); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Min/max envelope across members
  if (_chartState.showBand && memberKeys.length >= 2 && blend.length) {
    const byDate = new Map();
    for (const k of memberKeys) {
      for (const p of _chartState.members[k]) {
        const t = Date.parse(p.date);
        const v = Number(p.snow_depth_in);
        if (!Number.isFinite(v)) continue;
        const arr = byDate.get(t) || [];
        arr.push(v); byDate.set(t, arr);
      }
    }
    const sorted = [...byDate.entries()].sort((a, b) => a[0] - b[0]);
    if (sorted.length >= 2) {
      ctx.fillStyle = "rgba(140, 197, 255, 0.18)";
      ctx.beginPath();
      const upper = sorted.map(([t, arr]) => [t, Math.max(...arr)]);
      const lower = sorted.map(([t, arr]) => [t, Math.min(...arr)]);
      ctx.moveTo(xAt(upper[0][0]), yAt(upper[0][1]));
      for (const [t, v] of upper) ctx.lineTo(xAt(t), yAt(v));
      for (let i = lower.length - 1; i >= 0; i--) ctx.lineTo(xAt(lower[i][0]), yAt(lower[i][1]));
      ctx.closePath(); ctx.fill();
    }
  }

  // Individual members (faint, dashed)
  if (_chartState.showMembers) {
    const colors = ["#ff8a4c", "#4cc8ff", "#b889ff", "#ffd166", "#5fe5a8", "#ff5c8a"];
    let i = 0;
    for (const k of memberKeys) {
      const pts = _chartState.members[k].filter(p => p.snow_depth_in != null);
      if (!pts.length) continue;
      ctx.strokeStyle = colors[i % colors.length];
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      pts.forEach((p, idx) => {
        const x = xAt(Date.parse(p.date)), y = yAt(Number(p.snow_depth_in));
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      i++;
    }
    ctx.setLineDash([]);
  }

  // History (solid white)
  if (hist.length) {
    ctx.strokeStyle = "#e7ecf3"; ctx.lineWidth = 1.8;
    ctx.beginPath();
    hist.forEach((p, idx) => {
      const x = xAt(Date.parse(p.date)), y = yAt(Number(p.snow_depth_in));
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Ensemble blend (solid gold)
  if (blend.length) {
    ctx.strokeStyle = "#ffd166"; ctx.lineWidth = 2.2;
    ctx.beginPath();
    blend.forEach((p, idx) => {
      const x = xAt(Date.parse(p.date)), y = yAt(Number(p.snow_depth_in));
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Legend
  ctx.fillStyle = "#aab7d4"; ctx.font = "11px Inter, sans-serif";
  ctx.fillText("— observed", padL + 6, padT + 12);
  ctx.fillStyle = "#ffd166";
  ctx.fillText("— ensemble blend", padL + 80, padT + 12);
  if (_chartState.showBand) {
    ctx.fillStyle = "#8ec5ff";
    ctx.fillText("□ member min/max", padL + 188, padT + 12);
  }
}

function renderSummary(payload) {
  const el = document.getElementById("forecast-summary");
  const blend = payload.blend || [];
  const last = payload.last_observed;
  const last7 = blend[Math.min(6, blend.length - 1)];
  const allDepths = blend.map(p => Number(p.snow_depth_in)).filter(v => Number.isFinite(v));
  const maxFc = allDepths.length ? Math.max(...allDepths) : null;
  const minFc = allDepths.length ? Math.min(...allDepths) : null;
  const delta = last != null && last7 != null ? last7.snow_depth_in - last : null;
  el.innerHTML = `
    <table>
      <tr><td>Current depth</td><td><b>${fmtNumber(last, 1)} in</b></td></tr>
      <tr><td>Day +7 (blend)</td><td><b>${fmtNumber(last7?.snow_depth_in, 1)} in</b> ${delta != null ? `(${delta >= 0 ? "+" : ""}${delta.toFixed(1)} in)` : ""}</td></tr>
      <tr><td>Min over horizon</td><td>${fmtNumber(minFc, 1)} in</td></tr>
      <tr><td>Max over horizon</td><td>${fmtNumber(maxFc, 1)} in</td></tr>
    </table>
  `;
}

function renderMemberTable(payload) {
  const el = document.getElementById("member-table");
  const weights = payload.weights || {};
  const mae = payload.rolling_mae || {};
  const members = payload.members || {};
  const rows = [];
  for (const k of Object.keys(members)) {
    const last = members[k][members[k].length - 1];
    rows.push(`<tr><td>${k}</td><td>${fmtNumber(weights[k] * 100, 1)}%</td><td>${fmtNumber(mae[k], 2)}</td><td>${fmtNumber(last?.snow_depth_in, 1)}</td></tr>`);
  }
  const blendLast = (payload.blend || []).slice(-1)[0];
  rows.push(`<tr class="ensemble-row"><td>ensemble_blend</td><td>—</td><td>${fmtNumber(mae.ensemble_blend, 2)}</td><td>${fmtNumber(blendLast?.snow_depth_in, 1)}</td></tr>`);
  el.innerHTML = `<table>
    <thead><tr><th>Member</th><th>Weight</th><th>MAE (in)</th><th>Day +${payload.horizon_days}</th></tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`;
}

async function selectStation(station) {
  currentStation = station;
  document.getElementById("panel-empty").style.display = "none";
  document.getElementById("panel-content").style.display = "block";
  document.getElementById("forecast-status").textContent = "loading forecast…";
  try {
    const r = await fetch(`forecasts/${station.id}.json`);
    if (!r.ok) throw new Error(`${r.status}`);
    const payload = await r.json();
    renderMeta(station, payload);
    renderSummary(payload);
    renderMemberTable(payload);
    drawChart(document.getElementById("chart"), payload.history, payload.members, payload.blend, payload.last_observed_date);
    document.getElementById("forecast-status").textContent = `issued ${payload.issued_at} · horizon ${payload.horizon_days} d`;
  } catch (e) {
    document.getElementById("forecast-status").textContent = `error: ${e.message}`;
  }
}

document.getElementById("toggle-members").addEventListener("click", (ev) => {
  const next = ev.currentTarget.getAttribute("aria-pressed") !== "true";
  ev.currentTarget.setAttribute("aria-pressed", String(next));
  _chartState.showMembers = next;
  _renderChart();
});
document.getElementById("toggle-band").addEventListener("click", (ev) => {
  const next = ev.currentTarget.getAttribute("aria-pressed") !== "true";
  ev.currentTarget.setAttribute("aria-pressed", String(next));
  _chartState.showBand = next;
  _renderChart();
});
document.getElementById("refresh-btn").addEventListener("click", () => {
  if (currentStation) selectStation(currentStation);
});
window.addEventListener("resize", _renderChart);

loadStations();
