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
let _latestDaily = [];

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

  // Show the full snow-water year: clip history to Oct 1 of the current
  // season (Oct 1 of the previous calendar year if we're before Oct, else
  // Oct 1 of the current year).
  const histAll = _chartState.history.filter(p => p.snow_depth_in != null);
  const now = new Date();
  const seasonStart = new Date(Date.UTC(
    now.getUTCMonth() >= 9 ? now.getUTCFullYear() : now.getUTCFullYear() - 1,
    9, 1,
  )).getTime();
  const hist = histAll.filter(p => Date.parse(p.date) >= seasonStart);
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

function _sizeCanvas(c, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = c.clientWidth || 640;
  c.width = Math.round(cssW * dpr);
  c.height = Math.round(cssH * dpr);
  c.style.height = cssH + "px";
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  return { ctx, cssW, cssH };
}

function drawSnowfallChart(canvas, daily) {
  if (!canvas) return;
  const { ctx, cssW, cssH } = _sizeCanvas(canvas, 160);
  if (!daily || !daily.length) {
    ctx.fillStyle = "#7c87a8"; ctx.font = "12px Inter, sans-serif";
    ctx.fillText("no snowfall forecast available", 12, 24); return;
  }
  const padL = 44, padR = 12, padT = 12, padB = 28;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;
  const values = daily.map(d => Number(d.snowfall_in) || 0);
  const ymax = Math.max(0.5, ...values) * 1.15;

  ctx.strokeStyle = "#1f2a44"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 3; i++) {
    const y = padT + (i / 3) * plotH;
    ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#7c87a8"; ctx.font = "11px Inter, sans-serif";
  for (let i = 0; i <= 3; i++) {
    const v = ymax - (i / 3) * ymax;
    ctx.fillText(v.toFixed(1), 4, padT + (i / 3) * plotH + 4);
  }

  const barW = (plotW / daily.length) * 0.66;
  const slot = plotW / daily.length;
  daily.forEach((d, i) => {
    const v = Number(d.snowfall_in) || 0;
    const h = (v / ymax) * plotH;
    const x = padL + i * slot + (slot - barW) / 2;
    const y = padT + plotH - h;
    // gradient blue->white based on amount
    const intensity = Math.min(1, v / 6);
    const r = Math.round(140 + 100 * intensity);
    const g = Math.round(197 + 50 * intensity);
    const b = 255;
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, y, barW, h);
    if (v >= 0.1) {
      ctx.fillStyle = "#e7ecf3";
      ctx.font = "10px Inter, sans-serif";
      const lbl = v.toFixed(1);
      const lw = ctx.measureText(lbl).width;
      ctx.fillText(lbl, x + barW / 2 - lw / 2, y - 3);
    }
    // date tick
    ctx.fillStyle = "#7c87a8";
    ctx.font = "10px Inter, sans-serif";
    const dd = new Date(Date.parse(d.date));
    const tickLbl = `${dd.toLocaleString(undefined, { month: "short", timeZone: "UTC" })} ${dd.getUTCDate()}`;
    const tw = ctx.measureText(tickLbl).width;
    ctx.fillText(tickLbl, x + barW / 2 - tw / 2, cssH - 8);
  });

  ctx.fillStyle = "#8ec5ff"; ctx.font = "11px Inter, sans-serif";
  const total = values.reduce((s, v) => s + v, 0);
  ctx.fillText(`Σ ${total.toFixed(1)} in over ${daily.length}d`, padL + 6, padT + 12);
}

function drawTempChart(canvas, daily) {
  if (!canvas) return;
  const { ctx, cssW, cssH } = _sizeCanvas(canvas, 180);
  if (!daily || !daily.length) {
    ctx.fillStyle = "#7c87a8"; ctx.font = "12px Inter, sans-serif";
    ctx.fillText("no temperature forecast available", 12, 24); return;
  }
  const padL = 44, padR = 12, padT = 14, padB = 28;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;

  const tmins = daily.map(d => Number(d.tmin_f)).filter(Number.isFinite);
  const tmaxs = daily.map(d => Number(d.tmax_f)).filter(Number.isFinite);
  const allT = [...tmins, ...tmaxs];
  if (!allT.length) {
    ctx.fillStyle = "#7c87a8"; ctx.font = "12px Inter, sans-serif";
    ctx.fillText("no temperature forecast available", 12, 24); return;
  }
  let ymin = Math.min(...allT), ymax = Math.max(...allT);
  const pad = Math.max(4, (ymax - ymin) * 0.15);
  ymin -= pad; ymax += pad;
  // always show the freeze line
  ymin = Math.min(ymin, 28); ymax = Math.max(ymax, 36);

  const xs = daily.map((_, i) => padL + (i + 0.5) * (plotW / daily.length));
  const yAt = v => padT + (1 - (v - ymin) / (ymax - ymin || 1)) * plotH;

  // grid
  ctx.strokeStyle = "#1f2a44"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padT + (i / 4) * plotH;
    ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#7c87a8"; ctx.font = "11px Inter, sans-serif";
  for (let i = 0; i <= 4; i++) {
    const v = ymax - (i / 4) * (ymax - ymin);
    ctx.fillText(v.toFixed(0) + "°", 4, padT + (i / 4) * plotH + 4);
  }

  // freezing-line at 32°F
  if (ymin < 32 && ymax > 32) {
    const yF = yAt(32);
    ctx.strokeStyle = "rgba(140, 197, 255, 0.45)";
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(padL, yF); ctx.lineTo(padL + plotW, yF); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#8ec5ff"; ctx.font = "10px Inter, sans-serif";
    ctx.fillText("32°F", padL + plotW - 30, yF - 3);
  }

  // tmin/tmax envelope
  ctx.fillStyle = "rgba(255, 138, 76, 0.18)";
  ctx.beginPath();
  daily.forEach((d, i) => {
    const v = Number(d.tmax_f);
    if (Number.isFinite(v)) ctx.lineTo(xs[i], yAt(v));
  });
  for (let i = daily.length - 1; i >= 0; i--) {
    const v = Number(daily[i].tmin_f);
    if (Number.isFinite(v)) ctx.lineTo(xs[i], yAt(v));
  }
  ctx.closePath(); ctx.fill();

  function drawLine(field, color, width) {
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); let started = false;
    daily.forEach((d, i) => {
      const v = Number(d[field]);
      if (!Number.isFinite(v)) return;
      const x = xs[i], y = yAt(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  drawLine("tmax_f", "#ff8a4c", 1.6);
  drawLine("tmin_f", "#8ec5ff", 1.6);
  drawLine("tmean_f", "#ffd166", 2.2);

  // point markers + labels for tmean
  ctx.fillStyle = "#ffd166";
  daily.forEach((d, i) => {
    const v = Number(d.tmean_f);
    if (!Number.isFinite(v)) return;
    const x = xs[i], y = yAt(v);
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
  });

  // date ticks
  ctx.fillStyle = "#7c87a8"; ctx.font = "10px Inter, sans-serif";
  daily.forEach((d, i) => {
    const dd = new Date(Date.parse(d.date));
    const lbl = `${dd.toLocaleString(undefined, { month: "short", timeZone: "UTC" })} ${dd.getUTCDate()}`;
    const tw = ctx.measureText(lbl).width;
    ctx.fillText(lbl, xs[i] - tw / 2, cssH - 8);
  });

  // legend
  ctx.font = "11px Inter, sans-serif";
  ctx.fillStyle = "#ff8a4c"; ctx.fillText("— tmax", padL + 6, padT + 12);
  ctx.fillStyle = "#ffd166"; ctx.fillText("— tmean", padL + 60, padT + 12);
  ctx.fillStyle = "#8ec5ff"; ctx.fillText("— tmin", padL + 120, padT + 12);
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

function renderNwsDivergence(payload) {
  const el = document.getElementById("nws-divergence");
  const nd = payload.nws_divergence;
  if (!el) return;
  if (!nd || !nd.available || !nd.summary) {
    el.innerHTML = "";
    return;
  }
  const s = nd.summary;
  const delta = s.final_delta_in;
  const deltaTxt = delta != null ? `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} in` : "—";
  const rows = (nd.daily || []).map(d => `
    <tr>
      <td>${d.date}</td>
      <td>${fmtNumber(d.sw_depth_in, 1)}</td>
      <td>${fmtNumber(d.nws_implied_depth_in, 1)}</td>
      <td>${d.delta_in != null ? (d.delta_in >= 0 ? "+" : "") + d.delta_in.toFixed(1) : "—"}</td>
      <td>${fmtNumber(d.nws_snowfall_in, 1)}${d.above_nws_snow_line ? " ⚠" : ""}</td>
      <td>${fmtNumber(d.sw_snowfall_in, 1)}</td>
    </tr>`).join("");
  el.innerHTML = `
    <details class="mae-explainer">
      <summary>vs NWS official forecast${s.office ? ` (${s.office})` : ""}: ${deltaTxt} at day +${s.horizon_days}</summary>
      <p>${s.headline || ""}</p>
      <table>
        <thead><tr><th>Date</th><th>SW depth</th><th>NWS depth</th><th>Δ in</th><th>NWS snow</th><th>SW snow</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p style="opacity:0.7">"NWS depth" applies NWS snowfall + degree-day melt to today's observed depth — what you'd expect trusting the public forecast verbatim. ⚠ marks days the NWS snow level sits above the station (forecast snowfall may verify as rain).</p>
    </details>
  `;
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
    renderNwsDivergence(payload);
    drawChart(document.getElementById("chart"), payload.history, payload.members, payload.blend, payload.last_observed_date);
    _latestDaily = payload.daily_forecast || [];
    drawSnowfallChart(document.getElementById("snowfall-chart"), _latestDaily);
    drawTempChart(document.getElementById("temp-chart"), _latestDaily);
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
window.addEventListener("resize", () => {
  _renderChart();
  if (_latestDaily.length) {
    drawSnowfallChart(document.getElementById("snowfall-chart"), _latestDaily);
    drawTempChart(document.getElementById("temp-chart"), _latestDaily);
  }
});

loadStations();
