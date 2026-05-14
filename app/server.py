"""Local Flask app for live SnowWatch forecasts.

Run from the project root:
    python -m app.server --host 0.0.0.0 --port 8000

Endpoints
---------
GET /                       index.html with the Leaflet map
GET /api/stations           full SNOTEL station list
GET /api/forecast/<id>      run the ensemble for one station
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from . import snotel
from .forecast import forecast_station

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATIONS_PATH = Path(os.environ.get("SW_STATIONS_FILE") or (DATA / "stations.json"))

app = Flask(__name__, template_folder=str(ROOT / "app" / "templates"),
            static_folder=str(ROOT / "app" / "static"))

_CACHE: dict = {}
_TTL = 1800  # 30 min


def _load_stations() -> list[dict]:
    if STATIONS_PATH.exists():
        payload = json.loads(STATIONS_PATH.read_text())
        return payload.get("stations", [])
    return snotel.list_active_stations()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stations")
def api_stations():
    return jsonify({"stations": _load_stations()})


@app.route("/api/forecast/<sid>")
def api_forecast(sid: str):
    force = request.args.get("refresh") == "1"
    now = time.time()
    if not force and sid in _CACHE and now - _CACHE[sid]["t"] < _TTL:
        return jsonify(_CACHE[sid]["payload"])
    stations = _load_stations()
    match = next((s for s in stations if str(s.get("id")) == str(sid)), None)
    if not match:
        return jsonify({"error": f"unknown station {sid}"}), 404
    fc = forecast_station(match)
    payload = asdict(fc)
    payload["station"] = match
    _CACHE[sid] = {"t": now, "payload": payload}
    return jsonify(payload)


@app.route("/static/<path:p>")
def static_proxy(p):
    return send_from_directory(str(ROOT / "app" / "static"), p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
