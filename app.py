"""
app.py — TNSDMA Flood Intelligence System
==========================================
All live/forecast data is served from cache_manager.py (pre-fetched background).
Page loads are now instant — no API calls on request.
"""

import atexit
import json
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request

# ── Core engine (historical / physics) ───────────────────────────────────────
from flood_engine import (
    CYCLONE_PATHS, CYCLONES, analyze_flood_risk, daily,
    get_daily_frame_data, get_heatmap_data, tn_map,
)

# ── Cache manager (live + forecast + cyclone) ─────────────────────────────────
from cache_manager import (
    get_cache_status, get_cyclone_data, get_forecast_data,
    get_live_data, start_scheduler, stop_scheduler,
)

# ── Cyclone banner helper ─────────────────────────────────────────────────────
from cyclone_tracker import get_cyclone_banner

# ── Forecast chart helper ─────────────────────────────────────────────────────
from forecast_engine import get_forecast_chart_data

# ── Live alert helper ─────────────────────────────────────────────────────────
from live_engine import generate_alerts

app = Flask(__name__)

# ── Start background scheduler once at startup ────────────────────────────────
start_scheduler()
atexit.register(stop_scheduler)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _available_cyclones():
    available = []
    for name, (s, e, _) in CYCLONES.items():
        year = int(s.split("-")[0])
        if not daily.empty and year in daily.index.year:
            available.append(name)
        elif daily.empty:
            available.append(name)
    return available


def _package_historical(start, end, speed, blockage, track_path):
    raw_frames  = get_daily_frame_data(start, end, speed, blockage, track=track_path)
    heat_frames = get_heatmap_data(start, end, speed, blockage, track=track_path)
    sim_data = {}
    dates    = []
    if raw_frames:
        dates     = [d["Date"].iloc[0] for d in raw_frames]
        districts = raw_frames[0]["District"].unique()
        for d in districts:
            sim_data[d] = []
        for frame in raw_frames:
            for _, row in frame.iterrows():
                sim_data[row["District"]].append({
                    "Accumulation": row["Accumulation"],
                    "Daily_Rain":   row["Daily_Rain"],
                    "Total_Rain":   row["Total_Rain"],
                    "Risk":         row["Risk"],
                    "Color":        row["Color"],
                    "Debug_W":      row["Debug_W"],
                    "Debug_Dist":   row["Debug_Dist"],
                    "depth":      row["Accumulation"],
                    "daily_rain": row["Daily_Rain"],
                    "total_rain": row["Total_Rain"],
                    "risk":       row["Risk"],
                    "color":      row["Color"],
                })
    return sim_data, dates, heat_frames


def _package_live(live_map: dict):
    today    = datetime.now().strftime("%Y-%m-%d")
    sim_data = {}
    for district, data in live_map.items():
        if not isinstance(data, dict):
            continue
        sim_data[district] = [data]
    return sim_data, [today]


def _package_forecast(forecast_map: dict):
    sim_data = {}
    dates    = []
    first    = True
    for district, data in forecast_map.items():
        if not isinstance(data, dict):
            continue
        sim_data[district] = []
        for day in data.get("days", []):
            sim_data[district].append(day)
            if first:
                dates.append(day["date"])
        first = False
    return sim_data, dates


# ── Main route ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    active_mode     = "live"
    result          = None
    data_meta       = None
    alerts          = []
    cyclone_banner  = None
    confidence_note = None
    cache_status    = get_cache_status()

    cyclone    = "Michaung 2023"
    speed      = 12
    blockage   = 0
    start      = pd.Timestamp.today() - pd.Timedelta(days=7)
    end        = pd.Timestamp.today()
    track_path = []

    if request.method == "POST":
        active_mode = request.form.get("mode", "live")
        if active_mode == "historical":
            cyclone  = request.form.get("cyclone", "Michaung 2023")
            blockage = int(request.form.get("blockage", 0))
            if cyclone in CYCLONES:
                start, end, speed = CYCLONES[cyclone]
                track_path = CYCLONE_PATHS.get(cyclone, [])
        elif active_mode == "custom":
            start    = request.form.get("start")
            end      = request.form.get("end")
            speed    = float(request.form.get("speed", 12))
            blockage = int(request.form.get("blockage", 0))

    if active_mode == "live":
        live_map         = get_live_data()
        sim_data, dates  = _package_live(live_map)
        heat_frames      = []
        data_meta        = live_map.get("_meta", {})
        alerts           = generate_alerts(live_map)
        cyclone_scan     = get_cyclone_data()
        cyclone_banner   = get_cyclone_banner(cyclone_scan)
        if cyclone_banner:
            cyclone_banner["approximate_track"] = cyclone_scan.get("approximate_track", [])
        chennai = live_map.get("Chennai", {})
        result  = {
            "risk":              chennai.get("risk", "N/A"),
            "total_rain":        chennai.get("rain_today_mm", 0),
            "peak_accumulation": chennai.get("flood_depth_mm", 0),
            "wind_speed":        chennai.get("wind_kmh", 0),
            "chart_labels":      [datetime.now().strftime("%b %d")],
            "chart_rain":        [chennai.get("rain_today_mm", 0)],
            "chart_water":       [chennai.get("flood_depth_mm", 0)],
            "source":            data_meta.get("source", "Open-Meteo"),
            "fetched_at":        data_meta.get("fetched_at", ""),
        }

    elif active_mode == "forecast":
        forecast_map     = get_forecast_data()
        sim_data, dates  = _package_forecast(forecast_map)
        heat_frames      = []
        data_meta        = forecast_map.get("_meta", {})
        confidence_note  = data_meta.get("note", "")
        cyclone_scan     = get_cyclone_data()
        cyclone_banner   = get_cyclone_banner(cyclone_scan)
        if cyclone_banner:
            cyclone_banner["approximate_track"] = cyclone_scan.get("approximate_track", [])
        chennai_fc    = forecast_map.get("Chennai", {})
        chennai_chart = get_forecast_chart_data("Chennai", forecast_map)
        if chennai_fc and chennai_fc.get("days"):
            peak_day = max(chennai_fc["days"], key=lambda d: d.get("flood_depth_mm", 0))
            result = {
                "risk":              chennai_fc.get("peak_risk", "N/A"),
                "total_rain":        round(sum(d.get("rain_mm", 0) for d in chennai_fc["days"]), 1),
                "peak_accumulation": peak_day.get("flood_depth_mm", 0),
                "wind_speed":        max(d.get("wind_kmh", 0) for d in chennai_fc["days"]),
                "chart_labels":      chennai_chart.get("labels", []),
                "chart_rain":        chennai_chart.get("rain_mid", []),
                "chart_water":       chennai_chart.get("depth_mid", []),
                "chart_rain_low":    chennai_chart.get("rain_low", []),
                "chart_rain_high":   chennai_chart.get("rain_high", []),
                "chart_depth_low":   chennai_chart.get("depth_low", []),
                "chart_depth_high":  chennai_chart.get("depth_high", []),
                "confidence_labels": chennai_chart.get("confidence", []),
                "conf_colors":       chennai_chart.get("conf_colors", []),
                "source":            chennai_fc.get("source", "Open-Meteo"),
                "fetched_at":        chennai_fc.get("fetched_at", ""),
            }

    else:  # historical
        sim_data, dates, heat_frames = _package_historical(
            start, end, speed, blockage, track_path
        )
        data_meta = {
            "source":     "CHIRPS v2.0 Satellite Rainfall (UCSB)",
            "mode":       "HISTORICAL",
            "note":       (
                "CHIRPS v2.0 (Climate Hazards Group InfraRed Precipitation with Station data), "
                "0.05° resolution. Funk et al. (2015), Sci. Data 2, 150066."
            ),
            "fetched_at": f"Preprocessed — {datetime.now().strftime('%d %b %Y')}",
        }
        result = analyze_flood_risk(
            "Chennai", start, end, speed, blockage, track=track_path
        )

    frontend_data = {
        "map_geo":    json.loads(tn_map.to_json()),
        "simulation": sim_data,
        "dates":      dates,
        "heatmap":    heat_frames,
    }

    return render_template(
        "index.html",
        result           = result,
        frontend_data    = frontend_data,
        track_path       = track_path,
        cyclones         = _available_cyclones(),
        selected_cyclone = cyclone,
        active_mode      = active_mode,
        blockage         = blockage,
        now              = datetime.now().strftime("%d %b %Y"),
        data_meta        = data_meta,
        alerts           = alerts,
        cyclone_banner   = cyclone_banner,
        confidence_note  = confidence_note,
        cache_status     = cache_status,
    )


# ── JSON API endpoints ────────────────────────────────────────────────────────

@app.route("/api/live")
def api_live():
    return jsonify(get_live_data())

@app.route("/api/forecast")
def api_forecast():
    return jsonify(get_forecast_data())

@app.route("/api/cyclone")
def api_cyclone():
    return jsonify(get_cyclone_data())

@app.route("/api/alerts")
def api_alerts():
    return jsonify(generate_alerts(get_live_data()))

@app.route("/api/cache-status")
def api_cache_status():
    return jsonify(get_cache_status())


if __name__ == "__main__":
    app.run(debug=True)