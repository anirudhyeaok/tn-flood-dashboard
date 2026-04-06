"""
cyclone_tracker.py — Live Cyclone Detection for Tamil Nadu
==========================================================
Replaces the single-point storm_tracker.py with a district-level wind scan.

Detects which TN districts are under cyclone-force winds in real time,
approximates a track from the wind field, and returns structured alert data.

Classification basis:
  IMD Cyclone Intensity Scale (India Meteorological Department):
  - Depression:        Wind 17–27 km/h
  - Deep Depression:   Wind 28–61 km/h
  - Cyclonic Storm:    Wind 62–88 km/h
  - Severe CS:         Wind 89–117 km/h
  - Very Severe CS:    Wind 118–167 km/h
  - Extremely Severe:  Wind 168–221 km/h
  Source: IMD Technical Document No. ESSO/IMD/NHAC/CYCLONE/2020/01

Data Source: Open-Meteo Real-Time API (GFS 0.25°)
API: https://open-meteo.com/en/docs
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from live_engine import DISTRICT_CENTROIDS, OPEN_METEO_BASE

# ── IMD wind classification ───────────────────────────────────────────────────
# (min_kmh, label, severity_index)
IMD_SCALE = [
    (168, "Extremely Severe Cyclonic Storm", 5),
    (118, "Very Severe Cyclonic Storm", 4),
    (89, "Severe Cyclonic Storm", 3),
    (62, "Cyclonic Storm", 2),
    (28, "Deep Depression", 1),
    (17, "Depression", 0),
]


def classify_wind(wind_kmh: float) -> dict:
    """Returns IMD classification for a given wind speed."""
    for threshold, label, severity in IMD_SCALE:
        if wind_kmh >= threshold:
            return {"label": label, "severity": severity, "threshold_kmh": threshold}
    return {"label": "No Significant System", "severity": -1, "threshold_kmh": 0}


# ── Fetch wind data ───────────────────────────────────────────────────────────

def _fetch_wind(lat: float, lon: float) -> dict:
    """Fetches current max wind speed for a lat/lon."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "windspeed_10m,winddirection_10m,windgusts_10m",
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }
    try:
        resp = requests.get(OPEN_METEO_BASE, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        winds = data.get("windspeed_10m", [0])
        gusts = data.get("windgusts_10m", [0])
        dirs = data.get("winddirection_10m", [0])
        max_idx = int(np.argmax(winds))
        return {
            "max_wind_kmh": round(max(winds), 1),
            "max_gust_kmh": round(max(gusts), 1),
            "direction_deg": dirs[max_idx],
        }
    except Exception as e:
        return {"max_wind_kmh": 0.0, "max_gust_kmh": 0.0, "direction_deg": 0, "error": str(e)}


# ── Main tracker ──────────────────────────────────────────────────────────────

def scan_cyclone_activity() -> dict:
    """
    Scans all TN districts for cyclone-force winds.

    Returns:
        {
          "is_active": bool,
          "system_type": str (IMD classification of strongest reading),
          "severity": int (0–5, -1 = none),
          "affected_districts": [ { district, wind_kmh, classification }, ... ],
          "approximate_track": [ [lat, lon], ... ],  # centroid path of >60 km/h districts
          "max_wind_district": str,
          "max_wind_kmh": float,
          "fetched_at": str,
          "source": str,
          "imd_reference": str,
        }
    """
    district_winds = {}
    for district, (lat, lon) in DISTRICT_CENTROIDS.items():
        district_winds[district] = {
            "lat": lat, "lon": lon,
            **_fetch_wind(lat, lon)
        }

    # Find districts under cyclone-force wind (IMD: ≥62 km/h = Cyclonic Storm)
    cyclone_threshold = 60  # slightly below formal 62 to catch edges
    affected = []
    for district, data in district_winds.items():
        wind = data["max_wind_kmh"]
        classification = classify_wind(wind)
        if wind >= cyclone_threshold:
            affected.append({
                "district": district,
                "wind_kmh": wind,
                "gust_kmh": data.get("max_gust_kmh", 0),
                "lat": data["lat"],
                "lon": data["lon"],
                "classification": classification["label"],
                "severity": classification["severity"],
            })

    # Sort by wind speed descending
    affected.sort(key=lambda x: x["wind_kmh"], reverse=True)

    is_active = len(affected) > 0
    max_severity = max((a["severity"] for a in affected), default=-1)
    system_type = classify_wind(
        max((a["wind_kmh"] for a in affected), default=0)
    )["label"]

    # Approximate track: order affected districts by lon (west→east landfall pattern)
    # This gives a rough line showing where the system is moving through
    approx_track = []
    if affected:
        track_districts = sorted(affected, key=lambda x: x["lon"])
        approx_track = [[a["lat"], a["lon"]] for a in track_districts]

    # Highest wind district
    max_district = affected[0]["district"] if affected else None
    max_wind = affected[0]["wind_kmh"] if affected else 0.0

    return {
        "is_active": is_active,
        "system_type": system_type if is_active else "No Active System",
        "severity": max_severity,
        "affected_districts": affected,
        "approximate_track": approx_track,
        "max_wind_district": max_district,
        "max_wind_kmh": max_wind,
        "district_winds": district_winds,  # full data for map layer
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "Open-Meteo GFS 0.25° Real-Time Wind",
        "imd_reference": "IMD Cyclone Scale: ESSO/IMD/NHAC/CYCLONE/2020/01",
    }


# ── Alert banner data ─────────────────────────────────────────────────────────

SEVERITY_BANNERS = {
    5: {"color": "#7c3aed", "text": "EXTREMELY SEVERE CYCLONIC STORM", "bg": "#4c1d95"},
    4: {"color": "#dc2626", "text": "VERY SEVERE CYCLONIC STORM", "bg": "#7f1d1d"},
    3: {"color": "#ea580c", "text": "SEVERE CYCLONIC STORM", "bg": "#7c2d12"},
    2: {"color": "#d97706", "text": "CYCLONIC STORM", "bg": "#78350f"},
    1: {"color": "#ca8a04", "text": "DEEP DEPRESSION", "bg": "#713f12"},
    0: {"color": "#65a30d", "text": "DEPRESSION", "bg": "#365314"},
}


def get_cyclone_banner(scan_result: dict) -> dict | None:
    """Returns a banner dict if a system is active, else None."""
    if not scan_result["is_active"]:
        return None
    severity = scan_result["severity"]
    banner = SEVERITY_BANNERS.get(severity, {})
    return {
        **banner,
        "system_type": scan_result["system_type"],
        "max_district": scan_result["max_wind_district"],
        "max_wind_kmh": scan_result["max_wind_kmh"],
        "affected_count": len(scan_result["affected_districts"]),
        "source": scan_result["source"],
        "fetched_at": scan_result["fetched_at"],
    }