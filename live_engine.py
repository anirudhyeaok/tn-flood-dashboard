"""
live_engine.py — Real-Time Flood Risk Engine
=============================================
Fetches live hourly rainfall from Open-Meteo API for all Tamil Nadu districts.
Feeds real data into the existing flood physics engine.

Data Source: Open-Meteo (https://open-meteo.com) — Free, no API key required.
Reference: Hersbach et al. (2020), ERA5 reanalysis; Open-Meteo forecast model.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from flood_engine import (
    calculate_flood_depth, determine_risk,
    daily_drainage_capacity, COASTAL_DISTRICTS, tn_map
)

# ── District centroids (lat, lon) for API calls ───────────────────────────────
# Derived from TamilNadu.geojson centroids (NAME_2 column)
# These are approximate geographic centers used solely for weather data retrieval.
DISTRICT_CENTROIDS = {
    "Chennai":          (13.082, 80.270),
    "Thiruvallur":      (13.143, 79.906),
    "Chengalpattu":     (12.692, 79.982),
    "Kancheepuram":     (12.832, 79.703),
    "Ranipet":          (12.922, 79.333),
    "Vellore":          (12.916, 79.132),
    "Tirupattur":       (12.497, 78.573),
    "Krishnagiri":      (12.519, 78.213),
    "Dharmapuri":       (12.127, 78.158),
    "Salem":            (11.665, 78.146),
    "Namakkal":         (11.219, 78.167),
    "Erode":            (11.341, 77.717),
    "Tiruppur":         (11.108, 77.340),
    "Coimbatore":       (11.016, 76.955),
    "The Nilgiris":     (11.412, 76.695),
    "Dindigul":         (10.362, 77.972),
    "Karur":            (10.960, 78.080),
    "Tiruchirappalli":  (10.790, 78.704),
    "Perambalur":       (11.233, 78.880),
    "Ariyalur":         (11.140, 79.078),
    "Cuddalore":        (11.748, 79.768),
    "Villupuram":       (11.939, 79.493),
    "Kallakurichi":     (11.739, 78.959),
    "Mayiladuthurai":   (11.103, 79.652),
    "Nagapattinam":     (10.765, 79.844),
    "Thiruvarur":       (10.773, 79.637),
    "Thanjavur":        (10.787, 79.138),
    "Tiruvarur":        (10.773, 79.637),
    "Pudukottai":       (10.379, 78.820),
    "Pudukkottai":      (10.379, 78.820),
    "Sivaganga":        (9.846,  78.480),
    "Madurai":          (9.925,  78.120),
    "Ramanathapuram":   (9.371,  78.830),
    "Virudhunagar":     (9.587,  77.952),
    "Tenkasi":          (8.960,  77.315),
    "Tirunelveli":      (8.727,  77.700),
    "Thoothukudi":      (8.764,  78.135),
    "Kanyakumari":      (8.088,  77.552),
}

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# ── API fetch ─────────────────────────────────────────────────────────────────

def _fetch_district_rain_today(lat: float, lon: float) -> dict:
    """
    Calls Open-Meteo for a single lat/lon.
    Returns today's cumulative rainfall (mm) and current wind speed (km/h).
    
    API Docs: https://open-meteo.com/en/docs
    Variables used:
      - precipitation: hourly precip in mm (WMO indicator 4677)
      - windspeed_10m: 10m wind in km/h
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation,windspeed_10m",
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }
    try:
        resp = requests.get(OPEN_METEO_BASE, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        precip = hourly.get("precipitation", [0] * 24)
        wind   = hourly.get("windspeed_10m", [0] * 24)
        return {
            "daily_total_mm": float(sum(precip)),
            "hourly_precip":  precip,
            "max_wind_kmh":   float(max(wind)),
            "source":         "Open-Meteo ERA5-seamless + GFS",
            "fetched_at":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
    except Exception as e:
        return {
            "daily_total_mm": 0.0,
            "hourly_precip":  [0.0] * 24,
            "max_wind_kmh":   0.0,
            "source":         "Open-Meteo (fetch failed)",
            "fetched_at":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "error":          str(e),
        }


def fetch_all_districts_live() -> dict:
    """
    Fetches today's live rainfall for every TN district.
    Returns a dict keyed by district name with rain data + metadata.
    """
    results = {}
    for district, (lat, lon) in DISTRICT_CENTROIDS.items():
        results[district] = _fetch_district_rain_today(lat, lon)
    return results


# ── Flood risk from live data ─────────────────────────────────────────────────

def get_live_risk_map() -> dict:
    """
    Main entry point for the Live module.
    
    Returns:
        {
          district_name: {
            "risk": "HIGH" | "MODERATE" | "LOW" | "NO FLOOD",
            "color": hex string,
            "rain_today_mm": float,
            "flood_depth_mm": float,
            "wind_kmh": float,
            "is_cyclone_wind": bool,
            "source": str,
            "fetched_at": str,
            "confidence": "LIVE"
          },
          ...
          "_meta": { "fetched_at": ..., "source": ... }
        }
    """
    live_data = fetch_all_districts_live()
    output = {}

    for district, rain_info in live_data.items():
        rain_mm = rain_info["daily_total_mm"]
        wind    = rain_info["max_wind_kmh"]

        # Build a 1-day rain series for the flood engine
        rain_series = pd.Series(
            [rain_mm],
            index=pd.date_range(pd.Timestamp.today().normalize(), periods=1)
        )

        # Run through existing physics (no spatial track for live, dist_km=9999)
        water_s, eff_r, dist_km = calculate_flood_depth(
            rain_series, district, speed=wind if wind > 0 else 12, blockage=0, track=None
        )

        depth = float(water_s.iloc[0]) if not water_s.empty else 0.0
        risk, color = determine_risk(depth, dist_km)

        output[district] = {
            "risk":           risk,
            "color":          color,
            "rain_today_mm":  round(rain_mm, 1),
            "flood_depth_mm": round(depth, 1),
            "wind_kmh":       round(wind, 1),
            "is_cyclone_wind": wind >= 60,
            "source":         rain_info["source"],
            "fetched_at":     rain_info["fetched_at"],
            "confidence":     "LIVE",
            "Accumulation":   round(depth, 1),    # legacy key for frontend
            "Daily_Rain":     round(rain_mm, 1),  # legacy key for frontend
            "Total_Rain":     round(rain_mm, 1),
            "Risk":           risk,
            "Color":          color,
            "Debug_W":        1.0,
            "Debug_Dist":     0,
        }

    # Global metadata for the attribution panel
    sample = next(iter(live_data.values()), {})
    output["_meta"] = {
        "fetched_at": sample.get("fetched_at", "Unknown"),
        "source":     sample.get("source", "Open-Meteo"),
        "mode":       "LIVE",
        "note":       "Rainfall: Open-Meteo ERA5-seamless (hourly). Wind: GFS 0.25°.",
    }

    return output


# ── Alert generation ──────────────────────────────────────────────────────────

ALERT_THRESHOLDS = {
    "HIGH":     {"level": "RED ALERT",    "icon": "🔴", "action": "Evacuate low-lying areas immediately"},
    "MODERATE": {"level": "ORANGE ALERT", "icon": "🟠", "action": "Restrict movement, monitor waterways"},
    "LOW":      {"level": "YELLOW ALERT", "icon": "🟡", "action": "Stay alert, avoid flood-prone zones"},
}

def generate_alerts(risk_map: dict) -> list:
    """
    Generates structured alert objects from a live risk map.
    Suitable for rendering an alert feed in the UI.
    
    Threshold basis: NDMA India colour-coded alert system.
    Reference: NDMA Guidelines on Flood Management (2008), Chapter 4.
    """
    alerts = []
    now_str = datetime.now().strftime("%d %b %Y, %H:%M IST")

    for district, data in risk_map.items():
        if not isinstance(data, dict):
            continue
        risk = data.get("risk", "NO FLOOD")
        if risk in ALERT_THRESHOLDS:
            t = ALERT_THRESHOLDS[risk]
            alerts.append({
                "district":   district,
                "level":      t["level"],
                "icon":       t["icon"],
                "action":     t["action"],
                "rain_mm":    data.get("rain_today_mm", 0),
                "depth_mm":   data.get("flood_depth_mm", 0),
                "wind_kmh":   data.get("wind_kmh", 0),
                "timestamp":  now_str,
                "source":     data.get("source", "Open-Meteo"),
            })

    # Sort by severity
    order = {"RED ALERT": 0, "ORANGE ALERT": 1, "YELLOW ALERT": 2}
    alerts.sort(key=lambda x: order.get(x["level"], 3))
    return alerts