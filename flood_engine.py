import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString

# CONFIG
PROCESSED_DATA_FILE = "tn_district_rain_history.csv"
if os.path.exists("TamilNadu.geojson"):
    MAP_PATH = "TamilNadu.geojson"
else:
    MAP_PATH = r"C:\Users\crade\Desktop\project something yeah\maps\TamilNadu.geojson"

API_TIMEOUT_SECONDS = 12
DEFAULT_FORECAST_DAYS = 7

# DATA
POPULATION_DENSITY = {
    "Chennai": 26000,
    "Kancheepuram": 800,
    "Thiruvallur": 700,
    "Cuddalore": 600,
    "Madurai": 800,
    "Kanyakumari": 1100,
}
DEFAULT_DENSITY = 400

COASTAL_DISTRICTS = {
    "Chennai",
    "Thiruvallur",
    "Chengalpattu",
    "Kancheepuram",
    "Villupuram",
    "Cuddalore",
    "Mayiladuthurai",
    "Nagapattinam",
    "Thiruvarur",
    "Thanjavur",
    "Pudukkottai",
    "Ramanathapuram",
    "Thoothukudi",
    "Tirunelveli",
    "Kanyakumari",
}

masks = {}
daily = pd.DataFrame()
daily_drainage_capacity = {}
tn_map = None


def init_engine():
    global daily, daily_drainage_capacity, tn_map

    try:
        tn_map = gpd.read_file(MAP_PATH).to_crs(epsg=4326)
    except Exception as exc:
        raise FileNotFoundError("CRITICAL: TamilNadu.geojson map not found.") from exc

    if os.path.exists(PROCESSED_DATA_FILE):
        daily = pd.read_csv(PROCESSED_DATA_FILE, index_col=0, parse_dates=True)
        daily.columns = [c.strip() for c in daily.columns]
    else:
        raise FileNotFoundError("Run preprocess_v2.py first!")

    for district in daily.columns:
        wet_days = daily[district][daily[district] > 1]
        base = wet_days.quantile(0.75) if not wet_days.empty else 15.0
        daily_drainage_capacity[district] = max(30, min(base, 100))


init_engine()


# --- SPATIAL ---
def calculate_proximity_weight(district_name, cyclone_track):
    if not cyclone_track:
        return 1.0, 0

    clean_name = str(district_name).strip()
    try:
        district_geom = tn_map[tn_map["NAME_2"] == clean_name].geometry.values[0]
    except Exception:
        return 1.0, 0

    track_points = [(p[1], p[0]) for p in cyclone_track]
    track_line = LineString(track_points)
    dist_deg = district_geom.distance(track_line)
    min_dist_km = dist_deg * 111

    if min_dist_km < 100:
        return 1.0, min_dist_km
    if min_dist_km > 400:
        return 0.2, min_dist_km
    return 1.0 - ((min_dist_km - 100) / 300) * 0.8, min_dist_km


# --- SIMULATION CORE ---
def get_data_safe(district, start, end):
    if daily.empty:
        return pd.Series(0, index=pd.date_range(start, end))
    try:
        return daily.loc[start:end, district]
    except Exception:
        return pd.Series(0, index=pd.date_range(start, end))


def calculate_flood_depth(rain_series, district, speed=12, blockage=0, track=None):
    base_rate = daily_drainage_capacity.get(district, 30.0)
    blockage_factor = (100 - blockage) / 100.0
    drainage = base_rate * blockage_factor

    _, dist_km = calculate_proximity_weight(district, track)
    is_coastal = district in COASTAL_DISTRICTS

    standing = 0.0
    levels = []
    eff_rain_list = []

    for rain in rain_series:
        effective_rain = rain

        if is_coastal and dist_km < 80 and effective_rain > 20:
            effective_rain *= 1.35

        current_drainage = drainage
        if effective_rain > 120:
            current_drainage *= 0.1
        elif effective_rain > 60:
            if is_coastal and dist_km < 100:
                current_drainage *= 0.2
            else:
                current_drainage *= 0.5

        runoff_bonus = standing * 0.20 if standing > 80 else 0

        standing = standing + effective_rain - current_drainage - runoff_bonus
        if standing < 0:
            standing = 0

        levels.append(standing)
        eff_rain_list.append(effective_rain)

    return (
        pd.Series(levels, index=rain_series.index),
        pd.Series(eff_rain_list, index=rain_series.index),
        dist_km,
    )


def determine_risk(depth, dist_km):
    if dist_km < 150:
        if depth > 50:
            return "HIGH", "#ef4444"
        if depth > 20:
            return "MODERATE", "#f97316"
        if depth > 5:
            return "LOW", "#facc15"
        return "NO FLOOD", "#10b981"

    if depth > 150:
        return "HIGH", "#ef4444"
    if depth > 80:
        return "MODERATE", "#f97316"
    if depth > 30:
        return "LOW", "#facc15"
    return "NO FLOOD", "#10b981"


def _build_frames_from_rain(date_range, district_rain_df, speed=12, blockage=0, track=None):
    district_water = {}
    district_eff_rain = {}
    meta_dist = {}
    meta_w = {}

    for district in daily.columns:
        rain_values = district_rain_df.get(district, pd.Series(0.0, index=date_range)).reindex(date_range).fillna(0.0)
        water_s, eff_r, dist = calculate_flood_depth(rain_values, district, speed, blockage, track)
        district_water[district] = water_s
        district_eff_rain[district] = eff_r
        meta_dist[district] = dist
        w, _ = calculate_proximity_weight(district, track)
        meta_w[district] = w

    frames = []
    cum_rain_tracker = {district: 0.0 for district in daily.columns}

    for date in date_range:
        day_stats = []
        for district in daily.columns:
            depth = district_water[district].get(date, 0)
            daily_r = district_eff_rain[district].get(date, 0)
            cum_rain_tracker[district] += daily_r
            dist = meta_dist.get(district, 9999)
            w = meta_w.get(district, 1.0)
            risk, color = determine_risk(depth, dist)

            day_stats.append(
                {
                    "District": district,
                    "Accumulation": float(f"{depth:.1f}"),
                    "Daily_Rain": float(f"{daily_r:.1f}"),
                    "Rain": float(f"{daily_r:.1f}"),
                    "Total_Rain": float(f"{cum_rain_tracker[district]:.1f}"),
                    "Risk": risk,
                    "Color": color,
                    "Date": date.strftime("%Y-%m-%d"),
                    "Debug_W": float(f"{w:.2f}"),
                    "Debug_Dist": int(dist),
                }
            )
        frames.append(pd.DataFrame(day_stats))

    return frames


def get_daily_frame_data(start, end, speed=12, blockage=0, track=None):
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    date_range = pd.date_range(start_ts, end_ts)
    hist_df = pd.DataFrame(index=date_range)
    for district in daily.columns:
        hist_df[district] = get_data_safe(district, start_ts, end_ts).reindex(date_range).fillna(0.0).values
    return _build_frames_from_rain(date_range, hist_df, speed=speed, blockage=blockage, track=track)


def _district_centroids_for_api():
    centroids = {}
    for _, row in tn_map.iterrows():
        district = str(row["NAME_2"]).strip()
        c = row.geometry.centroid
        centroids[district] = (round(float(c.y), 4), round(float(c.x), 4))
    return centroids


def _climatology_fallback(date_range):
    fallback = pd.DataFrame(index=date_range)
    if daily.empty:
        for district in tn_map["NAME_2"]:
            fallback[str(district).strip()] = 0.0
        return fallback

    for district in daily.columns:
        vals = []
        for ts in date_range:
            same_day = daily[(daily.index.month == ts.month) & (daily.index.day == ts.day)][district]
            vals.append(float(same_day.mean()) if not same_day.empty else 0.0)
        fallback[district] = vals
    return fallback.fillna(0.0)


def _fetch_open_meteo_daily(date_range, horizon_days):
    district_coords = _district_centroids_for_api()
    rain_df = pd.DataFrame(index=date_range)
    wind_samples = []
    failures = 0

    for district, (lat, lon) in district_coords.items():
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,wind_speed_10m_max",
            "forecast_days": horizon_days,
            "timezone": "Asia/Kolkata",
        }
        try:
            res = requests.get(url, params=params, timeout=API_TIMEOUT_SECONDS)
            res.raise_for_status()
            payload = res.json()
            daily_data = payload.get("daily", {})
            times = pd.to_datetime(daily_data.get("time", []))
            rain = daily_data.get("precipitation_sum", [])
            wind = daily_data.get("wind_speed_10m_max", [])

            rain_series = pd.Series(rain, index=times, dtype=float).reindex(date_range).fillna(0.0)
            rain_df[district] = rain_series.values
            if wind:
                wind_samples.extend([float(w) for w in wind if w is not None])
        except Exception:
            failures += 1
            rain_df[district] = np.nan

    if failures > 0:
        fallback = _climatology_fallback(date_range)
        for district in rain_df.columns:
            rain_df[district] = rain_df[district].fillna(fallback.get(district, 0.0))

    avg_wind = float(np.mean(wind_samples)) if wind_samples else 0.0
    if failures == len(district_coords):
        source = "Historical fallback (API unavailable)"
    elif failures > 0:
        source = "Open-Meteo + historical fallback"
    else:
        source = "Open-Meteo forecast API"

    meta = {
        "source": source,
        "failed_districts": failures,
        "district_count": len(district_coords),
        "avg_wind_speed": round(avg_wind, 1),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return rain_df.fillna(0.0), meta


def get_live_frame_data(speed=12, blockage=0, track=None):
    date_range = pd.date_range(pd.Timestamp.today().normalize(), periods=1, freq="D")
    rain_df, meta = _fetch_open_meteo_daily(date_range, horizon_days=1)
    frames = _build_frames_from_rain(date_range, rain_df, speed=speed, blockage=blockage, track=track)
    return frames, meta


def get_forecast_frame_data(speed=12, blockage=0, track=None, days=DEFAULT_FORECAST_DAYS):
    safe_days = max(1, min(int(days), 10))
    date_range = pd.date_range(pd.Timestamp.today().normalize(), periods=safe_days, freq="D")
    rain_df, meta = _fetch_open_meteo_daily(date_range, horizon_days=safe_days)
    frames = _build_frames_from_rain(date_range, rain_df, speed=speed, blockage=blockage, track=track)
    return frames, meta


# --- HEATMAP & ANALYSIS ---
def get_heatmap_data(start, end, speed, blockage, track=None):
    heatmap_frames = []
    frames = get_daily_frame_data(start, end, speed, blockage, track)
    centroids = {}
    for _, row in tn_map.iterrows():
        cent = row.geometry.centroid
        centroids[str(row["NAME_2"]).strip()] = [cent.y, cent.x]

    for day_df in frames:
        day_points = []
        for _, row in day_df.iterrows():
            if row["Accumulation"] > 10:
                lat, lon = centroids.get(row["District"], [0, 0])
                intensity = min(row["Accumulation"] / 200.0, 1.0)
                day_points.append([lat, lon, intensity])
        heatmap_frames.append(day_points)
    return heatmap_frames


def get_heatmap_data_from_frames(frames):
    heatmap_frames = []
    centroids = {}
    for _, row in tn_map.iterrows():
        cent = row.geometry.centroid
        centroids[str(row["NAME_2"]).strip()] = [cent.y, cent.x]

    for day_df in frames:
        day_points = []
        for _, row in day_df.iterrows():
            if row["Accumulation"] > 10:
                lat, lon = centroids.get(row["District"], [0, 0])
                intensity = min(row["Accumulation"] / 200.0, 1.0)
                day_points.append([lat, lon, intensity])
        heatmap_frames.append(day_points)
    return heatmap_frames


def analyze_flood_risk(district, start, end, speed=12, blockage=0, track=None):
    frames = get_daily_frame_data(start, end, speed, blockage, track)
    return analyze_flood_risk_from_frames(district, frames, speed=speed)


def analyze_flood_risk_from_frames(district, frames, speed=12):
    total_rain = 0.0
    peak_water = 0.0
    chart_labels = []
    chart_rain = []
    chart_water = []

    for frame in frames:
        if frame.empty:
            continue

        chart_labels.append(pd.to_datetime(frame["Date"].iloc[0]).strftime("%b %d"))
        row = frame[frame["District"] == district]
        if row.empty:
            chart_rain.append(0.0)
            chart_water.append(0.0)
            continue

        total_rain = float(row["Total_Rain"].iloc[0])
        water = float(row["Accumulation"].iloc[0])
        peak_water = max(peak_water, water)

        chart_rain.append(float(row["Daily_Rain"].iloc[0]))
        chart_water.append(water)

    risk = "HIGH" if peak_water > 100 else "MODERATE" if peak_water > 50 else "LOW" if peak_water > 20 else "NO FLOOD"
    return {
        "risk": risk,
        "total_rain": float(f"{total_rain:.1f}"),
        "peak_accumulation": float(f"{peak_water:.1f}"),
        "affected_pop": 0,
        "wind_speed": float(f"{speed:.1f}"),
        "chart_labels": chart_labels,
        "chart_rain": chart_rain,
        "chart_water": chart_water,
    }


def forecast_7day():
    frames, _ = get_forecast_frame_data(days=7)
    return analyze_flood_risk_from_frames("Chennai", frames)


# --- TRACKS ---
CYCLONE_PATHS = {
    "Fengal 2024": [[10.0, 85.0], [11.5, 82.0], [12.0, 80.0], [12.2, 79.8]],
    "Michaung 2023": [[10.5, 83.0], [11.2, 82.5], [12.5, 81.5], [13.2, 80.5], [14.5, 80.2], [15.8, 80.4]],
    "Mandous 2022": [[9.5, 84.5], [10.5, 82.5], [11.8, 80.5], [12.6, 80.1]],
    "Nivar 2020": [[9.5, 84.0], [10.8, 81.5], [11.9, 79.8], [12.5, 79.0]],
    "Burevi 2020": [[8.5, 83.0], [9.0, 81.0], [9.2, 79.5], [9.1, 78.5]],
    "Gaja 2018": [[13.5, 87.0], [12.0, 84.0], [11.0, 81.0], [10.4, 79.8]],
    "Ockhi 2017": [[7.5, 77.5], [8.5, 76.0], [9.5, 74.0], [12.0, 72.0]],
    "Vardah 2016": [[11.0, 88.0], [12.5, 84.0], [13.0, 81.0], [13.1, 80.2], [13.2, 79.5]],
    "Nilam 2012": [[9.5, 83.5], [10.5, 81.5], [11.5, 80.5], [12.6, 80.1]],
    "Thane 2011": [[12.0, 86.0], [12.5, 83.0], [12.0, 80.5], [11.6, 79.8]],
}

CYCLONES = {
    "Fengal 2024": ("2024-11-25", "2024-12-04", 90),
    "Michaung 2023": ("2023-12-01", "2023-12-08", 110),
    "Mandous 2022": ("2022-12-06", "2022-12-12", 85),
    "Nivar 2020": ("2020-11-20", "2020-11-30", 120),
    "Burevi 2020": ("2020-11-30", "2020-12-05", 85),
    "Gaja 2018": ("2018-11-10", "2018-11-20", 140),
    "Ockhi 2017": ("2017-11-29", "2017-12-06", 155),
    "Vardah 2016": ("2016-12-06", "2016-12-14", 130),
    "Nilam 2012": ("2012-10-28", "2012-11-01", 85),
    "Thane 2011": ("2011-12-25", "2011-12-31", 140),
}
