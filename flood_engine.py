# ============================================================
# FLOOD ENGINE — FINAL V10 (Coastal Sensitivity & Saturation)
# ============================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from datetime import datetime
import json
import math

# CONFIG
PROCESSED_DATA_FILE = "tn_district_rain_history.csv" 
if os.path.exists("TamilNadu.geojson"):
    MAP_PATH = "TamilNadu.geojson"
else:
    MAP_PATH = r"C:\Users\crade\Desktop\project something yeah\maps\TamilNadu.geojson"

# DATA
POPULATION_DENSITY = { "Chennai": 26000, "Kancheepuram": 800, "Thiruvallur": 700, "Cuddalore": 600, "Madurai": 800, "Kanyakumari": 1100 }
DEFAULT_DENSITY = 400

# COASTAL DISTRICTS (High Sensitivity to Surge/Spikes)
COASTAL_DISTRICTS = {
    "Chennai", "Thiruvallur", "Chengalpattu", "Kancheepuram", "Villupuram", 
    "Cuddalore", "Mayiladuthurai", "Nagapattinam", "Thiruvarur", "Thanjavur", 
    "Pudukkottai", "Ramanathapuram", "Thoothukudi", "Tirunelveli", "Kanyakumari"
}

masks = {}
daily = pd.DataFrame()
daily_drainage_capacity = {} 
tn_map = None

def init_engine():
    global daily, daily_drainage_capacity, tn_map
    try: tn_map = gpd.read_file(MAP_PATH).to_crs(epsg=4326)
    except: raise FileNotFoundError("CRITICAL: TamilNadu.geojson map not found.")

    if os.path.exists(PROCESSED_DATA_FILE):
        daily = pd.read_csv(PROCESSED_DATA_FILE, index_col=0, parse_dates=True)
        daily.columns = [c.strip() for c in daily.columns]
    else: raise FileNotFoundError("Run preprocess_v2.py first!")

    for d in daily.columns:
        wet_days = daily[d][daily[d] > 1]
        base = wet_days.quantile(0.75) if not wet_days.empty else 15.0
        # Reasonable drainage limits (30mm - 100mm)
        daily_drainage_capacity[d] = max(30, min(base, 100))

init_engine()

# --- SPATIAL ---
def calculate_proximity_weight(district_name, cyclone_track):
    if not cyclone_track: return 1.0, 0
    clean_name = str(district_name).strip()
    try: district_geom = tn_map[tn_map['NAME_2'] == clean_name].geometry.values[0]
    except: return 1.0, 0 
    
    # Fix: Ensure track is (Lon, Lat) for calculation
    track_points = [(p[1], p[0]) for p in cyclone_track]
    track_line = LineString(track_points)
    dist_deg = district_geom.distance(track_line)
    min_dist_km = dist_deg * 111
    
    # Spatial Decay: <100km is core, >400km is background
    if min_dist_km < 100: return 1.0, min_dist_km
    elif min_dist_km > 400: return 0.2, min_dist_km
    else: return (1.0 - ((min_dist_km - 100) / 300) * 0.8), min_dist_km

# --- SIMULATION CORE ---
def get_data_safe(district, start, end):
    if daily.empty: return pd.Series(0, index=pd.date_range(start, end))
    try: return daily.loc[start:end, district]
    except: return pd.Series(0, index=pd.date_range(start, end))

def calculate_flood_depth(rain_series, district, speed=12, blockage=0, track=None):
    base_rate = daily_drainage_capacity.get(district, 30.0)
    blockage_factor = (100 - blockage) / 100.0
    drainage = base_rate * blockage_factor
    
    proximity_weight, dist_km = calculate_proximity_weight(district, track)
    is_coastal = district in COASTAL_DISTRICTS
    
    standing = 0.0
    levels = []
    eff_rain_list = []
    
    for rain in rain_series:
        effective_rain = rain
        
        # 1. COASTAL PHYSICS (The Fix)
        # If district is coastal AND near track (<80km), we assume satellite under-read
        # and storm surge effects, so we boost the rain input.
        if is_coastal and dist_km < 80 and effective_rain > 20:
            effective_rain *= 1.35 # +35% Intensity Boost
            
        # 2. DRAINAGE COLLAPSE
        # Coastal areas with heavy rain fail harder due to sea-level pressure
        current_drainage = drainage
        
        if effective_rain > 120:
            current_drainage *= 0.1 # Total Collapse (Flash Flood)
        elif effective_rain > 60:
            if is_coastal and dist_km < 100:
                current_drainage *= 0.2 # Coastal Collapse (Surge block)
            else:
                current_drainage *= 0.5 # Inland Saturation
                
        # 3. Runoff
        runoff_bonus = standing * 0.20 if standing > 80 else 0
        
        standing = standing + effective_rain - current_drainage - runoff_bonus
        if standing < 0: standing = 0
        
        levels.append(standing)
        eff_rain_list.append(effective_rain)
        
    return pd.Series(levels, index=rain_series.index), pd.Series(eff_rain_list, index=rain_series.index), dist_km

def determine_risk(depth, dist_km):
    # Context Aware Risk
    if dist_km < 150:
        if depth > 50: return "HIGH", "#ef4444"
        elif depth > 20: return "MODERATE", "#f97316"
        elif depth > 5: return "LOW", "#facc15"
        else: return "NO FLOOD", "#10b981"
    else:
        if depth > 150: return "HIGH", "#ef4444"
        elif depth > 80: return "MODERATE", "#f97316"
        elif depth > 30: return "LOW", "#facc15"
        else: return "NO FLOOD", "#10b981"

def get_daily_frame_data(start, end, speed=12, blockage=0, track=None):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    date_range = pd.date_range(s, e)
    
    district_water = {}
    district_eff_rain = {}
    meta_dist = {}
    meta_w = {} # Added to track weight for debug

    for d in daily.columns:
        rain_s = get_data_safe(d, s, e)
        # Note: We calculate weight inside calculate_flood_depth but can also grab it directly
        water_s, eff_r, dist = calculate_flood_depth(rain_s, d, speed, blockage, track)
        district_water[d] = water_s
        district_eff_rain[d] = eff_r
        meta_dist[d] = dist
        # Recalculate just for debug display (cheap operation)
        w, _ = calculate_proximity_weight(d, track)
        meta_w[d] = w

    frames = []
    cum_rain_tracker = {d: 0.0 for d in daily.columns} 

    for date in date_range:
        day_stats = []
        for d in daily.columns:
            depth = district_water[d].get(date, 0)
            daily_r = district_eff_rain[d].get(date, 0)
            cum_rain_tracker[d] += daily_r
            dist = meta_dist.get(d, 9999)
            w = meta_w.get(d, 1.0)
            
            risk, color = determine_risk(depth, dist)
            
            day_stats.append({
                "District": d, 
                "Accumulation": float(f"{depth:.1f}"), 
                "Daily_Rain": float(f"{daily_r:.1f}"),
                "Rain": float(f"{daily_r:.1f}"), # Legacy key
                "Total_Rain": float(f"{cum_rain_tracker[d]:.1f}"),
                "Risk": risk, "Color": color,
                "Date": date.strftime("%Y-%m-%d"),
                "Debug_W": float(f"{w:.2f}"), 
                "Debug_Dist": int(dist)
            })
        frames.append(pd.DataFrame(day_stats))
    return frames

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
            if row['Accumulation'] > 10: 
                lat, lon = centroids.get(row['District'], [0,0])
                intensity = min(row['Accumulation'] / 200.0, 1.0)
                day_points.append([lat, lon, intensity])
        heatmap_frames.append(day_points)
    return heatmap_frames

def analyze_flood_risk(district, start, end, speed=12, blockage=0, track=None):
    frames = get_daily_frame_data(start, end, speed, blockage, track)
    total_rain = 0
    peak_water = 0
    
    for frame in frames:
        row = frame[frame['District'] == district]
        if not row.empty:
            total_rain = row['Total_Rain'].iloc[0]
            peak_water = max(peak_water, row['Accumulation'].iloc[0])

    risk = "HIGH" if peak_water > 100 else "MODERATE" if peak_water > 50 else "LOW" if peak_water > 20 else "NO FLOOD"
    
    return {
        "risk": risk, "total_rain": float(f"{total_rain:.1f}"),
        "peak_accumulation": float(f"{peak_water:.1f}"), "affected_pop": 0,
        "wind_speed": speed,
        "chart_labels": [d.strftime("%b %d") for d in pd.date_range(start, end)],
        "chart_rain": [float(f[f['District']==district]['Daily_Rain'].iloc[0]) for f in frames if not f[f['District']==district].empty],
        "chart_water": [float(f[f['District']==district]['Accumulation'].iloc[0]) for f in frames if not f[f['District']==district].empty]
    }

def forecast_7day():
    return analyze_flood_risk("Chennai", pd.Timestamp.today(), pd.Timestamp.today()+pd.Timedelta(days=7))

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
    "Thane 2011": [[12.0, 86.0], [12.5, 83.0], [12.0, 80.5], [11.6, 79.8]]
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
    "Thane 2011": ("2011-12-25", "2011-12-31", 140)
}
