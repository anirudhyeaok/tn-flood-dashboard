import os
import pandas as pd
import xarray as xr
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

# --- CONFIG ---
CHIRPS_DIR = r"C:\Users\crade\Desktop\project something yeah\notebooks\chirps_nc"
TN_MAP_PATH = r"C:\Users\crade\Desktop\project something yeah\maps\TamilNadu.geojson"
OUTPUT_FILE = "tn_district_rain_history.csv"
CHIRPS_YEARS = [2011, 2012, 2015, 2016, 2017, 2018, 2020, 2022, 2023, 2024] 

print("--- 1. Loading Map ---")
tn_map = gpd.read_file(TN_MAP_PATH).to_crs(epsg=4326)
DISTRICT_COL = "NAME_2"

print("--- 2. Building Masks ---")
# Use the first available file to build the grid mask
first_file = None
for y in CHIRPS_YEARS:
    f = os.path.join(CHIRPS_DIR, f"chirps-v2.0.{y}.days_p05.nc")
    if os.path.exists(f):
        first_file = f
        break

if not first_file:
    raise FileNotFoundError("No .nc files found! Download them first.")

ds_ref = xr.open_dataset(first_file).sel(longitude=slice(76, 81.5), latitude=slice(8, 13.8))
lats = ds_ref.latitude.values
lons = ds_ref.longitude.values
lon2d, lat2d = np.meshgrid(lons, lats)

masks = {}
for _, row in tn_map.iterrows():
    name = row[DISTRICT_COL]
    geom = row.geometry.buffer(0.08)
    gprep = prep(geom)
    m = np.zeros(lon2d.shape, dtype=bool)
    for i in range(lon2d.shape[0]):
        for j in range(lon2d.shape[1]):
            if gprep.intersects(Point(lon2d[i, j], lat2d[i, j])):
                m[i, j] = True
    # Fallback for tiny districts
    if m.sum() == 0:
        cx, cy = geom.centroid.xy
        m[np.argmin(np.abs(lats - cy[0])), np.argmin(np.abs(lons - cx[0]))] = True
    masks[name] = m

print("--- 3. Extracting PEAK RAINFALL (95th Percentile) ---")
all_series = {}

for y in CHIRPS_YEARS:
    fpath = os.path.join(CHIRPS_DIR, f"chirps-v2.0.{y}.days_p05.nc")
    if not os.path.exists(fpath):
        print(f"Skipping {y}...")
        continue
    
    print(f"Processing {y}...")
    ds = xr.open_dataset(fpath).sel(longitude=slice(76, 81.5), latitude=slice(8, 13.8))
    arr = ds["precip"].values
    dates = pd.date_range(f"{y}-01-01", periods=arr.shape[0])
    
    for d_name, m in masks.items():
        # SCIENTIFIC FIX: Use 95th Percentile instead of Mean
        # This captures the "Storm Core" hitting the district, ignoring dry areas.
        # If a district gets hit, we want to know the WORST part of it.
        # We use 'nanpercentile' to ignore missing ocean data.
        district_pixels = arr[:, m]
        
        # Optimization: Only calculate if there's rain in the grid to save time
        # axis=1 means we reduce the pixels to a single value per day
        peak_rain = np.nanpercentile(district_pixels, 95, axis=1)
        
        series = pd.Series(peak_rain, index=dates)
        all_series.setdefault(d_name, []).append(series)

print("--- 4. Saving High-Fidelity Data ---")
final_data = {d: pd.concat(val) for d, val in all_series.items()}
df = pd.DataFrame(final_data).sort_index()

df.to_csv(OUTPUT_FILE)
print("DONE. This file now contains 'Worst Case' rainfall for every district.")
