from flask import Flask, render_template, request
import pandas as pd
import json
from datetime import datetime
from flood_engine import (
    analyze_flood_risk, forecast_7day, get_daily_frame_data, get_heatmap_data,
    CYCLONES, CYCLONE_PATHS, tn_map, daily
)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    active_mode = "live"
    result = None
    frontend_data = None
    track_path = []
    
    # DEFAULTS
    cyclone = "Michaung 2023"
    speed = 12; blockage = 0
    start = pd.Timestamp.today() - pd.Timedelta(days=7)
    end = pd.Timestamp.today()

    # 1. HANDLE FORM INPUT
    if request.method == "POST": 
        active_mode = request.form.get("mode", "live")
        
        if active_mode == "historical":
            cyclone = request.form.get("cyclone", "Michaung 2023")
            blockage = int(request.form.get("blockage", 0))
            
            if cyclone in CYCLONES:
                start, end, speed = CYCLONES[cyclone]
                # CRITICAL: Get the track for filtering
                track_path = CYCLONE_PATHS.get(cyclone, [])
                
        elif active_mode == "custom":
            start = request.form.get("start")
            end = request.form.get("end")
            speed = float(request.form.get("speed", 12))
            blockage = int(request.form.get("blockage", 0))

    # 2. GENERATE DATA (FORCE TRACK PASSING)
    # We pass 'track_path' even if empty. The engine handles empty lists correctly.
    raw_frontend = get_daily_frame_data(start, end, speed, blockage, track=track_path)
    heat_frames = get_heatmap_data(start, end, speed, blockage, track=track_path)
    
    # 3. PACKAGING FOR FRONTEND
    sim_data = {}
    dates = []
    if raw_frontend:
        dates = [d['Date'].iloc[0] for d in raw_frontend]
        districts = raw_frontend[0]['District'].unique()
        for d in districts:
            sim_data[d] = []
        for frame in raw_frontend:
            for _, row in frame.iterrows():
                sim_data[row['District']].append({
                    "depth": row['Accumulation'],
                    "daily_rain": row['Daily_Rain'], # Matches new key
                    "total_rain": row['Total_Rain'], # Matches new key
                    "risk": row['Risk'],
                    "color": row['Color'],
                    "debug_w": row['Debug_W'], 
                    "debug_dist": row['Debug_Dist']
                })

    frontend_data = {
        "map_geo": json.loads(tn_map.to_json()),
        "simulation": sim_data,
        "dates": dates,
        "heatmap": heat_frames
    }
    
    # 4. ANALYSIS CARD
    # We use Chennai as default, or the first district available
    target = "Chennai"
    result = analyze_flood_risk(target, start, end, speed, blockage, track=track_path)

    # 5. SMART DROPDOWN FILTER
    available_cyclones = []
    for name, (s, e, _) in CYCLONES.items():
        year = int(s.split("-")[0])
        if not daily.empty and year in daily.index.year:
            available_cyclones.append(name)
        elif daily.empty:
             available_cyclones.append(name)

    return render_template(
        "index.html",
        result=result,
        frontend_data=frontend_data,
        track_path=track_path,
        cyclones=available_cyclones,
        selected_cyclone=cyclone,
        active_mode=active_mode,
        now=datetime.now().strftime("%d %b %Y")
    )

if __name__ == "__main__":
    app.run(debug=True)
