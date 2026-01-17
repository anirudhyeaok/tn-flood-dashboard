import pandas as pd
import folium
import json
import geopandas as gpd

from flood_engine import predict_event_flood, TN_MAP_PATH

# Load Tamil Nadu map once
tn_map = gpd.read_file(TN_MAP_PATH).to_crs(epsg=4326)
DISTRICT_COL = "NAME_2"

def risk_color(risk):
    """Map risk level to clear colors."""
    return {
        "NO FLOOD": "#2ecc71",   # green
        "LOW": "#f1c40f",        # yellow
        "MODERATE": "#e67e22",   # orange
        "HIGH": "#e74c3c"        # red
    }.get(risk, "#95a5a6")

def build_flood_map(start, end, speed):
    """
    Rebuilds flood_map.html with REAL values for every district.
    """

    results = []

    for _, row in tn_map.iterrows():
        d = row[DISTRICT_COL]

        out = predict_event_flood(d, start, end, speed)

        results.append({
            "District": d,
            "Risk": out["risk"],
            "Rain": out["rain"],
            "Ratio": out["ratio"]
        })

    df = pd.DataFrame(results)

    # Save CSV (useful for debugging)
    df.to_csv("static/district_risk.csv", index=False)

    # Create map centered on Tamil Nadu
    m = folium.Map(location=[11.0, 78.5], zoom_start=7, tiles="cartodb dark_matter")

    # Add district polygons
    for _, row in tn_map.iterrows():
        d = row[DISTRICT_COL]

        info = df[df["District"] == d].iloc[0]

        tooltip_html = f"""
        <b>{d}</b><br>
        Rain: {info["Rain"]} mm<br>
        Ratio: {info["Ratio"]}<br>
        Risk: <b>{info["Risk"]}</b>
        """

        folium.GeoJson(
            row.geometry,
            style_function=lambda x, r=info["Risk"]: {
                "fillColor": risk_color(r),
                "color": "white",
                "weight": 1,
                "fillOpacity": 0.7
            },
            tooltip=tooltip_html
        ).add_to(m)

    # Save the map
    m.save("static/flood_map.html")
