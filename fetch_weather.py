import requests
import pandas as pd
from datetime import datetime

# Major TN Districts and their coordinates to keep it fast
CITIES = {
    "Chennai": (13.0827, 80.2707),
    "Kancheepuram": (12.8342, 79.7036),
    "Cuddalore": (11.7480, 79.7714),
    "Nagapattinam": (10.7656, 79.8424),
    "Madurai": (9.9252, 78.1198),
    "Coimbatore": (11.0168, 76.9558),
    "Kanyakumari": (8.0883, 77.5385)
}

def fetch_open_meteo():
    print("Fetching live satellite telemetry...")
    live_data = []

    for district, (lat, lon) in CITIES.items():
        # Open-Meteo Free API (No keys needed)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=rain_sum&timezone=auto"
        response = requests.get(url).json()

        # Extract Today (Live) and Tomorrow (Forecast)
        today_rain = response['daily']['rain_sum'][0]
        tomorrow_rain = response['daily']['rain_sum'][1]

        # Save Today
        live_data.append({"District": district, "Date": str(datetime.now().date()), "Rainfall": today_rain, "Type": "Live"})
        # Save Tomorrow
        live_data.append({"District": district, "Date": "Forecast", "Rainfall": tomorrow_rain, "Type": "Forecast"})

    df = pd.DataFrame(live_data)
    df.to_csv("live_telemetry.csv", index=False)
    print("Success! Live and Forecast data saved to live_telemetry.csv")

if __name__ == "__main__":
    fetch_open_meteo()