import requests

def is_cyclone_active():
    url = "https://api.open-meteo.com/v1/forecast?latitude=13.0&longitude=80.2&hourly=windspeed_10m"
    r = requests.get(url).json()
    max_wind = max(r["hourly"]["windspeed_10m"])
    return max_wind > 30, max_wind
