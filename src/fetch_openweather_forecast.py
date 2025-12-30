import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
OUT_CSV = "data/forecast_96h.csv"  # OpenWeather forecast ~4 days

FEATURES = ["hour","day","month","pm2_5","pm10","no2","so2","co","o3"]

def aqi_to_class_openweather(aqi):
    mapping = {1:"Good", 2:"Fair", 3:"Moderate", 4:"Poor", 5:"Very Poor"}
    try:
        return mapping.get(int(aqi))
    except:
        return None

def main():
    api_key = os.getenv("OPENWEATHER_API_KEY")
    lat = float(os.getenv("CITY_LAT", "31.5497"))
    lon = float(os.getenv("CITY_LON", "74.3436"))

    if not api_key:
        raise ValueError("Missing OPENWEATHER_API_KEY")

    r = requests.get(API_URL, params={"lat":lat,"lon":lon,"appid":api_key}, timeout=60)
    r.raise_for_status()
    payload = r.json()

    rows = []
    for item in payload.get("list", []):
        dt_utc = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        comp = item.get("components", {})
        aqi = item.get("main", {}).get("aqi")

        rows.append({
            "timestamp_utc": dt_utc.isoformat(),
            "hour": dt_utc.hour,
            "day": dt_utc.day,
            "month": dt_utc.month,
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "no2": comp.get("no2"),
            "so2": comp.get("so2"),
            "co": comp.get("co"),
            "o3": comp.get("o3"),
            "ow_aqi": aqi,
            "ow_aqi_class": aqi_to_class_openweather(aqi),
            "lat": lat,
            "lon": lon,
            "source": "openweather_forecast",
        })

    df = pd.DataFrame(rows)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("✅ Saved forecast:", OUT_CSV, "Rows:", len(df))
    print(df.head(3))

if __name__ == "__main__":
    main()
