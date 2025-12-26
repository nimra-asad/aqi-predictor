import os
import json
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

LAT = float(os.getenv("CITY_LAT", "31.5497"))
LON = float(os.getenv("CITY_LON", "74.3436"))
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def main():
    if not API_KEY:
        print("ERROR: OPENWEATHER_API_KEY missing in .env")
        return

    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY}

    r = requests.get(url, params=params, timeout=30)
    print("HTTP status:", r.status_code)

    if r.status_code != 200:
        print("Response:", r.text[:300])
        return

    data = r.json()
    data["_fetched_at_utc"] = datetime.now(timezone.utc).isoformat()

    os.makedirs("data/raw", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = f"data/raw/airpollution_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
