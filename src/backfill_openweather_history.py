import os
import time
import math
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

OUT_CSV = os.getenv("OUT_CSV", "data/historical_features_1y.csv")
PROGRESS_FILE = os.getenv("PROGRESS_FILE", "data/backfill_progress_1y.txt")

EXPECTED_COLS = [
    "timestamp_utc", "hour", "day", "month",
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "aqi", "aqi_change", "aqi_class",
    "source", "lat", "lon",
]

def aqi_to_class_openweather(aqi) -> str:
    # OpenWeather AQI scale: 1..5
    mapping = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    try:
        return mapping.get(int(aqi), "Unknown")
    except Exception:
        return "Unknown"

def fetch_history(lat: float, lon: float, start_unix: int, end_unix: int, api_key: str) -> dict:
    params = {"lat": lat, "lon": lon, "start": start_unix, "end": end_unix, "appid": api_key}
    timeouts = [30, 60, 90]
    for attempt, to in enumerate(timeouts, start=1):
        try:
            r = requests.get(API_URL, params=params, timeout=to)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ReadTimeout:
            print(f"⚠️ Timeout (attempt {attempt}/{len(timeouts)}) for {start_unix}->{end_unix}, retrying...")
            time.sleep(2 * attempt)
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error: {e} (attempt {attempt}/{len(timeouts)}), retrying...")
            time.sleep(2 * attempt)
    raise RuntimeError(f"❌ Failed after retries for {start_unix}->{end_unix}")

def payload_to_df(payload: dict, lat: float, lon: float) -> pd.DataFrame:
    rows = []
    for item in payload.get("list", []):
        dt_utc = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        comp = item.get("components", {})
        aqi = item.get("main", {}).get("aqi", None)

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
            "aqi": aqi,
            "aqi_change": pd.NA,  # computed later
            "aqi_class": aqi_to_class_openweather(aqi) if aqi is not None else None,
            "source": "openweather_history",
            "lat": lat,
            "lon": lon,
        })
    df = pd.DataFrame(rows)

    # Ensure expected columns exist and order
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[EXPECTED_COLS]

def compute_aqi_change_inplace(csv_path: str) -> None:
    # Compute aqi_change on the FULL file (safe and simple)
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    change = df["aqi"].diff().fillna(0).round(0)
    df["aqi_change"] = pd.to_numeric(change, errors="coerce").fillna(0).astype("int64")
    df.to_csv(csv_path, index=False)

def main():
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENWEATHER_API_KEY in .env or environment.")

    lat = float(os.getenv("CITY_LAT", "31.5497"))
    lon = float(os.getenv("CITY_LON", "74.3436"))

    # ✅ DEFAULT = 365 (1 year). You can override via env BACKFILL_DAYS.
    days_back = int(os.getenv("BACKFILL_DAYS", "365"))

    # Chunk size (days)
    chunk_days = int(os.getenv("CHUNK_DAYS", "3"))
    chunk_seconds = chunk_days * 24 * 3600

    now = int(time.time())
    start_all = now - days_back * 24 * 3600

    out_path = Path(OUT_CSV)
    prog_path = Path(PROGRESS_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prog_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ Resume logic (but WITHOUT overwriting output)
    if prog_path.exists():
        saved = prog_path.read_text(encoding="utf-8").strip()
        if saved.isdigit():
            start_all = int(saved)
            print(f"🔁 Resuming from: {datetime.fromtimestamp(start_all, tz=timezone.utc)}")

    chunks = max(1, math.ceil((now - start_all) / chunk_seconds))
    print(f"Backfilling Lahore history: {days_back} days, {chunks} chunks of {chunk_days} days...")
    print(f"Output file: {OUT_CSV}")

    # ✅ Incremental write: append each chunk to CSV
    wrote_header = not out_path.exists()

    t0 = start_all
    for i in range(chunks):
        t1 = min(t0 + chunk_seconds, now)
        print(f"Chunk {i+1}/{chunks}: {datetime.fromtimestamp(t0, tz=timezone.utc)} -> {datetime.fromtimestamp(t1, tz=timezone.utc)}")

        payload = fetch_history(lat, lon, t0, t1, api_key)
        df_chunk = payload_to_df(payload, lat, lon)

        if len(df_chunk) > 0:
            df_chunk.to_csv(out_path, mode="a", header=wrote_header, index=False)
            wrote_header = False

        # Save progress AFTER successful chunk
        prog_path.write_text(str(t1), encoding="utf-8")

        time.sleep(1.5)
        t0 = t1

    # ✅ Final cleanup & compute aqi_change over the full CSV
    if out_path.exists():
        compute_aqi_change_inplace(str(out_path))
        df = pd.read_csv(out_path)
        print("✅ Saved:", OUT_CSV)
        print("Rows:", len(df))
        print(df.head(3))
        print(df["aqi_class"].value_counts(dropna=False))
    else:
        raise RuntimeError("No output file created. Check API key/limits.")

if __name__ == "__main__":
    main()
