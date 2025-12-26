import os, glob, json
import pandas as pd
from datetime import datetime, timezone

RAW_PATTERN = "data/raw/airpollution_*.json"
OUT_CSV = "data/features.csv"

def load_latest_raw():
    files = sorted(glob.glob(RAW_PATTERN))
    if not files:
        raise FileNotFoundError("No raw files found. Run fetch_data.py first.")
    latest = files[-1]
    with open(latest, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["_source_file"] = latest
    return raw

def make_row(raw) -> pd.DataFrame:
    item = raw["list"][0]
    comp = item["components"]
    aqi = item["main"]["aqi"]  # 1..5 category
    dt_utc = datetime.fromtimestamp(item["dt"], tz=timezone.utc)

    row = {
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
        "source_file": raw.get("_source_file"),
        "fetched_at_utc": raw.get("_fetched_at_utc"),
    }
    return pd.DataFrame([row])

def add_aqi_change(df_new: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    # change rate = current AQI - previous AQI
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        if len(df_old) > 0:
            prev = df_old.iloc[-1]["aqi"]
            df_new["aqi_change"] = df_new["aqi"] - prev
        else:
            df_new["aqi_change"] = 0
    else:
        df_new["aqi_change"] = 0
    return df_new

def append_csv(df: pd.DataFrame, csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    raw = load_latest_raw()
    df = make_row(raw)
    df = add_aqi_change(df, OUT_CSV)
    append_csv(df, OUT_CSV)
    print("✅ Appended 1 row to:", OUT_CSV)
    print(df)
