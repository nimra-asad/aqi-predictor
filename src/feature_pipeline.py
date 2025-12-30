import os, glob, json
import pandas as pd
from datetime import datetime, timezone

EXPECTED_COLS = [
    "timestamp_utc", "hour", "day", "month",
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "aqi", "aqi_change", "aqi_class",
    "source_file", "fetched_at_utc",
]

RAW_PATTERN = "data/raw/airpollution_*.json"
OUT_CSV = "data/features.csv"


# ✅ Correct mapping for OpenWeather AQI categories (1..5)
def aqi_to_class_openweather(aqi) -> str:
    mapping = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor",
    }
    try:
        return mapping.get(int(aqi), None)
    except Exception:
        return None


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
    aqi = item["main"]["aqi"]  # 1..5 category from OpenWeather
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
        "aqi_change": pd.NA,  # will be filled in add_aqi_change()
        "aqi_class": aqi_to_class_openweather(aqi),

        "source_file": raw.get("_source_file"),
        "fetched_at_utc": raw.get("_fetched_at_utc"),
    }

    df = pd.DataFrame([row])

    # Force column order NOW (prevents shifting later)
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[EXPECTED_COLS]

    return df


def add_aqi_change(df_new: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """
    aqi_change = current_aqi - previous_aqi
    Ensure numeric to avoid weird strings.
    """
    # Ensure current aqi numeric
    df_new["aqi"] = pd.to_numeric(df_new["aqi"], errors="coerce")

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)

        # Ensure old df has expected columns in correct order
        for c in EXPECTED_COLS:
            if c not in df_old.columns:
                df_old[c] = pd.NA
        df_old = df_old[EXPECTED_COLS]

        if len(df_old) > 0:
            prev = pd.to_numeric(df_old.iloc[-1]["aqi"], errors="coerce")
            if pd.isna(prev) or pd.isna(df_new.loc[0, "aqi"]):
                df_new["aqi_change"] = 0
            else:
                df_new["aqi_change"] = (df_new["aqi"] - prev).astype("int64")
        else:
            df_new["aqi_change"] = 0
    else:
        df_new["aqi_change"] = 0

    return df_new


def append_csv(df: pd.DataFrame, csv_path: str) -> None:
    """
    SAFE append:
    - reads existing csv (if present)
    - ensures BOTH old and new dataframes have same columns/order
    - rewrites csv completely (prevents corruption)
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Ensure df has all expected columns and correct order
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[EXPECTED_COLS]

    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)

        for c in EXPECTED_COLS:
            if c not in old_df.columns:
                old_df[c] = pd.NA
        old_df = old_df[EXPECTED_COLS]

        combined = pd.concat([old_df, df], ignore_index=True)
    else:
        combined = df

    # Final safety: enforce numeric types where needed
    combined["aqi"] = pd.to_numeric(combined["aqi"], errors="coerce")
    combined["aqi_change"] = pd.to_numeric(combined["aqi_change"], errors="coerce").fillna(0).astype("int64")

    combined.to_csv(csv_path, index=False)


if __name__ == "__main__":
    raw = load_latest_raw()
    df = make_row(raw)
    df = add_aqi_change(df, OUT_CSV)
    append_csv(df, OUT_CSV)
    print("✅ Appended 1 row to:", OUT_CSV)
    print(df)
