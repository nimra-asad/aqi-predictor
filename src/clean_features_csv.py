import pandas as pd

CSV_PATH = "data/features.csv"

EXPECTED_COLS = [
    "timestamp_utc", "hour", "day", "month",
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "aqi", "aqi_change", "aqi_class",
    "source_file", "fetched_at_utc",
]

def aqi_to_class(aqi):
    if pd.isna(aqi):
        return None
    try:
        aqi = float(aqi)
    except Exception:
        return None

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "USG"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

def main():
    df = pd.read_csv(CSV_PATH)

    # --- Ensure all expected columns exist (create missing ones as empty) ---
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # --- Fill missing aqi_class from aqi ---
    df["aqi_class"] = df["aqi_class"].replace("", pd.NA)
    missing_mask = df["aqi_class"].isna()
    df.loc[missing_mask, "aqi_class"] = df.loc[missing_mask, "aqi"].apply(aqi_to_class)

    # --- Fix types safely ---
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], errors="coerce")
    df["timestamp_utc"] = df["timestamp_utc"].astype(str)

    # --- Remove duplicates by timestamp (keep latest fetch) ---
    df = df.sort_values("fetched_at_utc")
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="last")

    # --- Reorder columns exactly (prevents misalignment later) ---
    df = df[EXPECTED_COLS]

    df.to_csv(CSV_PATH, index=False)
    print("✅ Cleaned and rewrote data/features.csv")
    print("Rows:", len(df))
    print(df[["aqi", "aqi_class"]].tail(10))

if __name__ == "__main__":
    main()
