import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks

load_dotenv()

FEATURES_PATH = "data/features.csv"

EXPECTED_COLS = [
    "timestamp_utc", "hour", "day", "month",
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "aqi", "aqi_change", "aqi_class",
    "source_file", "fetched_at_utc",
]

def main():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT", "aqi-lahore")

    if not api_key:
        raise ValueError("Missing HOPSWORKS_API_KEY in .env")

    df = pd.read_csv(FEATURES_PATH)

    # Ensure all expected columns exist + correct order
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[EXPECTED_COLS]

    # Basic cleanup
    df = df.dropna(subset=["timestamp_utc"]).drop_duplicates(subset=["timestamp_utc"])
    df["timestamp_utc"] = df["timestamp_utc"].astype(str)

    # ✅ Fix types so schema is stable
    # numeric columns
    num_cols = ["hour", "day", "month", "pm2_5", "pm10", "no2", "so2", "co", "o3", "aqi", "aqi_change"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # make integer-like columns integers (nullable)
    df["hour"] = df["hour"].astype("Int64")
    df["day"] = df["day"].astype("Int64")
    df["month"] = df["month"].astype("Int64")
    df["aqi"] = df["aqi"].astype("Int64")
    df["aqi_change"] = df["aqi_change"].fillna(0).astype("Int64")

    # string columns
    df["aqi_class"] = df["aqi_class"].astype(str)
    df["source_file"] = df["source_file"].astype(str)
    df["fetched_at_utc"] = df["fetched_at_utc"].astype(str)

    project = hopsworks.login(project=project_name, api_key_value=api_key)
    fs = project.get_feature_store()

    # ✅ Create a NEW version so schema mismatch disappears
    fg = fs.get_or_create_feature_group(
        name="lahore_aqi_features",
        version=2,
        primary_key=["timestamp_utc"],
        description="Lahore AQI features v2 (adds aqi_class, stable types)",
        online_enabled=False,
    )

    fg.insert(
        df,
        write_options={
            "wait_for_job": True,
            "kafka_producer": False,
        },
    )

    print(f"✅ Uploaded rows to lahore_aqi_features v2: {len(df)}")

if __name__ == "__main__":
    main()
