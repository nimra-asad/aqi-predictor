import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks

load_dotenv()

FEATURES_PATH = "data/features.csv"

def main():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT", "aqi-lahore")

    if not api_key:
        raise ValueError("Missing HOPSWORKS_API_KEY in .env")

    df = pd.read_csv(FEATURES_PATH)
    df = df.dropna(subset=["timestamp_utc"]).drop_duplicates(subset=["timestamp_utc"])
    df["timestamp_utc"] = df["timestamp_utc"].astype(str)

    project = hopsworks.login(project=project_name, api_key_value=api_key)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="lahore_aqi_features",
        version=1,
        primary_key=["timestamp_utc"],
        description="Lahore AQI features (offline only; avoids Kafka on Windows)",
        online_enabled=False,
    )

    # ✅ Force OFFLINE write (prevents Kafka/jks path)
    fg.insert(
        df,
        write_options={
            "wait_for_job": True,
            "kafka_producer": False,   # key line: do not use Kafka
        },
    )

    print(f"✅ Uploaded rows: {len(df)}")

if __name__ == "__main__":
    main()
