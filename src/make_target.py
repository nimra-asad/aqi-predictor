import pandas as pd

def aqi_to_class(aqi: float) -> str:
    """Convert numeric AQI into a categorical class."""
    if pd.isna(aqi):
        return None
    aqi = float(aqi)

    # Simple 5-class buckets (works well for multi-class demo)
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "USG"  # Unhealthy for Sensitive Groups
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

if __name__ == "__main__":
    path = "data/features.csv"
    df = pd.read_csv(path)

    if "aqi" not in df.columns:
        raise SystemExit("❌ No 'aqi' column found in data/features.csv")

    df["aqi_class"] = df["aqi"].apply(aqi_to_class)

    df.to_csv(path, index=False)

    print("✅ Saved 'aqi_class' into data/features.csv")
    print(df["aqi_class"].value_counts(dropna=False))
