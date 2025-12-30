import pandas as pd

PATH = "data/features.csv"

def aqi_to_class_openweather(aqi):
    mapping = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor",
    }
    try:
        return mapping.get(int(aqi))
    except:
        return None

df = pd.read_csv(PATH)

# Recompute for ALL rows
df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
df["aqi_class"] = df["aqi"].apply(aqi_to_class_openweather)

df.to_csv(PATH, index=False)

print("✅ Recomputed aqi_class for all live rows")
print(df[["aqi", "aqi_class"]].value_counts())
