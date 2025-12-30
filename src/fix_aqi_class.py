import pandas as pd

VALID = {"Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy"}

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

df = pd.read_csv("data/features.csv")

# normalize blanks
df["aqi_class"] = df["aqi_class"].astype(str).str.strip()
df.loc[df["aqi_class"].isin(["", "nan", "None"]), "aqi_class"] = None

# if aqi_class is not valid, recompute from aqi
bad = ~df["aqi_class"].isin(VALID)
df.loc[bad, "aqi_class"] = df.loc[bad, "aqi"].apply(aqi_to_class)

# drop any rows still invalid after recompute (rare corrupted rows)
bad2 = ~df["aqi_class"].isin(VALID)
df = df.loc[~bad2].copy()

df.to_csv("data/features.csv", index=False)

print("✅ Fixed aqi_class. Rows now:", len(df))
print(df[["aqi", "aqi_class"]].tail(10))
