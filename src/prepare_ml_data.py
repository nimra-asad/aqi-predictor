import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/historical_features_1y.csv"

FEATURES = [
    "hour", "day", "month",
    "pm2_5", "pm10", "no2", "so2", "co", "o3"
]
TARGET = "aqi_class"

df = pd.read_csv(DATA_PATH)

# Clean: keep only rows with all required fields
df = df.dropna(subset=FEATURES + [TARGET]).copy()

X = df[FEATURES]
y = df[TARGET]

# Stratified split keeps class proportions the same
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Loaded:", DATA_PATH)
print("Total rows:", len(df))
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("\nClass distribution (full):")
print(y.value_counts())
print("\nClass distribution (train):")
print(y_train.value_counts())
