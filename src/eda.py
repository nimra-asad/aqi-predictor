import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/features.csv")

# Drop non-useful columns
df = df.drop(columns=["timestamp_utc", "source_file", "fetched_at_utc"])

print("Shape:", df.shape)
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# AQI distribution
plt.figure()
sns.countplot(x="aqi", data=df)
plt.title("AQI Class Distribution")
plt.tight_layout()
plt.savefig("outputs/aqi_distribution.png")
plt.close()

print("EDA plots saved in outputs/")
