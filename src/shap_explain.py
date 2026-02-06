import os
import numpy as np
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "outputs/best_model.joblib"
DATA_PATH = "data/historical_features_1y.csv"  # <- big dataset

FEATURES = [
    "hour", "day", "month",
    "pm2_5", "pm10", "no2", "so2", "co", "o3"
]

os.makedirs("outputs", exist_ok=True)

# Load model + data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Keep only needed columns and drop missing
X = df[FEATURES].dropna()

# Keep SHAP fast + stable
background_size = min(30, len(X))
explain_size = min(100, len(X))

background = X.sample(background_size, random_state=42)
X_explain = X.sample(explain_size, random_state=7)

# KernelExplainer works for any model (incl. multiclass)
explainer = shap.KernelExplainer(model.predict_proba, background)
shap_values = explainer.shap_values(X_explain)

# -------- shape-safe handling --------
# We want a matrix of shape (n_samples, n_features) for ONE class
def get_class_matrix(sv, class_index=0):
    # Case 1: list of arrays [class0, class1, ...]
    if isinstance(sv, list):
        return sv[class_index]

    sv = np.array(sv)

    # Case 2: (n_samples, n_features, n_classes)
    if sv.ndim == 3 and sv.shape[0] == len(X_explain) and sv.shape[1] == len(FEATURES):
        return sv[:, :, class_index]

    # Case 3: (n_samples, n_classes, n_features)
    if sv.ndim == 3 and sv.shape[0] == len(X_explain) and sv.shape[2] == len(FEATURES):
        return sv[:, class_index, :]

    # Fallback: try to squeeze
    sv = np.squeeze(sv)
    return sv

sv_matrix = get_class_matrix(shap_values, class_index=0)

# Final safety check: match features
if sv_matrix.shape[1] != X_explain.shape[1]:
    raise ValueError(
        f"SHAP matrix shape {sv_matrix.shape} does not match X shape {X_explain.shape}. "
        "Please paste this error back to ChatGPT."
    )

# Plot summary for class 0
plt.figure()
shap.summary_plot(sv_matrix, X_explain, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=200)
plt.close()

print("✅ SHAP summary plot saved to outputs/shap_summary.png")
