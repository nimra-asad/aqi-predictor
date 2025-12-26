import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


DATA_PATH = "data/features.csv"
OUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUT_DIR, "best_model.joblib")
REPORT_PATH = os.path.join(OUT_DIR, "model_report.txt")


def ensure_outputs_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def ridge_predict_as_class(model, X):
    """
    Ridge outputs continuous values. Convert to AQI class 1..5 by rounding and clipping.
    """
    preds = model.predict(X)
    preds = np.rint(preds).astype(int)          # round to nearest integer
    preds = np.clip(preds, 1, 5)                # keep within [1, 5]
    return preds


def main():
    ensure_outputs_dir()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run feature_pipeline first.")

    df = pd.read_csv(DATA_PATH)

    # Keep only the columns we need
    # Drop non-feature columns if present
    drop_cols = [c for c in ["timestamp_utc", "source_file", "fetched_at_utc"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    if "aqi" not in df.columns:
        raise ValueError("Target column 'aqi' not found in data/features.csv")

    # Basic clean
    df = df.dropna()
    df["aqi"] = df["aqi"].astype(int)

    # Features + target
    X = df.drop(columns=["aqi"])
    y = df["aqi"]

    classes = sorted(y.unique().tolist())
    print("Classes in dataset:", classes)
    print("Total rows:", len(df))

    # If too few rows, training becomes unstable
    if len(df) < 10:
        print("⚠ Very small dataset (<10 rows). Collect more data for meaningful training.")
        return

    # Stratify only if we have 2+ classes
    stratify = y if len(classes) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    results = []

    # -------------------------
    # 1) Baseline
    # -------------------------
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    base_preds = baseline.predict(X_test)

    base_acc = accuracy_score(y_test, base_preds)
    base_f1 = f1_score(y_test, base_preds, average="weighted")
    results.append(("Baseline (most_frequent)", base_acc, base_f1))

    # -------------------------
    # 2) Random Forest (Classifier)
    # -------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced" if len(classes) > 1 else None
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average="weighted")
    results.append(("RandomForestClassifier", rf_acc, rf_f1))

    # -------------------------
    # 3) Ridge Regression (Regression -> rounded to class)
    # -------------------------
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])
    ridge.fit(X_train, y_train)
    ridge_preds = ridge_predict_as_class(ridge, X_test)

    ridge_acc = accuracy_score(y_test, ridge_preds)
    ridge_f1 = f1_score(y_test, ridge_preds, average="weighted")
    results.append(("Ridge (rounded->class)", ridge_acc, ridge_f1))

    # -------------------------
    # Pick best model by F1-weighted
    # -------------------------
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    best_name, best_acc, best_f1 = results_sorted[0]

    if best_name == "RandomForestClassifier":
        best_model = rf
        best_preds = rf_preds
    elif best_name == "Ridge (rounded->class)":
        best_model = ridge
        best_preds = ridge_preds
    else:
        best_model = baseline
        best_preds = base_preds

    # -------------------------
    # Save report
    # -------------------------
    cm = confusion_matrix(y_test, best_preds)
    report = classification_report(y_test, best_preds, zero_division=0)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== Model Comparison (Test Set) ===\n")
        for name, acc, f1w in results:
            f.write(f"{name:25s} | Accuracy={acc:.4f} | F1-weighted={f1w:.4f}\n")

        f.write("\n=== Best Model ===\n")
        f.write(f"Best: {best_name}\n")
        f.write(f"Accuracy: {best_acc:.4f}\n")
        f.write(f"F1-weighted: {best_f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    # Save best model
    joblib.dump(best_model, MODEL_PATH)

    # Print a small comparison table in terminal
    print("\n=== Model Comparison ===")
    for name, acc, f1w in results:
        print(f"{name:25s} | Accuracy={acc:.4f} | F1-weighted={f1w:.4f}")

    print("\n✅ Best model:", best_name)
    print("✅ Saved model to:", MODEL_PATH)
    print("✅ Saved report to:", REPORT_PATH)

    if len(classes) == 1:
        print("\n⚠ Only ONE AQI class present. Models will mostly predict that class.")
        print("✅ Your pipeline is correct—collect more data for meaningful improvements.")


if __name__ == "__main__":
    main()
