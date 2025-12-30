import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import joblib
import os



DATA_PATH = "data/historical_features_1y.csv"
OUT_DIR = "outputs"
BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_model.joblib")

FEATURES = [
    "hour", "day", "month",
    "pm2_5", "pm10", "no2", "so2", "co", "o3"
]
TARGET = "aqi_class"

def main():
    df = pd.read_csv(DATA_PATH)

    # Keep only necessary columns and drop missing
    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    # Make sure features are numeric
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES).copy()

    X = df[FEATURES]
    y = df[TARGET]

    # Stratified split = keeps class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("✅ Dataset:", DATA_PATH)
    print("Total rows:", len(df))
    print("Train:", len(X_train), " Test:", len(X_test))
    print("\nClass distribution (train):")
    print(y_train.value_counts())

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=600,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
    }

    results = []
    best_name, best_model, best_acc = None, None, -1

    for name, model in models.items():
        print("\n" + "="*60)
        print("Training:", name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results.append((name, acc))
        print("Accuracy:", round(acc, 4))
        print("\nClassification Report:")
        print(classification_report(y_test, preds, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    print("\n" + "="*60)
    print("✅ Summary (Accuracy):")
    for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:18s}  {acc:.4f}")

    # Save best model
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\n🏆 Best model: {best_name} (accuracy={best_acc:.4f})")
    print("✅ Saved:", BEST_MODEL_PATH)

if __name__ == "__main__":
    main()
