import os
import json
import joblib
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Lahore AQI Forecasting", layout="wide")

HIST_PATH = "data/historical_features_1y.csv"
MODEL_PATH = "outputs/best_model.joblib"
METRICS_PATH = "outputs/model_metrics.json"

FEATURES = ["hour", "day", "month", "pm2_5", "pm10", "no2", "so2", "co", "o3"]
TARGET = "aqi_class"

# OpenWeather forecast endpoint
FORECAST_URL = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"


# ---------------- Helpers ----------------
def safe_load_csv(path):
    """Return dataframe if exists, else None (so app does not break on deployment)."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Missing model: {MODEL_PATH}. Train first and commit the file.")
        st.stop()
    return joblib.load(MODEL_PATH)

def get_api_key():
    # Try Streamlit secrets safely (won't crash if secrets.toml is missing)
    try:
        key = st.secrets.get("OPENWEATHER_API_KEY", None)
        if key:
            return key
    except Exception:
        pass

    # Fallback to env var (works locally if you set it)
    return os.getenv("OPENWEATHER_API_KEY")



def get_city_coords():
    # Defaults (Lahore)
    lat, lon = "31.5497", "74.3436"

    try:
        lat = st.secrets.get("CITY_LAT", lat)
        lon = st.secrets.get("CITY_LON", lon)
    except Exception:
        pass

    lat = os.getenv("CITY_LAT", lat)
    lon = os.getenv("CITY_LON", lon)

    return float(lat), float(lon)


def aqi_to_class_openweather(aqi):
    mapping = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    try:
        return mapping.get(int(aqi), "Unknown")
    except Exception:
        return "Unknown"

@st.cache_data(ttl=900)  # cache 15 minutes
def fetch_forecast_df(lat: float, lon: float, api_key: str) -> pd.DataFrame:
    """Fetch OpenWeather air pollution forecast and return as DataFrame."""
    r = requests.get(
        FORECAST_URL,
        params={"lat": lat, "lon": lon, "appid": api_key},
        timeout=60
    )
    r.raise_for_status()
    payload = r.json()

    rows = []
    for item in payload.get("list", []):
        dt_utc = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        comp = item.get("components", {})
        aqi = item.get("main", {}).get("aqi", None)

        rows.append({
            "timestamp_utc": dt_utc.isoformat(),
            "hour": dt_utc.hour,
            "day": dt_utc.day,
            "month": dt_utc.month,
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "no2": comp.get("no2"),
            "so2": comp.get("so2"),
            "co": comp.get("co"),
            "o3": comp.get("o3"),
            "ow_aqi": aqi,
            "ow_aqi_class": aqi_to_class_openweather(aqi),
            "source": "openweather_forecast",
            "lat": lat,
            "lon": lon,
        })

    df = pd.DataFrame(rows)

    # Ensure numeric
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Parse timestamp as UTC datetime
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    return df

def filter_next_72_hours(df_forecast: pd.DataFrame) -> pd.DataFrame:
    """Filter forecast rows from now to now+72 hours, display local time too."""
    now_utc = pd.Timestamp.utcnow()
    end_utc = now_utc + pd.Timedelta(hours=72)

    df = df_forecast.copy()
    df = df.dropna(subset=FEATURES)  # must have all features
    df = df[(df["timestamp_utc"] >= now_utc) & (df["timestamp_utc"] <= end_utc)].copy()

    if len(df) == 0:
        return df

    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert("Asia/Karachi")
    df = df.sort_values("timestamp_utc")
    return df


# ---------------- UI ----------------
st.title("🌫️ Lahore AQI Forecasting Dashboard")
st.caption("Real-time + 3-day forecast using OpenWeather + ML model")

page = st.sidebar.radio("Menu", ["🌍 Overview", "📈 3-Day Forecast", "🧪 Model Performance", "🗂️ Data"])


# ---------- Overview ----------
if page == "🌍 Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("City", "Lahore")
    c2.metric("Model", "Best Saved Model (outputs/best_model.joblib)")
    c3.metric("Forecast Horizon", "Next 72 hours")

    st.write("### What this app does")
    st.write(
        "- Fetches **live OpenWeather forecast** data\n"
        "- Filters exactly **now → next 72 hours**\n"
        "- Predicts AQI class using your trained ML model\n"
        "- Shows model performance (live if dataset exists, otherwise from saved metrics JSON)"
    )

    st.info("Open **📈 3-Day Forecast** to see real-time predictions.")


# ---------- 3-Day Forecast ----------
if page == "📈 3-Day Forecast":
    st.subheader("📈 Next 3 Days AQI Forecast (Hourly, from NOW)")

    api_key = get_api_key()
    if not api_key:
        st.error("Missing OPENWEATHER_API_KEY. Add it to environment (.env) locally or Streamlit Secrets in cloud.")
        st.stop()

    lat, lon = get_city_coords()
    model = load_model()

    with st.spinner("Fetching live forecast from OpenWeather..."):
        fc = fetch_forecast_df(lat, lon, api_key)

    fc_72 = filter_next_72_hours(fc)

    if len(fc_72) == 0:
        st.error("No forecast rows found for next 72 hours. Try again in a few minutes.")
        st.stop()

    # Predict with your ML model
    fc_72["predicted_aqi_class"] = model.predict(fc_72[FEATURES])

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Next Hour (Pred)", str(fc_72["predicted_aqi_class"].iloc[0]))
    k2.metric("Most Common (72h)", str(fc_72["predicted_aqi_class"].mode().iloc[0]))
    k3.metric("Rows Used", str(len(fc_72)))
    k4.metric("Last Forecast Time", str(fc_72["timestamp_local"].iloc[-1]))

    st.write("### Forecast table (first 72 rows)")
    st.dataframe(
        fc_72[["timestamp_local", "predicted_aqi_class", "ow_aqi_class"] + FEATURES].reset_index(drop=True),
        use_container_width=True
    )

    # Plot: map classes to numeric scale
    mapping = {"Good": 1, "Fair": 2, "Moderate": 3, "Poor": 4, "Very Poor": 5, "Unknown": None}
    plot_df = fc_72[["timestamp_local", "predicted_aqi_class", "ow_aqi_class"]].copy()
    plot_df["pred_num"] = plot_df["predicted_aqi_class"].map(mapping)
    plot_df["ow_num"] = plot_df["ow_aqi_class"].map(mapping)

    st.write("### Forecast chart (AQI class scale 1..5)")
    st.line_chart(plot_df.set_index("timestamp_local")[["pred_num", "ow_num"]])

    st.caption("Note: 'ow_num' is OpenWeather's own forecast AQI class; 'pred_num' is your ML model prediction.")


# ---------- Model Performance ----------
if page == "🧪 Model Performance":
    st.subheader("🧪 Model Performance")

    df = safe_load_csv(HIST_PATH)

    # If dataset missing (deployment case): show saved metrics
    if df is None:
        st.warning("Historical dataset not included in deployment. Showing saved evaluation metrics.")

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)

            st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            st.metric("Test Samples", metrics.get("test_samples", "N/A"))

            st.subheader("Classification Report")
            report_df = pd.DataFrame(metrics["classification_report"]).T
            st.dataframe(report_df, use_container_width=True)

        else:
            st.error(f"Saved metrics not found: {METRICS_PATH}. Commit it to GitHub to enable this page.")

        st.stop()

    # If dataset exists (local case): compute live metrics
    model = load_model()

    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    st.metric("Accuracy", f"{acc*100:.2f}%")

    st.subheader("Classification Report")
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)


# ---------- Data ----------
if page == "🗂️ Data":
    st.subheader("🗂️ Files and Data Preview")

    st.write("Historical dataset:")
    if os.path.exists(HIST_PATH):
        st.dataframe(pd.read_csv(HIST_PATH).head(10), use_container_width=True)
    else:
        st.warning(f"Dataset not deployed: {HIST_PATH}")

    st.write("Model file:")
    st.write("✅ Found" if os.path.exists(MODEL_PATH) else f"❌ Missing {MODEL_PATH}")

    st.write("Saved metrics file:")
    st.write("✅ Found" if os.path.exists(METRICS_PATH) else f"❌ Missing {METRICS_PATH}")

    st.write("Tip: On Streamlit Cloud, add secrets OPENWEATHER_API_KEY, CITY_LAT, CITY_LON.")
