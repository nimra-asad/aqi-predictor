# app/app.py
import os
import joblib
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Lahore AQI Forecasting", layout="wide")

# ---------------- PATHS (Streamlit Cloud safe) ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "outputs" / "best_model.joblib"

# ---------------- CONSTANTS ----------------
FEATURES = ["hour", "day", "month", "pm2_5", "pm10", "no2", "so2", "co", "o3"]
FORECAST_URL = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"

# ---------------- SAFE SECRETS ----------------
def secrets_get(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def get_api_key():
    return secrets_get("OPENWEATHER_API_KEY", None) or os.getenv("OPENWEATHER_API_KEY")

def get_city_coords():
    lat = secrets_get("CITY_LAT", "31.5497")
    lon = secrets_get("CITY_LON", "74.3436")
    lat = os.getenv("CITY_LAT", lat)
    lon = os.getenv("CITY_LON", lon)
    return float(lat), float(lon)

def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Missing model file: {MODEL_PATH}\n\n✅ Fix: commit outputs/best_model.joblib to GitHub.")
        st.stop()
    return joblib.load(MODEL_PATH)

# ---------------- OPENWEATHER AQI ----------------
def ow_aqi_to_label(aqi_1to5):
    mapping = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    try:
        return mapping.get(int(aqi_1to5), "Unknown")
    except Exception:
        return "Unknown"

def ow_advice(aqi_1to5):
    try:
        aqi = int(aqi_1to5)
    except Exception:
        aqi = 0
    if aqi == 1:
        return "Air is clean. Great for outdoor activities."
    if aqi == 2:
        return "Generally okay. Sensitive people stay aware."
    if aqi == 3:
        return "Moderate pollution. Sensitive groups should reduce long outdoor exposure."
    if aqi == 4:
        return "Poor air quality. Limit outdoor exertion and consider a mask."
    if aqi == 5:
        return "Very poor air quality. Avoid outdoor activities if possible."
    return "Data uncertain. Please refresh in a few minutes."

def ow_icon(aqi_1to5):
    try:
        aqi = int(aqi_1to5)
    except Exception:
        return "❓"
    return {1:"😊", 2:"🙂", 3:"😷", 4:"🥵", 5:"☠️"}.get(aqi, "❓")

# ---------------- DATA FETCH ----------------
@st.cache_data(ttl=900)
def fetch_forecast_df(lat: float, lon: float, api_key: str) -> pd.DataFrame:
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
            "timestamp_utc": dt_utc,
            "hour": dt_utc.hour,
            "day": dt_utc.day,
            "month": dt_utc.month,
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "no2": comp.get("no2"),
            "so2": comp.get("so2"),
            "co": comp.get("co"),
            "o3": comp.get("o3"),
            "ow_aqi": aqi,  # 1..5
            "ow_aqi_label": ow_aqi_to_label(aqi),
        })

    df = pd.DataFrame(rows)

    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ow_aqi"] = pd.to_numeric(df["ow_aqi"], errors="coerce")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert("Asia/Karachi")
    return df

def filter_next_hours(df: pd.DataFrame, hours: int = 72) -> pd.DataFrame:
    now_utc = pd.Timestamp.utcnow()
    end_utc = now_utc + pd.Timedelta(hours=hours)
    out = df.dropna(subset=FEATURES).copy()
    out = out[(out["timestamp_utc"] >= now_utc) & (out["timestamp_utc"] <= end_utc)].copy()
    out = out.sort_values("timestamp_utc")
    return out

def daily_cards_openweather(df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_hourly.empty:
        return pd.DataFrame()

    local = df_hourly.copy()
    local["date_local"] = local["timestamp_local"].dt.date

    today_local = pd.Timestamp.now(tz="Asia/Karachi").date()
    target_dates = [today_local + timedelta(days=i) for i in (1, 2, 3)]

    rows = []
    for d in target_dates:
        day_df = local[local["date_local"] == d].copy()
        if day_df.empty:
            continue

        aqi_mode = day_df["ow_aqi"].dropna().mode()
        aqi_day = int(aqi_mode.iloc[0]) if len(aqi_mode) else None

        rows.append({
            "date": d,
            "ow_aqi": aqi_day,
            "ow_aqi_label": ow_aqi_to_label(aqi_day),
            "pm2_5_avg": float(day_df["pm2_5"].mean()),
            "pm10_avg": float(day_df["pm10"].mean()),
        })

    return pd.DataFrame(rows)

# ---------------- THEME (Lavender/Purple/Black/White) ----------------
# Color tokens
PURPLE_BG_1 = "#07050F"     # near black
PURPLE_BG_2 = "#140A2E"     # deep purple
HERO_PURPLE = "#3B1A7A"     # strong purple
LAVENDER = "#B9A7FF"        # lavender
LAVENDER_SOFT = "#DCD6FF"   # softer lavender for details
CARD_STROKE = "rgba(255,255,255,0.10)"
TEXT_WHITE = "#FFFFFF"
TEXT_MUTED = "#CFC7FF"

# Streamlit base theme overrides
st.markdown(
    f"""
    <style>
      /* App background */
      .stApp {{
        background: radial-gradient(1200px 700px at 15% 10%, {PURPLE_BG_2} 0%, {PURPLE_BG_1} 55%, #000 100%);
        color: {TEXT_WHITE};
      }}

      /* Remove extra padding top */
      .block-container {{ padding-top: 1.2rem; }}

      /* Titles */
      h1, h2, h3, h4, h5, h6 {{ color: {TEXT_WHITE}; }}
      .subtle {{ color: {TEXT_MUTED}; margin-top: -6px; }}

      /* Card styles */
      .card {{
        border-radius: 18px;
        padding: 18px 18px;
        border: 1px solid {CARD_STROKE};
        box-shadow: 0 12px 28px rgba(0,0,0,0.40);
      }}
      .hero {{
        background: linear-gradient(135deg, {HERO_PURPLE} 0%, #2B0F5C 55%, #1B0A3F 100%);
      }}
      .lav {{
        background: linear-gradient(135deg, rgba(185,167,255,0.22) 0%, rgba(185,167,255,0.12) 60%, rgba(255,255,255,0.06) 100%);
      }}

      .big {{ font-size: 34px; font-weight: 850; letter-spacing: 0.2px; }}
      .mid {{ font-size: 18px; font-weight: 700; opacity: 0.96; }}
      .small {{ font-size: 14px; color: {LAVENDER_SOFT}; margin-top: 8px; line-height: 1.35; }}
      .pill {{
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        border: 1px solid {CARD_STROKE};
        font-weight: 700;
        font-size: 13px;
        margin-top: 10px;
        color: {TEXT_WHITE};
      }}

      .hr {{
        height:1px;
        background: rgba(255,255,255,0.12);
        margin: 16px 0;
      }}

      /* Expander */
      details {{
        border-radius: 14px;
        border: 1px solid {CARD_STROKE};
        background: rgba(255,255,255,0.04);
        padding: 4px 8px;
      }}

      /* Dataframe styling wrapper */
      .stDataFrame {{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid {CARD_STROKE};
        background: rgba(255,255,255,0.03);
      }}

      /* Metric cards */
      [data-testid="stMetric"] {{
        background: rgba(255,255,255,0.04);
        border: 1px solid {CARD_STROKE};
        padding: 14px 14px;
        border-radius: 16px;
      }}
      [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}
      [data-testid="stMetricValue"] {{ color: {TEXT_WHITE} !important; }}

      /* Make captions visible */
      .stCaption {{ color: {TEXT_MUTED} !important; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- APP ----------------
st.title("🌫️ Lahore AQI Forecasting Dashboard")
st.markdown(
    '<div class="subtle">OpenWeather AQI (1–5) for next 72 hours + next 3 days cards (tomorrow → day+3).</div>',
    unsafe_allow_html=True
)

api_key = get_api_key()
if not api_key:
    st.error("Missing OPENWEATHER_API_KEY. Add it in Streamlit Cloud Secrets.")
    st.stop()

lat, lon = get_city_coords()
model = load_model()

with st.spinner("Fetching OpenWeather air pollution forecast..."):
    fc = fetch_forecast_df(lat, lon, api_key)

fc_72 = filter_next_hours(fc, hours=72)
if fc_72.empty:
    st.error("No usable forecast rows for the next 72 hours. Please refresh in a few minutes.")
    st.stop()

# ML predictions kept (for project value)
fc_72["predicted_aqi_class"] = model.predict(fc_72[FEATURES])

# ---------- HERO ----------
current = fc_72.iloc[0]
ow_aqi_now = int(current["ow_aqi"]) if pd.notna(current["ow_aqi"]) else None
ow_label_now = ow_aqi_to_label(ow_aqi_now)
advice_now = ow_advice(ow_aqi_now)
icon_now = ow_icon(ow_aqi_now)

pm25 = float(current["pm2_5"])
pm10 = float(current["pm10"])
ts_now = pd.to_datetime(current["timestamp_local"])

st.markdown(
    f"""
    <div class="card hero">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:14px;">
        <div>
          <div class="mid">Current (OpenWeather AQI)</div>
          <div class="big">AQI: {ow_aqi_now if ow_aqi_now is not None else "?"} — {ow_label_now}</div>
          <div class="small">{advice_now}</div>
          <div class="pill">Time: {ts_now.strftime("%d %b %Y, %H:%M")} (PKT)</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:50px; line-height:1;">{icon_now}</div>
          <div class="small" style="margin-top:10px;">
            <b style="color:{TEXT_WHITE};">Main Pollutants</b><br/>
            PM2.5: {pm25:.2f} μg/m³<br/>
            PM10: {pm10:.2f} μg/m³
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- 3 DAY CARDS ----------
st.subheader("📅 Next 3 Days AQI Forecast (OpenWeather 1–5)")
cards = daily_cards_openweather(fc_72)

if cards.empty:
    st.warning("Could not build daily cards (not enough forecast points).")
else:
    cols = st.columns(3)
    for i, row in enumerate(cards.itertuples(index=False)):
        date_str = pd.to_datetime(row.date).strftime("%d %b %Y")
        ow_aqi = row.ow_aqi
        ow_label = row.ow_aqi_label
        icon = ow_icon(ow_aqi)
        advice = ow_advice(ow_aqi)

        with cols[i]:
            st.markdown(
                f"""
                <div class="card lav" style="min-height:170px;">
                  <div class="mid" style="color:{TEXT_WHITE};">{date_str}</div>
                  <div class="big" style="margin-top:6px;">AQI: {ow_aqi if ow_aqi is not None else "?"} — {ow_label} {icon}</div>
                  <div class="small">
                    Avg PM2.5: {row.pm2_5_avg:.2f} μg/m³ &nbsp;|&nbsp;
                    Avg PM10: {row.pm10_avg:.2f} μg/m³
                  </div>
                  <div class="small" style="margin-top:10px;">{advice}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------- HOURLY FORECAST ----------
st.subheader("⏱️ Hourly Forecast (Next 72 Hours)")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Next Hour (OW AQI)", f"{int(fc_72['ow_aqi'].iloc[0]) if pd.notna(fc_72['ow_aqi'].iloc[0]) else '—'}")
k2.metric("Most Common OW AQI (72h)", f"{int(fc_72['ow_aqi'].dropna().mode().iloc[0]) if fc_72['ow_aqi'].dropna().shape[0] else '—'}")
k3.metric("Rows Used", str(len(fc_72)))
k4.metric("Last Forecast Time", str(fc_72["timestamp_local"].iloc[-1])[:16])

# Chart: OW AQI (1..5) and ML class mapped to 1..5
ml_map = {"Good": 1, "Fair": 2, "Moderate": 3, "Poor": 4, "Very Poor": 5, "Unknown": 0}
plot_df = fc_72[["timestamp_local", "ow_aqi", "predicted_aqi_class"]].copy()
plot_df["ow_aqi_num"] = pd.to_numeric(plot_df["ow_aqi"], errors="coerce").fillna(0).astype(float)
plot_df["ml_num"] = plot_df["predicted_aqi_class"].map(ml_map).fillna(0).astype(float)

st.write("### Forecast Chart (OpenWeather AQI vs ML Class)")
st.line_chart(plot_df.set_index("timestamp_local")[["ow_aqi_num", "ml_num"]])
st.caption("ow_aqi_num = OpenWeather AQI (1–5). ml_num = your ML predicted class mapped to 1–5.")

with st.expander("Show hourly forecast table (first 72 rows)"):
    st.dataframe(
        fc_72[["timestamp_local", "ow_aqi", "ow_aqi_label", "predicted_aqi_class"] + FEATURES].reset_index(drop=True),
        use_container_width=True
    )
