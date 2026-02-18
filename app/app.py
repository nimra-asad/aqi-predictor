# app/app.py
import os
import joblib
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Lahore AQI Forecasting", page_icon="🌫️", layout="wide")

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
        return "Generally okay. Sensitive people should stay aware."
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
    return {1:"😊", 2:"🙂", 3:"😷", 4:"😵‍💫", 5:"☠️"}.get(aqi, "❓")

def aqi_badge_class(label: str) -> str:
    m = {
        "Good": "good",
        "Fair": "fair",
        "Moderate": "moderate",
        "Poor": "poor",
        "Very Poor": "verypoor",
    }
    return m.get(label, "neutral")

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

# ---------------- PROFESSIONAL LIGHT THEME ----------------
st.markdown("""
<style>
/* Base */
.stApp { background: #F7F8FA; color: #0F172A; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1200px; }
html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }

/* Optional: remove Streamlit UI chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Headings */
h1 { letter-spacing: -0.03em; margin-bottom: 0.2rem; }
h2 { letter-spacing: -0.02em; }
.subtle { color: #64748B; margin-top: -6px; }

/* Card */
.card {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
}
.card-header {
  display:flex; justify-content:space-between; align-items:flex-start; gap:14px;
}
.title-sm { font-size: 0.95rem; font-weight: 800; color: #334155; margin-bottom: 8px; }
.big { font-size: 2.05rem; font-weight: 900; color: #0F172A; margin: 0; line-height: 1.15; }
.small { font-size: 0.95rem; color: #475569; margin-top: 10px; line-height: 1.45; }
.pill {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  background: #F8FAFC; border: 1px solid #E5E7EB;
  font-weight: 700; font-size: 0.85rem; color: #0F172A; margin-top: 10px;
}

/* Badges */
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  font-weight: 900; font-size: 0.85rem; border: 1px solid #E5E7EB;
  background: #F8FAFC; color: #0F172A;
}
.neutral { background:#F8FAFC; border-color:#E5E7EB; color:#0F172A; }
.good { background:#ECFDF5; border-color:#A7F3D0; color:#065F46; }
.fair { background:#EFF6FF; border-color:#BFDBFE; color:#1D4ED8; }
.moderate { background:#FFFBEB; border-color:#FDE68A; color:#92400E; }
.poor { background:#FFF7ED; border-color:#FDBA74; color:#9A3412; }
.verypoor { background:#FEF2F2; border-color:#FECACA; color:#991B1B; }

/* Divider */
hr { border:none; border-top: 1px solid #E5E7EB; margin: 18px 0; }

/* Dataframe */
.stDataFrame { border-radius: 14px; overflow: hidden; border: 1px solid #E5E7EB; background: #FFFFFF; }

/* Metric cards */
[data-testid="stMetric"] {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  padding: 14px;
  border-radius: 14px;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
}
[data-testid="stMetricLabel"] { color: #64748B !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- APP HEADER ----------------
st.title("🌫️ Lahore AQI Forecasting Dashboard")
st.markdown('<div class="subtle">OpenWeather AQI (1–5) for next 72 hours + next 3 days cards (tomorrow → day+3).</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

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

# ML predictions kept
fc_72["predicted_aqi_class"] = model.predict(fc_72[FEATURES])

# ---------- CURRENT HERO (CLEAN CARD) ----------
current = fc_72.iloc[0]
ow_aqi_now = int(current["ow_aqi"]) if pd.notna(current["ow_aqi"]) else None
ow_label_now = ow_aqi_to_label(ow_aqi_now)
advice_now = ow_advice(ow_aqi_now)
icon_now = ow_icon(ow_aqi_now)

pm25 = float(current["pm2_5"])
pm10 = float(current["pm10"])
ts_now = pd.to_datetime(current["timestamp_local"])
badge_css = aqi_badge_class(ow_label_now)

col_left, col_right = st.columns([2.1, 1], gap="large")

with col_left:
    st.markdown(
        f"""
        <div class="card">
          <div class="card-header">
            <div style="flex:1;">
              <div class="title-sm">Current AQI (OpenWeather)</div>
              <div class="big">AQI: {ow_aqi_now if ow_aqi_now is not None else "?"} — {ow_label_now}</div>
              <div class="small">{advice_now}</div>
              <div class="pill">🕒 {ts_now.strftime("%d %b %Y, %H:%M")} (PKT)</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:44px; line-height:1;">{icon_now}</div>
              <div style="margin-top:10px;">
                <span class="badge {badge_css}">{ow_label_now}</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_right:
    st.markdown(
        f"""
        <div class="card">
          <div class="title-sm">Main Pollutants</div>
          <div class="small" style="margin-top:0;">
            <b>PM2.5:</b> {pm25:.2f} μg/m³<br/>
            <b>PM10:</b> {pm10:.2f} μg/m³
          </div>
          <div class="pill">📍 Lahore</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------- 3 DAY CARDS ----------
st.subheader("📅 Next 3 Days AQI Forecast (OpenWeather 1–5)")
cards = daily_cards_openweather(fc_72)

if cards.empty:
    st.warning("Could not build daily cards (not enough forecast points).")
else:
    cols = st.columns(3, gap="large")
    for i, row in enumerate(cards.itertuples(index=False)):
        date_str = pd.to_datetime(row.date).strftime("%d %b %Y")
        ow_aqi = row.ow_aqi
        ow_label = row.ow_aqi_label
        icon = ow_icon(ow_aqi)
        advice = ow_advice(ow_aqi)
        css = aqi_badge_class(ow_label)

        with cols[i]:
            st.markdown(
                f"""
                <div class="card" style="min-height:190px;">
                  <div class="card-header">
                    <div>
                      <div class="title-sm">{date_str}</div>
                      <div class="big" style="font-size:1.6rem;">AQI: {ow_aqi if ow_aqi is not None else "?"} — {ow_label}</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:36px; line-height:1;">{icon}</div>
                      <div style="margin-top:10px;">
                        <span class="badge {css}">{ow_label}</span>
                      </div>
                    </div>
                  </div>
                  <div class="small">
                    Avg PM2.5: {row.pm2_5_avg:.2f} μg/m³ &nbsp;|&nbsp; Avg PM10: {row.pm10_avg:.2f} μg/m³
                    <br/><br/>
                    {advice}
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------- HOURLY FORECAST ----------
st.subheader("⏱️ Hourly Forecast (Next 72 Hours)")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Next Hour (OW AQI)", f"{int(fc_72['ow_aqi'].iloc[0]) if pd.notna(fc_72['ow_aqi'].iloc[0]) else '—'}")
k2.metric("Most Common OW AQI (72h)", f"{int(fc_72['ow_aqi'].dropna().mode().iloc[0]) if fc_72['ow_aqi'].dropna().shape[0] else '—'}")
k3.metric("Rows Used", str(len(fc_72)))
k4.metric("Last Forecast Time", str(fc_72['timestamp_local'].iloc[-1])[:16])

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
        use_container_width=True,
        hide_index=True
    )
