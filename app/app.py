import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="AQI Predictor", layout="centered")

st.title("🌫️ Lahore AQI Predictor")

# Load data
df = pd.read_csv("data/features.csv")

st.subheader("📊 Latest AQI Records")
st.dataframe(df.tail(5))

# Load model
model = joblib.load("outputs/best_model.joblib")

st.subheader("🔮 AQI Prediction")

latest_row = df.tail(1).drop(columns=["aqi"])
prediction = model.predict(latest_row)[0]

st.metric("Predicted AQI Level", int(prediction))
