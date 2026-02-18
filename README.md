# Lahore AQI Forecasting Dashboard (End-to-End MLOps Project)

## 📌 Project Overview

This project is a complete end-to-end Machine Learning Operations (MLOps) system that forecasts the Air Quality Index (AQI) for Lahore, Pakistan.

The system:

- Fetches air pollution forecast data from the OpenWeather API
- Builds structured features (time-based + pollutant features)
- Trains a machine learning model
- Explains predictions using SHAP
- Automates pipelines via GitHub Actions
- Deploys an interactive Streamlit dashboard
- Supports local feature-store fallback mode

This project satisfies all requirements including:
- Feature engineering
- Model training & evaluation
- Model explainability
- Pipeline automation
- Web-based deployment

---

## 🌍 Live Deployment

- 🔗 **GitHub Repository**: (Paste your GitHub link here)
- 🚀 **Streamlit Live App**: (Paste your Streamlit Cloud link here)

---

## 🏗️ System Architecture

### 1️⃣ Data Source

Data is collected from:
- OpenWeather Air Pollution Forecast API (AQI index 1–5)

Pollutants used:
- PM2.5
- PM10
- NO2
- SO2
- CO
- O3

---

### 2️⃣ Feature Engineering

Generated features include:

- `hour`
- `day`
- `month`
- Pollutant concentrations
- AQI change
- AQI class
- Timestamp metadata

Feature generation scripts:
src/fetch_data.py
src/feature_pipeline.py
src/prepare_ml_data.py

yaml
Copy code

---

### 3️⃣ Model Training & Evaluation

Training scripts:
src/train_model.py
src/train_and_evaluate.py

yaml
Copy code

Outputs:
- `outputs/best_model.joblib`
- `outputs/model_metrics.json`
- `outputs/model_report.txt`

Model performance metrics include:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

### 4️⃣ Model Explainability (SHAP)

Explainability implemented using SHAP:

src/shap_explain.py

yaml
Copy code

Output:
- `outputs/shap_summary.png`

This ensures interpretability and transparency of AQI predictions.

---

### 5️⃣ Feature Store Integration

Feature push script:
src/push_to_feature_store.py

bash
Copy code

Supports two modes:

- `local` mode → saves snapshot locally
- `hopsworks` mode → pushes to remote feature store

Safe local execution:

```powershell
$env:FEATURE_STORE_MODE="local"
python -m src.push_to_feature_store
6️⃣ CI/CD Automation
Automated pipelines configured in:

bash
Copy code
.github/workflows/ci.yml
Automation includes:

Feature generation

Model training

Evaluation

Scheduled pipeline runs

7️⃣ Streamlit Web Dashboard
Main file:

bash
Copy code
app/app.py
Dashboard features:

72-hour AQI forecast

Next 3-day AQI summary cards

Pollutant breakdown

Model-based predictions

SHAP explanation visualization

Professional lavender/purple UI theme

Safe secrets handling
📁 Project Structure
AQI-PREDICTOR
│
├── .github/workflows/
├── app/
│   └── app.py
├── data/
├── outputs/
│   ├── best_model.joblib
│   ├── model_metrics.json
│   ├── model_report.txt
│   └── shap_summary.png
├── src/
│   ├── fetch_data.py
│   ├── feature_pipeline.py
│   ├── train_model.py
│   ├── train_and_evaluate.py
│   ├── shap_explain.py
│   └── push_to_feature_store.py
├── requirements.txt
├── runtime.txt
└── README.md

⚙️ Local Setup Instructions
1️⃣ Clone Repository
git clone <your-repo-link>
cd aqi-predictor

2️⃣ Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt

3️⃣ Add API Key Locally

Create:

.streamlit/secrets.toml


Add:

OPENWEATHER_API_KEY = "YOUR_KEY"
CITY_LAT = "31.5497"
CITY_LON = "74.3436"

4️⃣ Run the App
python -m streamlit run app/app.py

☁️ Streamlit Cloud Deployment

Connect GitHub repository.

Set main file path to:

app/app.py


Add secrets in Streamlit Cloud:

OPENWEATHER_API_KEY
CITY_LAT
CITY_LON


Deploy and reboot the app.

🔐 Security Notes

The following files are NOT committed:

.env

.streamlit/secrets.toml

.venv

Secrets are managed securely via:

Streamlit Cloud Secrets

Environment variables

📊 Key Achievements

✔ End-to-end ML pipeline
✔ Automated CI/CD
✔ Feature engineering
✔ Model evaluation
✔ SHAP explainability
✔ Professional dashboard
✔ Production-ready project structure

🎯 Conclusion

This project demonstrates a fully automated, production-style AQI forecasting system built using modern MLOps principles, structured feature engineering, explainable machine learning, and cloud deployment.

It fulfills all project requirements including:

Data ingestion

Feature engineering

Model training & evaluation

Explainability

Automation

Deployment