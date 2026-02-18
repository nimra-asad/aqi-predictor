# Lahore AQI Forecasting System

## Project Overview

This project implements a complete end-to-end Machine Learning pipeline to forecast the Air Quality Index (AQI) for Lahore, Pakistan.

The system integrates data ingestion, feature engineering, model training, explainability, automation, and deployment into a structured MLOps workflow.

The objective was to design a production-style AQI prediction system that satisfies requirements including feature engineering, model evaluation, explainable AI, and CI/CD automation.

---

## Data Source

AQI forecast data is retrieved from the OpenWeather Air Pollution API (AQI scale 1–5).

Pollutants used in the model:

- PM2.5  
- PM10  
- NO₂  
- SO₂  
- CO  
- O₃  

---

## Feature Engineering

The feature pipeline transforms raw API data into structured model-ready features, including:

- Time-based features: hour, day, month  
- Pollutant concentration features  
- AQI change indicator  
- AQI classification label  

Feature preparation scripts are located in the `src/` directory.

---

## Model Development

A supervised machine learning model was trained using engineered features to predict AQI levels.

The training workflow includes:

- Data preprocessing  
- Model fitting  
- Performance evaluation  
- Model selection  

Evaluation metrics:

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- R² Score  

The final trained model is stored in:

outputs/best_model.joblib

yaml
Copy code

---

## Model Explainability

SHAP (SHapley Additive exPlanations) was implemented to interpret model predictions.

The SHAP summary plot highlights feature importance and provides transparency into how pollutant levels influence AQI predictions.

Output file:

outputs/shap_summary.png

yaml
Copy code

---

## Automation (CI/CD)

Pipeline automation is configured using GitHub Actions.

Automated workflows include:

- Feature generation  
- Model training  
- Evaluation  

This ensures reproducibility and continuous integration of the ML pipeline.

Workflow configuration is located in:

.github/workflows/ci.yml

yaml
Copy code

---

## Feature Store Integration

A feature push script supports both:

- Local snapshot mode  
- Remote feature store mode  

Safe local execution:

```powershell
$env:FEATURE_STORE_MODE="local"
python -m src.push_to_feature_store
Streamlit Deployment
An interactive Streamlit dashboard was developed to:

Display 72-hour AQI forecasts

Show 3-day AQI summary

Present pollutant breakdown

Provide model-based predictions

Visualize SHAP explainability

Main application file:

bash
Copy code
app/app.py
Project Structure
bash
Copy code
aqi-predictor/
├── app/
├── src/
├── outputs/
├── .github/workflows/
├── requirements.txt
└── README.md
Local Setup
Clone repository

Create virtual environment

Install dependencies

Add API key in .streamlit/secrets.toml

Run:

arduino
Copy code
python -m streamlit run app/app.py
Conclusion
This project demonstrates a structured, automated, and explainable AQI forecasting system built using modern MLOps principles.

It successfully integrates:

Data ingestion

Feature engineering

Model training and evaluation

Explainability

CI/CD automation

Cloud deployment