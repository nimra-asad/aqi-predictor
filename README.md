# AQI Forecasting System – Lahore

## 1. Problem Statement
This project predicts Air Quality Index (AQI) categories for Lahore using historical and real-time air pollution data. The system provides real-time AQI monitoring and a 3-day forecast via a Streamlit dashboard.

## 2. Data Sources
- OpenWeather Air Pollution API
- Historical backfill (~1 year of data)
- Live data fetched via API

## 3. Feature Pipeline
The feature pipeline fetches raw pollutant data and computes ML-ready features.

Key features:
- Time-based: hour, day, month
- Pollutants: PM2.5, PM10, NO₂, SO₂, CO, O₃
- Target variable: AQI Class (Good, Fair, Moderate, Poor, Very Poor)

Relevant files:
- src/fetch_data.py
- src/feature_pipeline.py
- src/clean_features_csv.py
- src/make_target.py

## 4. Historical Data Backfill
Historical data is generated using:
- src/backfill_openweather_history.py

This creates a dataset used for model training and evaluation.

## 5. Model Training & Evaluation
Multiple models were trained and evaluated:
- Random Forest
- Gradient Boosting
- Extra Trees
- Logistic Regression

The best model is selected and saved.

Relevant files:
- src/train_model.py
- src/train_and_evaluate.py

Artifacts:
- outputs/best_model.joblib
- outputs/model_report.txt
- EDA and evaluation plots

## 6. Web Application
The Streamlit dashboard:
- Shows real-time AQI for Lahore
- Displays a 3-day AQI forecast
- Uses secure secrets management

Entry point:
- app/app.py

Deployed on Streamlit Cloud.

## 7. Feature Store (Hopsworks)
A Feature Store integration is implemented using Hopsworks to store processed features.

Relevant file:
- src/push_to_feature_store.py

This enables consistent feature usage for training and inference.

## 8. Model Registry
The project is designed to register trained models in a Model Registry for versioning and reuse. The best model is currently saved locally and prepared for registry integration.

## 9. CI/CD Pipeline
GitHub Actions is used for CI/CD:
- Automated checks and training steps
- Workflow defined in .github/workflows/ci.yml

## 10. Explainability
Model explainability is addressed using feature importance analysis for tree-based models. SHAP integration is planned to provide deeper insights.

## 11. Future Improvements
- Full SHAP integration
- Automated retraining schedules
- AQI hazard alerts
- Support for additional cities
