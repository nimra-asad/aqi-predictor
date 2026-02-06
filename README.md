# 🌫️ AQI Forecasting System – Lahore

## 1. Problem Statement
This project forecasts **Air Quality Index (AQI) categories for Lahore** using historical and real-time air pollution data.  
The system provides **real-time AQI monitoring** and a **3-day AQI forecast** through an interactive Streamlit dashboard.

The goal is to demonstrate an **end-to-end ML system**, covering data ingestion, feature engineering, model training, explainability, CI/CD, and deployment.

---

## 2. Data Sources
- **OpenWeather Air Pollution API**
- **Historical backfill** (~1 year of hourly data, ~8,500 records)
- **Live data ingestion** via API calls

---

## 3. Feature Pipeline
The feature pipeline converts raw API responses into ML-ready features.

### Features
- **Time-based:** hour, day, month  
- **Pollutants:** PM2.5, PM10, NO₂, SO₂, CO, O₃  
- **Target:** AQI Class *(Good, Fair, Moderate, Poor, Very Poor)*

### Key Files
- `src/fetch_data.py`
- `src/feature_pipeline.py`
- `src/clean_features_csv.py`
- `src/make_target.py`

---

## 4. Historical Data Backfill
Historical training data is generated using automated backfill scripts.

### Key File
- `src/backfill_openweather_history.py`

This process creates a consistent dataset for **model training and evaluation**.

---

## 5. Model Training & Evaluation
Multiple machine learning models were trained and compared:

- Random Forest  
- Gradient Boosting  
- Extra Trees  
- Logistic Regression  

The **best-performing model** was selected based on evaluation metrics and saved for inference.

### Key Files
- `src/train_model.py`
- `src/train_and_evaluate.py`

### Artifacts
- `outputs/best_model.joblib`
- `outputs/model_report.txt`
- EDA and evaluation plots (correlation heatmap, AQI distribution)

---

## 6. Web Application
An interactive **Streamlit dashboard** provides:

- Real-time AQI for Lahore  
- 3-day AQI forecast  
- Secure secrets management (API keys not stored in GitHub)

### Entry Point
- `app/app.py`

### Deployment
- Deployed on **Streamlit Cloud**

---

## 7. Feature Store (Hopsworks Architecture)
The project implements a **Feature Store–based design** inspired by Hopsworks.

### Implementation
- Feature Store logic: `src/push_to_feature_store.py`
- **Local development:** feature snapshot stored as Parquet  
  - `outputs/feature_store_snapshot.parquet`
- **Cloud-ready:** Hopsworks integration available for production environments

This approach ensures **consistent features** between training and inference while remaining Windows-friendly for local development.

---

## 8. Model Registry
A **Model Registry–style workflow** is implemented for versioning trained models.

### Implementation
- Versioned model snapshot:
  - `outputs/model_registry_snapshot/`
  - `model_v1.joblib`
  - `metadata.txt`

The structure mirrors how managed registries (e.g., Hopsworks Model Registry) track models, versions, and metadata.

---

## 9. CI/CD Pipeline
**GitHub Actions** is used for CI/CD automation.

### Capabilities
- Runs on every push and pull request
- Scheduled workflows simulate:
  - Periodic feature pipeline execution
  - Daily model retraining

### Workflow File
- `.github/workflows/ci.yml`

This satisfies automated pipeline requirements without requiring Airflow.

---

## 10. Explainability (SHAP)
Model explainability is implemented using **SHAP (SHapley Additive exPlanations)**.

### Details
- **Kernel SHAP** is used to support multiclass classification
- Global feature importance is visualized using a SHAP summary plot

### Artifact
- `outputs/shap_summary.png`

This provides transparency into which pollutants and temporal features most influence AQI predictions.

---

## 11. Future Improvements
- Full production deployment of Hopsworks Feature Store & Model Registry  
- Automated retraining with monitoring and alerts  
- AQI hazard alerts for unhealthy air conditions  
- Multi-city AQI forecasting support  

---

## ✅ Project Highlights
- End-to-end ML pipeline  
- Feature Store & Model Registry architecture  
- CI/CD with scheduled workflows  
- Model explainability with SHAP  
- Deployed real-time forecasting app  
