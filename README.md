# CAPSURE Healthcare Analytics Pipeline
A reproducible, auditable clinical data pipeline with ML-powered hospital surge prediction — built in Python as part of a healthcare analytics research collaboration at Northeastern University.

## Project Overview
This pipeline processes structured clinical datasets to support healthcare analytics workflows. It automates data validation, feature engineering, surge prediction modeling, and produces an interactive Streamlit dashboard for live predictions.
Inspired by the HSyE Hospital Capacity Surge Model at Northeastern University.

## What This Project Covers
FeatureDescriptionData PreprocessingMissing value handling, outlier clipping, type conversionValidation PipelineRange checks, duplicate detection, null reportingFeature Engineering5 clinical features auto-generated from raw dataML ModelRandom Forest surge predictor — AUC: 0.91Streamlit DashboardLive surge predictor + interactive chartsAudit LogJSON run log with timestamp, record count, feature listTest Suite10 automated pytest test cases

## Project Structure
capsure_pipeline/
│
├── CAPSURE_Healthcare_Pipeline.ipynb  # Main pipeline notebook
├── app.py                             # Streamlit dashboard
├── test_pipeline.py                   # Pytest test suite
│
├── outputs/
│   ├── featured_hospital_data.csv     # Final cleaned dataset
│   ├── validation_report.csv          # Feature quality report
│   ├── pipeline_run_log.json          # Auditable run log
│   ├── surge_model.pkl                # Trained ML model
│   ├── capsure_dashboard.png          # Pipeline output charts
│   └── ml_model_results.png           # ML evaluation charts
│
└── README.md

## Tech Stack

Python — pandas, numpy, scikit-learn, matplotlib
Machine Learning — Random Forest, Logistic Regression, StandardScaler
Dashboard — Streamlit
Testing — pytest
Data — COVID-19 US hospital dataset (816 records)


## How to Run
1. Install dependencies
bashpip install pandas numpy scikit-learn matplotlib streamlit pytest
2. Run the full pipeline
Open CAPSURE_Healthcare_Pipeline.ipynb in Jupyter and run all cells top to bottom.
3. Launch the dashboard
bashpython -m streamlit run app.py
4. Run test suite
bashpython -m pytest test_pipeline.py -v

## Pipeline Stages
Stage 1 — Load
Downloads structured clinical COVID-19 dataset (816 US records, 2020–2022)
Stage 2 — Validate
Checks for missing values, duplicates, negative values, and clinical range violations
Stage 3 — Preprocess
Parses dates, converts types, fills nulls with median, clips outliers
Stage 4 — Feature Engineering
Automatically generates 5 clinical features:

daily_new_cases — day-over-day case growth
case_fatality_rate — deaths / confirmed cases
recovery_rate — recovered / confirmed cases
active_cases — proxy for inpatient bed demand
surge_flag — 1 if cases exceed 7-day rolling average × 1.5

Stage 5 — ML Modeling
Trains Random Forest and Logistic Regression, selects best model by AUC score

Best model: Random Forest (AUC: 0.9123)

Stage 6 — Dashboard
Streamlit app with live surge predictor, feature distributions, and audit log viewer
Stage 7 — Save Outputs
Saves all outputs + timestamped JSON run log for full reproducibility
