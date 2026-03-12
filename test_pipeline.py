"""
CAPSURE Healthcare Analytics Pipeline — Test Suite
Run with: python -m pytest test_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
import os, pickle

# ============================================================
# TEST 1 — Data Loading
# ============================================================
def test_data_loads():
    df = pd.read_csv("outputs/featured_hospital_data.csv")
    assert len(df) > 0, "Dataset should not be empty"
    assert len(df.columns) > 0, "Dataset should have columns"
    print(f"✅ Data loads correctly: {len(df)} records")


# ============================================================
# TEST 2 — Required Columns Exist
# ============================================================
def test_required_columns():
    df = pd.read_csv("outputs/featured_hospital_data.csv")
    required = ["collection_week", "surge_flag",
                "case_fatality_rate", "recovery_rate",
                "active_cases", "daily_new_cases"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    print("✅ All required columns present")


# ============================================================
# TEST 3 — No Missing Values After Cleaning
# ============================================================
def test_no_missing_values():
    df = pd.read_csv("outputs/featured_hospital_data.csv")
    num_cols = ["case_fatality_rate", "recovery_rate", "active_cases", "surge_flag"]
    for col in num_cols:
        nulls = df[col].isnull().sum()
        assert nulls == 0, f"{col} has {nulls} missing values"
    print("✅ No missing values in key columns")


# ============================================================
# TEST 4 — No Negative Values
# ============================================================
def test_no_negative_values():
    df = pd.read_csv("outputs/featured_hospital_data.csv")
    cols = ["active_cases", "recovery_rate", "surge_flag"]
    for col in cols:
        negs = (df[col] < 0).sum()
        assert negs == 0, f"{col} has {negs} negative values"
    print("✅ No negative values in clinical columns")


# ============================================================
# TEST 5 — Surge Flag is Binary (0 or 1 only)
# ============================================================
def test_surge_flag_binary():
    df = pd.read_csv("outputs/featured_hospital_data.csv")
    unique_vals = set(df["surge_flag"].unique())
    assert unique_vals.issubset({0, 1}), f"surge_flag has non-binary values: {unique_vals}"
    print(f"✅ surge_flag is binary: {df['surge_flag'].sum()} surge days")


# ============================================================
# TEST 6 — Feature Value Ranges Are Valid
# ============================================================
def test_feature_ranges():
    df = pd.read_csv("outputs/featured_hospital_data.csv")
    assert df["case_fatality_rate"].max() <= 1.0, "Fatality rate should be <= 1.0"
    assert df["recovery_rate"].max() <= 1.0,      "Recovery rate should be <= 1.0"
    assert df["active_cases"].min() >= 0,         "Active cases should be >= 0"
    print("✅ All feature ranges are valid")


# ============================================================
# TEST 7 — Output Files Exist
# ============================================================
def test_output_files_exist():
    files = [
        "outputs/featured_hospital_data.csv",
        "outputs/validation_report.csv",
        "outputs/pipeline_run_log.json",
        "outputs/surge_model.pkl",
    ]
    for f in files:
        assert os.path.exists(f), f"Missing output file: {f}"
    print("✅ All output files exist")


# ============================================================
# TEST 8 — ML Model Loads and Predicts
# ============================================================
def test_model_loads_and_predicts():
    with open("outputs/surge_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    model   = model_data["model"]
    scaler  = model_data["scaler"]
    features = model_data["features"]

    # Test with a dummy input
    test_input = np.array([[0.05, 0.2, 0.5, 0.4]])
    scaled = scaler.transform(test_input)
    pred   = model.predict(scaled)
    proba  = model.predict_proba(scaled)

    assert pred[0] in [0, 1],         "Prediction should be 0 or 1"
    assert 0 <= proba[0][1] <= 1,     "Probability should be between 0 and 1"
    print(f"✅ Model predicts correctly: surge={pred[0]}, prob={proba[0][1]:.2f}")


# ============================================================
# TEST 9 — Run Log Has Required Keys
# ============================================================
def test_run_log_structure():
    import json
    with open("outputs/pipeline_run_log.json") as f:
        log = json.load(f)
    required_keys = ["run_time", "total_records", "surge_days_detected"]
    for key in required_keys:
        assert key in log, f"Missing key in run log: {key}"
    print("✅ Run log structure is valid")


# ============================================================
# TEST 10 — Dataset Has Enough Records
# ============================================================
def test_sufficient_records():
    df = pd.read_csv("outputs/featured_hospital_data.csv")
    assert len(df) >= 100, f"Dataset too small: {len(df)} records"
    print(f"✅ Sufficient records: {len(df)}")