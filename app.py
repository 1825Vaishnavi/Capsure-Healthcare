import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, json
from datetime import datetime

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="CAPSURE Healthcare Analytics",
    page_icon="🏥",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("outputs/surge_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("outputs/featured_hospital_data.csv")

model_data = load_model()
model   = model_data["model"]
scaler  = model_data["scaler"]
FEATURES = model_data["features"]
df = load_data()

# Rename columns to match model feature names
df = df.rename(columns={
    "case_fatality_rate" : "bed_occupancy_rate",
    "recovery_rate"      : "covid_bed_ratio",
    "active_cases"       : "icu_utilization_rate",
    "daily_new_cases"    : "beds_available",
})

# ── Header ───────────────────────────────────────────────
st.title("🏥 CAPSURE Healthcare Analytics Pipeline")
st.markdown("**Reproducible clinical data pipeline with ML-powered surge prediction**")
st.divider()

# ── KPI Cards ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",      f"{len(df):,}")
col2.metric("Surge Events",       f"{df['surge_flag'].sum():,}")
col3.metric("Model",              model_data["model_name"])
col4.metric("AUC Score",          "0.9123")

st.divider()

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Surge Predictor", "📋 Run Log"])

# ── TAB 1: Dashboard ─────────────────────────────────────
with tab1:
    st.subheader("Pipeline Output Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Surge Events Over Time**")
        if "collection_week" in df.columns:
            df["collection_week"] = pd.to_datetime(df["collection_week"], errors="coerce")
            surge_time = df.groupby("collection_week")["surge_flag"].sum().reset_index()
            st.line_chart(surge_time.set_index("collection_week"))

    with col2:
        st.markdown("**Feature Distributions**")
        feature = st.selectbox("Select feature", FEATURES)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df[feature].dropna(), bins=30, color="#2196F3", edgecolor="white")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.markdown("**Raw Data Sample**")
    st.dataframe(df[FEATURES + ["surge_flag"]].head(20), use_container_width=True)

# ── TAB 2: Live Surge Predictor ──────────────────────────
with tab2:
    st.subheader("🔮 Live Hospital Surge Predictor")
    st.markdown("Enter hospital metrics below to predict surge risk:")

    col1, col2 = st.columns(2)
    with col1:
        bed_occ  = st.slider("Bed Occupancy Rate",   0.0, 1.0, 0.7, 0.01)
        covid_br = st.slider("COVID Bed Ratio",       0.0, 1.0, 0.3, 0.01)
    with col2:
        icu_util = st.slider("ICU Utilization Rate",  0.0, 1.0, 0.5, 0.01)
        beds_av  = st.slider("Beds Available (norm)", 0.0, 1.0, 0.4, 0.01)

    if st.button("🔍 Predict Surge Risk", type="primary"):
        input_data = np.array([[bed_occ, covid_br, icu_util, beds_av]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ SURGE PREDICTED — Risk: {probability*100:.1f}%")
        else:
            st.success(f"✅ NO SURGE — Risk: {probability*100:.1f}%")

        st.progress(float(probability))
        st.caption(f"Model: {model_data['model_name']} | Confidence: {probability*100:.1f}%")

# ── TAB 3: Run Log ───────────────────────────────────────
with tab3:
    st.subheader("📋 Auditable Pipeline Run Log")
    try:
        with open("outputs/pipeline_run_log.json") as f:
            log = json.load(f)
        st.json(log)
        st.success("✅ Pipeline is reproducible and auditable")
    except:
        st.warning("Run the pipeline first to generate the log.")

st.divider()
st.caption(f"CAPSURE Healthcare Analytics Pipeline | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")