# ============================================================
#  AMG STYLE CAR PRICE PREDICTION DASHBOARD
#  Author: Dam Pham
#  Purpose: Machine Learning + Luxury UI (Streamlit)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from category_encoders import TargetEncoder

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AMG Car Price Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# GLOBAL CSS (AMG / MERCEDES STYLE)
# ============================================================

HERO_IMAGE = (
    "https://images.unsplash.com/photo-1617814076231-6c5f93f87e6b"
    "?auto=format&fit=crop&w=1920&q=80"
)

st.markdown(f"""
<style>

/* ===== RESET ===== */
.block-container {{
    padding: 0 !important;
}}
html, body, [class*="css"] {{
    font-family: "Helvetica Neue", Arial, sans-serif;
}}

/* ===== HERO SECTION ===== */
.hero {{
    position: relative;
    height: 90vh;
    background-image: url("{HERO_IMAGE}");
    background-size: cover;
    background-position: center;
}}
.hero::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(
        to right,
        rgba(0,0,0,0.85),
        rgba(0,0,0,0.25)
    );
}}
.hero-content {{
    position: absolute;
    top: 32%;
    left: 6%;
    max-width: 560px;
    color: #fff;
}}
.hero h1 {{
    font-size: 52px;
    font-weight: 600;
    margin-bottom: 18px;
}}
.hero p {{
    font-size: 19px;
    line-height: 1.6;
    opacity: 0.9;
}}
.hero-btn {{
    margin-top: 26px;
    padding: 14px 34px;
    background: #ffffff;
    color: #000000;
    border: none;
    font-weight: 600;
    font-size: 15px;
    cursor: pointer;
}}

/* ===== SECTIONS ===== */
.section {{
    background: #0f0f0f;
    padding: 70px 6%;
    color: #e5e5e5;
}}
.section-title {{
    font-size: 32px;
    margin-bottom: 32px;
}}

/* ===== CARDS ===== */
.cards {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 28px;
}}
.card {{
    background: #1c1c1c;
    padding: 28px;
    border-radius: 8px;
}}
.card h3 {{
    margin-bottom: 14px;
    font-size: 20px;
}}
.card p {{
    font-size: 14px;
    opacity: 0.8;
    line-height: 1.6;
}}

/* ===== METRICS ===== */
.metrics {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 28px;
}}
.metric {{
    background: #1c1c1c;
    padding: 28px;
    border-radius: 8px;
    text-align: center;
}}
.metric h2 {{
    margin: 0;
    font-size: 34px;
}}
.metric span {{
    font-size: 14px;
    opacity: 0.7;
}}

/* ===== FORM ===== */
.form-box {{
    background: #1c1c1c;
    padding: 32px;
    border-radius: 8px;
}}

</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"
DATA_PATH = "data.csv"

# ============================================================
# DATA LOADING & PREPROCESSING (GIỮ NGUYÊN LOGIC CŨ)
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df[df['highway MPG'] < 60]
    df = df[df['city mpg'] < 40]

    df['MSRP'] = pd.to_numeric(
        df['MSRP'].replace('[$,]', '', regex=True),
        errors='coerce'
    )
    df['Engine HP'] = pd.to_numeric(df['Engine HP'], errors='coerce')

    df.dropna(subset=['Engine HP', 'MSRP'], inplace=True)

    df['Number of Doors'].fillna(df['Number of Doors'].median(), inplace=True)
    df['Engine Fuel Type'].fillna(df['Engine Fuel Type'].mode()[0], inplace=True)
    df['Engine Cylinders'].fillna(4, inplace=True)

    if 'Market Category' in df.columns:
        df.drop('Market Category', axis=1, inplace=True)

    df['Years Of Manufacture'] = 2025 - df['Year']
    return df

data = load_data()

# ============================================================
# MODEL TRAINING (GIỮ NGUYÊN)
# ============================================================

def train_and_save_model(df):
    X = df.drop("MSRP", axis=1)
    y = df["MSRP"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    encoder = TargetEncoder(cols=["Make", "Model"])
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_train_num = X_train_enc.select_dtypes(include=np.number)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_num, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    return model, encoder

if not os.path.exists(MODEL_PATH):
    with st.spinner("Training model for the first time..."):
        model, encoder = train_and_save_model(data)
else:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# ============================================================
# HERO LANDING PAGE
# ============================================================

st.markdown(f"""
<div class="hero">
    <div class="hero-content">
        <h1>The legacy never fades.</h1>
        <p>
            Precision engineering meets machine intelligence.<br>
            Predict premium automobile prices with confidence.
        </p>
        <button class="hero-btn">Explore Intelligence</button>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# OVERVIEW SECTION
# ============================================================

st.markdown("""
<div class="section">
    <div class="section-title">Performance Overview</div>
    <div class="metrics">
        <div class="metric">
            <h2>{cars}</h2>
            <span>Total Vehicles</span>
        </div>
        <div class="metric">
            <h2>${avg_price}</h2>
            <span>Average Price</span>
        </div>
        <div class="metric">
            <h2>{hp} HP</h2>
            <span>Average Engine Power</span>
        </div>
    </div>
</div>
""".format(
    cars=f"{len(data):,}",
    avg_price=f"{int(data['MSRP'].mean()):,}",
    hp=f"{int(data['Engine HP'].mean())}"
), unsafe_allow_html=True)

# ============================================================
# FEATURE SECTION
# ============================================================

st.markdown("""
<div class="section">
    <div class="cards">
        <div class="card">
            <h3>AMG Data</h3>
            <p>
                Curated automotive dataset with thousands of premium vehicles,
                cleaned and optimized for machine learning analysis.
            </p>
        </div>
        <div class="card">
            <h3>Experience AI</h3>
            <p>
                Gradient Boosting regression model trained on real-world pricing
                patterns and technical specifications.
            </p>
        </div>
        <div class="card">
            <h3>Smart Prediction</h3>
            <p>
                Instant price estimation based on brand, model, year, and engine
                performance.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown("""
<div class="section">
    <div class="section-title">Price Prediction Engine</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='section'>", unsafe_allow_html=True)

with st.form("prediction_form"):
    st.markdown("<div class='form-box'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    make = col1.selectbox("Brand", sorted(data["Make"].unique()))
    model_name = col2.selectbox(
        "Model",
        sorted(data[data["Make"] == make]["Model"].unique())
    )

    hp = col1.slider(
        "Engine Power (HP)",
        int(data["Engine HP"].min()),
        int(data["Engine HP"].max()),
        int(data["Engine HP"].median())
    )

    year = col2.slider("Year of Manufacture", 1990, 2025, 2018)

    submit = st.form_submit_button("Predict Price")
    st.markdown("</div>", unsafe_allow_html=True)

if submit:
    input_df = data.drop("MSRP", axis=1).iloc[:1].copy()

    for col in input_df.columns:
        if input_df[col].dtype == "object":
            input_df[col] = data[col].mode()[0]
        else:
            input_df[col] = data[col].median()

    input_df["Make"] = make
    input_df["Model"] = model_name
    input_df["Engine HP"] = hp
    input_df["Year"] = year
    input_df["Years Of Manufacture"] = 2025 - year

    input_enc = encoder.transform(input_df)
    input_num = input_enc.select_dtypes(include=np.number)

    price = model.predict(input_num)[0]

    st.success(f"💰 Estimated Vehicle Price: **${price:,.2f}**")

st.markdown("</div>", unsafe_allow_html=True)
