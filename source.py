import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from category_encoders import TargetEncoder

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AMG Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.block-container { padding: 0 !important; }
html, body, [class*="css"] {
    font-family: 'Helvetica Neue', Arial, sans-serif;
}

/* ===== HERO ===== */
.hero {
    position: relative;
    height: 85vh;
    background-image: url("assets/hero.jpg");
    background-size: cover;
    background-position: center;
}
.hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        to right,
        rgba(0,0,0,0.8),
        rgba(0,0,0,0.2)
    );
}
.hero-content {
    position: absolute;
    top: 30%;
    left: 6%;
    max-width: 520px;
    color: white;
}
.hero h1 {
    font-size: 46px;
    font-weight: 600;
    margin-bottom: 14px;
}
.hero p {
    font-size: 18px;
    opacity: 0.85;
}
.hero-btn {
    margin-top: 22px;
    padding: 12px 28px;
    background: white;
    color: black;
    border: none;
    font-weight: 600;
    cursor: pointer;
}

/* ===== SECTIONS ===== */
.section {
    background: #111;
    padding: 60px 6%;
    color: #eee;
}

/* ===== CARDS ===== */
.cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
}
.card {
    background: #1c1c1c;
    padding: 24px;
    border-radius: 6px;
}
.card h3 {
    margin-bottom: 10px;
}
.card p {
    font-size: 14px;
    opacity: 0.75;
}
</style>
""", unsafe_allow_html=True)

# ================= CONSTANTS =================
MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"

# ================= DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df = df[df['highway MPG'] < 60]
    df = df[df['city mpg'] < 40]
    df['MSRP'] = pd.to_numeric(df['MSRP'].replace('[$,]', '', regex=True))
    df['Engine HP'] = pd.to_numeric(df['Engine HP'])
    df.dropna(subset=['Engine HP', 'MSRP'], inplace=True)
    df['Number of Doors'].fillna(df['Number of Doors'].median(), inplace=True)
    df['Engine Fuel Type'].fillna(df['Engine Fuel Type'].mode()[0], inplace=True)
    df['Engine Cylinders'].fillna(4, inplace=True)
    if 'Market Category' in df.columns:
        df.drop('Market Category', axis=1, inplace=True)
    df['Years Of Manufacture'] = 2025 - df['Year']
    return df

data = load_data()

# ================= MODEL =================
def train_model(data):
    X = data.drop("MSRP", axis=1)
    y = data["MSRP"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    encoder = TargetEncoder(cols=['Make', 'Model'])
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_train_num = X_train_enc.select_dtypes(include=np.number)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_num, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    return model, encoder

if not os.path.exists(MODEL_PATH):
    with st.spinner("Training model for the first time..."):
        model, encoder = train_model(data)
else:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# ================= HERO =================
st.markdown("""
<div class="hero">
    <div class="hero-overlay"></div>
    <div class="hero-content">
        <h1>The legacy never fades.</h1>
        <p>
        Precision engineering meets machine learning.<br>
        Predict premium car prices with confidence.
        </p>
        <button class="hero-btn">Explore Intelligence</button>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= INFO SECTION =================
st.markdown("""
<div class="section">
    <div class="cards">
        <div class="card">
            <h3>AMG Data</h3>
            <p>Historical pricing data from premium automotive brands.</p>
        </div>
        <div class="card">
            <h3>Experience AI</h3>
            <p>Machine learning models trained on real-world car datasets.</p>
        </div>
        <div class="card">
            <h3>Smart Prediction</h3>
            <p>Instant price estimation using advanced regression models.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= PREDICTION =================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("🤖 Predict Car Price")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    make = col1.selectbox("Brand", sorted(data['Make'].unique()))
    model_name = col2.selectbox(
        "Model",
        sorted(data[data['Make'] == make]['Model'].unique())
    )

    hp = col1.slider(
        "Engine Power (HP)",
        int(data['Engine HP'].min()),
        int(data['Engine HP'].max()),
        int(data['Engine HP'].median())
    )

    year = col2.slider("Year of Manufacture", 1990, 2025, 2018)

    submit = st.form_submit_button("🚀 Predict")

if submit:
    input_df = data.drop("MSRP", axis=1).iloc[:1].copy()

    for col in input_df.columns:
        if input_df[col].dtype == "object":
            input_df[col] = data[col].mode()[0]
        else:
            input_df[col] = data[col].median()

    input_df['Make'] = make
    input_df['Model'] = model_name
    input_df['Engine HP'] = hp
    input_df['Year'] = year
    input_df['Years Of Manufacture'] = 2025 - year

    input_enc = encoder.transform(input_df)
    input_num = input_enc.select_dtypes(include=np.number)

    price = model.predict(input_num)[0]
    st.success(f"💰 Estimated Price: **${price:,.2f}**")

st.markdown("</div>", unsafe_allow_html=True)
