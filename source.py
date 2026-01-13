import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from category_encoders import TargetEncoder

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AMG Car Price Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# HERO IMAGE (MÀY YÊU CẦU)
# =====================================================
HERO_IMAGE = (
    "https://img.tripi.vn/cdn-cgi/image/width=1600/"
    "https://gcs.tripi.vn/public-tripi/tripi-feed/img/482791EyF/anh-mo-ta.png"
)

# =====================================================
# CSS – FIX TRẮNG / ĐEN + HERO IMAGE FULL
# =====================================================
st.markdown(f"""
<style>
.block-container {{
    padding: 0 !important;
}}

html, body, [class*="css"] {{
    font-family: Helvetica, Arial, sans-serif;
}}

.hero {{
    position: relative;
    height: 90vh;
    width: 100%;
    background-image: url("{HERO_IMAGE}");
    background-size: cover;
    background-position: center right;
}}

.hero::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(
        to right,
        rgba(0,0,0,0.75),
        rgba(0,0,0,0.15)
    );
}}

.hero-content {{
    position: absolute;
    top: 30%;
    left: 6%;
    max-width: 520px;
    color: white;
}}

.hero h1 {{
    font-size: 50px;
    font-weight: 600;
    margin-bottom: 18px;
}}

.hero p {{
    font-size: 18px;
    line-height: 1.6;
    opacity: 0.9;
}}

.hero-btn {{
    margin-top: 24px;
    padding: 14px 34px;
    background: white;
    color: black;
    border: none;
    font-weight: 600;
    cursor: pointer;
}}

.section {{
    background: #0f0f0f;
    padding: 70px 6%;
    color: #e5e5e5;
}}

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

.form-box {{
    background: #1c1c1c;
    padding: 32px;
    border-radius: 8px;
}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA (GIỮ NGUYÊN)
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

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

# =====================================================
# MODEL (GIỮ NGUYÊN)
# =====================================================
MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"

def train_model(df):
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
    model, encoder = train_model(data)
else:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# =====================================================
# HERO SECTION (ĐÃ FIX TRỐNG ĐEN)
# =====================================================
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

# =====================================================
# PREDICTION FORM (FULL COLUMNS)
# =====================================================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("🤖 Price Prediction")

with st.form("prediction_form"):
    st.markdown("<div class='form-box'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    make = c1.selectbox("Make", sorted(data["Make"].unique()))
    model_name = c2.selectbox(
        "Model", sorted(data[data["Make"] == make]["Model"].unique())
    )
    year = c3.slider("Year", 1990, 2025, 2018)

    engine_fuel = c1.selectbox(
        "Engine Fuel Type", sorted(data["Engine Fuel Type"].unique())
    )
    engine_hp = c2.slider(
        "Engine HP", int(data["Engine HP"].min()),
        int(data["Engine HP"].max()),
        int(data["Engine HP"].median())
    )
    engine_cyl = c3.selectbox(
        "Engine Cylinders", sorted(data["Engine Cylinders"].unique())
    )

    highway_mpg = c1.slider(
        "Highway MPG",
        int(data["highway MPG"].min()),
        int(data["highway MPG"].max()),
        int(data["highway MPG"].median())
    )
    city_mpg = c2.slider(
        "City MPG",
        int(data["city mpg"].min()),
        int(data["city mpg"].max()),
        int(data["city mpg"].median())
    )
    popularity = c3.slider(
        "Popularity",
        int(data["Popularity"].min()),
        int(data["Popularity"].max()),
        int(data["Popularity"].median())
    )

    submit = st.form_submit_button("🚀 Predict Price")
    st.markdown("</div>", unsafe_allow_html=True)

if submit:
    input_df = data.drop("MSRP", axis=1).iloc[:1].copy()

    for col in input_df.columns:
        if input_df[col].dtype == "object":
            input_df[col] = data[col].mode()[0]
        else:
            input_df[col] = data[col].median()

    input_df.update({
        "Make": make,
        "Model": model_name,
        "Year": year,
        "Engine Fuel Type": engine_fuel,
        "Engine HP": engine_hp,
        "Engine Cylinders": engine_cyl,
        "highway MPG": highway_mpg,
        "city mpg": city_mpg,
        "Popularity": popularity,
        "Years Of Manufacture": 2025 - year
    })

    price = model.predict(
        encoder.transform(input_df).select_dtypes(include=np.number)
    )[0]

    st.success(f"💰 Estimated Price: **${price:,.2f}**")

st.markdown("</div>", unsafe_allow_html=True)
