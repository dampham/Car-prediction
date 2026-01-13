import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Luxury Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
body {
    background-color: #0e0e0e;
}
.hero {
    background-image: url("https://img.tripi.vn/cdn-cgi/image/width=1600/https://gcs.tripi.vn/public-tripi/tripi-feed/img/482791EyF/anh-mo-ta.png");
    background-size: cover;
    background-position: center;
    height: 80vh;
    display: flex;
    align-items: center;
    padding-left: 80px;
}
.hero-box {
    background: rgba(0,0,0,0.6);
    padding: 50px;
    border-radius: 12px;
    max-width: 620px;
}
.hero h1 {
    color: white;
    font-size: 52px;
    font-weight: 700;
}
.hero p {
    color: #cccccc;
    font-size: 18px;
    margin-top: 15px;
}
.hero button {
    margin-top: 25px;
    padding: 14px 32px;
    background: white;
    border: none;
    font-size: 16px;
    border-radius: 6px;
}
.card {
    background: #111;
    padding: 30px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class="hero">
    <div class="hero-box">
        <h1>The legacy never fades.</h1>
        <p>
            Precision engineering meets machine intelligence.<br>
            Predict premium automobile prices with confidence.
        </p>
        <a href="#predict">
            <button>Explore Intelligence</button>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL & DATA
# =====================================================
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")
    data = pd.read_csv("data.csv")
    return model, encoder, data

model, encoder, data = load_assets()

# =====================================================
# TEMPLATE DATAFRAME (QUAN TRỌNG NHẤT)
# =====================================================
X_template = data.drop(columns=["MSRP"])

# =====================================================
# FORM
# =====================================================
st.markdown("<div id='predict'></div>", unsafe_allow_html=True)
st.markdown("## 🚘 Vehicle Configuration")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        make = st.selectbox("Make", sorted(data["Make"].unique()))
        model_name = st.text_input("Model")
        year = st.slider("Year", 1990, 2025, 2018)
        fuel = st.selectbox("Engine Fuel Type", data["Engine Fuel Type"].dropna().unique())

    with c2:
        hp = st.number_input("Engine HP", 50, 1500, 250)
        cyl = st.selectbox("Engine Cylinders", sorted(data["Engine Cylinders"].dropna().unique()))
        trans = st.selectbox("Transmission Type", data["Transmission Type"].dropna().unique())
        drive = st.selectbox("Driven Wheels", data["Driven_Wheels"].dropna().unique())

    with c3:
        doors = st.selectbox("Number of Doors", sorted(data["Number of Doors"].dropna().unique()))
        size = st.selectbox("Vehicle Size", data["Vehicle Size"].dropna().unique())
        style = st.selectbox("Vehicle Style", data["Vehicle Style"].dropna().unique())
        market = st.selectbox("Market Category", data["Market Category"].dropna().unique())

    c4, c5 = st.columns(2)

    with c4:
        highway = st.number_input("Highway MPG", 5, 80, 30)
        city = st.number_input("City MPG", 5, 60, 22)

    with c5:
        popularity = st.slider("Popularity", 1, 5000, 1000)

    submit = st.form_submit_button("🔮 Predict Price")

# =====================================================
# PREDICTION (FIX 100%)
# =====================================================
if submit:
    input_df = X_template.copy()

    input_df.iloc[0] = {
        "Make": make,
        "Model": model_name,
        "Year": year,
        "Engine Fuel Type": fuel,
        "Engine HP": hp,
        "Engine Cylinders": cyl,
        "Transmission Type": trans,
        "Driven_Wheels": drive,
        "Number of Doors": doors,
        "Market Category": market,
        "Vehicle Size": size,
        "Vehicle Style": style,
        "highway MPG": highway,
        "city mpg": city,
        "Popularity": popularity,
        "Years Of Manufacture": 2025 - year
    }

    input_encoded = encoder.transform(input_df)
    input_encoded = input_encoded.select_dtypes(include=np.number)

    price = model.predict(input_encoded)[0]

    st.markdown("---")
    st.markdown("## 💰 Estimated Market Value")
    st.success(f"${price:,.2f}")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr>
<center>
<p style="color:#777">
Luxury Automotive Intelligence © 2026
</p>
</center>
""", unsafe_allow_html=True)
