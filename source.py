import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Premium Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.hero {
    background-image: url("https://img.tripi.vn/cdn-cgi/image/width=1600/https://gcs.tripi.vn/public-tripi/tripi-feed/img/482791EyF/anh-mo-ta.png");
    background-size: cover;
    background-position: center;
    height: 75vh;
    padding: 80px;
    display: flex;
    align-items: center;
}
.hero-box {
    background: rgba(0,0,0,0.55);
    padding: 50px;
    max-width: 600px;
    border-radius: 12px;
}
.hero h1 {
    color: white;
    font-size: 48px;
    margin-bottom: 20px;
}
.hero p {
    color: #dddddd;
    font-size: 18px;
    margin-bottom: 30px;
}
.hero button {
    background: white;
    color: black;
    padding: 14px 26px;
    font-size: 16px;
    border-radius: 6px;
    border: none;
}
.section {
    padding: 60px 0px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION
# =========================
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

# =========================
# LOAD MODEL & DATA
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")
    data = pd.read_csv("data.csv")
    return model, encoder, data

model, encoder, data = load_assets()

# =========================
# PREDICTION FORM
# =========================
st.markdown("<div id='predict'></div>", unsafe_allow_html=True)
st.markdown("## 🚘 Vehicle Specification")

with st.form("car_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        make = st.selectbox("Make", sorted(data["Make"].unique()))
        model_name = st.text_input("Model")
        year = st.slider("Year", 1990, 2025, 2018)
        engine_fuel = st.selectbox("Engine Fuel Type", data["Engine Fuel Type"].dropna().unique())

    with col2:
        engine_hp = st.number_input("Engine HP", 50, 1500, 250)
        engine_cyl = st.selectbox("Engine Cylinders", sorted(data["Engine Cylinders"].dropna().unique()))
        transmission = st.selectbox("Transmission Type", data["Transmission Type"].dropna().unique())
        driven_wheels = st.selectbox("Driven Wheels", data["Driven_Wheels"].dropna().unique())

    with col3:
        doors = st.selectbox("Number of Doors", sorted(data["Number of Doors"].dropna().unique()))
        vehicle_size = st.selectbox("Vehicle Size", data["Vehicle Size"].dropna().unique())
        vehicle_style = st.selectbox("Vehicle Style", data["Vehicle Style"].dropna().unique())
        market_category = st.selectbox("Market Category", data["Market Category"].dropna().unique())

    col4, col5, col6 = st.columns(3)

    with col4:
        highway_mpg = st.number_input("Highway MPG", 5, 80, 30)
        city_mpg = st.number_input("City MPG", 5, 60, 22)

    with col5:
        popularity = st.slider("Popularity", 1, 5000, 1000)

    with col6:
        msrp = st.number_input("MSRP (optional)", 0, 500000, 30000)

    submit = st.form_submit_button("🔮 Predict Price")

# =========================
# PREDICTION LOGIC (FIXED)
# =========================
if submit:
    input_df = pd.DataFrame([{
        "Make": make,
        "Model": model_name,
        "Year": year,
        "Engine Fuel Type": engine_fuel,
        "Engine HP": engine_hp,
        "Engine Cylinders": engine_cyl,
        "Transmission Type": transmission,
        "Driven_Wheels": driven_wheels,
        "Number of Doors": doors,
        "Market Category": market_category,
        "Vehicle Size": vehicle_size,
        "Vehicle Style": vehicle_style,
        "highway MPG": highway_mpg,
        "city mpg": city_mpg,
        "Popularity": popularity,
        "MSRP": msrp,
        "Years Of Manufacture": 2025 - year
    }])

    input_encoded = encoder.transform(input_df)
    input_encoded = input_encoded.select_dtypes(include=np.number)

    prediction = model.predict(input_encoded)[0]

    st.markdown("---")
    st.markdown("## 💰 Estimated Vehicle Price")
    st.success(f"${prediction:,.2f}")

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<center>
<p style="color:gray">
© 2026 Premium Automotive Intelligence • Built with Streamlit & Machine Learning
</p>
</center>
""", unsafe_allow_html=True)
