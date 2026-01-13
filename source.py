import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
.hero {
    background-image: url("https://img.tripi.vn/cdn-cgi/image/width=1600/https://gcs.tripi.vn/public-tripi/tripi-feed/img/482791EyF/anh-mo-ta.png");
    background-size: cover;
    background-position: center;
    height: 75vh;
    display: flex;
    align-items: center;
    padding-left: 80px;
}
.hero-box {
    background: rgba(0,0,0,0.6);
    padding: 50px;
    border-radius: 14px;
    max-width: 620px;
}
.hero h1 {
    color: white;
    font-size: 50px;
    font-weight: 700;
}
.hero p {
    color: #dddddd;
    font-size: 18px;
    margin-top: 15px;
}
.hero button {
    margin-top: 25px;
    padding: 14px 30px;
    background: white;
    border: none;
    font-size: 16px;
    border-radius: 6px;
}
.section {
    padding: 60px 0px;
}
.card {
    background: #111;
    padding: 30px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class="hero">
    <div class="hero-box">
        <h1>The legacy never fades.</h1>
        <p>
            Predict car prices using machine learning<br>
            with stable deployment and clean design.
        </p>
        <a href="#predict">
            <button>Start Prediction</button>
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
# TEMPLATE (CỰC KỲ QUAN TRỌNG)
# =====================================================
X_template = data.drop(columns=["MSRP"])

# =====================================================
# FORM
# =====================================================
st.markdown("<div id='predict'></div>", unsafe_allow_html=True)
st.markdown("## 🚘 Vehicle Information")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        make = st.selectbox("Make", sorted(data["Make"].dropna().unique()))
        year = st.slider("Year", 1990, 2025, 2018)
        fuel = st.selectbox("Engine Fuel Type", data["Engine Fuel Type"].dropna().unique())

    with c2:
        hp = st.number_input("Engine HP", 50, 1500, 250)
        cyl = st.selectbox(
            "Engine Cylinders",
            sorted(data["Engine Cylinders"].dropna().unique())
        )
        transmission = st.selectbox(
            "Transmission Type",
            data["Transmission Type"].dropna().unique()
        )

    with c3:
        drive = st.selectbox(
            "Driven Wheels",
            data["Driven_Wheels"].dropna().unique()
        )
        size = st.selectbox(
            "Vehicle Size",
            data["Vehicle Size"].dropna().unique()
        )
        style = st.selectbox(
            "Vehicle Style",
            data["Vehicle Style"].dropna().unique()
        )

    mpg1, mpg2, pop = st.columns(3)
    with mpg1:
        highway = st.number_input("Highway MPG", 5, 80, 30)
    with mpg2:
        city = st.number_input("City MPG", 5, 60, 22)
    with pop:
        popularity = st.slider("Popularity", 1, 5000, 1000)

    submit = st.form_submit_button("🔮 Predict Price")

# =====================================================
# PREDICTION LOGIC (STABLE – NO ERROR)
# =====================================================
if submit:
    # Copy full template to keep all columns
    input_df = X_template.copy()

    # User-controlled features
    input_df["Make"] = make
    input_df["Year"] = year
    input_df["Engine Fuel Type"] = fuel
    input_df["Engine HP"] = hp
    input_df["Engine Cylinders"] = cyl
    input_df["Transmission Type"] = transmission
    input_df["Driven_Wheels"] = drive
    input_df["Vehicle Size"] = size
    input_df["Vehicle Style"] = style
    input_df["highway MPG"] = highway
    input_df["city mpg"] = city
    input_df["Popularity"] = popularity
    input_df["Years Of Manufacture"] = 2025 - year

    # Fill remaining columns safely
    for col in input_df.columns:
        if input_df[col].isna().any():
            if input_df[col].dtype == "object":
                input_df[col].fillna(data[col].mode()[0], inplace=True)
            else:
                input_df[col].fillna(data[col].median(), inplace=True)

    # Encode
    input_encoded = encoder.transform(input_df)
    input_encoded = input_encoded.select_dtypes(include=np.number)

    # Align features with model
    input_encoded = input_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Predict
    price = model.predict(input_encoded)[0]

    st.markdown("---")
    st.markdown("## 💰 Estimated Price")
    st.success(f"${price:,.2f}")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr>
<center>
<p style="color:gray">
Car Price Prediction System © 2026<br>
Machine Learning • Streamlit
</p>
</center>
""", unsafe_allow_html=True)
