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
    font-size: 48px;
}
.hero p {
    color: #dddddd;
    font-size: 18px;
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
        <p>Stable ML deployment for car price prediction</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD ASSETS
# =====================================================
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")
    data = pd.read_csv("data.csv")
    return model, encoder, data

model, encoder, data = load_assets()

# =====================================================
# GET ENCODER INPUT COLUMNS (CỰC KỲ QUAN TRỌNG)
# =====================================================
if hasattr(encoder, "feature_names_in_"):
    encoder_input_cols = list(encoder.feature_names_in_)
elif hasattr(encoder, "cols"):
    encoder_input_cols = list(encoder.cols)
else:
    st.error("❌ Cannot determine encoder input columns")
    st.stop()

# =====================================================
# FORM (GIẢM THUỘC TÍNH)
# =====================================================
st.markdown("## 🚘 Vehicle Information")

with st.form("predict_form"):
    make = st.selectbox("Make", sorted(data["Make"].dropna().unique()))
    year = st.slider("Year", 1990, 2025, 2018)
    hp = st.number_input("Engine HP", 50, 1500, 250)
    fuel = st.selectbox(
        "Engine Fuel Type",
        data["Engine Fuel Type"].dropna().unique()
    )

    submit = st.form_submit_button("🔮 Predict Price")

# =====================================================
# PREDICTION (100% SAFE)
# =====================================================
if submit:
    # Tạo input đúng CỘT encoder cần
    input_df = pd.DataFrame(columns=encoder_input_cols)

    for col in encoder_input_cols:
        if col == "Make":
            input_df.loc[0, col] = make
        elif col == "Year":
            input_df.loc[0, col] = year
        elif col == "Engine HP":
            input_df.loc[0, col] = hp
        elif col == "Engine Fuel Type":
            input_df.loc[0, col] = fuel
        elif col == "Years Of Manufacture":
            input_df.loc[0, col] = 2025 - year
        else:
            # Fill mặc định từ data train
            if data[col].dtype == "object":
                input_df.loc[0, col] = data[col].mode()[0]
            else:
                input_df.loc[0, col] = data[col].median()

    # Encode
    input_encoded = encoder.transform(input_df)
    input_encoded = input_encoded.select_dtypes(include=np.number)

    # Align with model
    input_encoded = input_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    price = model.predict(input_encoded)[0]

    st.success(f"💰 Estimated Price: ${price:,.2f}")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr>
<center style="color:gray">
Car Price Prediction • Stable ML Deployment
</center>
""", unsafe_allow_html=True)
