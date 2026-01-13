import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIG (GIỮ NGUYÊN)
# =====================================================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# CSS (GIỮ NGUYÊN)
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
# HERO (GIỮ NGUYÊN)
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
# FORM (GIỮ NGUYÊN)
# =====================================================
st.markdown("## 🚘 Vehicle Information")

with st.form("predict_form"):
    make = st.selectbox("Make", sorted(data["Make"].dropna().unique()))
    model_name = st.selectbox(
        "Model",
        sorted(data[data["Make"] == make]["Model"].dropna().unique())
    )
    year = st.slider("Year", 1990, 2025, 2018)
    hp = st.number_input("Engine HP", 50, 1500, int(data["Engine HP"].median()))
    fuel = st.selectbox(
        "Engine Fuel Type",
        data["Engine Fuel Type"].dropna().unique()
    )

    submit = st.form_submit_button("🔮 Predict Price")

# =====================================================
# PREDICTION LOGIC (ĐÃ CHỈNH – GIỐNG FILE TRAIN)
# =====================================================
if submit:
    # 1️⃣ TẠO 1 DÒNG MẪU CÓ ĐỦ CỘT NHƯ LÚC TRAIN
    input_df = data.drop(columns=["MSRP"]).iloc[:1].copy()

    # 2️⃣ ĐIỀN GIÁ TRỊ MẶC ĐỊNH (median / mode)
    for col in input_df.columns:
        if input_df[col].dtype == "object":
            input_df[col] = data[col].mode()[0]
        else:
            input_df[col] = data[col].median()

    # 3️⃣ GHI ĐÈ GIÁ TRỊ USER NHẬP
    input_df["Make"] = make
    input_df["Model"] = model_name
    input_df["Year"] = year
    input_df["Engine HP"] = hp
    input_df["Engine Fuel Type"] = fuel
    input_df["Years Of Manufacture"] = 2025 - year

    # 4️⃣ ENCODE (KHÔNG BAO GIỜ LỖI DIMENSION)
    input_enc = encoder.transform(input_df)

    # 5️⃣ LẤY CỘT SỐ (GIỐNG LÚC TRAIN)
    input_num = input_enc.select_dtypes(include=[np.number])

    # 6️⃣ PREDICT
    price = model.predict(input_num)[0]

    st.success(f"💰 Estimated Price: ${price:,.2f}")

# =====================================================
# FOOTER (GIỮ NGUYÊN)
# =====================================================
st.markdown("""
<hr>
<center style="color:gray">
Car Price Prediction • Stable ML Deployment
</center>
""", unsafe_allow_html=True)
