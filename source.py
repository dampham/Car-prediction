# ============================================================
# CORE IMPORTS (KHÔNG ĐỔI LOGIC ML)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from category_encoders import TargetEncoder

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SESSION STATE FOR NAVIGATION (LOGIC CHUẨN)
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ============================================================
# GLOBAL CSS – FULLSCREEN HERO + BUTTONS
# ============================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    margin: 0;
    padding: 0;
    background-color: #0e1117;
    color: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
}
header, footer {visibility: hidden;}

.hero-full {
    width: 100%;
    height: 100vh;
    background-image: url("https://img.tripi.vn/cdn-cgi/image/width=1600/https://gcs.tripi.vn/public-tripi/tripi-feed/img/482791EyF/anh-mo-ta.png");
    background-size: cover;
    background-position: center;
}

.hero-buttons {
    position: absolute;
    top: 70%;
    left: 50%;
    transform: translate(-50%, -50%);
    display: flex;
    gap: 30px;
}

.hero-buttons button {
    background: rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(8px);
    padding: 18px 30px;
    border-radius: 14px;
    font-size: 18px;
    color: white;
    transition: 0.3s;
}

.hero-buttons button:hover {
    background: linear-gradient(135deg, #ff4b4b, #ff9068);
    transform: scale(1.05);
}

.section {
    padding: 60px 80px;
}

.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
}

.footer {
    text-align: center;
    color: #777;
    padding: 40px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO IMAGE
# ============================================================
st.markdown('<div class="hero-full"></div>', unsafe_allow_html=True)

# ============================================================
# HOMEPAGE FEATURE BUTTONS
# ============================================================
if st.session_state.page == "home":
    st.markdown('<div class="hero-buttons">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Dataset Overview"):
            st.session_state.page = "overview"
            st.rerun()

    with col2:
        if st.button("📈 EDA Analysis"):
            st.session_state.page = "eda"
            st.rerun()

    with col3:
        if st.button("🤖 Price Prediction"):
            st.session_state.page = "predict"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MODEL PATH
# ============================================================
MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"

# ============================================================
# LOAD DATA (GIỮ NGUYÊN)
# ============================================================
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    data = data[data["highway MPG"] < 60]
    data = data[data["city mpg"] < 40]
    data["MSRP"] = pd.to_numeric(data["MSRP"].replace("[\$,]", "", regex=True), errors="coerce")
    data["Engine HP"] = pd.to_numeric(data["Engine HP"], errors="coerce")
    data = data.dropna(subset=["Engine HP", "MSRP"])
    data["Number of Doors"].fillna(data["Number of Doors"].median(), inplace=True)
    data["Engine Fuel Type"].fillna(data["Engine Fuel Type"].mode()[0], inplace=True)
    data["Engine Cylinders"].fillna(4, inplace=True)
    if "Market Category" in data.columns:
        data.drop(["Market Category"], axis=1, inplace=True)
    data["Years Of Manufacture"] = 2025 - data["Year"]
    return data

data = load_data()

# ============================================================
# TRAIN MODEL (GIỮ NGUYÊN)
# ============================================================
def train_and_save_model(data):
    X = data.drop(["MSRP"], axis=1)
    y = data["MSRP"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=100)
    encoder = TargetEncoder(cols=["Make", "Model"])
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_train_num = X_train_enc.select_dtypes(include=[np.number])
    model = GradientBoostingRegressor(n_estimators=100, random_state=100)
    model.fit(X_train_num, y_train)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    return model, encoder

if not os.path.exists(MODEL_PATH):
    model, encoder = train_and_save_model(data)
else:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# ============================================================
# SIDEBAR NAVIGATION (SYNC WITH HOMEPAGE)
# ============================================================
menu = st.sidebar.radio(
    "📌 Navigation",
    ["Home", "Overview", "EDA Analysis", "Price Prediction"],
    index=["home", "overview", "eda", "predict"].index(st.session_state.page)
)

mapping = {
    "Home": "home",
    "Overview": "overview",
    "EDA Analysis": "eda",
    "Price Prediction": "predict"
}
st.session_state.page = mapping[menu]

# ============================================================
# OVERVIEW + FILTER
# ============================================================
if st.session_state.page == "overview":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 📊 Dataset Overview")

    filtered_data = data.copy()
    with st.expander("🔎 Filter by column", expanded=True):
        for col in filtered_data.columns:
            if col == "MSRP":
                continue
            if filtered_data[col].dtype == "object":
                selected = st.multiselect(col, filtered_data[col].unique(), default=filtered_data[col].unique())
                filtered_data = filtered_data[filtered_data[col].isin(selected)]
            else:
                minv, maxv = float(filtered_data[col].min()), float(filtered_data[col].max())
                r = st.slider(col, minv, maxv, (minv, maxv))
                filtered_data = filtered_data[(filtered_data[col] >= r[0]) & (filtered_data[col] <= r[1])]

    st.dataframe(filtered_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# EDA
# ============================================================
elif st.session_state.page == "eda":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x="Engine HP", y="MSRP", alpha=0.4)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        data.groupby("Year")["MSRP"].mean().plot()
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PRICE PREDICTION
# ============================================================
elif st.session_state.page == "predict":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🤖 Price Prediction")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        make = col1.selectbox("Make", sorted(data["Make"].unique()))
        model_name = col2.selectbox("Model", sorted(data[data["Make"] == make]["Model"].unique()))
        hp = col1.number_input("Engine HP", value=int(data["Engine HP"].median()))
        year = col2.number_input("Year", 1990, 2025, 2015)
        submit = st.form_submit_button("🚀 Predict")

    if submit:
        input_df = data.drop(["MSRP"], axis=1).iloc[:1].copy()
        for col in input_df.columns:
            input_df[col] = data[col].mode()[0] if input_df[col].dtype == "object" else data[col].median()
        input_df["Make"] = make
        input_df["Model"] = model_name
        input_df["Engine HP"] = hp
        input_df["Year"] = year
        input_df["Years Of Manufacture"] = 2025 - year
        input_enc = encoder.transform(input_df)
        input_num = input_enc.select_dtypes(include=[np.number])
        price = model.predict(input_num)[0]
        st.success(f"💰 Estimated Price: ${price:,.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
Car Price Prediction System • ML Deployment
</div>
""", unsafe_allow_html=True)
