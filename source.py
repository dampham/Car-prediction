# ============================================================
# CORE IMPORTS (KHÔNG ĐỔI)
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
# GLOBAL CSS – FULLSCREEN HERO IMAGE (KHÔNG ĐỔI)
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
header {visibility: hidden;}
footer {visibility: hidden;}

.hero-full {
    width: 100%;
    height: 100vh;
    background-image: url("https://img.tripi.vn/cdn-cgi/image/width=1600/https://gcs.tripi.vn/public-tripi/tripi-feed/img/482791EyF/anh-mo-ta.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
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

.stButton>button {
    background: linear-gradient(135deg, #ff4b4b, #ff9068);
    color: white;
    border-radius: 12px;
    padding: 0.7em 2.2em;
    border: none;
    font-size: 16px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

.footer {
    text-align: center;
    color: #777;
    padding: 40px 0;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO – FULL SCREEN IMAGE ONLY
# ============================================================
st.markdown('<div class="hero-full"></div>', unsafe_allow_html=True)

# ============================================================
# MODEL PATH
# ============================================================
MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"

# ============================================================
# LOAD DATA (KHÔNG ĐỔI)
# ============================================================
@st.cache_data
def load_data():
    if not os.path.exists("data.csv"):
        st.error("Không tìm thấy file data.csv")
        return pd.DataFrame()

    data = pd.read_csv("data.csv")
    data = data[data["highway MPG"] < 60]
    data = data[data["city mpg"] < 40]

    data["MSRP"] = pd.to_numeric(
        data["MSRP"].replace("[\$,]", "", regex=True),
        errors="coerce"
    )
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
# TRAIN MODEL (KHÔNG ĐỔI)
# ============================================================
def train_and_save_model(data):
    X = data.drop(["MSRP"], axis=1)
    y = data["MSRP"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    encoder = TargetEncoder(cols=["Make", "Model"])
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_train_num = X_train_enc.select_dtypes(include=[np.number])

    model = GradientBoostingRegressor(n_estimators=100, random_state=100)
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
# SIDEBAR
# ============================================================
menu = st.sidebar.radio(
    "📌 Navigation",
    ["Overview", "EDA Analysis", "Price Prediction"],
)

# ============================================================
# OVERVIEW + DATASET FILTER (TÍNH NĂNG MỚI – KHÔNG ĐỘNG CŨ)
# ============================================================
if menu == "Overview":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 📊 Dataset Overview")

    filtered_data = data.copy()

    with st.expander("🔎 Filter by columns", expanded=True):
        for col in filtered_data.columns:
            if col == "MSRP":
                continue

            if filtered_data[col].dtype == "object":
                options = sorted(filtered_data[col].dropna().unique())
                selected = st.multiselect(col, options, default=options)
                if selected:
                    filtered_data = filtered_data[filtered_data[col].isin(selected)]
            else:
                min_val = float(filtered_data[col].min())
                max_val = float(filtered_data[col].max())
                selected_range = st.slider(
                    col, min_val, max_val, (min_val, max_val)
                )
                filtered_data = filtered_data[
                    (filtered_data[col] >= selected_range[0]) &
                    (filtered_data[col] <= selected_range[1])
                ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(filtered_data):,}")
    col2.metric("Avg MSRP", f"${filtered_data['MSRP'].mean():,.0f}")
    col3.metric("Manufacturers", filtered_data["Make"].nunique())

    st.markdown("### 📋 Filtered Dataset Preview")
    st.dataframe(filtered_data, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# EDA (KHÔNG ĐỔI)
# ============================================================
elif menu == "EDA Analysis":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x="Engine HP", y="MSRP", alpha=0.4)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        data.groupby("Year")["MSRP"].mean().plot(kind="line")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PRICE PREDICTION (KHÔNG ĐỔI)
# ============================================================
elif menu == "Price Prediction":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🤖 Predict Car Price")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        make = col1.selectbox("Make", sorted(data["Make"].unique()))
        model_name = col2.selectbox(
            "Model",
            sorted(data[data["Make"] == make]["Model"].unique())
        )
        hp = col1.number_input(
            "Engine HP", value=int(data["Engine HP"].median())
        )
        year = col2.number_input(
            "Year", min_value=1990, max_value=2025, value=2015
        )

        submit = st.form_submit_button("🚀 Predict")

    if submit:
        input_df = data.drop(["MSRP"], axis=1).iloc[:1].copy()

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
        input_num = input_enc.select_dtypes(include=[np.number])

        price = model.predict(input_num)[0]
        st.success(f"💰 Estimated Price: ${price:,.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
Car Price Prediction System • Machine Learning Deployment
</div>
""", unsafe_allow_html=True)
