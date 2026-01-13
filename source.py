# ============================================================
# CORE IMPORTS (GIỮ NGUYÊN LOGIC)
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from category_encoders import TargetEncoder

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL STYLE (GLASSMORPHISM + MODERN UI)
# ============================================================
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #0e1117;
    color: #f5f5f5;
}

/* ---------- HERO ---------- */
.hero {
    background-image: url("https://img.tripi.vn/cdn-cgi/image/width=1600/https://gcs.tripi.vn/public-tripi/tripi-feed/img/482791EyF/anh-mo-ta.png");
    background-size: cover;
    background-position: center;
    height: 85vh;
    display: flex;
    align-items: center;
    padding-left: 80px;
}

.hero-box {
    background: rgba(0,0,0,0.65);
    backdrop-filter: blur(10px);
    padding: 60px;
    border-radius: 18px;
    max-width: 650px;
    animation: fadeIn 1.2s ease-in-out;
}

.hero h1 {
    font-size: 52px;
    font-weight: 700;
    color: #ffffff;
}

.hero p {
    font-size: 18px;
    color: #dddddd;
    margin-top: 12px;
}

/* ---------- ANIMATION ---------- */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(30px);}
    to {opacity: 1; transform: translateY(0);}
}

/* ---------- CARD ---------- */
.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* ---------- KPI ---------- */
.kpi {
    text-align: center;
    padding: 25px;
    border-radius: 14px;
    background: rgba(255,255,255,0.05);
}

.kpi h2 {
    font-size: 32px;
    margin: 0;
}

.kpi p {
    color: #aaaaaa;
    margin-top: 5px;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background: linear-gradient(135deg, #ff4b4b, #ff9068);
    color: white;
    border-radius: 12px;
    padding: 0.7em 2em;
    border: none;
    font-size: 16px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #ff9068, #ff4b4b);
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    color: #777;
    margin-top: 80px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO SECTION
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-box">
        <h1>The legacy never fades.</h1>
        <p>Professional car price prediction system powered by Machine Learning.</p>
    </div>
</div>
""", unsafe_allow_html=True)

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
# TRAINING LOGIC (GIỮ NGUYÊN)
# ============================================================
def train_and_save_model(data):
    X = data.drop(["MSRP"], axis=1)
    y = data["MSRP"]

    X_train, X_test, y_train, y_test = train_test_split(
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
# OVERVIEW
# ============================================================
if menu == "Overview":
    st.markdown("## 📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div class='kpi'><h2>{len(data):,}</h2><p>Total Cars</p></div>",
        unsafe_allow_html=True
    )
    col2.markdown(
        f"<div class='kpi'><h2>{data['Make'].nunique()}</h2><p>Manufacturers</p></div>",
        unsafe_allow_html=True
    )
    col3.markdown(
        f"<div class='kpi'><h2>${data['MSRP'].mean():,.0f}</h2><p>Avg Price</p></div>",
        unsafe_allow_html=True
    )

    st.markdown("### Preview Data")
    st.dataframe(data.head(15))

# ============================================================
# EDA
# ============================================================
elif menu == "EDA Analysis":
    st.markdown("## 📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x="Engine HP", y="MSRP", alpha=0.4)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        data.groupby("Year")["MSRP"].mean().plot(kind="line")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICTION (LOGIC GIỮ NGUYÊN)
# ============================================================
elif menu == "Price Prediction":
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

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
Car Price Prediction System • Machine Learning Deployment
</div>
""", unsafe_allow_html=True)
