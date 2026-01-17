import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from category_encoders import TargetEncoder

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================
MODEL_DIR = "models"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

MODEL_PATHS = {
    "Gradient Boosting (Main)": os.path.join(MODEL_DIR, "gbr.pkl"),
    "Linear Regression (Baseline)": os.path.join(MODEL_DIR, "linear.pkl"),
    "Ridge Regression": os.path.join(MODEL_DIR, "ridge.pkl"),
    "Lasso Regression": os.path.join(MODEL_DIR, "lasso.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "rf.pkl"),
}

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATA (KEEP ORIGINAL LOGIC)
# =====================================================
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")

    data = data[data["highway MPG"] < 60]
    data = data[data["city mpg"] < 40]

    data["MSRP"] = pd.to_numeric(
        data["MSRP"].replace("[$,]", "", regex=True),
        errors="coerce"
    )

    data["Engine HP"] = pd.to_numeric(data["Engine HP"], errors="coerce")
    data = data.dropna(subset=["Engine HP", "MSRP"])

    data["Number of Doors"].fillna(data["Number of Doors"].median(), inplace=True)
    data["Engine Fuel Type"].fillna(data["Engine Fuel Type"].mode()[0], inplace=True)
    data["Engine Cylinders"].fillna(4, inplace=True)

    if "Market Category" in data.columns:
        data.drop(columns=["Market Category"], inplace=True)

    data["Years Of Manufacture"] = 2025 - data["Year"]
    return data

data = load_data()

# =====================================================
# TRAIN ALL MODELS (ADD ONLY)
# =====================================================
def train_models(data):
    X = data.drop("MSRP", axis=1)
    y = data["MSRP"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    encoder = TargetEncoder(cols=["Make", "Model"])
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_train_num = X_train_enc.select_dtypes(include=[np.number])

    models = {
        "Gradient Boosting (Main)": GradientBoostingRegressor(
            n_estimators=100, random_state=100
        ),
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=100, n_jobs=-1
        ),
    }

    for name, model in models.items():
        model.fit(X_train_num, y_train)
        joblib.dump(model, MODEL_PATHS[name])

    joblib.dump(encoder, ENCODER_PATH)

# =====================================================
# FIRST RUN TRAINING
# =====================================================
if not os.path.exists(ENCODER_PATH):
    with st.status("Training models for the first time..."):
        train_models(data)
        st.success("Models trained successfully")

encoder = joblib.load(ENCODER_PATH)

@st.cache_resource
def load_models():
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

models = load_models()

# =====================================================
# HERO IMAGE
# =====================================================
st.markdown("""
<style>
.hero {
    background-image: url("https://img.tripi.vn/cdn-cgi/image/width=1600/https://gcs.tripi.vn/public-tripi/tripi-feed/img/482791EyF/anh-mo-ta.png");
    background-size: cover;
    background-position: center;
    height: 90vh;
}
</style>
<div class="hero"></div>
""", unsafe_allow_html=True)

# =====================================================
# NAVIGATION
# =====================================================
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset Overview", "EDA", "Prediction"]
)

# =====================================================
# HOME
# =====================================================
if menu == "Home":
    st.title("Car Price Prediction System")
    st.write(
        "Predict car prices using multiple machine learning models "
        "trained on historical vehicle data."
    )

# =====================================================
# DATASET OVERVIEW + FILTER
# =====================================================
elif menu == "Dataset Overview":
    st.header("Dataset Overview")

    with st.expander("Filter columns"):
        selected_cols = st.multiselect(
            "Select columns",
            data.columns.tolist(),
            default=data.columns.tolist()
        )

    st.dataframe(data[selected_cols])

# =====================================================
# EDA
# =====================================================
elif menu == "EDA":
    st.header("Exploratory Data Analysis")

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    data.groupby("Year")["MSRP"].mean().plot(kind="bar", ax=ax[0])
    sns.scatterplot(data=data, x="Engine HP", y="MSRP", ax=ax[1], alpha=0.4)
    st.pyplot(fig)

# =====================================================
# PREDICTION
# =====================================================
elif menu == "Prediction":
    st.header("Car Price Prediction")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        model_choice = col1.selectbox(
            "Select Model",
            list(models.keys())
        )

        make = col1.selectbox(
            "Car Make",
            sorted(data["Make"].unique())
        )

        model_name = col2.selectbox(
            "Car Model",
            sorted(data[data["Make"] == make]["Model"].unique())
        )

        hp = col1.number_input(
            "Horsepower (HP)",
            value=int(data["Engine HP"].median())
        )

        year = col2.number_input(
            "Year of Manufacture",
            min_value=1990,
            max_value=2025,
            value=2015
        )

        submit = st.form_submit_button("Predict Price")

    if submit:
        input_df = data.drop("MSRP", axis=1).iloc[:1].copy()

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

        prediction = models[model_choice].predict(input_num)[0]

        st.success(
            f"Predicted price using **{model_choice}**: "
            f"${prediction:,.2f}"
        )

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    "<hr><center style='color:gray'>© 2026 Car Price Prediction</center>",
    unsafe_allow_html=True
)
