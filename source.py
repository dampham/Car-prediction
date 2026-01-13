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

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ================= CONSTANTS =================
MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"

# ================= DATA =================
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    data = data[data['highway MPG'] < 60]
    data = data[data['city mpg'] < 40]
    data['MSRP'] = pd.to_numeric(data['MSRP'].replace('[$,]', '', regex=True))
    data['Engine HP'] = pd.to_numeric(data['Engine HP'])
    data.dropna(subset=['Engine HP', 'MSRP'], inplace=True)
    data['Number of Doors'].fillna(data['Number of Doors'].median(), inplace=True)
    data['Engine Fuel Type'].fillna(data['Engine Fuel Type'].mode()[0], inplace=True)
    data['Engine Cylinders'].fillna(4, inplace=True)
    if 'Market Category' in data.columns:
        data.drop('Market Category', axis=1, inplace=True)
    data['Years Of Manufacture'] = 2025 - data['Year']
    return data

data = load_data()

# ================= MODEL =================
def train_and_save(data):
    X = data.drop("MSRP", axis=1)
    y = data["MSRP"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    encoder = TargetEncoder(cols=['Make', 'Model'])
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_train_num = X_train_enc.select_dtypes(include=np.number)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_num, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    return model, encoder

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔧 Training model for the first time..."):
        model, encoder = train_and_save(data)
else:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# ================= SIDEBAR =================
st.sidebar.title("🚘 Car Prediction App")
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Data Analysis", "🤖 Price Prediction"]
)

# ================= OVERVIEW =================
if menu == "🏠 Overview":
    st.title("🚗 Car Price Prediction Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cars", f"{len(data):,}")
    c2.metric("Average Price", f"${data['MSRP'].mean():,.0f}")
    c3.metric("Avg Engine HP", f"{data['Engine HP'].mean():.0f} HP")

    st.markdown("### 📄 Dataset Preview")
    st.dataframe(data.head(20), use_container_width=True)

# ================= EDA =================
elif menu == "📊 Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Average Price by Year")
        fig, ax = plt.subplots()
        data.groupby("Year")["MSRP"].mean().plot(ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("#### Engine Power vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=data,
            x="Engine HP",
            y="MSRP",
            alpha=0.4,
            ax=ax
        )
        st.pyplot(fig)

# ================= PREDICTION =================
elif menu == "🤖 Price Prediction":
    st.title("🤖 Predict Car Price")
    st.caption("Fill in the car specifications to estimate price")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        make = col1.selectbox("Car Brand", sorted(data['Make'].unique()))
        model_name = col2.selectbox(
            "Car Model",
            sorted(data[data['Make'] == make]['Model'].unique())
        )

        hp = col1.slider(
            "Engine Power (HP)",
            int(data['Engine HP'].min()),
            int(data['Engine HP'].max()),
            int(data['Engine HP'].median())
        )

        year = col2.slider("Year of Manufacture", 1990, 2025, 2015)

        submitted = st.form_submit_button("🚀 Predict Price")

    if submitted:
        input_df = data.drop("MSRP", axis=1).iloc[:1].copy()

        for col in input_df.columns:
            if input_df[col].dtype == "object":
                input_df[col] = data[col].mode()[0]
            else:
                input_df[col] = data[col].median()

        input_df['Make'] = make
        input_df['Model'] = model_name
        input_df['Engine HP'] = hp
        input_df['Year'] = year
        input_df['Years Of Manufacture'] = 2025 - year

        input_enc = encoder.transform(input_df)
        input_num = input_enc.select_dtypes(include=np.number)

        price = model.predict(input_num)[0]

        st.success(f"💰 Estimated Car Price: **${price:,.2f}**")
