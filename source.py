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

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Dự đoán giá xe ô tô",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = 'model.pkl'
ENCODER_PATH = 'encoder.pkl'

# =====================================================
# UI THEME (CSS + EFFECT) — UI ONLY
# =====================================================
st.markdown("""
<style>
html, body {
    scroll-behavior: smooth;
    background-color: #0f172a;
    color: white;
}

.block-container {
    padding-top: 0rem !important;
}

/* HERO */
.hero {
    height: 85vh;
    background-image:
        linear-gradient(120deg, rgba(0,0,0,0.75), rgba(0,0,0,0.2)),
        url("https://img.tripi.vn/cdn-cgi/image/width=1600/https://gcs.tripi.vn/public-tripi/tripi-feed/img/482791EyF/anh-mo-ta.png");
    background-size: cover;
    background-position: center;
    display: flex;
    align-items: center;
    padding-left: 80px;
}

.hero-box {
    backdrop-filter: blur(14px);
    background: rgba(0,0,0,0.55);
    padding: 60px;
    border-radius: 22px;
    max-width: 650px;
    box-shadow: 0 40px 120px rgba(0,0,0,0.7);
}

.hero-title {
    font-size: 56px;
    font-weight: 800;
}

.hero-sub {
    font-size: 20px;
    margin-top: 16px;
    color: #d1d5db;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.06);
    padding: 28px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    transition: all 0.35s ease;
}

.card:hover {
    transform: translateY(-6px);
}

/* RESULT */
.result {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    padding: 28px;
    border-radius: 18px;
    font-size: 26px;
    font-weight: 800;
    color: black;
    text-align: center;
    margin-top: 20px;
}

/* BUTTON */
button {
    border-radius: 12px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
  <div class="hero-box">
    <div class="hero-title">Car Price Prediction</div>
    <div class="hero-sub">
        Machine Learning • Data Science • Streamlit<br>
        Ứng dụng dự đoán giá xe ô tô thông minh
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA — GIỮ NGUYÊN LOGIC
# =====================================================
@st.cache_data
def load_data():
    if not os.path.exists('data.csv'):
        st.error("Không tìm thấy file data.csv!")
        return pd.DataFrame()

    data = pd.read_csv('data.csv')
    data = data[data['highway MPG'] < 60]
    data = data[data['city mpg'] < 40]

    data['MSRP'] = pd.to_numeric(
        data['MSRP'].replace('[$,]', '', regex=True),
        errors='coerce'
    )
    data['Engine HP'] = pd.to_numeric(data['Engine HP'], errors='coerce')

    data = data.dropna(subset=['Engine HP', 'MSRP'])
    data['Number of Doors'].fillna(data['Number of Doors'].median(), inplace=True)
    data['Engine Fuel Type'].fillna(data['Engine Fuel Type'].mode()[0], inplace=True)
    data['Engine Cylinders'].fillna(4, inplace=True)

    if 'Market Category' in data.columns:
        data.drop(['Market Category'], axis=1, inplace=True)

    data['Years Of Manufacture'] = 2025 - data['Year']
    return data

data = load_data()

# =====================================================
# TRAIN MODEL — GIỮ NGUYÊN LOGIC
# =====================================================
def train_and_save_model(data):
    X = data.drop(['MSRP'], axis=1)
    y = data['MSRP']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    encoder = TargetEncoder(cols=['Make', 'Model'])
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_train_num = X_train_enc.select_dtypes(include=[np.number])

    model = GradientBoostingRegressor(
        n_estimators=100,
        random_state=100
    )
    model.fit(X_train_num, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    return model, encoder

if not os.path.exists(MODEL_PATH):
    with st.status("🚀 Đang huấn luyện mô hình lần đầu..."):
        model, encoder = train_and_save_model(data)
        st.success("Huấn luyện thành công!")
else:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# =====================================================
# SIDEBAR
# =====================================================
menu = st.sidebar.selectbox(
    "📌 Chức năng",
    ["Tổng quan dữ liệu", "Phân tích (EDA)", "Dự đoán giá"]
)

# =====================================================
# PAGE: OVERVIEW
# =====================================================
if menu == "Tổng quan dữ liệu":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Xem trước dữ liệu")
    st.dataframe(data.head(10))
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PAGE: EDA
# =====================================================
elif menu == "Phân tích (EDA)":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Phân tích dữ liệu")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    data.groupby('Year')['MSRP'].mean().plot(kind='bar', ax=ax[0])
    sns.scatterplot(data=data, x='Engine HP', y='MSRP', ax=ax[1], alpha=0.5)
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PAGE: PREDICT — GIỮ NGUYÊN LOGIC
# =====================================================
elif menu == "Dự đoán giá":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🤖 Dự đoán giá xe")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        make = col1.selectbox("Hãng xe", sorted(data['Make'].unique()))
        model_name = col2.selectbox(
            "Dòng xe",
            sorted(data[data['Make'] == make]['Model'].unique())
        )

        hp = col1.number_input(
            "Mã lực (HP)",
            value=int(data['Engine HP'].median())
        )
        year = col2.number_input(
            "Năm sản xuất",
            min_value=1990,
            max_value=2025,
            value=2015
        )

        submit = st.form_submit_button("🚀 Dự đoán")

    if submit:
        input_df = data.drop(['MSRP'], axis=1).iloc[:1].copy()

        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                input_df[col] = data[col].mode()[0]
            else:
                input_df[col] = data[col].median()

        input_df['Make'] = make
        input_df['Model'] = model_name
        input_df['Engine HP'] = hp
        input_df['Year'] = year
        input_df['Years Of Manufacture'] = 2025 - year

        input_enc = encoder.transform(input_df)
        input_num = input_enc.select_dtypes(include=[np.number])

        prediction = model.predict(input_num)

        st.markdown(
            f"<div class='result'>💰 Giá dự đoán: ${prediction[0]:,.2f}</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
