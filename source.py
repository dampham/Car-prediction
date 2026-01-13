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
from sklearn.preprocessing import MinMaxScaler

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Car Prices Prediction.", layout="wide")

# =====================================================
# UI – CSS (CHỈ THÊM, KHÔNG ĐỤNG LOGIC)
# =====================================================
st.markdown("""
<style>
.hero {
    background-image: url("https://img.tripi.vn/cdn-cgi/image/width=1600/https://gcs.tripi.vn/public-tripi/tripi-feed/img/482791EyF/anh-mo-ta.png");
    background-size: cover;
    background-position: center;
    height: 65vh;
    display: flex;
    align-items: center;
    padding-left: 80px;
    margin-bottom: 40px;
}
.hero-box {
    background: rgba(0,0,0,0.6);
    padding: 50px;
    border-radius: 14px;
    max-width: 620px;
}
.hero h1 {
    color: white;
    font-size: 46px;
    font-weight: 700;
}
.hero p {
    color: #dddddd;
    font-size: 18px;
    margin-top: 15px;
}
.sidebar .sidebar-content {
    background-color: #f7f7f7;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO UI
# =====================================================
st.markdown("""
<div class="hero">
    <div class="hero-box">
        <h1>Car Price Prediction</h1>
        <p>
            Machine learning application predicts car prices.<br>
            based on actual data
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = 'model.pkl'
ENCODER_PATH = 'encoder.pkl'

# =====================================================
# LOAD & PREPROCESS DATA (GIỮ NGUYÊN)
# =====================================================
@st.cache_data
def load_data():
    if not os.path.exists('data.csv'):
        st.error("Không tìm thấy file data.csv!")
        return pd.DataFrame()
    
    data = pd.read_csv('data.csv')
    data = data[data['highway MPG'] < 60]
    data = data[data['city mpg'] < 40]
    data['MSRP'] = pd.to_numeric(data['MSRP'].replace('[$,]', '', regex=True), errors='coerce')
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
# TRAIN & SAVE MODEL (GIỮ NGUYÊN)
# =====================================================
def train_and_save_model(data):
    X = data.drop(['MSRP'], axis=1)
    y = data['MSRP']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
    )
    
    te = TargetEncoder(cols=['Make', 'Model'])
    X_train_enc = te.fit_transform(X_train, y_train)
    X_train_num = X_train_enc.select_dtypes(include=[np.number])
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=100)
    model.fit(X_train_num, y_train)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(te, ENCODER_PATH)
    return model, te

if not os.path.exists(MODEL_PATH):
    with st.status("🚀 Lần đầu khởi chạy: Đang huấn luyện mô hình..."):
        model, encoder = train_and_save_model(data)
        st.success("Đã huấn luyện và lưu model.pkl thành công!")
else:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# =====================================================
# APP CONTENT
# =====================================================
st.title("🚗 Car Price Prediction and Analysis App.")

menu = st.sidebar.selectbox(
    "📌 Select function",
    ["Data Overview", "Analysis (EDA)", "Price prediction"]
)

if menu == "Data Overview":
    st.subheader("📊 Preview data")
    st.dataframe(data.head(10))

elif menu == "Analysis (EDA)":
    st.subheader("📈 Price trend analysis")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    data.groupby('Year')['MSRP'].mean().plot(kind='bar', ax=ax[0])
    sns.scatterplot(data=data, x='Engine HP', y='MSRP', ax=ax[1], alpha=0.5)
    st.pyplot(fig)

elif menu == "Price prediction":
    st.subheader("🤖 Predict car prices")
    st.info("Trạng thái: Đã tải mô hình từ file `model.pkl`")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        make = col1.selectbox("Hãng xe", sorted(data['Make'].unique()))
        model_name = col2.selectbox(
            "Dòng xe",
            sorted(data[data['Make'] == make]['Model'].unique())
        )
        hp = col1.number_input("Horsepower (HP)", value=int(data['Engine HP'].median()))
        year = col2.number_input("Year of manufacture", min_value=1990, max_value=2025, value=2015)
        
        if st.form_submit_button("🚀 Predict now"):
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

            st.success(f"💰 The predicted price of the car is: ${prediction[0]:,.2f}")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr>
<center style="color:gray">
© 2026 • Car Price Prediction App • Streamlit & Machine Learning
</center>
""", unsafe_allow_html=True)

