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

# --- Cấu hình trang ---
st.set_page_config(page_title="Dự đoán giá xe ô tô", layout="wide")

MODEL_PATH = 'model.pkl'
ENCODER_PATH = 'encoder.pkl'

# --- Hàm Load và Preprocess Data ---
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

# --- Logic Huấn luyện và Lưu trữ ---
def train_and_save_model(data):
    X = data.drop(['MSRP'], axis=1)
    y = data['MSRP']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    
    # Huấn luyện Encoder
    te = TargetEncoder(cols=['Make', 'Model'])
    X_train_enc = te.fit_transform(X_train, y_train)
    
    # Lọc cột số
    X_train_num = X_train_enc.select_dtypes(include=[np.number])
    
    # Huấn luyện Model
    model = GradientBoostingRegressor(n_estimators=100, random_state=100)
    model.fit(X_train_num, y_train)
    
    # Lưu file
    joblib.dump(model, MODEL_PATH)
    joblib.dump(te, ENCODER_PATH)
    return model, te

# Kiểm tra xem đã có model chưa, nếu chưa thì tự train ngay khi mở app
if not os.path.exists(MODEL_PATH):
    with st.status("🚀 Lần đầu khởi chạy: Đang huấn luyện mô hình..."):
        model, encoder = train_and_save_model(data)
        st.success("Đã huấn luyện và lưu model.pkl thành công!")
else:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# --- Giao diện Streamlit ---
st.title("🚗 Ứng dụng Dự đoán và Phân tích Giá Xe")

menu = st.sidebar.selectbox("Chọn chức năng", ["Tổng quan dữ liệu", "Phân tích (EDA)", "Dự đoán giá"])

if menu == "Tổng quan dữ liệu":
    st.subheader("📊 Xem trước dữ liệu")
    st.dataframe(data.head(10))

elif menu == "Phân tích (EDA)":
    st.subheader("📈 Phân tích xu hướng giá")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    data.groupby('Year')['MSRP'].mean().plot(kind='bar', ax=ax[0], color='skyblue')
    sns.scatterplot(data=data, x='Engine HP', y='MSRP', ax=ax[1], alpha=0.5)
    st.pyplot(fig)

elif menu == "Dự đoán giá":
    st.subheader("🤖 Dự đoán giá xe")
    st.info("Trạng thái: Đã tải mô hình từ file `model.pkl`")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        # 1. Người dùng nhập các thông số chính
        make = col1.selectbox("Hãng xe", sorted(data['Make'].unique()))
        model_name = col2.selectbox("Dòng xe", sorted(data[data['Make'] == make]['Model'].unique()))
        hp = col1.number_input("Mã lực (HP)", value=int(data['Engine HP'].median()))
        year = col2.number_input("Năm sản xuất", min_value=1990, max_value=2025, value=2015)
        
        if st.form_submit_button("Dự đoán ngay"):
            # --- CÁCH FIX LỖI DIMENSION ---
            # Tạo 1 dòng dữ liệu trống có đầy đủ tất cả các cột như lúc Train
            input_df = data.drop(['MSRP'], axis=1).iloc[:1].copy() 
            
            # Điền các giá trị trung bình/phổ biến vào tất cả các cột để tránh lỗi thiếu cột
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    input_df[col] = data[col].mode()[0]
                else:
                    input_df[col] = data[col].median()

            # Ghi đè các giá trị mà người dùng đã chọn vào dòng mẫu này
            input_df['Make'] = make
            input_df['Model'] = model_name
            input_df['Engine HP'] = hp
            input_df['Year'] = year
            input_df['Years Of Manufacture'] = 2025 - year
            
            # Thực hiện Encode và Predict trên dòng có đủ số cột (15 cột)
            input_enc = encoder.transform(input_df)
            input_num = input_enc.select_dtypes(include=[np.number])
            
            prediction = model.predict(input_num)
            st.success(f"Giá dự đoán của xe là: ${prediction[0]:,.2f}")