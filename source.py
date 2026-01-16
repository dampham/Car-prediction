# ================================
# IMPORT THƯ VIỆN
# ================================

# Streamlit: dùng để xây dựng web application cho Machine Learning
import streamlit as st

# Pandas & Numpy: xử lý dữ liệu dạng bảng và số học
import pandas as pd
import numpy as np

# Thư viện vẽ biểu đồ
import matplotlib.pyplot as plt
import seaborn as sns

# Joblib: lưu và load mô hình Machine Learning
import joblib

# OS: làm việc với file và thư mục
import os

# Chia tập dữ liệu train/test
from sklearn.model_selection import train_test_split

# Các mô hình Machine Learning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Encoder cho dữ liệu dạng category (Make, Model)
from category_encoders import TargetEncoder

# ================================
# CẤU HÌNH GIAO DIỆN WEB
# ================================

# Thiết lập tiêu đề và layout toàn màn hình
st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide"
)

# Đường dẫn lưu model và encoder
MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"

# ================================
# LOAD & TIỀN XỬ LÝ DỮ LIỆU
# ================================

@st.cache_data
def load_data():
    """
    Hàm load dữ liệu từ file CSV và thực hiện tiền xử lý.
    @st.cache_data giúp Streamlit không load lại dữ liệu nhiều lần.
    """

    # Kiểm tra file dữ liệu có tồn tại hay không
    if not os.path.exists("data.csv"):
        st.error("Không tìm thấy file data.csv")
        return pd.DataFrame()

    # Đọc dữ liệu
    data = pd.read_csv("data.csv")

    # Loại bỏ các giá trị MPG bất thường (outliers)
    data = data[data["highway MPG"] < 60]
    data = data[data["city mpg"] < 40]

    # Chuyển cột MSRP từ dạng chuỗi ($, ,) sang số
    data["MSRP"] = pd.to_numeric(
        data["MSRP"].replace("[$,]", "", regex=True),
        errors="coerce"
    )

    # Chuyển Engine HP sang dạng số
    data["Engine HP"] = pd.to_numeric(data["Engine HP"], errors="coerce")

    # Loại bỏ các dòng bị thiếu giá hoặc mã lực
    data = data.dropna(subset=["Engine HP", "MSRP"])

    # Điền giá trị thiếu
    data["Number of Doors"].fillna(data["Number of Doors"].median(), inplace=True)
    data["Engine Fuel Type"].fillna(data["Engine Fuel Type"].mode()[0], inplace=True)
    data["Engine Cylinders"].fillna(4, inplace=True)

    # Loại bỏ cột Market Category nếu tồn tại
    if "Market Category" in data.columns:
        data.drop(columns=["Market Category"], inplace=True)

    # Tạo feature mới: số năm đã sử dụng của xe
    data["Years Of Manufacture"] = 2025 - data["Year"]

    return data


# Load dữ liệu khi khởi động app
data = load_data()

# ================================
# HUẤN LUYỆN VÀ LƯU MÔ HÌNH
# ================================

def train_and_save_model(data):
    """
    Hàm huấn luyện mô hình Gradient Boosting và lưu model + encoder.
    """

    # Tách đặc trưng (X) và biến mục tiêu (y)
    X = data.drop("MSRP", axis=1)
    y = data["MSRP"]

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    # Target Encoding cho các biến phân loại
    encoder = TargetEncoder(cols=["Make", "Model"])
    X_train_enc = encoder.fit_transform(X_train, y_train)

    # Chỉ giữ lại các cột số (model ML chỉ học số)
    X_train_num = X_train_enc.select_dtypes(include=[np.number])

    # Khởi tạo mô hình Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=100,
        random_state=100
    )

    # Huấn luyện mô hình
    model.fit(X_train_num, y_train)

    # Lưu model và encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    return model, encoder


# Nếu chưa có model thì train
if not os.path.exists(MODEL_PATH):
    with st.status("Training model for the first time..."):
        model, encoder = train_and_save_model(data)
        st.success("Model trained and saved successfully!")
else:
    # Nếu đã có model thì load
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# ================================
# GIAO DIỆN CHÍNH
# ================================

st.title("🚗 Car Price Prediction System")

# Thanh menu bên trái
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Dataset Overview", "EDA", "Prediction"]
)

# ================================
# HOME
# ================================

if menu == "Home":
    st.markdown("""
    ### Welcome to Car Price Prediction System
    This system applies machine learning techniques to predict car prices
    based on technical specifications and historical data.
    """)

# ================================
# DATASET OVERVIEW
# ================================

elif menu == "Dataset Overview":
    st.subheader("Dataset Preview")

    # Cho phép người dùng chọn cột để hiển thị
    selected_cols = st.multiselect(
        "Select columns to display",
        data.columns.tolist(),
        default=data.columns.tolist()
    )

    st.dataframe(data[selected_cols])

# ================================
# EDA
# ================================

elif menu == "EDA":
    st.subheader("Exploratory Data Analysis")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Giá trung bình theo năm
    data.groupby("Year")["MSRP"].mean().plot(
        kind="bar", ax=ax[0], title="Average Price by Year"
    )

    # Quan hệ HP và giá
    sns.scatterplot(
        data=data,
        x="Engine HP",
        y="MSRP",
        ax=ax[1],
        alpha=0.5
    )

    st.pyplot(fig)

# ================================
# PREDICTION
# ================================

elif menu == "Prediction":
    st.subheader("Car Price Prediction")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        # Người dùng chọn hãng xe
        make = col1.selectbox("Car Make", sorted(data["Make"].unique()))

        # Model phụ thuộc vào hãng
        model_name = col2.selectbox(
            "Car Model",
            sorted(data[data["Make"] == make]["Model"].unique())
        )

        # Nhập mã lực
        hp = col1.number_input(
            "Engine Horsepower",
            value=int(data["Engine HP"].median())
        )

        # Nhập năm sản xuất
        year = col2.number_input(
            "Year of Manufacture",
            min_value=1990,
            max_value=2025,
            value=2015
        )

        submit = st.form_submit_button("Predict Price")

    if submit:
        # Tạo một dòng dữ liệu mẫu có đủ cột
        input_df = data.drop("MSRP", axis=1).iloc[:1].copy()

        # Điền giá trị mặc định để tránh lỗi thiếu cột
        for col in input_df.columns:
            if input_df[col].dtype == "object":
                input_df[col] = data[col].mode()[0]
            else:
                input_df[col] = data[col].median()

        # Ghi đè giá trị người dùng nhập
        input_df["Make"] = make
        input_df["Model"] = model_name
        input_df["Engine HP"] = hp
        input_df["Year"] = year
        input_df["Years Of Manufacture"] = 2025 - year

        # Encode và predict
        input_enc = encoder.transform(input_df)
        input_num = input_enc.select_dtypes(include=[np.number])

        prediction = model.predict(input_num)[0]

        st.success(f"Estimated Car Price: ${prediction:,.2f}")
