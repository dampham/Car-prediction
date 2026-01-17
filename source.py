# ================================
# IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ================================

# Streamlit: tạo web application cho Machine Learning
import streamlit as st

# Pandas: xử lý dữ liệu dạng bảng (DataFrame)
import pandas as pd

# Numpy: xử lý tính toán số học
import numpy as np

# Matplotlib & Seaborn: vẽ biểu đồ phân tích dữ liệu
import matplotlib.pyplot as plt
import seaborn as sns

# Joblib: lưu và load mô hình Machine Learning
import joblib

# OS: làm việc với file và thư mục trong hệ điều hành
import os

# Chia dữ liệu train/test
from sklearn.model_selection import train_test_split

# Các mô hình Machine Learning
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Target Encoder: mã hóa dữ liệu dạng category (Make, Model)
from category_encoders import TargetEncoder


# =====================================================
# PAGE CONFIG – CẤU HÌNH TRANG WEB
# =====================================================

# Thiết lập tiêu đề và layout toàn màn hình cho ứng dụng Streamlit
st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide"
)


# =====================================================
# PATHS – ĐƯỜNG DẪN LƯU MODEL
# =====================================================

# Thư mục chứa các model
MODEL_DIR = "models"

# Đường dẫn file encoder
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

# Đường dẫn cho từng mô hình Machine Learning
MODEL_PATHS = {
    "Gradient Boosting (Main)": os.path.join(MODEL_DIR, "gbr.pkl"),
    "Linear Regression (Baseline)": os.path.join(MODEL_DIR, "linear.pkl"),
    "Ridge Regression": os.path.join(MODEL_DIR, "ridge.pkl"),
    "Lasso Regression": os.path.join(MODEL_DIR, "lasso.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "rf.pkl"),
}

# Tạo thư mục models nếu chưa tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================
# LOAD DATA – ĐỌC & TIỀN XỬ LÝ DỮ LIỆU
# =====================================================

@st.cache_data
def load_data():
    """
    Hàm load dữ liệu từ file CSV và thực hiện tiền xử lý.
    @st.cache_data giúp Streamlit ghi nhớ dữ liệu, tránh load lại nhiều lần.
    """

    # Đọc dữ liệu từ file data.csv
    data = pd.read_csv("data.csv")

    # Loại bỏ các xe có MPG bất thường (outlier)
    data = data[data["highway MPG"] < 60]
    data = data[data["city mpg"] < 40]

    # Chuyển cột MSRP từ chuỗi ($, ,) sang dạng số
    data["MSRP"] = pd.to_numeric(
        data["MSRP"].replace("[$,]", "", regex=True),
        errors="coerce"
    )

    # Chuyển Engine HP sang dạng số
    data["Engine HP"] = pd.to_numeric(data["Engine HP"], errors="coerce")

    # Loại bỏ các dòng thiếu giá hoặc mã lực
    data = data.dropna(subset=["Engine HP", "MSRP"])

    # Điền giá trị thiếu cho các cột
    data["Number of Doors"].fillna(data["Number of Doors"].median(), inplace=True)
    data["Engine Fuel Type"].fillna(data["Engine Fuel Type"].mode()[0], inplace=True)
    data["Engine Cylinders"].fillna(4, inplace=True)

    # Loại bỏ cột Market Category nếu tồn tại
    if "Market Category" in data.columns:
        data.drop(columns=["Market Category"], inplace=True)

    # Tạo feature mới: số năm xe đã sử dụng
    data["Years"] = 2025 - data["Year"]

    return data


# Load dữ liệu khi ứng dụng khởi chạy
data = load_data()


# =====================================================
# TRAIN ALL MODELS – HUẤN LUYỆN TOÀN BỘ MÔ HÌNH
# =====================================================

def train_models(data):
    """
    Hàm huấn luyện tất cả các mô hình Machine Learning
    và lưu chúng vào thư mục models/
    """

    # Tách biến đầu vào (X) và biến mục tiêu (y)
    X = data.drop("MSRP", axis=1)
    y = data["MSRP"]

    # Chia tập train/test (chỉ dùng train)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    # Encode dữ liệu category (Make, Model)
    encoder = TargetEncoder(cols=["Make", "Model"])
    X_train_enc = encoder.fit_transform(X_train, y_train)

    # Chỉ giữ lại các cột số cho mô hình học
    X_train_num = X_train_enc.select_dtypes(include=[np.number])

    # Khởi tạo các mô hình Machine Learning
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

    # Huấn luyện từng mô hình và lưu lại
    for name, model in models.items():
        model.fit(X_train_num, y_train)
        joblib.dump(model, MODEL_PATHS[name])

    # Lưu encoder
    joblib.dump(encoder, ENCODER_PATH)


# =====================================================
# FIRST RUN TRAINING – HUẤN LUYỆN LẦN ĐẦU
# =====================================================

# Nếu chưa có encoder thì huấn luyện toàn bộ model
if not os.path.exists(ENCODER_PATH):
    with st.status("Training models for the first time..."):
        train_models(data)
        st.success("Models trained successfully")

# Load encoder đã lưu
encoder = joblib.load(ENCODER_PATH)


@st.cache_resource
def load_models():
    """
    Load toàn bộ model đã train để sử dụng khi dự đoán
    """
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}


# Load model vào bộ nhớ
models = load_models()


# =====================================================
# HERO IMAGE – ẢNH NỀN TRANG CHỦ
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
# NAVIGATION – MENU ĐIỀU HƯỚNG
# =====================================================

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset Overview", "EDA", "Prediction"]
)


# =====================================================
# HOME – TRANG CHỦ
# =====================================================

if menu == "Home":
    st.title("Car Price Prediction System")
    st.write(
        "Predict car prices using multiple machine learning models "
        "trained on historical vehicle data."
    )


# =====================================================
# DATASET OVERVIEW – XEM & FILTER DỮ LIỆU
# =====================================================

elif menu == "Dataset Overview":
    st.header("Dataset Overview")

    # Cho phép người dùng chọn cột để hiển thị
    with st.expander("Filter columns"):
        selected_cols = st.multiselect(
            "Select columns",
            data.columns.tolist(),
            default=data.columns.tolist()
        )

    st.dataframe(data[selected_cols])


# =====================================================
# EDA – PHÂN TÍCH KHÁM PHÁ DỮ LIỆU
# =====================================================

elif menu == "EDA":
    st.header("Exploratory Data Analysis")

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    # Biểu đồ giá trung bình theo năm
    data.groupby("Year")["MSRP"].mean().plot(kind="bar", ax=ax[0])

    # Biểu đồ quan hệ HP và giá
    sns.scatterplot(data=data, x="Engine HP", y="MSRP", ax=ax[1], alpha=0.4)

    st.pyplot(fig)


# =====================================================
# PREDICTION – DỰ ĐOÁN GIÁ XE
# =====================================================

elif menu == "Prediction":
    st.header("Car Price Prediction")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        # Chọn mô hình Machine Learning
        model_choice = col1.selectbox(
            "Select Model",
            list(models.keys())
        )

        # Chọn hãng xe
        make = col1.selectbox(
            "Car Make",
            sorted(data["Make"].unique())
        )

        # Chọn dòng xe (phụ thuộc hãng)
        model_name = col2.selectbox(
            "Car Model",
            sorted(data[data["Make"] == make]["Model"].unique())
        )

        # Nhập mã lực
        hp = col1.number_input(
            "Horsepower (HP)",
            value=int(data["Engine HP"].median())
        )

        # Nhập năm sản xuất
        year = col2.number_input(
            "Year",
            min_value=1990,
            max_value=2025,
            value=2015
        )

        submit = st.form_submit_button("Predict Price")

    if submit:
        # Tạo một dòng dữ liệu mẫu có đầy đủ cột như khi train
        input_df = data.drop("MSRP", axis=1).iloc[:1].copy()

        # Điền giá trị mặc định cho các cột còn thiếu
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
        input_df["Years"] = 2025 - year

        # Encode dữ liệu
        input_enc = encoder.transform(input_df)
        input_num = input_enc.select_dtypes(include=[np.number])

        # Dự đoán giá
        prediction = models[model_choice].predict(input_num)[0]

        st.success(
            f"Predicted price using **{model_choice}**: "
            f"${prediction:,.2f}"
        )


# =====================================================
# FOOTER – CHÂN TRANG
# =====================================================

st.markdown(
    "<hr><center style='color:gray'>© 2026 Car Price Prediction</center>",
    unsafe_allow_html=True
)

