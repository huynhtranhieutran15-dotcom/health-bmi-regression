"""Đọc dữ liệu, tính BMI và huấn luyện 3 mô hình hồi quy tuyến tính cho cholesterol."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "health.csv"

# Ba trường hợp theo đề bài: tên hiển thị -> danh sách biến đầu vào
CASES = {
    "Trường hợp 1: chol ~ tuổi": ["age"],
    "Trường hợp 2: chol ~ BMI": ["bmi"],
    "Trường hợp 3: chol ~ tuổi + BMI": ["age", "bmi"],
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Đọc health.csv, kiểm tra đủ cột và bỏ dòng thiếu dữ liệu."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    missing = {"age", "bmi", "chol"} - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột trong file dữ liệu: {missing}")

    return df.dropna(subset=["age", "bmi", "chol"]).reset_index(drop=True)


def calc_bmi(height_cm: float, weight_kg: float) -> float:
    """BMI = cân nặng (kg) / chiều cao (m)²."""
    height_m = height_cm / 100
    return weight_kg / height_m**2


def bmi_category(bmi: float) -> tuple[str, str]:
    """Phân loại BMI theo bảng chuẩn đề bài. Trả về (nhóm, mức rủi ro)."""
    if bmi < 18.5:
        return "Thiếu cân", "Thấp"
    if bmi < 25:
        return "Bình thường", "Trung bình"
    if bmi < 30:
        return "Thừa cân", "Tăng nhẹ"
    if bmi < 35:
        return "Béo phì độ I", "Vừa"
    if bmi < 40:
        return "Béo phì độ II", "Cao"
    return "Béo phì độ III", "Rất cao"


def correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Ma trận hệ số tương quan Pearson giữa tuổi, BMI và cholesterol."""
    return df[["age", "bmi", "chol"]].corr()


def train_model(df: pd.DataFrame, features: list[str]) -> dict:
    """Huấn luyện một mô hình hồi quy tuyến tính và tính các chỉ số đánh giá."""
    X = df[features].to_numpy()
    y = df["chol"].to_numpy()

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Cross-validation 5 fold: đo khả năng dự đoán trên dữ liệu chưa từng thấy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")

    return {
        "features": features,
        "model": model,
        "intercept": float(model.intercept_),
        "coef": dict(zip(features, model.coef_.astype(float))),
        "r2": r2_score(y, y_pred),
        "cv_r2": float(cv_r2.mean()),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "y_pred": y_pred,
    }


def train_all(df: pd.DataFrame) -> dict[str, dict]:
    """Huấn luyện đủ 3 trường hợp."""
    return {name: train_model(df, feats) for name, feats in CASES.items()}


def equation(result: dict) -> str:
    """Viết phương trình hồi quy dạng chuỗi, ví dụ: chol = 1.2 + 0.05×age ..."""
    text = f"chol = {result['intercept']:.3f}"
    for feature, coef in result["coef"].items():
        sign = "+" if coef >= 0 else "−"
        text += f" {sign} {abs(coef):.3f}×{feature}"
    return text


def predict(result: dict, age: float, bmi: float) -> float:
    """Dự đoán cholesterol từ tuổi và/hoặc BMI tùy theo mô hình."""
    inputs = {"age": age, "bmi": bmi}
    x = [[inputs[f] for f in result["features"]]]
    return float(result["model"].predict(x)[0])
