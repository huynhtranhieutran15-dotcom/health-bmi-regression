"""Dashboard web: khám phá dữ liệu, so sánh mô hình hồi quy và dự đoán cholesterol.

Chạy:  streamlit run main.py
"""

import pandas as pd
import streamlit as st

from src.charts import actual_vs_pred, corr_heatmap, plane_3d, simple_fit_plot
from src.model import (
    CASES,
    bmi_category,
    calc_bmi,
    correlation,
    equation,
    load_data,
    predict,
    train_all,
)

st.set_page_config(page_title="Hồi quy Cholesterol", page_icon="🩺", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_models() -> dict:
    return train_all(get_data())


df = get_data()
results = get_models()
corr = correlation(df)

# ----- Tiêu đề -----
st.title("🩺 Hồi quy tuyến tính: Cholesterol theo Tuổi & BMI")
st.caption(
    f"Dữ liệu: {len(df)} người · Cholesterol giả định đơn vị mmol/L · "
    "Mô hình huấn luyện trên toàn bộ dữ liệu, đánh giá bằng 5-fold CV"
)

tab_data, tab_model, tab_predict = st.tabs(
    ["📊 Khám phá dữ liệu", "📈 So sánh mô hình", "🧮 Tính BMI & Dự đoán"]
)

# ===================== TAB 1: KHÁM PHÁ DỮ LIỆU =====================
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("r (tuổi, cholesterol)", f"{corr.loc['age', 'chol']:.2f}", "tương quan rất mạnh")
    c2.metric("r (BMI, cholesterol)", f"{corr.loc['bmi', 'chol']:.2f}", "tương quan vừa")
    c3.metric("r (tuổi, BMI)", f"{corr.loc['age', 'bmi']:.2f}", "hai biến cũng liên quan nhau")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Bảng dữ liệu")
        st.dataframe(df, height=420)
    with right:
        st.subheader("Ma trận tương quan")
        st.pyplot(corr_heatmap(corr))

    st.info(
        "💡 **Nhận xét:** Tuổi gần như quyết định cholesterol trong bộ dữ liệu này. "
        "BMI có liên quan nhưng phần lớn trùng với ảnh hưởng của tuổi, vì tuổi và BMI "
        "cũng tương quan với nhau (r ≈ 0.69)."
    )

# ===================== TAB 2: SO SÁNH MÔ HÌNH =====================
with tab_model:
    summary = pd.DataFrame(
        [
            {
                "Mô hình": name,
                "Phương trình": equation(r),
                "R² (train)": r["r2"],
                "R² (CV 5-fold)": r["cv_r2"],
                "MAE": r["mae"],
                "RMSE": r["rmse"],
            }
            for name, r in results.items()
        ]
    ).round(3)
    st.subheader("Bảng so sánh 3 trường hợp")
    st.dataframe(summary, hide_index=True)

    best = max(results, key=lambda k: results[k]["cv_r2"])
    st.success(
        f"🏆 Mô hình có khả năng dự đoán tốt nhất trên dữ liệu mới (CV R² cao nhất): **{best}**"
    )

    choice = st.selectbox("Chọn mô hình để xem biểu đồ", list(CASES))
    chosen = results[choice]
    features = chosen["features"]

    g1, g2 = st.columns(2)
    with g1:
        if len(features) == 1:
            st.pyplot(simple_fit_plot(df, features[0], chosen))
        else:
            st.pyplot(plane_3d(df, chosen))
    with g2:
        st.pyplot(actual_vs_pred(df, chosen))

    st.warning(
        "⚠️ Chỉ có 18 mẫu nên kết quả chỉ mang tính minh họa. "
        "Hệ số R² và sai số có thể thay đổi đáng kể nếu có thêm dữ liệu."
    )

# ===================== TAB 3: BMI & DỰ ĐOÁN =====================
with tab_predict:
    st.subheader("Nhập thông tin của bạn")
    i1, i2, i3 = st.columns(3)
    age = i1.number_input("Tuổi", min_value=1, max_value=120, value=40, step=1)
    height = i2.number_input("Chiều cao (cm)", min_value=100.0, max_value=250.0, value=165.0, step=0.5)
    weight = i3.number_input("Cân nặng (kg)", min_value=20.0, max_value=250.0, value=60.0, step=0.5)

    bmi = calc_bmi(height, weight)
    category, risk = bmi_category(bmi)

    b1, b2 = st.columns(2)
    b1.metric("Chỉ số BMI", f"{bmi:.1f} kg/m²", category)
    b2.metric("Mức rủi ro sức khỏe (theo bảng BMI)", risk)
    # Thanh BMI: thang 15 đến 40 để dễ nhìn, vượt 40 thì thanh đầy
    st.progress(min(max((bmi - 15) / 25, 0.0), 1.0), text="Vị trí BMI trên thang 15 – 40")

    st.markdown("#### Dự đoán cholesterol")
    p_cols = st.columns(3)
    for col, (name, r) in zip(p_cols, results.items()):
        col.metric(name, f"{predict(r, age, bmi):.2f} mmol/L")

    best_pred = predict(results["Trường hợp 3: chol ~ tuổi + BMI"], age, bmi)
    st.success(f"Dự đoán đề xuất (trường hợp 3, dùng cả tuổi và BMI): **{best_pred:.2f} mmol/L**")

    # Cảnh báo khi đầu vào nằm ngoài khoảng dữ liệu huấn luyện (ngoại suy kém tin cậy)
    age_lo, age_hi = df["age"].min(), df["age"].max()
    bmi_lo, bmi_hi = df["bmi"].min(), df["bmi"].max()
    if not (age_lo <= age <= age_hi and bmi_lo <= bmi <= bmi_hi):
        st.warning(
            f"Đầu vào nằm ngoài khoảng dữ liệu huấn luyện (tuổi {age_lo:.0f}–{age_hi:.0f}, "
            f"BMI {bmi_lo:.1f}–{bmi_hi:.1f}). Kết quả dự đoán kém tin cậy."
        )

    st.caption(
        "Đây là kết quả thống kê từ dữ liệu mẫu nhỏ, không phải chẩn đoán y khoa. "
        "Hãy đi khám bác sĩ để có kết luận chính xác."
    )
