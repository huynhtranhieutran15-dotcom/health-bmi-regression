# 🩺 Hồi quy tuyến tính: Cholesterol theo Tuổi & BMI

Bài tập môn **Lập trình Python nâng cao** – Câu 3: Xây dựng mô hình hồi quy tuyến tính.
Project gồm mô hình **scikit-learn** và **dashboard web Streamlit** để dự đoán trực tiếp.

## ✨ Tính năng

- **Khám phá dữ liệu**: bảng dữ liệu, ma trận tương quan (heatmap), các chỉ số r.
- **So sánh 3 trường hợp hồi quy**:
  1. Cholesterol ~ Tuổi
  2. Cholesterol ~ BMI
  3. Cholesterol ~ Tuổi + BMI (mặt phẳng hồi quy 3D)
- **Đánh giá mô hình**: R², MAE, RMSE và 5-fold cross-validation để đo khả năng dự đoán trên dữ liệu mới.
- **Tính BMI**: nhập chiều cao (cm) và cân nặng (kg), nhận chỉ số BMI, nhóm cân nặng và mức rủi ro theo bảng chuẩn.
- **Dự đoán cholesterol**: nhập tuổi, chiều cao, cân nặng để xem dự đoán của cả 3 mô hình.

## 📁 Cấu trúc thư mục

```
health-bmi-regression/
├── main.py              # Entry point: dashboard Streamlit
├── src/
│   ├── __init__.py
│   ├── model.py         # Đọc dữ liệu, tính BMI, huấn luyện mô hình
│   └── charts.py        # Các hàm vẽ biểu đồ
├── data/
│   └── health.csv       # Dữ liệu: age, bmi, chol (18 người)
├── requirements.txt     # Thư viện cần cài
├── .gitignore
└── README.md
```

## 🚀 Cài đặt và chạy

Yêu cầu: **Python 3.10+**

```bash
# 1. Clone repo
git clone https://github.com/<username>/health-bmi-regression.git
cd health-bmi-regression

# 2. Tạo và kích hoạt môi trường ảo (venv)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy dashboard
streamlit run main.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`.

## 📐 Phương pháp

- **Mô hình**: `LinearRegression` của scikit-learn, huấn luyện trên toàn bộ dữ liệu.
- **BMI** = cân nặng (kg) / chiều cao (m)².
- **Đánh giá**: R² trên tập huấn luyện, và R² trung bình của 5-fold cross-validation (xáo trộn, `random_state=42`) để đánh giá khả năng dự đoán trên dữ liệu chưa thấy.
- **Phân loại BMI**: theo bảng chuẩn trong đề bài (Thiếu cân, Bình thường, Thừa cân, Béo phì độ I–III).

## 📊 Kết quả chính

| Mô hình | Phương trình | R² (train) | R² (CV 5-fold) | MAE | RMSE |
|---|---|---|---|---|---|
| TH1: chol ~ tuổi | chol = 1.089 + 0.058 × age | 0.877 | **0.788** | 0.243 | 0.285 |
| TH2: chol ~ BMI | chol = −2.832 + 0.264 × bmi | 0.481 | −0.105 | 0.465 | 0.587 |
| TH3: chol ~ tuổi + BMI | chol = 0.455 + 0.054 × age + 0.033 × bmi | 0.881 | 0.783 | 0.247 | 0.281 |

**Nhận xét:**

- **Tuổi** là yếu tố quyết định nhất: r(tuổi, cholesterol) = 0.94.
- **BMI một mình** dự đoán kém (CV R² âm). Nó có tương quan vừa phải (r = 0.69) nhưng phần lớn trùng với ảnh hưởng của tuổi, vì r(tuổi, BMI) = 0.69.
- **Thêm BMI vào mô hình tuổi** hầu như không cải thiện: R² tăng từ 0.877 lên 0.881, còn CV R² giảm nhẹ. Với bộ dữ liệu này, mô hình đơn giản (TH1) là đủ.

## ⚠️ Lưu ý

- Dữ liệu chỉ có **18 mẫu**, nên kết quả mang tính minh họa, chưa đủ để kết luận y khoa.
- Đơn vị cholesterol được **giả định là mmol/L** (phạm vi 1.9–4.6 phù hợp với đơn vị này).
- Widget dự đoán chỉ đáng tin khi tuổi và BMI nằm trong khoảng dữ liệu huấn luyện (tuổi 20–63, BMI 19.8–26.7). Ngoài khoảng này dashboard sẽ cảnh báo.
- Đây không phải công cụ chẩn đoán y khoa.

## 📄 Giấy phép

Dự án phục vụ mục đích học tập.
