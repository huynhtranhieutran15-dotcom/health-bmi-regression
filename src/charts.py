"""Các hàm vẽ biểu đồ cho dashboard (matplotlib)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#4C72B0"
RED = "#C44E52"
LABELS = ["Tuổi", "BMI", "Cholesterol"]


def corr_heatmap(corr: pd.DataFrame):
    """Heatmap ma trận tương quan, số hiển thị trực tiếp trên ô."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3), LABELS, rotation=30)
    ax.set_yticks(range(3), LABELS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def simple_fit_plot(df: pd.DataFrame, feature: str, result: dict):
    """Scatter + đường hồi quy cho mô hình một biến (trường hợp 1 hoặc 2)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[feature], df["chol"], color=BLUE, alpha=0.85, label="Dữ liệu thật")

    xs = np.linspace(df[feature].min(), df[feature].max(), 100)
    ys = result["intercept"] + result["coef"][feature] * xs
    ax.plot(xs, ys, color=RED, lw=2.5, label="Đường hồi quy")

    ax.set_xlabel("Tuổi" if feature == "age" else "BMI")
    ax.set_ylabel("Cholesterol (mmol/L)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plane_3d(df: pd.DataFrame, result: dict):
    """Điểm dữ liệu và mặt phẳng hồi quy 3D cho mô hình hai biến (trường hợp 3)."""
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(df["age"], df["bmi"], df["chol"], color=BLUE, s=40, label="Dữ liệu thật")

    xx, yy = np.meshgrid(
        np.linspace(df["age"].min(), df["age"].max(), 15),
        np.linspace(df["bmi"].min(), df["bmi"].max(), 15),
    )
    zz = result["intercept"] + result["coef"]["age"] * xx + result["coef"]["bmi"] * yy
    ax.plot_surface(xx, yy, zz, color=RED, alpha=0.35)

    ax.set_xlabel("Tuổi")
    ax.set_ylabel("BMI")
    ax.set_zlabel("Cholesterol")
    fig.tight_layout()
    return fig


def actual_vs_pred(df: pd.DataFrame, result: dict):
    """Biểu đồ thực tế vs dự đoán: điểm càng sát đường chéo thì mô hình càng tốt."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["chol"], result["y_pred"], color=BLUE, alpha=0.85)

    low = min(df["chol"].min(), result["y_pred"].min()) - 0.2
    high = max(df["chol"].max(), result["y_pred"].max()) + 0.2
    ax.plot([low, high], [low, high], "--", color="gray", label="Dự đoán = thực tế")

    ax.set_xlabel("Cholesterol thực tế")
    ax.set_ylabel("Cholesterol dự đoán")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
