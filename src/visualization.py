"""
Module visualization.py: Phase Visualization - Đồ họa hóa & Báo cáo Trực quan.
Cung cấp các biểu đồ khoa học chất lượng cao (300 DPI) phục vụ báo cáo:
1. So sánh đa chỉ số Train / Val / Test (R2 & RMSE).
2. Tương quan Giá trị Thực tế vs Dự đoán (Actual vs Predicted).
3. Phân tích phần dư sai số (Residual Analysis & Normality).
4. Độ quan trọng của các đặc trưng (Feature Importance & OLS Weights).
"""
# pylint: disable=too-many-locals,invalid-name,wrong-import-position
import os
import sys
import json
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_loader import load_dataset
from src.models import LinearRegressionModel, DecisionTreeRegressorModel


def set_plotting_style() -> None:
    """Thiết lập cấu hình giao diện Matplotlib thẩm mỹ và chuyên nghiệp."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7
    })


def plot_model_comparison(
    summary_data: Dict[str, Any],
    output_dir: Optional[str] = None
) -> str:
    """
    Biểu đồ 1: So sánh đa chỉ số (R2 & RMSE) trên cả 3 tập (Train, Val, Test).
    Chứng minh năng lực của Decision Tree và kiểm soát Overfitting.
    """
    set_plotting_style()
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, "outputs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    _, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    splits = ["train", "validation", "test"]
    split_labels = ["Train (70%)", "Validation (15%)", "Test (15%)"]
    x = np.arange(len(splits))
    width = 0.35

    lr_key = "Linear_Regression" if "Linear_Regression" in summary_data else "Linear Regression"
    dt_key = "Decision_Tree_Regressor" if "Decision_Tree_Regressor" in summary_data else "Decision Tree Regressor"

    lr_r2 = [summary_data[lr_key][s]["R2"] for s in splits]
    dt_r2 = [summary_data[dt_key][s]["R2"] for s in splits]

    lr_rmse = [summary_data[lr_key][s]["RMSE"] for s in splits]
    dt_rmse = [summary_data[dt_key][s]["RMSE"] for s in splits]

    # Panel A: R2 Score (Càng cao càng tốt)
    rects1 = axes[0].bar(x - width / 2, lr_r2, width, label="Linear Regression (OLS)", color="#2b5c8f", alpha=0.9)
    rects2 = axes[0].bar(x + width / 2, dt_r2, width, label="Decision Tree (CART)", color="#e27d60", alpha=0.9)

    axes[0].set_ylabel("Hệ số xác định ($R^2$ Score)", fontsize=11, fontweight="bold")
    axes[0].set_title("A. So sánh Hệ số Xác định ($R^2$ Score) trên 3 tập", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(split_labels, fontsize=10)
    axes[0].set_ylim(0, 0.8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    axes[0].legend(loc="upper right", framealpha=0.9)

    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: RMSE Score (Càng thấp càng tốt)
    rects3 = axes[1].bar(x - width / 2, lr_rmse, width, label="Linear Regression (OLS)", color="#2b5c8f", alpha=0.9)
    rects4 = axes[1].bar(x + width / 2, dt_rmse, width, label="Decision Tree (CART)", color="#e27d60", alpha=0.9)

    axes[1].set_ylabel("Sai số RMSE (USD)", fontsize=11, fontweight="bold")
    axes[1].set_title("B. So sánh Sai số Căn bậc hai Trung bình (RMSE)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(split_labels, fontsize=10)
    axes[1].set_ylim(0, 1500)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    axes[1].legend(loc="upper right", framealpha=0.9)

    for rect in rects3:
        h = rect.get_height()
        axes[1].annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects4:
        h = rect.get_height()
        axes[1].annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_actual_vs_predicted(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[str] = None
) -> str:
    """
    Biểu đồ 2: Đồ thị Giá trị Thực tế vs Dự đoán (Actual vs. Predicted).
    Hiển thị độ hội tụ so với đường chéo hoàn hảo y = x.
    """
    set_plotting_style()
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, "outputs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    _, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    y_t = y_test.to_numpy()
    max_val = max(np.max(y_t), 10000)
    colors = ["#2b5c8f", "#e27d60"]

    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        residuals = y_t - y_pred
        r2 = 1.0 - (np.sum(residuals ** 2) / np.sum((y_t - np.mean(y_t)) ** 2))
        rmse = np.sqrt(np.mean(residuals ** 2))

        ax.scatter(y_t, y_pred, alpha=0.35, color=colors[idx % len(colors)], edgecolors="none", s=25)
        ax.plot([0, max_val], [0, max_val], color="#d9534f", linestyle="--", linewidth=2.0, label="Đường kỳ vọng lý tưởng ($y = \\hat{y}$)")

        ax.set_title(f"{name} trên tập Test", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Doanh số thực tế ($y$ True - USD)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Doanh số dự đoán ($\\hat{y}$ Pred - USD)", fontsize=11, fontweight="bold")
        ax.set_xlim(0, max_val * 0.95)
        ax.set_ylim(0, max_val * 0.95)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Chú thích thông số đánh giá
        text_box = f"$R^2$: {r2:.4f}\nRMSE: {rmse:.2f} USD\nSamples: {len(y_t):,}"
        ax.text(0.05, 0.85, text_box, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9})
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_residual_analysis(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[str] = None
) -> str:
    """
    Biểu đồ 3: Phân tích phần dư sai số (Residual Plots & Phân phối phần dư).
    Kiểm tra giả định sai số quanh 0 và hiện tượng phân tán dị phương sai.
    """
    set_plotting_style()
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, "outputs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    _, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    y_t = y_test.to_numpy()
    colors = ["#2b5c8f", "#e27d60"]

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        residuals = y_t - y_pred

        # Hàng 1: Residuals vs Predicted
        ax_res = axes[0, idx]
        ax_res.scatter(y_pred, residuals, alpha=0.35, color=colors[idx % len(colors)], s=20)
        ax_res.axhline(y=0, color="#d9534f", linestyle="--", linewidth=1.5)
        ax_res.set_title(f"A{idx + 1}. Phần dư vs Dự đoán ({name})", fontsize=11, fontweight="bold")
        ax_res.set_xlabel("Doanh số dự đoán ($\\hat{y}$ - USD)", fontsize=10)
        ax_res.set_ylabel("Phần dư sai số ($y - \\hat{y}$)", fontsize=10)
        ax_res.grid(True, linestyle="--", alpha=0.7)

        # Hàng 2: Phân phối phần dư (Histogram & Mean)
        ax_dist = axes[1, idx]
        ax_dist.hist(residuals, bins=40, density=True, color=colors[idx % len(colors)], alpha=0.75, edgecolor="white")
        ax_dist.axvline(x=0, color="#d9534f", linestyle="--", linewidth=1.5, label="Sai số bằng 0")
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))
        ax_dist.axvline(x=mean_res, color="black", linestyle=":", linewidth=1.5, label=f"Trung bình: {mean_res:.1f}")

        ax_dist.set_title(f"B{idx + 1}. Phân phối phần dư ({name})", fontsize=11, fontweight="bold")
        ax_dist.set_xlabel("Độ lệch sai số (USD)", fontsize=10)
        ax_dist.set_ylabel("Mật độ tần suất (Density)", fontsize=10)
        ax_dist.legend(loc="upper right", fontsize=9)
        ax_dist.text(0.05, 0.85, f"Mean: {mean_res:.1f}\nStd: {std_res:.1f}", transform=ax_dist.transAxes,
                    fontsize=9, verticalalignment="top", bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cccccc"})
        ax_dist.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "residual_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_feature_importance(
    dt_model: DecisionTreeRegressorModel,
    lr_model: LinearRegressionModel,
    output_dir: Optional[str] = None
) -> str:
    """
    Biểu đồ 4: Độ quan trọng đặc trưng (Feature Importance & Trọng số hồi quy OLS).
    Giải thích nhân tố kinh doanh thúc đẩy doanh số bán lẻ BigMart.
    """
    set_plotting_style()
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, "outputs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    _, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. Decision Tree Feature Importance (Top 10)
    dt_importances = dt_model.get_feature_importances()
    top_dt = sorted(dt_importances.items(), key=lambda item: item[1], reverse=True)[:10]
    dt_names = [item[0] for item in top_dt][::-1]
    dt_vals = [item[1] for item in top_dt][::-1]

    axes[0].barh(dt_names, dt_vals, color="#e27d60", alpha=0.9, edgecolor="none", height=0.6)
    axes[0].set_title("A. Top 10 Đặc trưng quan trọng nhất (Decision Tree)", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Tỷ lệ đóng góp giảm phương sai (%)", fontsize=10, fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.7)

    for i, v in enumerate(dt_vals):
        axes[0].text(v + 0.5, i, f"{v:.2f}%", va="center", fontsize=9, fontweight="bold")
    axes[0].set_xlim(0, max(dt_vals) * 1.15)

    # 2. Linear Regression Coefficients (Top 5 Dương & Top 5 Âm)
    lr_params = lr_model.get_model_parameters()["weights"]
    sorted_weights = sorted(lr_params.items(), key=lambda item: item[1], reverse=True)
    top_pos = sorted_weights[:5]
    top_neg = sorted_weights[-5:]
    selected_weights = top_pos + top_neg
    selected_weights = sorted(selected_weights, key=lambda item: item[1])

    w_names = [item[0] for item in selected_weights]
    w_vals = [item[1] for item in selected_weights]
    bar_colors = ["#d9534f" if v < 0 else "#2b5c8f" for v in w_vals]

    axes[1].barh(w_names, w_vals, color=bar_colors, alpha=0.85, height=0.6)
    axes[1].axvline(0, color="black", linestyle="-", linewidth=0.8)
    axes[1].set_title("B. Trọng số OLS tác động mạnh nhất (Linear Regression)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("Hệ số hồi quy ($w_j$ - USD tác động)", fontsize=10, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.7)

    for i, v in enumerate(w_vals):
        offset = 15 if v >= 0 else -15
        ha = "left" if v >= 0 else "right"
        axes[1].text(v + offset, i, f"{v:+.1f}", va="center", ha=ha, fontsize=8.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def run_all_visualizations() -> Dict[str, str]:
    """
    Thực thi tạo và xuất toàn bộ 4 bộ đồ thị chất lượng cao (300 DPI) ra thư mục outputs/figures/.
    """
    print("\n" + "=" * 65)
    print("PHASE: VISUALIZATION PIPELINE (BigMart Model Insights)")
    print("=" * 65)

    figures_dir = os.path.join(REPO_ROOT, "outputs", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Nạp tóm tắt đánh giá
    eval_file = os.path.join(REPO_ROOT, "outputs", "models_evaluation_summary.json")
    if not os.path.exists(eval_file):
        from src.train import run_training  # pylint: disable=import-outside-toplevel
        print("[*] Chưa tìm thấy tóm tắt đánh giá. Đang tự động chạy huấn luyện...")
        run_training()

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 2. Nạp dữ liệu Test
    splits = load_dataset()
    (_, _), (_, _), (X_test, y_test) = splits

    # 3. Nạp mô hình đã huấn luyện
    lr_path = os.path.join(REPO_ROOT, "outputs", "linear_regression_trained_model.json")
    dt_path = os.path.join(REPO_ROOT, "outputs", "decision_tree_trained_model.json")

    lr_model = LinearRegressionModel.load_from_json(lr_path)
    dt_model = DecisionTreeRegressorModel.load_from_json(dt_path)

    models_dict = {
        "Linear Regression": lr_model,
        "Decision Tree Regressor": dt_model
    }

    generated_figures: Dict[str, str] = {}

    print("[1/4] Đang vẽ biểu đồ so sánh mô hình (model_comparison.png)...")
    p1 = plot_model_comparison(eval_data, figures_dir)
    generated_figures["model_comparison"] = p1
    print(f"      -> Đã xuất: {p1}")

    print("[2/4] Đang vẽ đồ thị Thực tế vs Dự đoán (actual_vs_predicted.png)...")
    p2 = plot_actual_vs_predicted(models_dict, X_test, y_test, figures_dir)
    generated_figures["actual_vs_predicted"] = p2
    print(f"      -> Đã xuất: {p2}")

    print("[3/4] Đang vẽ đồ thị phân tích phần dư sai số (residual_analysis.png)...")
    p3 = plot_residual_analysis(models_dict, X_test, y_test, figures_dir)
    generated_figures["residual_analysis"] = p3
    print(f"      -> Đã xuất: {p3}")

    print("[4/4] Đang vẽ biểu đồ tầm quan trọng đặc trưng (feature_importance.png)...")
    p4 = plot_feature_importance(dt_model, lr_model, figures_dir)
    generated_figures["feature_importance"] = p4
    print(f"      -> Đã xuất: {p4}")

    print("\n" + "=" * 65)
    print("HOÀN THÀNH XUẤT TOÀN BỘ 4 BIỂU ĐỒ TRỰC QUAN HÓA THÀNH CÔNG")
    print("=" * 65)
    return generated_figures


def main() -> None:
    """Hàm chạy chính từ dòng lệnh."""
    run_all_visualizations()


if __name__ == "__main__":
    main()
