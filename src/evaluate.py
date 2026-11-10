"""
Module evaluate.py: Phase 3 - Evaluation & Metrics Calculation.
Cung cấp các hàm tính toán chỉ số (MSE, RMSE, MAE, MAPE, R2)
và đánh giá độc lập mô hình trên các tập dữ liệu, đặc biệt là tập Test.
"""
# pylint: disable=invalid-name,wrong-import-position
import os
import sys
import json
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_loader import load_dataset
from src.models import LinearRegressionModel, DecisionTreeRegressorModel


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Tính toán toàn diện 5 chỉ số hồi quy chuẩn mực:
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE(%): Mean Absolute Percentage Error (tránh chia 0 qua epsilon 1e-8)
    - R2: Hệ số xác định Coefficient of Determination
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_t) != len(y_p):
        raise ValueError("Kích thước y_true và y_pred không khớp.")

    residuals = y_t - y_p
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    epsilon = 1e-8
    mape = float(np.mean(np.abs(residuals / np.maximum(np.abs(y_t), epsilon)))) * 100.0

    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE(%)": round(mape, 4),
        "R2": round(r2, 4)
    }


def evaluate_model(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """Đánh giá một mô hình trên một tập dữ liệu cụ thể (X, y)."""
    y_pred = model.predict(X)
    return calculate_metrics(y, y_pred)


def evaluate_model_splits(
    model: Any,
    splits: Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]
) -> Dict[str, Dict[str, float]]:
    """Đánh giá mô hình đồng thời trên cả 3 tập: train, validation, test."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
    return {
        "train": evaluate_model(model, X_train, y_train),
        "validation": evaluate_model(model, X_val, y_val),
        "test": evaluate_model(model, X_test, y_test)
    }


def compare_models(
    models_dict: Dict[str, Any],
    splits: Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """So sánh nhiều mô hình cùng lúc trên cả 3 tập dữ liệu."""
    results = {}
    for name, model in models_dict.items():
        results[name] = evaluate_model_splits(model, splits)
    return results


def save_evaluation_json(
    evaluation_data: Dict[str, Any],
    filepath: Optional[str] = None
) -> str:
    """Xuất kết quả đánh giá ra file JSON chuẩn."""
    if filepath is None:
        filepath = os.path.join(REPO_ROOT, "outputs", "models_evaluation_summary.json")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=4, ensure_ascii=False)
    return filepath

def run_evaluation() -> Dict[str, Any]:
    """
    Thực hiện quy trình đánh giá độc lập mô hình trên tập Test và đối sánh Train/Val/Test.
    """
    splits = load_dataset()
    (_, _), (_, _), (X_test, y_test) = splits

    lr_path = os.path.join(REPO_ROOT, "outputs", "linear_regression_trained_model.json")
    dt_path = os.path.join(REPO_ROOT, "outputs", "decision_tree_trained_model.json")

    models: Dict[str, Any] = {}
    if os.path.exists(lr_path):
        models["Linear Regression"] = LinearRegressionModel.load_from_json(lr_path)
    if os.path.exists(dt_path):
        models["Decision Tree Regressor"] = DecisionTreeRegressorModel.load_from_json(dt_path)

    if not models:
        from src.train import run_training  # pylint: disable=import-outside-toplevel
        print("[*] Chưa có mô hình lưu trữ. Đang tự động chạy huấn luyện...")
        run_training()
        if os.path.exists(lr_path):
            models["Linear Regression"] = LinearRegressionModel.load_from_json(lr_path)
        if os.path.exists(dt_path):
            models["Decision Tree Regressor"] = DecisionTreeRegressorModel.load_from_json(dt_path)

    print("\n" + "=" * 65)
    print("PHASE 3: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (FINAL EVALUATION)")
    print("=" * 65)
    print(f"Tổng số mẫu kiểm thử độc lập (Test Samples): {len(X_test):,}")

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        print(f"\n--- [KẾT QUẢ TẬP TEST] {name} ---")
        for metric_name, val in metrics.items():
            print(f"  + {metric_name:8s}: {val}")

    # Đánh giá toàn diện trên cả 3 tập để kiểm tra Overfitting
    comparison_results = compare_models(models, splits)
    save_evaluation_json(comparison_results)
    return comparison_results


def main() -> None:
    """Hàm thực thi chính khi gọi trực tiếp evaluate.py."""
    run_evaluation()


if __name__ == "__main__":
    main()
