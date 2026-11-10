import os
import sys
import json
from typing import Dict, Any, Tuple
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_loader import load_dataset
from src.models import LinearRegressionModel, DecisionTreeRegressorModel
from src.evaluate import evaluate_model_splits, save_evaluation_json


def train_linear_regression(
    splits: Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]
) -> Tuple[LinearRegressionModel, Dict[str, Any]]:
    (X_train, y_train), _, _ = splits

    hyperparameters = {
        "fit_intercept": True,
        "rcond": 1e-10
    }

    model = LinearRegressionModel()
    model.build_model(**hyperparameters)
    model.fit(X_train, y_train)

    evaluation = evaluate_model_splits(model, splits)
    learned_params = model.get_model_parameters()
    training_details = model.get_training_details()

    output_data = {
        "model_name": "Linear Regression (OLS via SVD)",
        "hyperparameters": hyperparameters,
        "learned_parameters": learned_params,
        "training_details": training_details,
        "evaluation_metrics": evaluation
    }

    return model, output_data


def train_decision_tree(
    splits: Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]
) -> Tuple[DecisionTreeRegressorModel, Dict[str, Any]]:
    (X_train, y_train), _, _ = splits

    hyperparameters = {
        "max_depth": 5,
        "min_samples_leaf": 20,
        "min_samples_split": 40
    }

    model = DecisionTreeRegressorModel()
    model.build_model(**hyperparameters)
    model.fit(X_train, y_train)

    evaluation = evaluate_model_splits(model, splits)
    nodes_list = model.get_nodes_list()
    total_nodes = len(nodes_list)
    leaf_nodes = sum(1 for n in nodes_list if n["node_type"] == "Leaf")

    tree_structure = {
        "total_nodes": total_nodes,
        "internal_nodes": total_nodes - leaf_nodes,
        "leaf_nodes": leaf_nodes,
        "max_depth": model.max_depth,
        "nodes": nodes_list
    }

    output_data = {
        "model_name": "Decision Tree Regressor (CART)",
        "hyperparameters": hyperparameters,
        "tree_structure": tree_structure,
        "feature_importances": model.get_feature_importances(),
        "evaluation_metrics": evaluation
    }

    return model, output_data


def save_json(data: Dict[str, Any], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[*] Đã xuất file JSON: {filepath}")


def run_training() -> Dict[str, Any]:
    splits = load_dataset()

    print("\n" + "=" * 60)
    print("1. Huấn luyện Linear Regression (OLS SVD)")
    print("=" * 60)
    _, lr_output = train_linear_regression(splits)
    lr_file = os.path.join(REPO_ROOT, "outputs", "linear_regression_trained_model.json")
    save_json(lr_output, lr_file)

    print("\n" + "=" * 60)
    print("2. Huấn luyện Decision Tree Regressor (CART)")
    print("=" * 60)
    _, dt_output = train_decision_tree(splits)
    dt_file = os.path.join(REPO_ROOT, "outputs", "decision_tree_trained_model.json")
    save_json(dt_output, dt_file)

    print("\n" + "=" * 60)
    print("3. Xuất bảng tổng hợp so sánh đánh giá (evaluate.py)")
    print("=" * 60)
    comparison_summary = {
        "Linear_Regression": lr_output["evaluation_metrics"],
        "Decision_Tree_Regressor": dt_output["evaluation_metrics"]
    }
    summary_file = os.path.join(REPO_ROOT, "outputs", "models_evaluation_summary.json")
    save_evaluation_json(comparison_summary, summary_file)
    print(f"[*] Đã xuất file JSON: {summary_file}")

    print("\n" + "=" * 60)
    print("HOÀN THÀNH QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ THÀNH CÔNG")
    print("=" * 60)

    return {
        "linear_regression": lr_output,
        "decision_tree": dt_output,
        "summary": comparison_summary
    }


def main():
    run_training()


if __name__ == "__main__":
    main()
