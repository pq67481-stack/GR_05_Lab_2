"""
Package src: Cung cấp đầy đủ 6 mô-đun cốt lõi của Machine Learning Pipeline:
1. data_loader: Tải, tiền xử lý và chia tập dữ liệu train/val/test
2. models: Định nghĩa kiến trúc mô hình LinearRegressionModel và DecisionTreeRegressorModel
3. train: Kịch bản huấn luyện (training loop, logging)
4. evaluate: Đánh giá mô hình, tính toán metrics và xuất báo cáo
5. inference: Dự báo và suy luận doanh số cho dữ liệu mới (Inference Phase)
6. visualization: Xuất đồ thị khoa học phân tích và đánh giá mô hình
"""
from src.data_loader import (
    load_dataset,
    load_processed_data,
    load_clean_data,
    clean_raw_data,
    split_data,
    fit_and_transform_pipeline
)
from src.models import (
    LinearRegressionModel,
    DecisionTreeRegressorModel,
    LinearRegression,
    DecisionTreeRegressor,
    TreeNode
)
from src.evaluate import (
    calculate_metrics,
    evaluate_model,
    evaluate_model_splits,
    compare_models,
    save_evaluation_json
)
from src.train import (
    run_training,
    train_linear_regression,
    train_decision_tree
)
from src.inference import (
    load_trained_models,
    preprocess_raw_input,
    predict_single_sample,
    predict_batch,
    save_inference_json
)
from src.visualization import (
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residual_analysis,
    plot_feature_importance,
    run_all_visualizations
)

__all__ = [
    "load_dataset",
    "load_processed_data",
    "load_clean_data",
    "clean_raw_data",
    "split_data",
    "fit_and_transform_pipeline",
    "LinearRegressionModel",
    "DecisionTreeRegressorModel",
    "LinearRegression",
    "DecisionTreeRegressor",
    "TreeNode",
    "calculate_metrics",
    "evaluate_model",
    "evaluate_model_splits",
    "compare_models",
    "save_evaluation_json",
    "run_training",
    "train_linear_regression",
    "train_decision_tree",
    "load_trained_models",
    "preprocess_raw_input",
    "predict_single_sample",
    "predict_batch",
    "save_inference_json",
    "plot_model_comparison",
    "plot_actual_vs_predicted",
    "plot_residual_analysis",
    "plot_feature_importance",
    "run_all_visualizations"
]
