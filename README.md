# GR_05_Lab_2: BigMart Sales Prediction ML Pipeline

Hệ thống Machine Learning dự đoán doanh số bán lẻ BigMart (`Item_Outlet_Sales`) theo chuẩn kiến trúc mô-đun hóa 4 pha (End-to-End Modular Machine Learning Pipeline), triệt tiêu hoàn toàn rò rỉ dữ liệu (Anti-Data Leakage), và hỗ trợ giải thích mô hình (Explainable AI - XAI).

---

## 1. Cấu trúc thư mục dự án

```text
GR_05_Lab_2/
├── requirements.txt
├── .gitignore
├── .vscode/
│   └── settings.json
├── data/
│   └── BigMart_Sales/
│       ├── raw/
│       │   ├── bigmart.csv
│       │   └── source.json
│       └── processed/
│           ├── bigmart_clean.csv
│           ├── preprocessing.json
│           ├── split_indices.csv
│           ├── test_processed.csv
│           ├── train_processed.csv
│           ├── val_processed.csv
│           ├── X_test.csv
│           ├── X_train.csv
│           ├── X_validation.csv
│           ├── y_test.csv
│           ├── y_train.csv
│           └── y_validation.csv
├── docs/
│   ├── Description.md
│   ├── Design.md
│   └── Algorithms/
│       ├── Decision_Tree.md
│       └── Linear_Regression.md
├── notebooks/
│   ├── 01_collection.ipynb
│   ├── 02_EDA.ipynb
│   └── report.ipynb
├── outputs/
│   ├── decision_tree_trained_model.json
│   ├── linear_regression_trained_model.json
│   ├── linear_regression_training_details.json
│   ├── models_evaluation_summary.json
│   ├── sample_inference_predictions.json
│   └── figures/
│       ├── actual_vs_predicted.png
│       ├── feature_importance.png
│       ├── model_comparison.png
│       └── residual_analysis.png
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── evaluate.py
    ├── inference.py
    ├── models.py
    ├── train.py
    └── visualization.py
```

---

## 2. Hướng dẫn cài đặt & Thực thi nhanh

### 2.1. Cài đặt môi trường
```bash
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Cài đặt thư viện:
pip install -r requirements.txt
```

### 2.2. Chạy quy trình 4 pha từ Terminal
```bash
# Phase 2: Huấn luyện mô hình (Linear Regression & Decision Tree)
python src/train.py

# Phase 3: Đánh giá độc lập trên tập Test
python src/evaluate.py

# Xuất 4 bộ biểu đồ khoa học 300 DPI
python src/visualization.py

# Phase 4: Thử nghiệm suy luận & dự báo (Inference)
python src/inference.py
```

---

## 3. Kết quả đối sánh mô hình (Benchmarking)

| Mô hình | Tập dữ liệu | $R^2$ Score | RMSE (USD) | MAE (USD) | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | Validation (15%) | 0.5583 | 1,149.25 | 847.13 | 110.15% |
| | **Test (15%)** | 0.5614 | 1,122.85 | 833.15 | 115.55% |
| **Decision Tree (CART)** | Validation (15%) | **0.6189** | **1,067.75** | **756.95** | **61.35%** |
| | **Test (15%)** | **0.5916** | **1,083.41** | **755.95** | **58.46%** |

*Mô hình Decision Tree Regressor đạt hiệu năng vượt trội và không bị Overfitting.*

