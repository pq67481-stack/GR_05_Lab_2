"""
Module inference.py: Phase 4 - Inference & Prediction Pipeline.
Cung cấp chức năng suy luận, dự báo doanh số bán hàng (Item_Outlet_Sales)
trên dữ liệu mới (chưa qua xử lý hoặc đã qua xử lý) bằng mô hình đã huấn luyện.
"""
# pylint: disable=too-many-locals,too-many-branches,invalid-name,wrong-import-position,import-outside-toplevel
import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import LinearRegressionModel, DecisionTreeRegressorModel


def load_preprocessing_metadata(meta_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Nạp các tham số tiền xử lý đóng băng từ Train (preprocessing.json).
    """
    if meta_path is None:
        p1 = os.path.join(REPO_ROOT, "data", "BigMart_Sales", "processed", "preprocessing.json")
        p2 = os.path.join(REPO_ROOT, "data", "tu_data", "processed", "preprocessing.json")
        meta_path = p1 if os.path.exists(p1) else p2

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Không tìm thấy file metadata '{meta_path}'. Vui lòng chạy train.py trước."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata: Dict[str, Any] = json.load(f)
    return metadata


def load_trained_models(
    model_type: str = "both"
) -> Tuple[Optional[LinearRegressionModel], Optional[DecisionTreeRegressorModel]]:
    """
    Nạp mô hình đã huấn luyện từ file JSON lưu trữ trong thư mục outputs"""

    lr_path = os.path.join(REPO_ROOT, "outputs", "linear_regression_trained_model.json")
    dt_path = os.path.join(REPO_ROOT, "outputs", "decision_tree_trained_model.json")

    if not os.path.exists(lr_path) or not os.path.exists(dt_path):
        from src.train import run_training
        print("[*] Chưa tìm thấy file mô hình huấn luyện. Đang tự động chạy kịch bản huấn luyện...")
        run_training()

    lr_model: Optional[LinearRegressionModel] = None
    dt_model: Optional[DecisionTreeRegressorModel] = None

    if model_type in ("lr", "both"):
        lr_model = LinearRegressionModel.load_from_json(lr_path)

    if model_type in ("dt", "both"):
        dt_model = DecisionTreeRegressorModel.load_from_json(dt_path)

    return lr_model, dt_model


def preprocess_raw_input(
    data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    
    if metadata is None:
        metadata = load_preprocessing_metadata()

    frozen_weight_median = float(metadata["frozen_weight_median"])
    frozen_size_mode = str(metadata["frozen_size_mode"])
    expected_features = list(metadata["feature_names"])

    # 1. Chuyển đổi đầu vào thành DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Dữ liệu đầu vào phải là Dict, List[Dict] hoặc pd.DataFrame.")

    # 2. Ép kiểu an toàn cho các cột số học (chuyển các giá trị không hợp lệ hoặc chuỗi rỗng thành NaN)
    numeric_base_cols = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"]
    for col in numeric_base_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Chuẩn hóa chuỗi text cho các biến phân loại (lọc bỏ các chuỗi rỗng/none/nan)
    cat_base_cols = [
        "Item_Identifier", "Item_Fat_Content", "Item_Type",
        "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"
    ]
    for col in cat_base_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x).strip().lower() not in ("none", "nan", "") else np.nan
            )

    # 4. Chuẩn hóa nhãn Item_Fat_Content
    if "Item_Fat_Content" in df.columns:
        fat_map = {
            "low fat": "Low Fat", "lowfat": "Low Fat", "lf": "Low Fat",
            "regular": "Regular", "reg": "Regular"
        }
        item_fat = df["Item_Fat_Content"].astype(str).str.lower().map(fat_map).fillna("Low Fat")
    else:
        item_fat = pd.Series("Low Fat", index=df.index)

    # 5. Tạo cờ chỉ báo missing & zero (theo đúng chuẩn xử lý dữ liệu)
    weight_missing = (
        df["Item_Weight"].isna().astype(float)
        if "Item_Weight" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    size_missing = (
        df["Outlet_Size"].isna().astype(float)
        if "Outlet_Size" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    visibility_zero = (
        (df["Item_Visibility"].eq(0) | df["Item_Visibility"].isna()).astype(float)
        if "Item_Visibility" in df.columns
        else pd.Series(1.0, index=df.index)
    )

    # 6. Điền khuyết theo tham số đóng băng từ Train
    item_weight = (
        df["Item_Weight"].fillna(frozen_weight_median).astype(float)
        if "Item_Weight" in df.columns
        else pd.Series(frozen_weight_median, index=df.index)
    )
    outlet_size = (
        df["Outlet_Size"].fillna(frozen_size_mode).astype(str)
        if "Outlet_Size" in df.columns
        else pd.Series(frozen_size_mode, index=df.index)
    )
    item_mrp = (
        df["Item_MRP"].fillna(0.0).astype(float)
        if "Item_MRP" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    item_vis = (
        df["Item_Visibility"].fillna(0.0).astype(float)
        if "Item_Visibility" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    outlet_year = (
        df["Outlet_Establishment_Year"].fillna(1999.0).astype(float)
        if "Outlet_Establishment_Year" in df.columns
        else pd.Series(1999.0, index=df.index)
    )

    # 7. Tạo DataFrame các đặc trưng số học cơ sở
    processed_df = pd.DataFrame({
        "Item_Weight": item_weight,
        "Item_Visibility": item_vis,
        "Item_MRP": item_mrp,
        "Outlet_Establishment_Year": outlet_year,
        "Item_Weight_Missing": weight_missing,
        "Outlet_Size_Missing": size_missing,
        "Item_Visibility_Zero": visibility_zero
    }, index=df.index)

    # 8. Tạo các cột One-Hot tương ứng đúng danh sách expected_features
    # Các biến danh mục: Item_Fat_Content, Item_Type, Outlet_Size, Outlet_Location_Type, Outlet_Type
    cat_sources: Dict[str, pd.Series] = {
        "Item_Fat_Content": item_fat,
        "Item_Type": (
            df["Item_Type"].fillna("").astype(str)
            if "Item_Type" in df.columns
            else pd.Series("", index=df.index)
        ),
        "Outlet_Size": outlet_size,
        "Outlet_Location_Type": (
            df["Outlet_Location_Type"].fillna("").astype(str)
            if "Outlet_Location_Type" in df.columns
            else pd.Series("", index=df.index)
        ),
        "Outlet_Type": (
            df["Outlet_Type"].fillna("").astype(str)
            if "Outlet_Type" in df.columns
            else pd.Series("", index=df.index)
        )
    }

    for feat in expected_features:
        if feat in processed_df.columns:
            continue

        # Tách tên đặc trưng one-hot: ví dụ "Item_Type_Dairy" -> prefix="Item_Type", value="Dairy"
        matched = False
        for prefix, col_series in cat_sources.items():
            if feat.startswith(prefix + "_"):
                val = feat[len(prefix) + 1:]
                processed_df[feat] = (col_series.astype(str) == val).astype(float)
                matched = True
                break

        if not matched:
            processed_df[feat] = 0.0

    # Đảm bảo đúng 30 đặc trưng và đúng thứ tự tuyệt đối
    processed_df = processed_df[expected_features]
    return processed_df


def predict_single_sample(
    raw_sample: Dict[str, Any],
    model_type: str = "both"
) -> Dict[str, Any]:
    """
    Dự báo doanh số cho 1 sản phẩm đơn lẻ (dạng Dict thô).
    Trả về dự báo của Linear Regression và Decision Tree (kèm quy tắc rẽ nhánh XAI).
    """
    metadata = load_preprocessing_metadata()
    X_mat = preprocess_raw_input(raw_sample, metadata)
    lr_model, dt_model = load_trained_models(model_type)

    result: Dict[str, Any] = {
        "input_sample": raw_sample
    }

    if lr_model is not None:
        lr_pred = float(lr_model.predict(X_mat)[0])
        result["linear_regression"] = {
            "predicted_sales": round(lr_pred, 2)
        }

    if dt_model is not None:
        dt_detail = dt_model.predict_single_with_path(X_mat.iloc[0].to_numpy())
        result["decision_tree"] = {
            "predicted_sales": round(float(dt_detail["predicted_value"]), 2),
            "leaf_node_id": dt_detail["leaf_node_id"],
            "decision_path": dt_detail["decision_path"]
        }

    return result


def predict_batch(
    data: Union[List[Dict[str, Any]], pd.DataFrame],
    model_type: str = "both"
) -> Dict[str, Any]:
    """
    Dự báo doanh số cho một tập dữ liệu thô nhiều mẫu.
    """
    metadata = load_preprocessing_metadata()
    X_mat = preprocess_raw_input(data, metadata)
    lr_model, dt_model = load_trained_models(model_type)

    results: Dict[str, Any] = {
        "n_samples": len(X_mat),
        "predictions": []
    }

    lr_preds = lr_model.predict(X_mat) if lr_model is not None else None
    X_np = X_mat.to_numpy()

    for i in range(len(X_mat)):
        item_pred: Dict[str, Any] = {"sample_index": i}
        if lr_preds is not None:
            item_pred["linear_regression_sales"] = round(float(lr_preds[i]), 2)
        if dt_model is not None:
            dt_detail = dt_model.predict_single_with_path(X_np[i])
            item_pred["decision_tree_sales"] = round(float(dt_detail["predicted_value"]), 2)
            item_pred["leaf_node_id"] = dt_detail["leaf_node_id"]
            item_pred["decision_path"] = dt_detail["decision_path"]
        results["predictions"].append(item_pred)

    return results


def save_inference_json(predictions_data: Dict[str, Any], filepath: Optional[str] = None) -> str:
    """
    Lưu kết quả dự báo ra file JSON.
    """
    if filepath is None:
        filepath = os.path.join(REPO_ROOT, "outputs", "sample_inference_predictions.json")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(predictions_data, f, indent=4, ensure_ascii=False)
    print(f"[*] Đã lưu kết quả suy luận ra file: {filepath}")
    return filepath


def main() -> None:
    """
    Hàm thực thi kiểm thử và mô phỏng pha suy luận (Inference).
    """
    print("\n" + "=" * 65)
    print("PHASE 4: INFERENCE / PREDICTION PIPELINE (BigMart Sales)")
    print("=" * 65)

    # 1. Thử nghiệm suy luận trên 1 mẫu sản phẩm thô mới chưa từng thấy
    sample_product: Dict[str, Any] = {
        "Item_Identifier": "FDA15",
        "Item_Weight": 9.30,
        "Item_Fat_Content": "Low Fat",
        "Item_Visibility": 0.016047,
        "Item_Type": "Dairy",
        "Item_MRP": 249.8092,
        "Outlet_Identifier": "OUT049",
        "Outlet_Establishment_Year": 1999,
        "Outlet_Size": "Medium",
        "Outlet_Location_Type": "Tier 1",
        "Outlet_Type": "Supermarket Type1"
    }

    print("\n[1] Thử nghiệm dự báo cho 1 sản phẩm mẫu (Single Raw Sample):")
    single_res = predict_single_sample(sample_product, model_type="both")
    if "linear_regression" in single_res:
        print(f" - Doanh số dự đoán (Linear Regression): {single_res['linear_regression']['predicted_sales']:,.2f}")
    if "decision_tree" in single_res:
        dt_res = single_res["decision_tree"]
        print(f" - Doanh số dự đoán (Decision Tree)    : {dt_res['predicted_sales']:,.2f}")
        print(f" - Rơi vào Leaf Node ID                : {dt_res['leaf_node_id']}")
        print(f" - Chuỗi quy tắc rẽ nhánh (Path)       : {dt_res['decision_path']}")

    # 2. Thử nghiệm dự báo trên 1 sản phẩm có dữ liệu thiếu (Missing Value Test)
    sample_missing: Dict[str, Any] = {
        "Item_Identifier": "FDX07",
        "Item_Weight": None,  # Thiếu cân nặng
        "Item_Fat_Content": "reg",  # Nhãn chưa chuẩn hóa
        "Item_Visibility": 0.0,  # Visibility = 0
        "Item_Type": "Snack Foods",
        "Item_MRP": 182.0950,
        "Outlet_Identifier": "OUT027",
        "Outlet_Establishment_Year": 1985,
        "Outlet_Size": None,  # Thiếu quy mô cửa hàng
        "Outlet_Location_Type": "Tier 3",
        "Outlet_Type": "Supermarket Type3"
    }

    print("\n[2] Thử nghiệm với mẫu có dữ liệu khuyết thiếu & nhãn rác:")
    missing_res = predict_single_sample(sample_missing, model_type="both")
    if "linear_regression" in missing_res:
        print(f" - Doanh số dự đoán (Linear Regression): {missing_res['linear_regression']['predicted_sales']:,.2f}")
    if "decision_tree" in missing_res:
        dt_res = missing_res["decision_tree"]
        print(f" - Doanh số dự đoán (Decision Tree)    : {dt_res['predicted_sales']:,.2f}")
        print(f" - Rơi vào Leaf Node ID                : {dt_res['leaf_node_id']}")
        print(f" - Chuỗi quy tắc rẽ nhánh (Path)       : {dt_res['decision_path']}")

    # 3. Xuất file JSON kết quả suy luận mẫu
    demo_output = {
        "inference_pipeline": "BigMart Sales Predictor",
        "sample_single_normal": single_res,
        "sample_single_with_missing": missing_res
    }
    save_inference_json(demo_output)
    print("\n" + "=" * 65)
    print("HOÀN THÀNH PHA SUY LUẬN (INFERENCE) THÀNH CÔNG")
    print("=" * 65)


if __name__ == "__main__":
    main()
