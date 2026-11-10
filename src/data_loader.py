import os
import json
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_processed_data_dir() -> str:
    """Trả về thư mục processed (ưu tiên data/BigMart_Sales/processed, sau đó data/tu_data/processed)."""
    p1 = os.path.join(REPO_ROOT, "data", "BigMart_Sales", "processed")
    if os.path.isdir(p1):
        return p1
    return os.path.join(REPO_ROOT, "data", "tu_data", "processed")


def load_clean_data(clean_path: Optional[str] = None) -> pd.DataFrame:
    """
    Nạp dữ liệu đã qua bước xử lý cục bộ và làm sạch ban đầu từ 02_EDA.ipynb (bigmart_clean.csv).
    """
    if clean_path is None:
        p1 = os.path.join(REPO_ROOT, "data", "BigMart_Sales", "processed", "bigmart_clean.csv")
        p2 = os.path.join(REPO_ROOT, "data", "tu_data", "processed", "bigmart_clean.csv")
        clean_path = p1 if os.path.exists(p1) else p2

    if not os.path.exists(clean_path):
        raise FileNotFoundError(
            f"Không tìm thấy file '{clean_path}'. Vui lòng chạy 02_EDA.ipynb để tạo dữ liệu sạch ban đầu."
        )

    clean_df = pd.read_csv(clean_path)
    if "source_row" not in clean_df.columns:
        clean_df.insert(0, "source_row", clean_df.index)

    return clean_df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.30,
    val_size: float = 0.50,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia tập dữ liệu thành 3 tập độc lập trước khi thực hiện bất kỳ phép biến đổi có trạng thái nào:
    - Train: 70%
    - Validation: 15%
    - Test: 15%
    """
    source_rows = df["source_row"].to_numpy()
    train_ids, remain_ids = train_test_split(source_rows, test_size=test_size, random_state=random_state)
    val_ids, test_ids = train_test_split(remain_ids, test_size=val_size, random_state=random_state)

    indexed = df.set_index("source_row")
    train_df = indexed.loc[train_ids].copy().reset_index()
    val_df = indexed.loc[val_ids].copy().reset_index()
    test_df = indexed.loc[test_ids].copy().reset_index()

    return train_df, val_df, test_df


def fit_and_transform_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_name: str = "Item_Outlet_Sales"
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Dict[str, Any]]:
    """
    Quy trình chuyển đổi đặc trưng tuần tự 3 giai đoạn:
    - Giai đoạn 1: FIT & TRANSFORM TRÊN TẬP TRAIN (Học median, mode, OneHotEncoder)
    - Giai đoạn 2: TRANSFORM TRÊN TẬP VALIDATION (Sử dụng tham số đóng băng từ Train)
    - Giai đoạn 3: TRANSFORM TRÊN TẬP TEST (Sử dụng tham số đóng băng từ Train)
    """
    numeric_columns = [
        "Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year",
        "Item_Weight_Missing", "Outlet_Size_Missing", "Item_Visibility_Zero"
    ]
    category_columns = [
        "Item_Fat_Content", "Item_Type", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"
    ]

    # =========================================================================
    # GIAI ĐOẠN 1: FIT & TRANSFORM TRÊN TẬP TRAIN
    # =========================================================================
    train_processed = train_df.copy()

    # 1.1. Học (Fit) trung vị Item_Weight và mode Outlet_Size DUY NHẤT trên Train
    train_weight_median = float(train_processed["Item_Weight"].median())
    train_size_mode = str(train_processed["Outlet_Size"].mode().iloc[0])

    # 1.2. Áp dụng điền khuyết cho Train
    train_processed["Item_Weight"] = train_processed["Item_Weight"].fillna(train_weight_median)
    train_processed["Outlet_Size"] = train_processed["Outlet_Size"].fillna(train_size_mode)

    # 1.3. Khởi tạo và Fit_Transform OneHotEncoder trên Train
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    train_encoded_cats = encoder.fit_transform(train_processed[category_columns])
    encoded_feature_names = encoder.get_feature_names_out(category_columns).tolist()
    all_feature_names = numeric_columns + encoded_feature_names

    # 1.4. Ghép đặc trưng thành X_train và y_train
    train_numeric_mat = train_processed[numeric_columns].astype(float).to_numpy()
    X_train_mat = np.column_stack([train_numeric_mat, train_encoded_cats])
    X_train = pd.DataFrame(X_train_mat, columns=all_feature_names)
    y_train = train_processed[target_name].copy().reset_index(drop=True)

    # =========================================================================
    # GIAI ĐOẠN 2: TRANSFORM TRÊN TẬP VALIDATION (THAM SỐ ĐÓNG BĂNG TỪ TRAIN)
    # =========================================================================
    val_processed = val_df.copy()

    # 2.1. Điền khuyết trên Validation bằng trung vị và mode của TRAIN
    val_processed["Item_Weight"] = val_processed["Item_Weight"].fillna(train_weight_median)
    val_processed["Outlet_Size"] = val_processed["Outlet_Size"].fillna(train_size_mode)

    # 2.2. Transform One-Hot bằng encoder ĐÃ FIT TRÊN TRAIN
    val_encoded_cats = encoder.transform(val_processed[category_columns])

    # 2.3. Ghép đặc trưng thành X_val và y_val
    val_numeric_mat = val_processed[numeric_columns].astype(float).to_numpy()
    X_val_mat = np.column_stack([val_numeric_mat, val_encoded_cats])
    X_val = pd.DataFrame(X_val_mat, columns=all_feature_names)
    y_val = val_processed[target_name].copy().reset_index(drop=True)

    # =========================================================================
    # GIAI ĐOẠN 3: TRANSFORM TRÊN TẬP TEST (THAM SỐ ĐÓNG BĂNG TỪ TRAIN)
    # =========================================================================
    test_processed = test_df.copy()

    # 3.1. Điền khuyết trên Test bằng trung vị và mode của TRAIN
    test_processed["Item_Weight"] = test_processed["Item_Weight"].fillna(train_weight_median)
    test_processed["Outlet_Size"] = test_processed["Outlet_Size"].fillna(train_size_mode)

    # 3.2. Transform One-Hot bằng encoder ĐÃ FIT TRÊN TRAIN
    test_encoded_cats = encoder.transform(test_processed[category_columns])

    # 3.3. Ghép đặc trưng thành X_test và y_test
    test_numeric_mat = test_processed[numeric_columns].astype(float).to_numpy()
    X_test_mat = np.column_stack([test_numeric_mat, test_encoded_cats])
    X_test = pd.DataFrame(X_test_mat, columns=all_feature_names)
    y_test = test_processed[target_name].copy().reset_index(drop=True)

    frozen_metadata = {
        "frozen_weight_median": train_weight_median,
        "frozen_size_mode": train_size_mode,
        "feature_names": all_feature_names,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test)
    }

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), frozen_metadata


def save_processed_data(
    train_split: Tuple[pd.DataFrame, pd.Series],
    val_split: Tuple[pd.DataFrame, pd.Series],
    test_split: Tuple[pd.DataFrame, pd.Series],
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    Lưu các tập dữ liệu X, y và metadata tiền xử lý ra thư mục data processed.
    """
    if output_dir is None:
        output_dir = get_processed_data_dir()
    os.makedirs(output_dir, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_split, val_split, test_split
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_validation.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_validation.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    if metadata is not None:
        meta_path = os.path.join(output_dir, "preprocessing.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)


def load_processed_data() -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """
    Nạp trực tiếp các tập dữ liệu X, y đã lưu trong thư mục processed.
    """
    proc_dir = get_processed_data_dir()

    X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv"))

    X_val = pd.read_csv(os.path.join(proc_dir, "X_validation.csv"))
    y_val = pd.read_csv(os.path.join(proc_dir, "y_validation.csv"))

    X_test = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv"))

    for df in [X_train, X_val, X_test, y_train, y_val, y_test]:
        if "source_row" in df.columns:
            df.drop(columns=["source_row"], inplace=True)

    return (X_train, y_train.iloc[:, 0]), (X_val, y_val.iloc[:, 0]), (X_test, y_test.iloc[:, 0])


def load_dataset(
    force_recompute: bool = False,
    save_processed: bool = True
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """
    Hàm giao diện chính nạp toàn bộ dataset cho train.py và evaluate.py:
    - Mặc định: Nạp nhanh các file CSV đã qua tiền xử lý trong data/tu_data/processed/
    - Nếu force_recompute=True: Nạp từ bigmart_clean.csv (kết quả làm sạch cục bộ từ EDA),
      thực hiện chia tập 70/15/15 và chạy lại tuần tự quy trình Fit (Train) / Transform (Val/Test).
    """
    proc_dir = get_processed_data_dir()
    has_processed = (
        os.path.exists(os.path.join(proc_dir, "X_train.csv")) and
        os.path.exists(os.path.join(proc_dir, "y_train.csv")) and
        os.path.exists(os.path.join(proc_dir, "X_validation.csv"))
    )

    if has_processed and not force_recompute:
        return load_processed_data()

    # Nạp từ dữ liệu đã làm sạch cục bộ từ EDA
    clean_df = load_clean_data()
    train_df, val_df, test_df = split_data(clean_df)
    train_split, val_split, test_split, metadata = fit_and_transform_pipeline(train_df, val_df, test_df)

    if save_processed:
        save_processed_data(train_split, val_split, test_split, metadata, proc_dir)

    return train_split, val_split, test_split


clean_raw_data = load_clean_data


def run_data_pipeline(force_recompute: bool = True) -> None:
    """
    Thực thi toàn bộ pipeline chia tập dữ liệu và tiền xử lý:
    1. Đọc dữ liệu sạch từ bigmart_clean.csv
    2. Chia tách độc lập thành 3 tập: Train (70%), Validation (15%), Test (15%)
    3. Fit tham số tiền xử lý trên Train, Transform sang Validation và Test
    4. Lưu kết quả ra thư mục data/BigMart_Sales/processed/
    5. In thông tin chi tiết và thống kê ra màn hình
    """
    print("=" * 65)
    print("QUY TRÌNH CHIA DỮ LIỆU & TIỀN XỬ LÝ (DATA SPLITTING & PIPELINE)")
    print("=" * 65)

    clean_df = load_clean_data()
    print("[*] Nạp dữ liệu thành công từ bigmart_clean.csv:")
    print(f"    - Tổng số mẫu: {len(clean_df):,} dòng x {clean_df.shape[1]} cột")

    print("\n[*] Thực hiện chia tách tập dữ liệu (70% Train - 15% Val - 15% Test)...")
    train_df, val_df, test_df = split_data(clean_df)
    print(f"    - Train set     : {len(train_df):,} dòng ({len(train_df)/len(clean_df)*100:.1f}%)")
    print(f"    - Validation set: {len(val_df):,} dòng ({len(val_df)/len(clean_df)*100:.1f}%)")
    print(f"    - Test set      : {len(test_df):,} dòng ({len(test_df)/len(clean_df)*100:.1f}%)")

    print("\n[*] Thực thi Fit bộ tiền xử lý trên Train & Transform sang Val/Test...")
    train_split, val_split, test_split, metadata = fit_and_transform_pipeline(train_df, val_df, test_df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_split, val_split, test_split

    print("    - Các tham số học được từ Train (đóng băng để tránh Data Leakage):")
    print(f"      + Median Item_Weight : {metadata['frozen_weight_median']:.2f}")
    print(f"      + Mode Outlet_Size   : {metadata['frozen_size_mode']}")
    print(f"      + Tổng số đặc trưng  : {len(metadata['feature_names'])} cột")

    proc_dir = get_processed_data_dir()
    print(f"\n[*] Đang lưu các file dữ liệu đã chia và xử lý vào:")
    print(f"    Thư mục: {proc_dir}")
    save_processed_data(train_split, val_split, test_split, metadata, proc_dir)

    print("\n[✓] DANH SÁCH CÁC FILE ĐÃ TẠO THÀNH CÔNG:")
    print(f"    ├── X_train.csv          : {X_train.shape} (Ma trận đặc trưng Train)")
    print(f"    ├── y_train.csv          : {y_train.shape} (Nhãn doanh số Train)")
    print(f"    ├── X_validation.csv     : {X_val.shape} (Ma trận đặc trưng Validation)")
    print(f"    ├── y_validation.csv     : {y_val.shape} (Nhãn doanh số Validation)")
    print(f"    ├── X_test.csv           : {X_test.shape} (Ma trận đặc trưng Test)")
    print(f"    ├── y_test.csv           : {y_test.shape} (Nhãn doanh số Test)")
    print("    └── preprocessing.json   : Lưu trữ các tham số tiền xử lý đóng băng")
    print("=" * 65)
    print("HOÀN THÀNH CHIA VÀ XỬ LÝ DỮ LIỆU THÀNH CÔNG!")
    print("=" * 65)


if __name__ == "__main__":
    run_data_pipeline(force_recompute=True)

