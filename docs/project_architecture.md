# Cấu trúc và luồng chạy

## 1. Dữ liệu gốc
Datasets/data_raw/bigmart.csv: nguyên bản, không ghi đè.
source.json: nguồn, ngày, SHA-256.

## 2. EDA và tiền xử lý
Datasets/data_processed/EDA.ipynb đọc, chuẩn hóa, vẽ, chia ba tập.
bigmart_clean.csv giữ ô thiếu, thêm source_row và cờ để truy vết.
split_indices.csv lưu chỉ mục ba tập dùng chung.
preprocessing.json lưu median, mode, danh mục, tên cột, seed và hash.
X_train/validation/test.csv: 30 cột số sau điền thiếu và mã hóa.
y_train/validation/test.csv: mục tiêu dương, cùng source_row với X.

## 3. Hai notebook mô hình
Models/Linear_Regression.ipynb: baseline, OLS, đánh giá, lỗi và demo.
Models/Decision_Tree.ipynb: validation 12 cấu hình, cây chọn, đánh giá, demo, so sánh.
Không có class/script .py dùng song song với notebook.

## 4. Đầu ra
outputs/figures/: EDA, split, tuning, lỗi, cấu trúc cây và so sánh.
*_metrics.csv: train/validation/test; *_predictions.csv: nhãn số và sai số test.
*_largest_errors.csv, *_errors_by_sales.csv: phân tích lỗi.
decision_tree_validation.csv: toàn bộ cấu hình thử trên validation.
model_comparison.csv: baseline và hai model trên cùng test.
*_run.json, environment.json, verification.json: dấu vết, phiên bản và kiểm chứng.

## 5. Tài liệu
README: cài đặt, thứ tự chạy, kết quả chính.
DATASET_DOCUMENTATION: nguồn và ý nghĩa dữ liệu.
docs/course_alignment: nguồn bài giảng và phần nhóm bổ sung.
docs/Description, detail_plan, pseudocode, implementation, progress_results.

Thứ tự: EDA → Linear Regression → Decision Tree.
Các file docs dưới 200 dòng. Bảng kết quả trong docs phải khớp CSV thực tế.
