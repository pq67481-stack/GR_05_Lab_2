# Nhóm 5 - Lab 2: Dự đoán doanh số BigMart

## 1. Bài toán và kết quả hiện tại
Dự đoán `Item_Outlet_Sales` cho sản phẩm tại cửa hàng từ thuộc tính sản phẩm/cửa hàng.
Đây là hồi quy trên dữ liệu bảng; không dự báo ngày/tuần vì dataset không có ngày bán.
Đã hoàn thành EDA, chia tập, tiền xử lý train-only, baseline, hai mô hình,
chọn cấu hình bằng validation, đánh giá test, biểu đồ, phân tích sai số và demo notebook.
Chưa triển khai website/API hoặc đo tốc độ thực nghiệm.

## 2. Cài đặt
Dùng Python 3.10 trở lên, mở terminal tại `GR_05_Lab_2`:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```
Chọn kernel `.venv` trong VS Code. Khi đã mở notebook, dùng Restart Kernel rồi Run All.
Các thư viện: pandas xử lý bảng, NumPy tính toán OLS, scikit-learn chia tập/mã hóa/cây/chỉ số,
Matplotlib vẽ hình. Không thêm script/class .py trùng notebook.

## 3. Thứ tự chạy bắt buộc
1. `Datasets/data_processed/EDA.ipynb`: tái tạo dữ liệu sạch, cùng split và tiền xử lý.
2. `Models/Linear_Regression.ipynb`: OLS tự viết và baseline, đánh giá, phân tích, demo.
3. `Models/Decision_Tree.ipynb`: thử cấu hình bằng validation, đánh giá và so sánh cuối.

Notebook đã có output và ảnh lưu sẵn. Khi đổi dữ liệu/cách chia/tiền xử lý,
phải chạy lại cả ba theo thứ tự để không so sánh kết quả cũ.
Các notebook model kiểm tra dấu vết dữ liệu; ô so sánh kiểm tra cùng test và tiền xử lý.

## 4. Cấu trúc thư mục
```text
Datasets/
  data_raw/bigmart.csv           # Nguyên bản Kaggle
  data_raw/source.json           # Nguồn, SHA-256
  data_processed/EDA.ipynb
  data_processed/bigmart_clean.csv
  data_processed/split_indices.csv
  data_processed/preprocessing.json
  data_processed/X_*.csv         # train, validation, test
  data_processed/y_*.csv
Models/
  Linear_Regression.ipynb
  Decision_Tree.ipynb
outputs/
  figures/                      # EDA, split, tuning, dự đoán, lỗi, cây, so sánh
  model_comparison.csv
  baseline_metrics.csv
  linear_regression_*            # metrics, predictions, errors, coefficients, run
  decision_tree_*                # metrics, predictions, errors, validation, run
  environment.json              # Phiên bản thư viện, seed
  verification.json             # Kết quả kiểm chứng
docs/                           # Mô tả, nguồn học, kế hoạch, mã giả, quá trình
```

## 5. Dữ liệu và tiền xử lý
- Gốc 8.523 mẫu, 12 cột, 1.559 sản phẩm và 10 cửa hàng; không trùng bản ghi.
- Chuẩn hóa 545 nhãn LF/low fat/reg về Low Fat/Regular; giữ nguyên số mẫu.
- Giữ 526 visibility bằng 0 và đánh dấu, không tự khẳng định là lỗi.
- Chia seed=42: train 5.966, validation 1.278, test 1.279.
- Từ train: median Item_Weight = 12,65, mode Outlet_Size = Medium.
- One-Hot học từ train, bỏ nhóm tham chiếu; tổng cộng 30 đặc trưng.
- Bỏ source_row, Item_Identifier và Outlet_Identifier khỏi đầu vào.
- Cờ thiếu và cờ visibility = 0 được giữ; không dùng target để điền thiếu.
- CSV clean còn ô thiếu có chủ đích; các X_train/validation/test đã không còn thiếu.

## 6. Hai mô hình và baseline
- Baseline: luôn dự đoán trung bình y_train, chỉ là mốc so sánh.
- OLS tự viết class bốn phương thức bằng NumPy `lstsq`; không gọi sklearn LinearRegression
  trong mô hình chính. Không dùng nghịch đảo ma trận, learning rate hoặc epoch.
- Cây: class bốn phương thức gọi sklearn DecisionTreeRegressor; không tuyên bố tự viết CART.
- Thử 12 cấu hình cây trên train/validation; chọn RMSE validation thấp nhất.
- Cấu hình chọn: max_depth=5, min_samples_leaf=20, random_state=42.
- Giữ model fit trên train; không refit train+validation trong kết quả báo cáo này.

## 7. Kết quả cuối trên cùng 1.279 mẫu test
| Model | MAE | MSE | RMSE | MAPE (%) |
|---|---|---|---|---|
| Baseline mean | 1347.81 | 2879273.71 | 1696.84 | 249.85 |
| Linear Regression OLS | 833.15 | 1260792.47 | 1122.85 | 115.55 |
| Decision Tree | 755.95 | 1173781.23 | 1083.41 | 58.46 |

Cây tốt hơn OLS theo cả bốn chỉ số trong lần thử này và có RMSE validation thấp hơn.
Cả hai tốt hơn baseline. Đây không phải kết luận cây luôn tốt hơn trên mọi dữ liệu.
OLS có 78 dự đoán test âm; giữ nguyên khi đánh giá, không cắt về 0.
MAPE có thể vượt 100%, không phải Accuracy và không đổi thành 100 - MAPE.
Đọc MAPE cùng MAE/RMSE, nhất là các mẫu doanh số nhỏ.

## 8. Cách đọc kết quả và demo
- `*_predictions.csv`: source_row, actual, prediction, residual=actual-prediction, APE.
- `*_largest_errors.csv`: 20 mẫu sai tuyệt đối nhiều nhất để tra dữ liệu gốc.
- `*_errors_by_sales.csv`: MAE/MAPE theo mức doanh số, chỉ phân tích sau đánh giá.
- Phần demo dùng dữ liệu giả định; tái sử dụng median/mode và danh mục đã học từ train.
- Demo báo lỗi nếu gặp danh mục chưa học; không fit lại bằng các sản phẩm demo.
- Đường 40 mẫu trong biểu đồ là thứ tự mẫu test, không phải thời gian.

## 9. Tài liệu
- `DATASET_DOCUMENTATION.md`: nguồn, cột và giới hạn.
- `docs/course_alignment.md`: đối chiếu trang bài giảng và phần nhóm bổ sung.
- `docs/pseudocode.md`, `docs/implementation.md`: thuật toán và cách đọc code.
- `docs/progress_results.md`: quá trình và kết quả thực tế.
- Nguồn dữ liệu: https://www.kaggle.com/datasets/yasserh/bigmartsalesdataset

## Chỉ số bổ sung R²

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat y_i)^2}{\sum_{i=1}^{n}(y_i-\bar y)^2}.$$

Ở đây, $\bar y$ là trung bình doanh số thật của **chính tập đang đánh giá**.
R² càng cao càng tốt: 1 là dự đoán hoàn hảo; 0 tương đương luôn dự đoán trung bình tập đánh giá;
âm là kém hơn mức đó. R² không phải Accuracy và không có đơn vị.
Ví dụ R² = 0,60 nghĩa là mô hình giải thích khoảng 60% biến thiên doanh số trên tập đó,
không phải dự đoán đúng 60% email/sản phẩm.
Baseline của lab dùng trung bình **train**, nên R² validation/test có thể âm, không bắt buộc bằng 0.
Chỉ bổ sung chỉ số đánh giá; vẫn chọn cấu hình cây theo RMSE validation, không theo R² test.


| Mô hình | R² test |
|---|---:|
| Baseline mean | -0.0017 |
| Linear Regression OLS | 0.5614 |
| Decision Tree | 0.5916 |

Hai notebook tính R² cho train/validation/test và baseline; CSV metrics và model_comparison.csv thêm cột `R2`.
Bảng thử cấu hình cây thêm train_R2/validation_R2 để quan sát, vẫn chọn bằng validation RMSE.
Biểu đồ riêng: `outputs/figures/model_comparison_r2.png` để tránh trộn R² với đơn vị sai số.
Đã chạy toàn bộ code hai notebook, đối chiếu công thức bằng tay và kiểm tra dự đoán/chỉ số cũ không đổi.
