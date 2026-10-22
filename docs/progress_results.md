# Quá trình thực hiện notebook và kết quả

## 1. Chọn và lưu dữ liệu - 22/10/2026
Chọn BigMart phù hợp bài hồi quy cơ bản.
Lưu bigmart.csv nguyên bản từ ZIP Kaggle đã tải trong bước kiểm tra.
Lưu nguồn, ngày và SHA-256 trong Datasets/data_raw/source.json.

## 2. Khám phá dữ liệu
8.523 mẫu, 12 cột gốc, 1.559 sản phẩm và 10 cửa hàng.
Không trùng toàn bộ dòng hoặc cặp sản phẩm/cửa hàng.
Mục tiêu Item_Outlet_Sales không thiếu/âm/0; nhỏ nhất 33,29, lớn nhất 13.086,9648.
Thiếu Item_Weight: 1.463; thiếu Outlet_Size: 2.410.
Item_Visibility bằng 0: 526 mẫu.

## 3. Chuẩn hóa
Bỏ khoảng trắng và thống nhất nhãn Item_Fat_Content.
545 nhãn LF/low fat/reg được đổi về Low Fat/Regular.
Không loại dòng nào, sau chuẩn hóa vẫn 8.523 mẫu.
Thêm source_row và ba cờ chất lượng dữ liệu.
Chưa điền thiếu bằng median/mode để không học thống kê từ toàn dataset.
Không cắt ngoại lệ tùy ý; giữ giá trị visibility bằng 0 và đánh dấu.

## 4. Trực quan hóa và xuất kết quả
Tạo histogram doanh số, biểu đồ thiếu, boxplot theo loại cửa hàng,
scatter giá niêm yết/doanh số. Lưu thành 2 PNG, hiển thị trong notebook.
Xuất bigmart_clean.csv cùng ba bảng thống kê chất lượng.

## 5. Kiểm chứng
Chạy đủ 6 ô code từ đầu đến cuối, các assert dữ liệu đều đạt.
Notebook hợp lệ và có output/ảnh đã lưu; dữ liệu gốc giữ nguyên.

## 6. Chưa thực hiện
Chưa chia tập, điền thiếu train-only, mã hóa, fit/predict hoặc tuning.
Chưa có điểm MAE/MSE/RMSE/MAPE của mô hình.
Bước tiếp theo xem detail_plan.md.

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
