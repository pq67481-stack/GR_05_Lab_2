# Giải thích cài đặt

## 1. Chuẩn bị đầu vào
EDA chuẩn hóa raw và giữ source_row để truy vết.
Chia ngẫu nhiên 70/15/15, seed=42; không phân tầng theo nhãn hồi quy.
Median Item_Weight=12,65; mode Outlet_Size=Medium, đều tính từ train.
Giữ ba cờ chất lượng. OneHotEncoder fit trên train, drop first, unknown ignore.
Hiện validation/test không có danh mục lạ. Nếu có, nhóm đó thành toàn 0,
có thể giống nhóm tham chiếu; demo chủ động báo lỗi cho danh mục chưa học.
Bỏ source_row và hai mã định danh; còn 30 cột số, không có ô thiếu.
Không cần scale cho OLS giải bằng least-squares và cây ở bản này.

## 2. LinearRegressionOLS
init: ready=False, intercept/coef=None.
build_model: đặt lại trạng thái và bật ready.
fit: thêm cột 1, giải least-squares, lấy intercept và coefficients.
predict: intercept + X @ coef; kiểm tra đã fit, số cột và giá trị hữu hạn.
Không gọi sklearn LinearRegression trong model chính; NumPy hỗ trợ giải hệ số.
Không lấy nghịch đảo XᵀX vì có thể suy biến; không dùng learning rate/epoch.
Hệ số khác đơn vị hoặc có phụ thuộc không được diễn giải trực tiếp như tầm quan trọng/nhân quả.

## 3. DecisionTreeSalesRegressor
init: lưu max_depth, min_samples_leaf và seed; model=None.
build_model: tạo DecisionTreeRegressor, fitted=False.
fit/predict: kiểm tra thứ tự gọi và chuyển cho thư viện.
Lá dự đoán trung bình y train tại lá; không phải bỏ phiếu phân loại.
Thử depth [3,5,8,None] và leaf [5,10,20]. Chọn RMSE validation thấp nhất.
Model chọn depth=5, leaf=20, seed=42; fit chỉ train, không refit train+validation.

## 4. Metrics và sai số
MAE = mean(abs(y - pred)); MSE = mean((y - pred)^2); RMSE = sqrt(MSE).
MAPE (%) = mean(abs((y - pred)/y))*100. Chặn trường hợp y<=0 trong bài này.
residual = actual - prediction; dương là dự đoán thấp hơn thực tế.
Giữ các dự đoán OLS âm khi đánh giá, có cột đếm số lượng.
MAPE có thể >100%, không phải Accuracy; không lấy 100 - MAPE.
Phân tích theo các khoảng doanh số cố định chỉ sau đánh giá, không thay cấu hình.

## 5. Đối chiếu công bằng và demo
Hai model đọc cùng X/y, thứ tự cột và hash preprocessing.
So sánh kiểm tra cùng source_row, cùng actual rồi tính lại metrics từ dự đoán.
Demo dùng median/mode/danh mục đã lưu; không fit trên dữ liệu mới.
One-Hot demo đã đối chiếu với encoder fit từ train, kết quả khớp.
Không lưu model pickle hoặc tạo script .py trùng notebook.

## 6. Kiểm chứng
Đã chạy cả ba notebook từ đầu đến cuối, không có error output.
OLS đúng với bài nhỏ có đáp án biết trước; reset xóa trạng thái đã học.
Đối chiếu dự đoán OLS với sklearn LinearRegression độc lập: khớp trong dung sai số học.
Cây fit lại cùng seed/cấu hình cho dự đoán test giống hệt.
Kiểm tra split không chồng chỉ mục, imputer train-only, JSON notebook hợp lệ.

## Chỉ số bổ sung R²
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
