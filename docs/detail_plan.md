# Quy trình thực hiện và trạng thái

## Đã hoàn thành
1. Chọn BigMart; giữ raw, nguồn và SHA-256.
2. EDA thiếu/trùng/phân bố doanh số, giá, loại sản phẩm và loại cửa hàng.
3. Chuẩn hóa nhãn, giữ ngoại lệ chưa chứng minh sai và đánh dấu chất lượng.
4. Chia 5.966 train/1.278 validation/1.279 test, seed=42, lưu chỉ mục.
5. Học median/mode và One-Hot chỉ từ train; xuất 30 đặc trưng số.
6. Tạo baseline trung bình train.
7. Viết mã giả, class OLS và fit/predict bằng NumPy least-squares.
8. Viết class bốn phương thức cho cây dùng thư viện, thử 12 cấu hình.
9. Chọn depth=5, min_samples_leaf=20 bằng RMSE validation.
10. Đánh giá cuối bằng MAE/MSE/RMSE/MAPE trên cùng test; không cắt dự đoán âm.
11. Vẽ dự đoán/lỗi, cấu trúc cây rút gọn, so sánh và phân tích lỗi theo doanh số.
12. Demo cùng thống kê/danh mục train; cập nhật README và docs.
13. Chạy lại notebook, kiểm chứng OLS và tính tái lập của cây.

## Quy tắc giữ nguyên khi chạy lại
Không chọn seed hoặc cấu hình theo test để có điểm đẹp hơn.
Thay dữ liệu/cách chia/feature phải chạy lại EDA rồi cả hai model.
Mã sản phẩm có thể xuất hiện ở nhiều tập; không tuyên bố kiểm thử sản phẩm mới hoàn toàn.
Giữ model train-only trong báo cáo; không đổi sang train+validation mà giữ điểm cũ.

## Ngoài phạm vi đã làm
Chưa triển khai production, đo thời gian hoặc chạy nhiều seed/cross-validation.
Chưa tối ưu MAPE bằng biến đổi target hay cắt dự đoán âm.
Nếu mở rộng sau này, chọn bằng validation hoặc một quy trình đánh giá mới được xác định trước.
