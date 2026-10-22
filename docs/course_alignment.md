# Đối chiếu với bài giảng

Tệp: Machine_Learning_Full_Course.pdf (198 trang).
Đã đối chiếu tại ../Present_and_Lab1_ML_G5/ trong thư mục Lab_Ml.
Số trang tính từ đầu PDF, không dựa trên số trang in trong hình slide.

| Nội dung | Trang PDF | Áp dụng |
|---|---:|---|
| Quy trình chuẩn bị, feature engineering, train/evaluate | 19 | EDA và toàn bộ notebook |
| pandas, NumPy, scikit-learn, Matplotlib | 21 | Vai trò thư viện |
| Ví dụ LinearRegression, fit, predict, MSE | 22 | Trình tự huấn luyện hồi quy |
| Vai trò train/validation/test | 27 | Cùng ba tập cho hai model |
| Hồi quy và MAE/MSE/RMSE | 29 | Đánh giá dạng số |
| Fit thống kê trên train, transform test | 35 | Nguyên tắc tránh rò rỉ khi tiền xử lý |
| Overfitting/underfitting | 68-69 | So train và validation khi chọn cây |
| Cây hồi quy, trung bình tại lá | 74 | Giải thích cây |
| Hồi quy tuyến tính đa biến | 75 | Công thức dự đoán OLS |
| OLS tối thiểu hóa tổng bình phương sai số | 76 | np.linalg.lstsq |
| Cây và rừng cho hồi quy | 80 | Model Decision Tree |

## Những lựa chọn triển khai của nhóm
- BigMart; tỷ lệ 70/15/15; seed=42.
- Median/mode, One-Hot drop nhóm đầu và xử lý danh mục mới.
- Baseline trung bình train, lưới 12 cấu hình cây, RMSE validation để chọn.
- Các biểu đồ cụ thể và lưu chỉ mục, cấu hình, dự đoán, kết quả kiểm chứng.
- MAPE do người dùng yêu cầu thêm; không ghi là yêu cầu nguyên văn bài giảng.
- Class bốn phương thức và docs dưới 200 dòng là yêu cầu bài làm đã cung cấp.

Trang 35 minh họa StandardScaler cho classification; ở đây áp dụng nguyên tắc
chỉ học từ train cho imputer/encoder, không nói trang này hướng dẫn riêng BigMart.
Không thêm deep learning, ensemble hoặc thuật toán ngoài phạm vi đã chốt.
