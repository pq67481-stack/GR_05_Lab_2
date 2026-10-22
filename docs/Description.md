# Mô tả bài toán Lab 2

## Bài toán
Dự đoán Item_Outlet_Sales cho sản phẩm tại cửa hàng bằng thuộc tính sản phẩm/cửa hàng.
Đây là bài hồi quy dữ liệu bảng; không có ngày bán để dự báo doanh số theo thời gian.
Không xác nhận bối cảnh thương mại điện tử; BigMart là dữ liệu bán lẻ.

## Đầu vào và đầu ra
- Đầu vào: khối lượng, visibility, MRP, năm thành lập, loại sản phẩm/cửa hàng,
  quy mô, vị trí, hàm lượng chất béo và các cờ chất lượng.
- Không dùng source_row và hai mã định danh làm feature.
- Đầu ra: giá trị doanh số, không gọi là số lượng bán; chưa tự gán đơn vị tiền.

## Mô hình
1. LinearRegressionOLS: tự viết học hệ số/predict bằng NumPy least-squares.
2. DecisionTreeSalesRegressor: class gọi DecisionTreeRegressor của scikit-learn.
Baseline trung bình train chỉ là mốc đối chiếu.

## Đánh giá
Cùng split 70/15/15, seed=42. Chọn độ phức tạp cây bằng RMSE validation.
Test chỉ đánh giá sau khi cấu hình đã chốt. Báo MAE, MSE, RMSE và MAPE.
Giữ dữ liệu gốc, notebook có giải thích/mã giả, hình và output đã chạy.
Chưa có API/website hoặc đánh giá sản phẩm hoàn toàn mới.
