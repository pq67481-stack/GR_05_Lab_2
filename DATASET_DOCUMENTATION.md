# Dữ liệu BigMart

## 1. Nguồn
https://www.kaggle.com/datasets/yasserh/bigmartsalesdataset
Tệp tải: bigmart.csv; 8.523 dòng và 12 cột.
CSV lấy nguyên bản từ ZIP Kaggle đã tải khi kiểm tra dataset.
Metadata nguồn và SHA-256 lưu trong Datasets/data_raw/source.json.
Kaggle ghi CC0; trang cuộc thi BigMart của Analytics Vidhya có điều khoản riêng.
Đối chiếu nguồn trước khi phân phối lại dữ liệu gốc công khai.

## 2. Ý nghĩa các cột
| Cột | Vai trò |
|---|---|
| Item_Identifier | Mã sản phẩm |
| Item_Weight | Khối lượng; đơn vị chưa xác minh |
| Item_Fat_Content | Nhóm hàm lượng chất béo |
| Item_Visibility | Mức độ trưng bày sản phẩm |
| Item_Type | Loại sản phẩm |
| Item_MRP | Giá niêm yết; không tự gán tiền tệ |
| Outlet_Identifier | Mã cửa hàng |
| Outlet_Establishment_Year | Năm thành lập, không phải ngày bán |
| Outlet_Size | Quy mô cửa hàng |
| Outlet_Location_Type | Nhóm vị trí |
| Outlet_Type | Loại cửa hàng |
| Item_Outlet_Sales | Mục tiêu doanh số; không gọi là số lượng bán |

## 3. Chất lượng thực tế
- 1.559 sản phẩm, 10 cửa hàng.
- Thiếu Item_Weight: 1.463 mẫu (khoảng 17,17%).
- Thiếu Outlet_Size: 2.410 mẫu (khoảng 28,28%).
- Item_Visibility = 0: 526 mẫu; giữ và đánh dấu, chưa tự điền.
- Không có dòng trùng hoàn toàn, không trùng cặp sản phẩm/cửa hàng.
- Mục tiêu không thiếu, không bằng 0, không âm.

## 4. Dữ liệu xử lý
- Chuẩn hóa khoảng trắng; LF/low fat thành Low Fat; reg thành Regular.
- 545 nhãn thay đổi, số dòng vẫn 8.523.
- Thêm source_row, Item_Weight_Missing, Outlet_Size_Missing, Item_Visibility_Zero.
- Còn ô thiếu có chủ đích: học thống kê điền thiếu chỉ từ train trong phần tiền xử lý của EDA.
- Không dùng source_row làm feature; target chỉ dùng làm y.

## 5. Giới hạn
Đây là bán lẻ tại cửa hàng; không xác nhận là e-commerce.
Không có ngày bán nên không làm chia tập/biểu đồ doanh số theo thời gian.
Không có tuổi, giới tính khách hàng hoặc chi phí chiến dịch marketing.
File không ghi rõ đơn vị tiền của Item_Outlet_Sales.
MAPE tính được với mục tiêu dương hiện tại, nhưng nhạy với mục tiêu nhỏ.

## 6. Dữ liệu đầu vào mô hình đã tạo
Chia seed=42, 5.966 train/1.278 validation/1.279 test; chỉ mục lưu split_indices.csv.
Median Item_Weight=12,65 và mode Outlet_Size=Medium đều học trên train.
X ba tập có 30 cột số và không còn thiếu; y vẫn là doanh số gốc, không log-transform.
CSV clean vẫn còn ô thiếu để tách rõ trước/sau imputation và tránh học thống kê toàn corpus.
Danh mục ngoài train được encoder bỏ qua; hiện không có ở validation/test.
Mô hình demo dùng lại danh mục đã học và báo lỗi nếu nhập nhóm chưa biết.
