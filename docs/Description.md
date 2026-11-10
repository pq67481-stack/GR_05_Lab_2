# Mô tả Dự án: Dự đoán Doanh số Bán lẻ BigMart (Lab 2)

## 1. Giới thiệu Dự án
Dự án tập trung vào bài toán **Hồi quy dữ liệu bảng (Tabular Regression)** nhằm dự đoán doanh số bán hàng (`Item_Outlet_Sales`) của từng mặt hàng tại các chi nhánh cửa hàng khác nhau thuộc chuỗi bán lẻ **BigMart**.

* **Mục tiêu chính**: Xây dựng mô hình học máy có khả năng dự đoán chính xác doanh thu bán ra dựa trên các thuộc tính của sản phẩm và đặc trưng của cửa hàng phân phối.
* **Ý nghĩa thực tiễn**: Giúp chuỗi bán lẻ tối ưu hóa kế hoạch nhập hàng, bố trí sản phẩm trên kệ, điều chỉnh mức giá niêm yết (MRP) và cải thiện hiệu quả hoạt động kinh doanh giữa các mô hình cửa hàng.

---

## 2. Tổng quan Tập dữ liệu (BigMart Dataset)
* **Nguồn dữ liệu**: Bộ dữ liệu BigMart Sales Dataset từ cuộc thi bán lẻ thực tế (Analytics Vidhya / Kaggle).
* **Quy mô**: Gồm `8,523` bản ghi (mỗi bản ghi là một cặp sản phẩm – cửa hàng) với `12` thuộc tính ban đầu.

### Phân nhóm thuộc tính:
1. **Thuộc tính sản phẩm (Product Attributes)**:
   * `Item_Identifier`: Mã định danh sản phẩm.
   * `Item_Weight`: Khối lượng sản phẩm (chứa dữ liệu khuyết cần điền).
   * `Item_Fat_Content`: Hàm lượng chất béo (Low Fat, Regular).
   * `Item_Visibility`: Tỷ lệ diện tích trưng bày của sản phẩm trong cửa hàng.
   * `Item_Type`: Danh mục nhóm hàng (16 nhóm: Thực phẩm, Đồ uống, Đồ gia dụng,...).
   * `Item_MRP`: Giá bán lẻ niêm yết tối đa của sản phẩm.

2. **Thuộc tính cửa hàng (Store Attributes)**:
   * `Outlet_Identifier`: Mã định danh cửa hàng (10 cửa hàng).
   * `Outlet_Establishment_Year`: Năm thành lập cửa hàng (trích xuất thành `Outlet_Age`).
   * `Outlet_Size`: Quy mô diện tích cửa hàng (Small, Medium, High).
   * `Outlet_Location_Type`: Phân cấp đô thị đặt cửa hàng (Tier 1, Tier 2, Tier 3).
   * `Outlet_Type`: Loại hình cửa hàng (Grocery Store, Supermarket Type 1, 2, 3).

3. **Biến mục tiêu (Target)**:
   * `Item_Outlet_Sales`: Doanh số bán hàng thực tế tại cửa hàng tương ứng.

> **Quy tắc phân chia & Tiền xử lý (Anti-Data Leakage)**:
> Dữ liệu được phân chia thành **Train (70%) - Validation (15%) - Test (15%)** với `random_state=42`. Toàn bộ các giá trị thống kê điền khuyết (median/mode) và bảng mã danh mục (One-Hot) **chỉ được học duy nhất trên tập Train**, sau đó áp dụng (transform) sang Validation và Test để đảm bảo tính khách quan thực tế.

---

## 3. Hai Mô hình Tiếp cận trong Dự án

Dự án lựa chọn và triển khai 2 thuật toán đại diện cho hai trường phái học máy kinh điển:

### 3.1. Hồi quy tuyến tính (Linear Regression)
* **Đặc trưng**: Mô hình tham số tuyến tính (Parametric Linear Model).
* **Cơ chế**: Tìm nghiệm tối ưu bằng phương pháp bình phương tối thiểu (Ordinary Least Squares - OLS), tối thiểu hóa tổng sai số phần dư (RSS).
* **Ưu điểm**: Đơn giản, thời gian tính toán nhanh, tính giải thích rất cao thông qua các hệ số trọng số ($w$) để biết đặc trưng nào ảnh hưởng tích cực hay tiêu cực đến doanh số.
* **Hạn chế**: Bị giới hạn bởi giả định quan hệ tuyến tính, khó nắm bắt được sự tương tác phức tạp hoặc quan hệ phi tuyến giữa sản phẩm và cửa hàng.

### 3.2. Cây quyết định hồi quy (Decision Tree Regressor)
* **Đặc trưng**: Mô hình phi tham số, phi tuyến tính (Non-parametric Non-linear Model).
* **Cơ chế**: Phân chia không gian dữ liệu thành các vùng chữ nhật nhỏ dựa trên tiêu chuẩn giảm thiểu độ phân tán (MSE reduction). Giá trị dự đoán tại mỗi nút lá là trung bình doanh số của các mẫu trong vùng đó.
* **Ưu điểm**: Bắt được các mối quan hệ phi tuyến tính phức tạp và sự kết hợp tương tác giữa vị trí, quy mô và giá sản phẩm mà không cần giả định trước dạng hàm.
* **Chiến lược kiểm soát**: Sử dụng tập **Validation** để dò tìm độ sâu tối ưu (`max_depth`) và kích thước lá tối thiểu (`min_samples_leaf`), ngăn chặn hiện tượng học vẹt (overfitting).

---

## 4. Mục tiêu Đánh giá và So sánh Mô hình

Dự án đặt mục tiêu đối chiếu trực diện hai mô hình trên cùng một tập kiểm thử (**Test set độc lập**):

1. **Bộ chỉ số đo lường hiệu năng**:
   * **RMSE (Root Mean Squared Error)**: Thước đo đánh giá cốt lõi, phạt nặng các sai số lớn trong bài toán doanh số bán lẻ.
   * **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình, dễ diễn giải theo thang đo tiền tệ thực tế.
   * **MAPE (Mean Absolute Percentage Error)**: Sai số phần trăm tương đối của từng dự đoán.
   * **$R^2$ Score (Hệ số xác định)**: Đánh giá tỷ lệ phương sai doanh số mà mô hình giải thích được.

2. **Bài toán so sánh**:
   * Xác định xem cấu trúc dữ liệu bán lẻ BigMart phù hợp hơn với quan hệ tuyến tính (Linear Regression) hay cấu trúc phân vùng phi tuyến (Decision Tree).
   * Đánh giá sự đánh đổi giữa **tính dễ giải thích (Interpretability)** của Linear Regression và **độ chính xác dự báo (Predictive Accuracy)** của Decision Tree.
   * Kết luận thuật toán nào hoạt động mạnh hơn, ổn định hơn và phù hợp hơn làm mô hình dự báo chính thức cho dự án.

