# Thiết Kế Hệ Thống Mô Hình (System Design)

Tài liệu này thiết kế khung kiến trúc tổng quan cho hệ thống học máy giải quyết bài toán hồi quy dự đoán doanh số bán lẻ BigMart (`Item_Outlet_Sales`). Thiết kế mang tính lý thuyết, chuẩn hóa quy trình và độc lập với thuật toán cụ thể (Model-agnostic).

---

## I. Purpose (Mục đích thiết kế)
1. **Chuẩn hóa quy trình**: Thiết lập một luồng xử lý học máy hoàn chỉnh (End-to-End Machine Learning Pipeline) từ dữ liệu thô đến mô hình hoàn thiện.
2. **Triệt tiêu rò rỉ dữ liệu (Zero Data Leakage)**: Đảm bảo tính toàn vẹn khoa học bằng cách cô lập tuyệt đối thông tin giữa quá trình huấn luyện và đánh giá.
3. **Khả năng tái lập (Reproducibility)**: Cố định các tham số phân chia và biến đổi để bất kỳ ai cũng có thể tái tạo cùng một kết quả trên cùng dữ liệu.
4. **Khung so sánh công bằng (Fair Comparison)**: Cung cấp một môi trường đánh giá nhất quán trên cùng tập dữ liệu để đối chiếu hiệu năng giữa các phương pháp tiếp cận khác nhau.

---

## II. Input / Output (Đầu vào & Đầu ra của Hệ thống)

### 1. Input (Dữ liệu đầu vào)
Dữ liệu đầu vào là dạng bảng (Tabular Data) mô tả các thực thể kinh doanh bán lẻ:
* **Thuộc tính số (Numerical Features)**: Các biến định lượng liên tục và rời rạc như khối lượng sản phẩm, mức độ trưng bày trên quầy kệ, giá bán lẻ niêm yết tối đa (MRP), tuổi hoạt động của cửa hàng.
* **Thuộc tính phân loại (Categorical Features)**: Các biến định danh phi số như nhóm hàng hóa, hàm lượng chất béo, quy mô cửa hàng, phân loại đô thị và loại hình kinh doanh (tiệm tạp hóa, siêu thị các cấp).
* **Mã định danh (Identifiers)**: Mã sản phẩm và mã cửa hàng — chỉ dùng để tra cứu truy vết dữ liệu gốc, được loại bỏ khỏi không gian đặc trưng của mô hình nhằm tránh overfitting.

### 2. Output (Kết quả đầu ra)
* **Giá trị dự đoán liên tục ($\hat{y}$)**: Doanh thu bán ra kỳ vọng (`Item_Outlet_Sales`) tương ứng với từng mặt hàng tại một chi nhánh cửa hàng cụ thể.
* **Báo cáo đo lường sai số**: Bảng tổng kết các chỉ số hiệu năng định lượng (RMSE, MAE, MAPE, $R^2$) và các đồ thị phân tích phần dư (Residual Analysis).
* **Mô hình triển khai (Inference Model)**: Đối tượng tiền xử lý và mô hình đã đóng băng tham số, sẵn sàng nhận dữ liệu mới để đưa ra dự đoán thời gian thực.

---

## III. Validation (Chiến lược Kiểm định & Thước đo)

### 1. Chiến lược phân tách dữ liệu (Data Splitting Strategy)
Hệ thống sử dụng chiến lược **Holdout Validation 3 tập độc lập**:
* **Tập Huấn luyện (Train set - 70%)**: Tập dữ liệu lớn nhất dùng để mô hình học các trọng số và học các tham số thống kê làm sạch dữ liệu.
* **Tập Kiểm định (Validation set - 15%)**: Đóng vai trò là tập "dữ liệu tương lai giả lập", dùng để dò tìm siêu tham số tối ưu, điều chỉnh độ phức tạp của mô hình và kiểm soát hiện tượng Overfitting/Underfitting.
* **Tập Kiểm thử (Test set - 15%)**: Tập dữ liệu được "giấu kín" hoàn toàn trong suốt quá trình xây dựng mô hình, chỉ dùng duy nhất một lần ở bước nghiệm thu cuối cùng để đo lường năng lực tổng quát hóa (Generalization).

### 2. Thước đo đánh giá (Evaluation Metrics)
* **RMSE (Root Mean Squared Error)**: Thước đo đánh giá chính; phạt nặng các sai số dự báo lớn.
* **MAE (Mean Absolute Error)**: Đo lường mức độ lệch tuyệt đối trung bình, dễ diễn giải trực tiếp theo đơn vị tiền tệ thực tế.
* **MAPE (Mean Absolute Percentage Error)**: Đo lường tỷ lệ phần trăm sai lệch tương đối so với giá trị thực tế.
* **$R^2$ Score (Hệ số xác định)**: Tỷ lệ phương sai của biến mục tiêu mà mô hình giải thích được so với mốc đối chiếu cơ bản (Baseline).

---

## IV. How to Do (Quy trình Thực hiện Tổng quan)

Quy trình vận hành hệ thống được thực hiện tuần tự qua 5 giai đoạn lý thuyết:

```text
┌─────────────────────────┐
│   1. DỮ LIỆU THÔ        │
│   (Raw BigMart Data)    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   2. CHIA TẬP ĐỘC LẬP   │ ──► [Train 70%] ──► [Validation 15%] ──► [Test 15%]
│   (Split First)         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   3. TIỀN XỬ LÝ &       │ ──► Fit trên Train ──► Đóng băng tham số ──► Transform Val/Test
│   ĐÓNG BĂNG THAM SỐ     │     (Imputation, Scaling, One-Hot Encoding)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   4. HUẤN LUYỆN &       │ ──► Train trên Train set
│   CHỌN LỰA MÔ HÌNH      │ ──► Tinh chỉnh siêu tham số trên Validation set
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   5. KIỂM THỬ & DEMO    │ ──► Đánh giá khách quan trên Test set
│   (Evaluation & Demo)   │ ──► Dự đoán mẫu sản phẩm mới (Inference Demo)
└─────────────────────────┘
```

### 1. Thu thập & Tiền xử lý dữ liệu thô (Data Collection & Stateless Cleaning)
* Kiểm tra tính toàn vẹn và cấu trúc của dữ liệu ban đầu.
* Thực hiện các phép làm sạch cục bộ không trạng thái (Stateless Processing): Chuẩn hóa các nhãn phân loại không nhất quán (chữ hoa, chữ thường, viết tắt) và thực hiện Feature Engineering cơ bản (ví dụ: chuyển đổi năm thành lập thành số năm hoạt động).
* Loại bỏ các trường định danh thuần túy không mang tính dự báo.

### 2. Phân chia tập dữ liệu (Data Splitting - Chống rò rỉ)
* Tách dữ liệu ngay từ đầu thành 3 phần Train (70%), Validation (15%), và Test (15%) với một hạt giống ngẫu nhiên cố định (`seed=42`).
* Đảm bảo tính rời nhau hoàn toàn giữa các tập để loại bỏ mọi nguy cơ rò rỉ thông tin trước khi áp dụng bất kỳ thuật toán thống kê nào.

### 3. Học tham số & Đóng băng chuyển đổi (Feature Transformation & Parameter Freezing)
* **Chỉ học từ Train (Fit on Train)**: Toàn bộ các giá trị thống kê mô tả (trung vị, yếu vị/mode, độ lệch chuẩn, bảng từ điển danh mục One-Hot) chỉ được tính toán duy nhất trên tập Train.
* **Đóng băng tham số (Freeze)**: Khóa cố định các giá trị này thành một bộ quy tắc chuyển đổi bất biến.
* **Áp dụng cho Val & Test (Transform Only)**: Sử dụng các tham số đóng băng từ Train để điền khuyết và chuyển đổi cho Validation set và Test set, bảo đảm dữ liệu kiểm thử được xử lý y như dữ liệu thực tế trong tương lai.

### 4. Huấn luyện & Tối ưu hóa mô hình (Training & Validation Tuning)
* Huấn luyện mô hình toán học trên tập Train bằng ma trận đặc trưng đã chuẩn hóa.
* Đánh giá sai số dự đoán trên Validation set để tìm cấu hình mô hình có sai số nhỏ nhất, tránh hiện tượng quá khớp (Overfitting) hoặc chưa khớp (Underfitting).
* Chốt cấu hình mô hình tốt nhất dựa trên hiệu năng Validation.

### 5. Kiểm thử cuối cùng & Trình diễn (Testing & Demo Inference)
* **Kiểm thử mù (Testing)**: Đưa tập Test (mô hình chưa từng nhìn thấy) qua bộ chuyển đổi tham số đóng băng và cho mô hình dự đoán. Tính toán các chỉ số RMSE, MAE, MAPE, $R^2$ để báo cáo năng lực thực tế.
* **Trình diễn dự đoán (Demo)**: Xây dựng quy trình suy luận (Inference Pipeline) nhận đầu vào là thông tin của một sản phẩm/cửa hàng mới bất kỳ, tự động làm sạch, chuyển đổi theo tham số đóng băng và xuất ra con số doanh số dự báo kèm độ tin cậy.

