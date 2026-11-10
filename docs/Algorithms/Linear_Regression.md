# Mô Hình Hồi Quy Tuyến Tính (Linear Regression - OLS)

Tài liệu này trình bày chi tiết về cơ sở lý thuyết, công thức toán học nghiệm giải tích, mã giả thuật toán và cấu trúc tham số của mô hình **Hồi quy tuyến tính bằng phương pháp Bình phương tối thiểu (Ordinary Least Squares - OLS)** trong bài toán dự đoán doanh số bán lẻ BigMart (`Item_Outlet_Sales`).

---

## 1. Giới thiệu Thuật toán & Vai trò

* **Bản chất**: Linear Regression OLS là mô hình học máy có giám sát dạng tham số tuyến tính (Parametric Linear Model). Mô hình giả định mối quan hệ giữa biến mục tiêu liên tục $y$ và các đặc trưng đầu vào $X$ là một tổ hợp tuyến tính.
* **Vai trò trong bài toán**:
  * **Mô hình chuẩn đối chiếu (Benchmark Model)**: Đóng vai trò là mốc chuẩn mực để so sánh hiệu năng với mô hình phi tuyến (Decision Tree Regressor) trên cùng tập dữ liệu.
  * **Tính minh bạch và giải thích cao (High Interpretability)**: Khác với các mô hình "hộp đen", OLS cung cấp trực tiếp vector trọng số $w$. Mỗi trọng số $w_j$ thể hiện trực tiếp mức độ và chiều hướng (tăng/giảm) tác động của biến $x_j$ lên doanh số `Item_Outlet_Sales`, phục vụ trực tiếp cho việc phân tích nghiệp vụ kinh doanh.

---

## 2. Cơ sở Toán học Chuyên sâu cho Hồi Quy Tuyến Tính (OLS Mathematics)

### 2.1. Phương trình Mô hình (Hypothesis)
Với mỗi mẫu dữ liệu gồm $p$ đặc trưng đã qua xử lý ($p = 30$):
$$x = [x_1, x_2, \ldots, x_p]^T \in \mathbb{R}^p$$

Hàm hồi quy tuyến tính xác định doanh số dự đoán $\hat{y}$ theo công thức:
$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_p x_p = w_0 + \sum_{j=1}^p w_j x_j
$$
Trong đó:
* $w_0 \in \mathbb{R}$: Hệ số tự do (Bias hay Intercept), đại diện cho mức doanh số cơ sở khi tất cả các đặc trưng bằng 0.
* $w_j \in \mathbb{R}$ ($j = 1, \ldots, p$): Trọng số (Weight / Slope Coefficient) ứng với đặc trưng thứ $j$.

Để thuận tiện cho tính toán đại số ma trận, ta mở rộng vector đặc trưng bằng cách thêm cột giá trị $1$ (cột bias):
$$\tilde{x} = [1, x_1, x_2, \ldots, x_p]^T \in \mathbb{R}^{p+1}, \quad w = [w_0, w_1, \ldots, w_p]^T \in \mathbb{R}^{p+1}$$

Khi đó phương trình dự đoán thu gọn thành tích vô hướng:
$$
\hat{y} = \tilde{x}^T w
$$

Đối với toàn bộ tập huấn luyện gồm $n$ mẫu ($n = 5,966$):
$$
\hat{y} = X w
$$
* $X \in \mathbb{R}^{n \times (p+1)}$: Ma trận thiết kế (Design Matrix), với cột đầu tiên là vector toàn số $1$.
* $y \in \mathbb{R}^{n \times 1}$: Vector nhãn mục tiêu thực tế (`Item_Outlet_Sales`).
* $\hat{y} \in \mathbb{R}^{n \times 1}$: Vector giá trị dự đoán của mô hình.

---

### 2.2. Hàm mất mát Bình phương tối thiểu (OLS Loss Function)
Phương pháp OLS tìm kiếm bộ trọng số $w$ sao cho tổng bình phương sai số phần dư (Residual Sum of Squares - RSS) trên tập Train là nhỏ nhất:
$$
\mathcal{L}(w) = \text{RSS}(w) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \|y - X w\|_2^2
$$

Khai triển đại số ma trận:
$$
\mathcal{L}(w) = (y - X w)^T (y - X w) = y^T y - 2 w^T X^T y + w^T X^T X w
$$

---

### 2.3. Nghiệm Giải tích và Tối ưu hóa Ổn định (Normal Equation & SVD)
Để tìm cực tiểu toàn cục của hàm lồi $\mathcal{L}(w)$, ta lấy vi phân ma trận theo vector $w$ và giải phương trình đạo hàm triệt tiêu:
$$
\nabla_w \mathcal{L}(w) = \frac{\partial \mathcal{L}(w)}{\partial w} = -2 X^T y + 2 X^T X w = 0
$$
$$
\iff X^T X w = X^T y \quad \text{(Hệ phương trình chuẩn - Normal Equation)}
$$

#### 1. Trường hợp lý tưởng (Không đa cộng tuyến):
Nếu ma trận hiệp phương sai mẫu $X^T X$ khả nghịch:
$$
w^* = (X^T X)^{-1} X^T y
$$

#### 2. Kỹ thuật ổn định số học trong thực tế (SVD Pseudo-Inverse):
Sau khi mã hóa One-Hot các biến danh mục (`Outlet_Type`, `Item_Type`), ma trận $X$ có thể xuất hiện hiện tượng đa cộng tuyến hoặc gần suy biến (Ill-conditioned), khiến việc tính ma trận nghịch đảo $(X^T X)^{-1}$ bị mất ổn định số học.

Giải thuật áp dụng phân tích suy biến số kỳ dị **SVD (Singular Value Decomposition)**:
$$
X = U \Sigma V^T
$$
Vector trọng số tối ưu được tính toán thông qua ma trận giả nghịch đảo Moore-Penrose $X^+$:
$$
w^* = X^+ y = V \Sigma^+ U^T y
$$
*(Trong đó $\Sigma^+$ thu được bằng cách nghịch đảo các giá trị kỳ dị khác 0 và loại bỏ các giá trị kỳ dị nhỏ hơn ngưỡng cắt `rcond=1e-10`)*.

---

## 3. Bản đồ Tham số Mô hình

| Thành phần | Ký hiệu Toán học | Ý nghĩa & Dạng dữ liệu | Vai trò |
| :--- | :--- | :--- | :--- |
| **Model Parameter Input (Dữ liệu học)** | $X_{train} \in \mathbb{R}^{n \times p}$<br>$y_{train} \in \mathbb{R}^{n \times 1}$ | - Ma trận đặc trưng Train ($n = 5,966, p = 30$).<br>- Vector nhãn mục tiêu doanh số thực tế. | Cung cấp không gian đặc trưng và nhãn thực tế để mô hình giải bài toán tối ưu bình phương sai số. |
| **Model Hyperparameters (Siêu tham số)** | `fit_intercept` (bool)<br>`rcond` ($\epsilon \approx 10^{-10}$) | - Quyết định có thêm hệ số tự do $w_0$ không.<br>- Ngưỡng cắt số kỳ dị SVD để chống đa cộng tuyến. | Thiết lập điều kiện giải thuật toán trước khi học nhằm đảm bảo tính ổn định số học. |
| **Model Parameter Output (Tham số được học)** | $w^* = [w_0, w_1, \ldots, w_p]^T \in \mathbb{R}^{p+1}$ | **Vector trọng số tối ưu**, gồm:<br>+ Bias: $w_0 \in \mathbb{R}$<br>+ Trọng số đặc trưng: $w_1, \ldots, w_p \in \mathbb{R}$ | **Kết quả cốt lõi của bước Train**. Bộ tham số được ĐÓNG BĂNG để đưa vào thực hiện suy luận. |
| **Inference Input (Đầu vào suy luận)** | $X_{val} \in \mathbb{R}^{m_{val} \times p}$<br>$X_{test} \in \mathbb{R}^{m_{test} \times p}$ | Ma trận đặc trưng của tập Validation hoặc Test đã được chuẩn hóa. | Dữ liệu mẫu mới cần ước tính doanh số. |
| **Inference Output (Dự đoán đầu ra)** | $\hat{y} = X_{design} \cdot w^* \in \mathbb{R}^{m \times 1}$ | Vector doanh số bán lẻ dự đoán liên tục. | Kết quả đầu ra dùng để tính toán sai số, đánh giá chất lượng mô hình và hỗ trợ ra quyết định. |

---

## 4. Mã giả Thuật toán (Pseudocode)

### 4.1. Thuật toán Huấn luyện (Training Procedure - OLS via SVD)
```text
Thuật toán: Train_Linear_Regression_OLS
Đầu vào:
    - X_train: Ma trận dữ liệu huấn luyện kích thước (n, p)
    - y_train: Vector mục tiêu kích thước (n, 1)
    - fit_intercept: boolean (mặc định = True)
    - rcond: float (mặc định = 1e-10)

Đầu ra:
    - w: Vector trọng số tối ưu đã đóng băng kích thước (p + 1, 1)

Các bước thực hiện:
    1. Kiểm tra kích thước ma trận: Đảm bảo số dòng n >= số cột p.
    2. Nếu fit_intercept == True:
           Tạo vector cột bias gồm n số 1: ones = [1, 1, ..., 1]^T (n, 1)
           Ghép ones vào cột đầu tiên của X_train:
           X_design = [ones, X_train]   # Kích thước (n, p+1)
       Ngược lại:
           X_design = X_train           # Kích thước (n, p)

    3. Phân rã ma trận X_design bằng SVD và giải bài toán bình phương tối thiểu:
           w, residuals, rank, s = Solve_Least_Squares_SVD(X_design, y_train, rcond)

    4. Phân tách và lưu trữ tham số:
           Nếu fit_intercept == True:
               bias_w0 = w[0]
               weights = w[1:]
           Ngược lại:
               bias_w0 = 0.0
               weights = w

    5. Khóa (Freeze) vector trọng số w, ghi nhận trạng thái mô hình 'Trained & Ready'.
    6. Trả về mô hình OLS đã hoàn thiện.
```

### 4.2. Thuật toán Suy luận & Dự đoán (Prediction Procedure)
```text
Thuật toán: Predict_Linear_Regression
Đầu vào:
    - X_new: Ma trận đặc trưng của các mẫu mới (Validation, Test hoặc thực tế), kích thước (m, p)
    - w: Vector trọng số đã học và đóng băng [w_0, w_1, ..., w_p]^T

Đầu ra:
    - y_pred: Vector giá trị doanh số dự đoán, kích thước (m, 1)

Các bước thực hiện:
    1. Kiểm tra tính toàn vẹn: Đảm bảo số cột của X_new bằng đúng p (khớp với Train).
    2. Bổ sung cột bias vào ma trận đầu vào:
           X_new_design = [ones(m, 1), X_new]   # Kích thước (m, p+1)
    3. Thực hiện phép nhân ma trận đại số tuyến tính:
           y_pred = X_new_design * w
    4. Trả về vector kết quả dự đoán y_pred.
```

---

## 5. Thước đo Đánh giá Mô hình (Evaluation Metrics)

Sau khi có vector dự đoán $\hat{y}$ và giá trị thực tế $y$, mô hình được đánh giá toàn diện dựa trên 4 chỉ số hồi quy:

1. **Root Mean Squared Error (RMSE)**:
   $$
   \text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2}
   $$
2. **Mean Absolute Error (MAE)**:
   $$
   \text{MAE} = \frac{1}{m} \sum_{i=1}^m |y_i - \hat{y}_i|
   $$
3. **Mean Absolute Percentage Error (MAPE)**:
   $$
   \text{MAPE} = \frac{100\%}{m} \sum_{i=1}^m \left| \frac{y_i - \hat{y}_i}{y_i} \right|
   $$
4. **Hệ số xác định ($R^2$ Score)**:
   $$
   R^2 = 1 - \frac{\sum_{i=1}^m (y_i - \hat{y}_i)^2}{\sum_{i=1}^m (y_i - \bar{y})^2}
   $$
   *(Trong đó $\bar{y}$ là trung bình cộng doanh số thực tế của tập đang đánh giá)*.

---

## 6. Đánh giá Mở rộng Kỹ thuật

* **Độ phức tạp tính toán**:
  * Huấn luyện (Train): $\mathcal{O}(n p^2 + p^3)$. Với bài toán BigMart ($n = 5,966, p = 30$), nghiệm được tính toán tức thì trong chưa đầy $0.05$ giây.
  * Suy luận (Predict): $\mathcal{O}(m \cdot p)$, chỉ gồm một phép nhân ma trận đơn giản, cực kỳ nhẹ và phù hợp triển khai trên các thiết bị tài nguyên hạn chế.
* **Ưu điểm**:
  * Nghiệm giải tích duy nhất (Global Optimum), không phụ thuộc vào khởi tạo ngẫu nhiên hay tỷ lệ học (Learning rate).
  * Tính giải thích trực tiếp theo ý nghĩa nghiệp vụ bán lẻ.
* **Hạn chế**:
  * Giả định quan hệ thuần túy tuyến tính, bỏ sót các tương tác phi tuyến phức tạp trong dữ liệu tiêu dùng bán lẻ.
