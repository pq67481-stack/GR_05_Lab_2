# Mô Hình Cây Quyết Định Hồi Quy (Decision Tree Regressor)

Tài liệu này trình bày chi tiết về cơ sở lý thuyết, công thức toán học phân nhánh hồi quy (CART Regressor), mã giả thuật toán và cấu trúc tham số của mô hình **Cây quyết định hồi quy (Decision Tree Regressor)** trong bài toán dự đoán doanh số bán lẻ BigMart (`Item_Outlet_Sales`).

---

## 1. Giới thiệu Thuật toán & Vai trò

* **Bản chất**: Decision Tree Regressor là mô hình học máy có giám sát phi tham số, phi tuyến tính (Non-parametric Non-linear Model) theo thuật toán CART (Classification and Regression Trees). Khác với bài toán phân loại (dùng Gini hay Entropy để chia nhãn rời rạc), **bài toán hồi quy (Regressor)** phân chia không gian dữ liệu nhằm cực tiểu hóa độ phân tán của giá trị liên tục (Variance / MSE Reduction).
* **Vai trò trong bài toán**:
  * **Mô hình học quan hệ phi tuyến (Non-linear Candidate)**: Đóng vai trò là đối trọng chính với Linear Regression OLS để xác định xem cấu trúc dữ liệu bán lẻ BigMart có phù hợp hơn với quan hệ phi tuyến và tương tác đa chiều hay không.
  * **Tự động bắt tương tác đặc trưng (Feature Interaction)**: Tự động kết hợp logic giữa các thuộc tính (ví dụ: `Outlet_Type` = Supermarket Type 3 kết hợp với `Item_MRP` cao) mà không cần can thiệp tạo đặc trưng thủ công.
  * **Cơ chế kiểm soát quá khớp (Tuning Overfitting)**: Cung cấp các siêu tham số linh hoạt (`max_depth`, `min_samples_leaf`) để tối ưu hóa độ sâu và độ khái quát trên tập Validation trước khi kiểm thử.

---

## 2. Cơ sở Toán học Chuyên sâu cho Cây Hồi Quy (Regression Tree Mathematics)

### 2.1. Phân vùng Không gian và Ước lượng Giá trị tại Nút Lá
Không gian đặc trưng $p$ chiều ($p = 30$) được cây quyết định phân chia thành $M$ siêu hộp trực giao (hyper-rectangles) không giao nhau: $R_1, R_2, \ldots, R_M$.

Hàm dự đoán hồi quy của toàn bộ cây được biểu diễn dưới dạng toán học:
$$
f(x) = \sum_{m=1}^M c_m \cdot \mathbb{I}(x \in R_m)
$$
Trong đó:
* $\mathbb{I}(x \in R_m) = 1$ nếu mẫu $x$ rơi vào vùng $R_m$, và bằng $0$ nếu ngược lại.
* $c_m \in \mathbb{R}$: Hằng số dự đoán doanh số ứng với vùng lá $R_m$.

#### Chứng minh toán học cho giá trị tại nút lá ($c_m$):
Để hàm dự đoán tối ưu theo tiêu chuẩn bình phương sai số nhỏ nhất (L2 Loss) trên vùng lá $R_m$:
$$
\min_{c_m} \sum_{i \in R_m} (y_i - c_m)^2
$$
Lấy đạo hàm bậc nhất theo $c_m$ và cho bằng 0:
$$
\frac{d}{dc_m} \sum_{i \in R_m} (y_i - c_m)^2 = -2 \sum_{i \in R_m} (y_i - c_m) = 0 \implies \sum_{i \in R_m} y_i = |R_m| \cdot c_m
$$
$$
\implies \hat{c}_m = \bar{y}_{R_m} = \frac{1}{|R_m|} \sum_{i \in R_m} y_i
$$
> **Kết luận toán học**: Trong bài toán hồi quy (Regressor), giá trị dự đoán tối ưu tại mỗi nút lá chính là **trung bình cộng số học của các nhãn mục tiêu thực tế** của tất cả các mẫu rơi vào lá đó.

---

### 2.2. Tiêu chuẩn Phân nhánh Tối ưu cho Hồi quy (Splitting Criterion - MSE / Variance Reduction)
Tại mỗi nút nội bộ chứa tập mẫu $D$ gồm $N$ phần tử ($D \subseteq \text{Train set}$), thuật toán cần chọn ra:
1. Một đặc trưng phân tách: $j \in \{1, 2, \ldots, p\}$.
2. Một ngưỡng cắt: $s \in \mathbb{R}$.

Cặp $(j, s)$ sẽ chia tập $D$ thành 2 tập con nhị phân:
$$
R_{\text{left}}(j, s) = \{x \in D \mid x_j \le s\} \quad \text{và} \quad R_{\text{right}}(j, s) = \{x \in D \mid x_j > s\}
$$

Sai số toàn phương trung bình (MSE) của một vùng dữ liệu $R$ được xác định bởi:
$$
\text{MSE}(R) = \frac{1}{|R|} \sum_{i \in R} (y_i - \bar{y}_R)^2 = \text{Var}(y_R)
$$

Thuật toán duyệt tham lam (Greedy Search) qua tất cả $p$ đặc trưng và các giá trị ngưỡng $s$ để tìm cặp tối ưu $(j^*, s^*)$ sao cho tổng sai số bình phương có trọng số của hai nút con là nhỏ nhất:
$$
(j^*, s^*) = \operatorname{argmin}_{j, s} \left[ \frac{|R_{\text{left}}|}{N} \text{MSE}(R_{\text{left}}) + \frac{|R_{\text{right}}|}{N} \text{MSE}(R_{\text{right}}) \right]
$$
$$
\iff \operatorname{argmin}_{j, s} \left[ \sum_{i \in R_{\text{left}}} (y_i - \bar{y}_{\text{left}})^2 + \sum_{i \in R_{\text{right}}} (y_i - \bar{y}_{\text{right}})^2 \right]
$$

Hoặc tương đương với việc tối đa hóa mức giảm phương sai (Variance Reduction):
$$
\Delta \text{Var}(j, s) = \text{Var}(D) - \left( \frac{|R_{\text{left}}|}{N} \text{Var}(R_{\text{left}}) + \frac{|R_{\text{right}}|}{N} \text{Var}(R_{\text{right}}) \right) \to \max
$$

---

### 2.3. Cơ chế Kiểm soát Overfitting trên Tập Validation
Nếu không có ràng buộc, cây sẽ phát triển sâu cho đến khi mỗi nút lá chỉ chứa 1 mẫu ($|R_m| = 1$), khi đó $\text{MSE}_{\text{train}} = 0$ nhưng cây bị **học vẹt (Overfitting nghiêm trọng)**.

Để kiểm soát cấu trúc hình học của cây, 2 siêu tham số quan trọng được điều chỉnh trên tập Validation:
1. **`max_depth` ($d_{\max}$)**: Giới hạn chiều sâu tối đa của cây (thử nghiệm các mức: 3, 4, 5, 6, 8, 10).
2. **`min_samples_leaf` ($n_{\min}$)**: Số lượng mẫu tối thiểu bắt buộc tại mỗi nút lá (thử nghiệm: 10, 20). Một điểm phân nhánh chỉ được chấp nhận nếu cả hai nhánh con thỏa mãn: $|R_{\text{left}}| \ge n_{\min}$ và $|R_{\text{right}}| \ge n_{\min}$.

Cấu hình tối ưu được lựa chọn khi đạt **$\text{RMSE}_{\text{val}}$ nhỏ nhất** trên tập Validation.

---

## 3. Bản đồ Tham số Mô hình

| Thành phần | Ký hiệu Toán học | Ý nghĩa & Dạng dữ liệu | Vai trò |
| :--- | :--- | :--- | :--- |
| **Model Parameter Input (Dữ liệu học)** | $X_{train} \in \mathbb{R}^{n \times p}$<br>$y_{train} \in \mathbb{R}^{n \times 1}$ | Ma trận đặc trưng Train ($n = 5,966, p = 30$) và nhãn doanh số thực tế. | Cung cấp không gian mẫu để thuật toán tìm kiếm các điểm cắt nhị phân và ước lượng giá trị trung bình tại các lá. |
| **Model Hyperparameters (Siêu tham số)** | `max_depth` ($d_{\max} \in \mathbb{N}^+$)<br>`min_samples_leaf` ($n_{\min} \in \mathbb{N}^+$)<br>`criterion='squared_error'` | Các giới hạn cấu trúc cây để kiểm soát độ phức tạp hình học. | Kiểm soát độ phức tạp của cây, ngăn chặn hiện tượng Overfitting và tối ưu hóa khả năng tổng quát hóa trên dữ liệu chưa thấy. |
| **Learned Model Parameters (Tham số được học)** | $\mathcal{T} = \{\text{Nodes}, \text{Thresholds}, \text{Leaves}\}$ | **Cấu trúc cây nhị phân hoàn chỉnh**, gồm:<br>1. Tập nút nội bộ: Lưu cặp $(j^*, s^*)$ (chỉ số cột và ngưỡng cắt).<br>2. Tập nút lá: Lưu giá trị dự đoán trung bình $\hat{c}_m = \bar{y}_{R_m} \in \mathbb{R}$ và số mẫu $|R_m|$.<br>3. Vector độ quan trọng đặc trưng (Feature Importances). | **Kết quả cốt lõi của bước Train**. Toàn bộ cấu trúc cây được ĐÓNG BĂNG để đưa vào thực hiện suy luận. |
| **Inference Input (Đầu vào suy luận)** | $X_{val} \in \mathbb{R}^{m_{val} \times p}$<br>$X_{test} \in \mathbb{R}^{m_{test} \times p}$ | Ma trận đặc trưng của các mẫu mới cần dự đoán. | Dữ liệu mẫu mới cần ước tính doanh số thông qua việc duyệt cây. |
| **Inference Output (Dự đoán đầu ra)** | $\hat{y} = [\hat{c}_{\text{leaf}(x_1)}, \ldots, \hat{c}_{\text{leaf}(x_m)}]^T$ | Vector các giá trị dự đoán doanh số dạng số thực liên tục. | Kết quả đầu ra dùng để tính toán sai số, đánh giá chất lượng mô hình và hỗ trợ ra quyết định. |

---

## 4. Mã giả Thuật toán (Pseudocode)

### 4.1. Thuật toán Huấn luyện & Dựng Cây Hồi Quy Đệ Quy (CART Regressor)
```text
Thuật toán: Build_Regression_Tree
Đầu vào:
    - Data: Tập dữ liệu tại nút hiện tại (X_node, y_node)
    - current_depth: Độ sâu hiện tại (bắt đầu từ 0 tại nút gốc)
    - max_depth: Chiều sâu tối đa cho phép
    - min_samples_leaf: Số mẫu tối thiểu ở nút lá

Đầu ra:
    - Node: Nút nội bộ (Internal Node) hoặc Nút lá (Leaf Node)

Các bước thực hiện:
    1. Tính số lượng mẫu N = len(Data) và giá trị dự đoán trung bình:
           c_leaf = Mean(y_node)

    2. Kiểm tra ĐIỀU KIỆN DỪNG:
           Nếu (current_depth >= max_depth) HOẶC 
              (N < 2 * min_samples_leaf) HOẶC 
              (Tất cả giá trị y_node hoàn toàn bằng nhau):
               Tạo Leaf_Node(predicted_value = c_leaf, num_samples = N)
               Trả về Leaf_Node

    3. Tìm điểm cắt tối ưu giảm thiểu MSE nhiều nhất (Best Split):
           best_feature = None
           best_threshold = None
           min_total_mse = Infinity

           Với mỗi đặc trưng j từ 1 đến p:
               Lấy danh sách các giá trị duy nhất đã sắp xếp của cột X_node[:, j]
               Với mỗi ngưỡng cắt s (trung điểm giữa hai giá trị liên tiếp):
                   Tách dữ liệu thành:
                       D_left  = {x in Data | x[j] <= s}
                       D_right = {x in Data | x[j] > s}

                   Nếu len(D_left) < min_samples_leaf HOẶC len(D_right) < min_samples_leaf:
                       Bỏ qua (vi phạm điều kiện lá tối thiểu)

                   Tính tổng phương sai sai số của 2 nhánh con:
                       mse_split = Sum_Squared_Errors(D_left) + Sum_Squared_Errors(D_right)

                   Nếu mse_split < min_total_mse:
                       min_total_mse = mse_split
                       best_feature = j
                       best_threshold = s

    4. Nếu không tìm được điểm cắt thỏa mãn điều kiện:
           Tạo Leaf_Node(predicted_value = c_leaf, num_samples = N)
           Trả về Leaf_Node

    5. Thực hiện phân nhánh đệ quy:
           left_child  = Build_Regression_Tree(D_left, current_depth + 1, max_depth, min_samples_leaf)
           right_child = Build_Regression_Tree(D_right, current_depth + 1, max_depth, min_samples_leaf)

           Tạo Internal_Node(feature = best_feature, threshold = best_threshold,
                             left = left_child, right = right_child)
           Trả về Internal_Node
```

### 4.2. Thuật toán Suy luận & Dự đoán (Tree Traversal Inference)
```text
Thuật toán: Predict_Tree_Sample
Đầu vào:
    - Node: Nút gốc của cây đã được đóng băng tham số
    - x: Vector đặc trưng của mẫu sản phẩm/cửa hàng mới kích thước (p,)

Đầu ra:
    - predicted_sales: Doanh số dự đoán

Các bước thực hiện:
    1. Nếu Node là Leaf_Node:
           Trả về Node.predicted_value

    2. So sánh đặc trưng của mẫu với ngưỡng cắt đã học:
           Nếu x[Node.feature] <= Node.threshold:
               Duyệt tiếp sang nhánh trái:
               Trả về Predict_Tree_Sample(Node.left, x)
           Ngược lại:
               Duyệt tiếp sang nhánh phải:
               Trả về Predict_Tree_Sample(Node.right, x)
```

---

## 5. Thước đo Đánh giá Mô hình (Evaluation Metrics)

Tương tự như OLS, mô hình Cây quyết định được đo lường đồng bộ trên tập Test độc lập thông qua 4 chỉ số hồi quy:

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

---

## 6. Đánh giá Mở rộng Kỹ thuật

* **Độ phức tạp tính toán**:
  * Huấn luyện (Train): $\mathcal{O}(p \cdot n \log n \cdot d_{\max})$ do cần sắp xếp các giá trị tại từng mức độ sâu.
  * Suy luận (Predict): $\mathcal{O}(d)$ với $d \le d_{\max}$. Cây quyết định nhị phân với $d = 5$ chỉ tốn tối đa **5 phép so sánh số học** cho mỗi mẫu dự đoán, tốc độ suy luận cực kỳ nhanh (dưới 1 microsecond).
* **Độ quan trọng đặc trưng (Feature Importance)**:
  * Thuật toán tự động đo lường đóng góp của từng đặc trưng dựa trên **tổng mức giảm phương sai (Variance Reduction)** tích lũy qua tất cả các nút mà đặc trưng đó được chọn:
    $$
    \text{Importance}(j) = \sum_{t \in \text{Internal Nodes}: v(t) = j} \left( I(D_t) - I(D_{t,\text{left}}) - I(D_{t,\text{right}}) \right)
    $$
  * Giúp giải thích biến nào (ví dụ: `Item_MRP` hay `Outlet_Type`) đóng vai trò quyết định nhất trong việc phân hóa doanh số bán lẻ.
* **Ưu điểm**:
  * Tự động xử lý quan hệ phi tuyến và tương tác giữa các thuộc tính.
  * Không nhạy cảm với thang đo của biến (không bắt buộc chuẩn hóa StandardScaler).
* **Hạn chế**:
  * Hàm dự đoán có dạng bậc thang rời rạc (Piecewise Constant), không thể ngoại suy vượt ra ngoài khoảng giá trị doanh số min/max đã gặp trong tập Train.
