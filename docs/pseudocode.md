# Mã giả Lab 2

## 1. Chuẩn bị và chia tập
```text
ĐẦU VÀO: bigmart.csv
CHUẨN HÓA chuỗi, nhãn chất béo; đánh dấu thiếu/visibility=0
KIỂM TRA chất lượng, lưu clean và source_row
CHIA train 70%, phần còn lại 30%, seed=42
CHIA phần còn lại thành validation/test bằng nhau, seed=42
HỌC median khối lượng và mode quy mô chỉ từ train
ĐIỀN thiếu ba tập bằng các giá trị đã học
HỌC danh mục One-Hot chỉ từ train, bỏ nhóm tham chiếu
BIẾN ĐỔI validation/test bằng encoder đã học
BỎ target, source_row và hai mã định danh khỏi X
LƯU X/y, chỉ mục và metadata để hai model dùng chung
```

## 2. OLS - tự viết
```text
ĐẦU VÀO: X_train(N×p), y_train(N), X_new(M×p)
ĐẦU RA: prediction(M)
INIT(): ready ← SAI; intercept, coef ← RỖNG
BUILD_MODEL(): xóa intercept/coef; ready ← ĐÚNG
FIT(X, y):
    NẾU chưa ready: BÁO LỖI
    KIỂM TRA X/y hợp lệ và hữu hạn
    A ← GHÉP_CỘT(1, X)
    w ← GIẢI_BÌNH_PHƯƠNG_TỐI_THIỂU(A, y)
    intercept ← w[0]; coef ← w[1:]
PREDICT(X_new):
    NẾU chưa fit: BÁO LỖI
    KIỂM TRA đúng số cột
    TRẢ VỀ intercept + X_new × coef
```

## 3. Cây hồi quy - dùng thư viện
```text
INIT(depth, min_leaf, seed): lưu cấu hình; model ← RỖNG
BUILD_MODEL(): model ← DecisionTreeRegressor(cấu hình); fitted ← SAI
FIT(X, y):
    NẾU chưa build: BÁO LỖI
    GỌI model.fit(X, y)
    # Thư viện chọn cách chia giảm sai số bình phương, lặp đến điều kiện dừng
    # Mỗi lá lưu trung bình mục tiêu các mẫu train tại lá
    fitted ← ĐÚNG
PREDICT(X_new):
    NẾU chưa fitted: BÁO LỖI
    TRẢ VỀ model.predict(X_new)  # đi theo điều kiện tới lá
VỚI depth TRONG [3,5,8,None]:
    VỚI min_leaf TRONG [5,10,20]:
        FIT trên train; TÍNH RMSE validation
CHỌN cấu hình có RMSE validation thấp nhất
```

## 4. Đánh giá cuối
```text
baseline ← trung bình y_train
GIỮ model đã fit train và cấu hình đã chốt
PREDICT trên cùng test cho hai model và baseline
TÍNH MAE, MSE, RMSE, MAPE
LƯU dự đoán, source_row, sai số; VẼ và PHÂN TÍCH
KHÔNG dùng kết quả test để sửa seed/tham số
```
