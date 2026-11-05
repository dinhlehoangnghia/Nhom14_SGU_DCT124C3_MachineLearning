# Bank Marketing - Dự đoán đăng ký tiền gửi có kỳ hạn

Dự án Machine Learning phân tích và dự đoán khả năng khách hàng đăng ký tiền gửi có kỳ hạn (Term Deposit) dựa trên dữ liệu tiếp thị trực tiếp của ngân hàng (UCI Bank Marketing Dataset).

> ⚠️ **LƯU Ý QUAN TRỌNG CHO THÀNH VIÊN NHÓM**: 
> - Toàn bộ các file CSV trong dự án đều được phân cách bằng **dấu chấm phẩy (`;`)**, KHÔNG PHẢI dấu phẩy. Khi đọc bằng pandas bắt buộc phải dùng: `pd.read_csv("...", sep=";")`.
> - Tuyệt đối **không dùng thuộc tính `duration`** khi huấn luyện mô hình triển khai thực tế để tránh lỗi Target Leakage (Rò rỉ dữ liệu).

## 1. Cấu trúc thư mục dự án
```text
Bank_Marketing_project/
├── Data/
│   ├── raw/                 # Dữ liệu gốc (bank-additional-full.csv)
│   └── processed/           # Dữ liệu sạch đã xử lý:
│       ├── train.csv        # Tập Train phân tầng (Stratified 80% - 32,940 mẫu)
│       ├── test.csv         # Tập Test phân tầng (Stratified 20% - 8,236 mẫu)
│       ├── time_train.csv   # Tập Train chia theo chuỗi thời gian (80% đầu)
│       ├── time_test.csv    # Tập Test chia theo chuỗi thời gian (20% sau)
│       └── split_info.json  # Metadata chi tiết về tỷ lệ và thông số chia dữ liệu
├── docs/                    # Báo cáo (.docx) và Slide (.pptx)
├── experiments/
│   └── experiment_log.csv   # Nhật ký ghi nhận các lần chạy mô hình
├── figures/                 # 8 biểu đồ EDA chất lượng cao (.png) phục vụ báo cáo
├── notebooks/
│   ├── 01_data_split_and_check.ipynb   # Làm sạch, lọc trùng, chia Train/Test
│   └── 02_train_eda.ipynb              # EDA chuyên sâu trên tập Train
├── requirements.txt         # Danh sách thư viện cần thiết
└── README.md
```

## 2. Thông số dữ liệu chính (từ `split_info.json`)
- **Dữ liệu gốc**: 41,188 mẫu. Đã phát hiện và loại bỏ chính xác **12 dòng trùng lặp** -> Còn lại 41,176 mẫu sạch.
- **Tập Train/Test (Stratified)**: Train 32,940 dòng (80%), Test 8,236 dòng (20%). Tỷ lệ nhãn `yes` trên cả 2 tập đạt đúng chuẩn **11.27%**.
- **Tập Time-based**: Giữ nguyên thứ tự thời gian gốc từ 05/2008 đến 06/2010.