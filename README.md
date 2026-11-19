# Bank Marketing - Dự đoán đăng ký tiền gửi có kỳ hạn

Đồ án Machine Learning phân tích và dự đoán khả năng khách hàng đăng ký tiền gửi có kỳ hạn (Term Deposit) dựa trên dữ liệu tiếp thị trực tiếp của ngân hàng (UCI Bank Marketing Dataset). Nguồn: https://archive.ics.uci.edu/dataset/222/bank+marketing

> ⚠️ **LƯU Ý QUAN TRỌNG CHO THÀNH VIÊN NHÓM**: 
> - Toàn bộ các file CSV trong đồ án đều được phân cách bằng **dấu chấm phẩy (`;`)**, KHÔNG PHẢI dấu phẩy. Khi đọc bằng pandas **bắt buộc** phải dùng: `pd.read_csv("...", sep=";")`.
> - Tuyệt đối **không dùng thuộc tính `duration`** khi huấn luyện mô hình triển khai thực tế để tránh lỗi Target Leakage (Rò rỉ dữ liệu).

## 1. Cấu trúc thư mục đồ án
```text
Nhom14_SGU_DCT124C3_MachineLearning/
├── Data/
│   ├── raw/                 # Dữ liệu gốc
│   │   └── bank-additional-full.csv
│   └── processed/           # Dữ liệu sạch đã xử lý:
│       ├── train.csv        # Tập Train phân tầng (Stratified 80% - 32,940 mẫu)
│       ├── test.csv         # Tập Test phân tầng (Stratified 20% - 8,236 mẫu)
│       ├── time_train.csv   # Tập Train chia theo chuỗi thời gian (80% đầu)
│       ├── time_test.csv    # Tập Test chia theo chuỗi thời gian (20% sau)
│       └── split_info.json  # Metadata chi tiết về tỷ lệ và thông số chia dữ liệu
├── docs/              # Báo cáo cuối kỳ (.docx) và Slide cuối kỳ (.pptx) (sẽ bổ sung sau)
├── experiments/
│   └── experiment_log.csv   # Nhật ký ghi nhận các lần chạy mô hình
├── figures/                 # 9 biểu đồ EDA chất lượng cao (.png) phục vụ báo cáo
├── notebooks/                          # Các script chạy thực nghiệm và phân tích theo từng bước:
│   ├── 01_data_split_and_check.ipynb   # Làm sạch, lọc trùng, chia Train/Test
│   ├── 02_train_eda.ipynb              # EDA chuyên sâu trên tập Train
│   ├── 03_baseline_models.py           # Chạy thử nghiệm các mô hình cơ sở
│   └── 04_modeling_evaluation.py       # Huấn luyện mô hình nâng cao, tinh chỉnh tham số và đánh giá chi tiết
├── src/                     # Mã nguồn module hóa (chứa các hàm dùng chung cho toàn đồ án):
    ├── baseline_models.py   # Định nghĩa hàm khởi tạo và huấn luyện các mô hình baseline
    ├── evaluate.py          # Tính toán các chỉ số (ROC-AUC, F1, Recall) và xuất biểu đồ đánh giá
    └── preprocess.py        # Xử lý dữ liệu thiếu, mã hóa biến, chuẩn hóa và chia tập (Random & Time-based)
├── requirements.txt         # Danh sách thư viện cần thiết
└── README.md
```

## 2. Thông số dữ liệu chính (từ `split_info.json`)
- **Dữ liệu gốc**: 41,188 mẫu. Đã phát hiện và loại bỏ chính xác **12 dòng trùng lặp** -> Còn lại 41,176 mẫu sạch.
- **Tập Train/Test (Stratified)**: Train 32,940 dòng (80%), Test 8,236 dòng (20%). Tỷ lệ nhãn `yes` trên cả 2 tập đạt đúng chuẩn **11.27%**.
- **Tập Time-based**: Giữ nguyên thứ tự thời gian gốc từ 05/2008 đến 06/2010.
