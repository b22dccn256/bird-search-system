# 🐦 Bird Search System

Hệ thống tìm kiếm chim dựa trên ảnh sử dụng machine learning để tìm những loài chim tương tự.

## Cấu trúc Dự án

```
bird-search-system/
│
├── data/
│   ├── raw/                     # 500+ ảnh gốc (chưa xử lý)
│   ├── processed/               # Ảnh đã resize, blur, tách nền
│   └── test_input/              # Ảnh truy vấn mới (bạn tự chụp/tải)
│
├── database/
│   ├── birds.db                 # SQLite: lưu metadata (tên, kích thước, đường dẫn)
│   ├── faiss.index              # FAISS: chỉ mục vector để tìm kiếm siêu nhanh
│   ├── features.npy             # Vector đặc trưng thô (trước khi chuẩn hóa)
│   ├── scaler.pkl               # Model StandardScaler đã fit
│   └── filenames.txt            # Ánh xạ index trong FAISS ↔ tên file ảnh
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Cấu hình đường dẫn & tham số
│   ├── preprocessing.py         # Resize, Gaussian Blur, GrabCut
│   ├── feature_extractor.py     # HSV + HOG + LBP trên lưới 4x4
│   ├── build_database.py        # Pipeline xây dựng CSDL & FAISS index
│   └── search_engine.py         # Logic tải index, chuẩn hóa query, tìm top-5
│
├── notebooks/                   # (Tùy chọn) Jupyter notebooks
│
├── app.py                       # Ứng dụng Streamlit
├── requirements.txt             # Danh sách thư viện cần cài
└── README.md                    # Mô tả dự án
```

## Cài đặt

### Yêu cầu
- Python 3.8+
- pip

### Các bước cài đặt

1. **Clone hoặc tải dự án:**
   ```bash
   cd bird-search-system
   ```

2. **Tạo virtual environment (tùy chọn nhưng được khuyến nghị):**
   ```bash
   python -m venv venv
   
   # Trên Windows
   venv\Scripts\activate
   
   # Trên macOS/Linux
   source venv/bin/activate
   ```

3. **Cài đặt các thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

## Hướng dẫn Sử dụng

### 1. Chuẩn bị Dữ liệu

- Đặt các ảnh chim gốc vào thư mục `data/raw/`
- Các ảnh sẽ được xử lý và lưu vào `data/processed/`

### 2. Xây dựng Database

```bash
python src/build_database.py
```

Điều này sẽ:
- Trích đặc trưng từ tất cả ảnh
- Chuẩn hóa các vector đặc trưng
- Tạo FAISS index
- Lưu các kết quả vào thư mục `database/`

### 3. Chạy Ứng dụng

```bash
streamlit run app.py
```

Truy cập ứng dụng tại: `http://localhost:8501`

## Bộ Thư viện Chính

- **Streamlit**: Xây dựng giao diện web
- **OpenCV**: Xử lý ảnh
- **NumPy**: Tính toán số học
- **FAISS**: Tìm kiếm vector hiệu năng cao
- **scikit-learn**: Machine learning tools (StandardScaler)
- **Pillow**: Xử lý ảnh

## Tính Năng

- ✅ Tải lên ảnh chim
- ✅ Tìm kiếm chim tương tự dựa trên đặc trưng hình ảnh
- ✅ Hiển thị độ tương tự
- ✅ Giao diện thân thiện người dùng

## Phát Triển Tiếp

- [ ] Thêm metadata về loài chim (tên, mô tả, phân loại)
- [ ] Tích hợp voice search
- [ ] Thêm thống kê sử dụng
- [ ] Đóng gói thành ứng dụng desktop/mobile

## Liên Hệ & Góp Ý

Để báo cáo lỗi hoặc đề xuất tính năng, vui lòng mở issue trên repository.

## Giấy Phép

MIT License - Xem file LICENSE để biết thêm chi tiết.
