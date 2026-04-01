# 🚀 Quick Start Guide - Bird Search System

## ✅ Dữ liệu mẫu đã được tạo!

### 📊 Thống kê Demo
- **Ảnh mẫu**: 15 ảnh chim
- **Thư mục chứa ảnh**: `data/raw/`
- **Database**: `database/`
- **Tính năng**: Tìm kiếm chim tương tự

### 🎯 Danh sách ảnh mẫu
```
1. eagle_01.jpg       - Chim ưng
2. sparrow_02.jpg     - Chim sẻ
3. owl_03.jpg         - Chim cú
4. parrot_04.jpg      - Chim vẹt
5. penguin_05.jpg     - Chim cánh cụt
6. flamingo_06.jpg    - Chim hạc
7. peacock_07.jpg     - Chim công
8. raven_08.jpg       - Chim quạ
9. swan_09.jpg        - Chim thiên nga
10. hummingbird_10.jpg - Chim ruồi
11. pigeon_11.jpg     - Chim bồ câu
12. crow_12.jpg       - Chim quạ đen
13. duck_13.jpg       - Chim vịt
14. hawk_14.jpg       - Chim diều hâu
15. dove_15.jpg       - Chim bồ câu bác
```

### 📁 Cấu trúc Database
```
database/
├── features.npy              # Vector đặc trưng gốc
├── features_scaled.npy       # Vector đặc trưng chuẩn hóa
├── scaler.pkl                # StandardScaler model
├── search_index.pkl          # Nearest neighbors index
└── filenames.txt             # Danh sách tên file
```

### 🔧 Công nghệ sử dụng
- **Feature Extraction**: Color histogram, Edge detection, Shape features
- **Normalization**: StandardScaler (scikit-learn)
- **Search**: NearestNeighbors (scikit-learn)
- **UI**: Streamlit
- **Storage**: NumPy, Pickle

### 🎮 Chạy ứng dụng

#### Cách 1: Chạy trực tiếp
```bash
streamlit run app.py
```

#### Cách 2: Dùng script wrapper
```bash
python run_app.py
```

Ứng dụng sẽ khởi động tại: `http://localhost:8501`

### 📝 Các bước đã hoàn thành

✅ **BƯỚC 1**: Tạo dữ liệu mẫu (15 ảnh chim)
- Tệp: `create_sample_data.py`

✅ **BƯỚC 2**: Xây dựng database
- Tệp: `build_database_script.py`
- Kết quả: Tất cả tệp trong folder `database/`

✅ **BƯỚC 3**: Triển khai backend
- `src/feature_extractor.py` - Trích đặc trưng
- `src/build_database.py` - Xây dựng index
- `src/search.py` - Tìm kiếm

✅ **BƯỚC 4**: Tạo giao diện web
- `app.py` - Ứng dụng Streamlit

### 🧪 Thử nghiệm ứng dụng

1. **Tải lên ảnh**: Click để chọn hoặc kéo thả ảnh chim
2. **Điều chỉnh kết quả**: Dùng slider để chọn số ảnh tương tự (1-20)
3. **Xem kết quả**: Danh sách các chim tương tự với độ tương tự

### 📊 Cách hoạt động
1. Ảnh được tải lên
2. Trích đặc trưng từ ảnh (36 chiều)
3. Chuẩn hóa đặc trưng
4. Tìm k neighbors gần nhất
5. Trả về danh sách với độ tương tự

### 🔄 Cập nhật dữ liệu

Để thêm ảnh mới:
1. Copy ảnh vào `data/raw/`
2. Chạy: `python build_database_script.py`
3. Ứng dụng sẽ load database mới tự động

### 🐛 Khắc phục sự cố

**Lỗi: "Database chưa được xây dựng"**
```bash
python build_database_script.py
```

**Lỗi: "Module not found"**
```bash
pip install -r requirements.txt
```

**Ứng dụng chạy chậm**
- Đơn giản hóa features (giảm bins từ 8)
- Sử dụng ảnh nhỏ hơn

### 📚 Thêm tài nguyên

- **Thêm ảnh**: Đặt trong `data/raw/` rồi chạy build script
- **Thay đổi features**: Edit `src/feature_extractor.py`
- **Tinh chỉnh tìm kiếm**: Sửa `src/search.py`

### 🎯 Bước tiếp theo

1. **Thêm ảnh thực tế** - Thay thế ảnh mẫu bằng ảnh chim thực
2. **Sử dụng model phức tạp** - Integrate ResNet, EfficientNet, VGG
3. **Thêm metadata** - Tên khoa học, đặc điểm, phân bố địa lý
4. **Deploy lên cloud** - Streamlit Cloud, Heroku, AWS

---

**Tạo lúc**: 31/03/2026
**Phiên bản**: 1.0 (Demo)
**Trạng thái**: ✅ Ready to use
