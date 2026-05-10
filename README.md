# 🐦 Bird Search System (CBIR)

Hệ thống tìm kiếm hình ảnh chim dựa trên nội dung (**Content-Based Image Retrieval - CBIR**) sử dụng kết hợp các thuật toán thị giác máy tính cổ điển (Classical Computer Vision) và học sâu (YOLOv8) để tiền xử lý.

![Streamlit App](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=Python&logoColor=white)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white)

---

## 📌 Tổng quan dự án
Dự án này cho phép người dùng tải lên một ảnh chim và tìm kiếm các loài chim có đặc điểm tương đồng nhất trong cơ sở dữ liệu. Thay vì chỉ sử dụng Deep Learning để trích xuất đặc trưng, hệ thống tập trung vào việc phân tích các đặc tính vật lý của ảnh như **Màu sắc, Kết cấu, Hình dạng và Bố cục không gian**.

### 🌟 Tính năng nổi bật
*   **Hybrid Preprocessing:** Kết hợp YOLOv8 để nhận diện đối tượng và GrabCut để tách nền chính xác.
*   **Rich Feature Set:** Trích xuất vector đặc trưng thô lên đến **1.460 chiều**.
*   **Tối ưu hóa PCA:** Giảm chiều dữ liệu xuống **512 chiều** để tăng tốc độ tìm kiếm nhưng vẫn giữ nguyên 95%+ thông tin.
*   **Giao diện trực quan:** Streamlit Dashboard hiển thị kết quả kèm độ tương tự (Similarity Score).

---

## 🏗️ Kiến trúc hệ thống

Dự án được chia thành 3 giai đoạn chính:

### 1. Tiền xử lý (`0_yolo_crop.py`)
Sử dụng mô hình **YOLOv8n** để tự động tìm chim trong ảnh gốc:
*   **Smart Padding:** Bù viền (`BORDER_REPLICATE`) để đưa ảnh về hình vuông mà không làm biến dạng tỷ lệ.
*   **Margin for GrabCut:** Mở rộng vùng biên để tạo "mồi" hậu cảnh, giúp thuật toán GrabCut ở bước sau hoạt động nét hơn.
*   **Resize:** Chuẩn hóa toàn bộ tập dữ liệu về kích thước $224 \times 224$.

### 2. Trích xuất đặc trưng Offline (`1_extract_offline.py`)
Xây dựng cơ sở dữ liệu đặc trưng cho hàng ngàn ảnh:
*   **Foreground Masking:** Sử dụng GrabCut để loại bỏ hoàn toàn nền.
*   **Feature Fusion:** Tổ hợp 4 loại đặc trưng:
    *   **Color (210 dim):** HSV/Lab Histograms & Color Moments.
    *   **Texture (179 dim):** LBP, GLCM, Gabor Filters, FFT.
    *   **Shape (527 dim):** Hu Moments, HOG, Radius Signature.
    *   **Spatial (544 dim):** Grid-based local histograms.
*   **Storage:** Lưu trữ vào SQLite (`bird.db`) và tệp nén PCA (`bird_features.npy`).

### 3. Tìm kiếm Online (`2_app.py`)
Giao diện người dùng:
*   Trích xuất đặc trưng ảnh truy vấn theo cùng quy trình offline.
*   Tính khoảng cách **Cosine Similarity** giữa ảnh truy vấn và cơ sở dữ liệu.
*   Hiển thị Top-K kết quả.

---

## 🛠️ Hướng dẫn cài đặt

1. **Clone dự án:**
   ```bash
   git clone https://github.com/b22dccn256/bird-search-system.git
   cd bird-search-system
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chuẩn bị dữ liệu:**
   *   Đặt ảnh gốc vào thư mục `data/raw/`.
   *   Chạy tiền xử lý: `python 0_yolo_crop.py`.

---

## 🚀 Cách vận hành

### Bước 1: Trích xuất đặc trưng (Chỉ chạy 1 lần)
```bash
python 1_extract_offline.py
```
*Script hỗ trợ tính năng **Resume**, nếu bạn dừng giữa chừng, lần sau chạy lại sẽ tự làm tiếp từ ảnh chưa xử lý.*

### Bước 2: Khởi chạy ứng dụng Web
```bash
streamlit run 2_app.py
```

---

## 📊 Chi tiết bộ trích xuất đặc trưng
Hệ thống sử dụng vector 1.460 chiều bao gồm:
| Loại đặc trưng | Phương pháp | Số chiều |
| :--- | :--- | :--- |
| **Màu sắc** | HSV/Lab Histograms, Moments | 210 |
| **Kết cấu** | LBP, GLCM, Gabor, EOH, FFT | 179 |
| **Hình dáng** | Hu Moments, HOG, Profiles, Radius | 527 |
| **Không gian** | 4x4 Grid Local Features | 544 |

---

## 📂 Cấu trúc thư mục
```text
├── data/               # Chứa ảnh thô và ảnh đã xử lý
├── database/           # Lưu trữ bird.db, features.npy, pca_model.pkl
├── 0_yolo_crop.py      # Tiền xử lý YOLO
├── 1_extract_offline.py# Trích xuất đặc trưng hàng loạt
├── 2_app.py            # Giao diện Streamlit
├── feature_extractors.py# "Trái tim" của hệ thống (chứa thuật toán CV)
└── requirements.txt    # Danh sách thư viện
```

---
**Author:** [Nguyen Duy Ha - Tran Trong Thai - Nguyen Manh Tuan]
**Project Status:** Hoàn thành kiến trúc lõi và giao diện demo.

## 📂 Cấu trúc thư mục chi tiết
bird-search-system/
├── data/                        # Chứa các tập dữ liệu ảnh
│   ├── raw/                     # Ảnh gốc thu thập được
│   ├── process_loc_tay/         # Ảnh đã qua lọc thủ công
│   └── processed_224/           # Ảnh đã được YOLO crop & resize chuẩn 224x224
│
├── database/                    # Lưu trữ kết quả sau khi trích xuất offline
│   ├── database.db              # SQLite: Lưu metadata và đường dẫn ảnh chim
│   ├── features.npy             # Ma trận đặc trưng đã giảm chiều (PCA 512-dim)
│   ├── pca_model.pkl            # Model PCA đã lưu để dùng cho ảnh truy vấn
│
├── .venv/                       # Môi trường ảo Python (Virtual Environment)
├── 0_yolo_crop.py               # Tiền xử lý: Dùng YOLOv8 để crop đối tượng chim
├── 1_extract_offline.py         # Pipeline: Trích xuất đặc trưng & Xây dựng CSDL
├── 2_app.py                     # Giao diện chính: Tìm kiếm ảnh bằng CV cổ điển
├── app.py                       # Giao diện phụ: Tìm kiếm bằng Deep Learning (ResNet)
├── feature_extractors.py        # Module thuật toán: Color, Texture, Shape, Spatial
├── requirements.txt             # Danh sách các thư viện cần thiết (OpenCV, YOLO,...)
├── .gitignore                   # Cấu hình bỏ qua các file không cần đẩy lên Git
└── README.md                    # Tài liệu hướng dẫn sử dụng dự án
