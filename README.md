# Hệ thống Tìm kiếm Ảnh Chim (Bird Image Search System)

Hệ thống **CBIR (Content-Based Image Retrieval)** tìm kiếm ảnh chim tương đồng dựa trên nội dung hình ảnh, sử dụng các đặc trưng Computer Vision cổ điển (không dùng Deep Learning cho bước trích xuất đặc trưng). Đầu vào là một ảnh chim bất kỳ; đầu ra là **5 ảnh giống nhất** trong cơ sở dữ liệu, xếp theo độ tương đồng giảm dần.

---

## Cấu trúc dự án

```
bird-search-system/
│
├── dataset/                     # Ảnh chim đã tiền xử lý chuẩn (1239 ảnh, kích thước 224×224)
├── database/
│   ├── database.db              # SQLite — quản lý ImageID và ImagePath tương đối
│   ├── features.npy             # Ma trận vector đặc trưng sau PCA (1239 × 512)
│   ├── features.csv             # Bảng đặc trưng CSV trực quan kết hợp file path (từ script chuyển đổi)
│   ├── pca_model.pkl            # Mô hình PCA đã huấn luyện
│   └── faiss.index              # Chỉ mục FAISS (IndexFlatIP) dùng để so khớp Cosine nhanh
│
├── data/
│   ├── process_loc_tay/         # Ảnh gốc chưa lọc
│   ├── process_loc_tay_2/       # Thư mục ảnh gốc bổ sung dùng để chạy lọc YOLOv8
│   ├── process_224_(dts)/       # Thư mục ảnh chim 224x224 đạt chuẩn đầu ra
│   └── result_grabcut/          # Ảnh trung gian kiểm tra mặt nạ GrabCut (phục vụ debug)
│
├── results/                     # Kết quả đánh giá chất lượng truy hồi
│   ├── per_query_metrics.csv    # Chi tiết độ chính xác của từng ảnh truy vấn
│   ├── quantitative_summary.csv # Kết quả đánh giá tổng hợp dạng CSV
│   └── quantitative_summary.md  # Báo cáo đánh giá tổng hợp dạng Markdown
│
├── file_support/                # Thư mục chứa các script tiện ích & tài liệu nhóm
│   ├── 0_yolo_shift_crop.py     # Script YOLOv8 crop phiên bản đầu
│   ├── build_faiss_only.py      # Build riêng faiss.index từ file features.npy có sẵn
│   ├── convert_features_to_csv.py # Chuyển đổi ma trận đặc trưng features.npy sang dạng CSV trực quan
│   ├── baocao_nhom11_hecsdldpt.txt # Báo cáo kết quả của Nhóm 11
│   └── dataset_old/             # Tập dữ liệu cũ lưu trữ dự phòng
│
├── 0_2_yolo_filter_images.py    # Script lọc ảnh gốc theo tỷ lệ aspect-ratio và cắt căn giữa YOLOv8
├── 1_extract_offline.py         # Trích xuất đặc trưng offline (GrabCut -> Trích đặc trưng -> PCA -> FAISS)
├── 2_app.py                     # Web App Streamlit tìm kiếm ảnh chim tương đồng trực quan
├── debug_grabcut_preview.py     # Script xem trước mặt nạ GrabCut phân đoạn chim/nền
├── evaluate_weak_labels.py      # Script đánh giá chất lượng hệ thống (Weak-label evaluation)
├── requirements.txt             # Danh sách thư viện Python cần thiết
└── yolov8n.pt                   # Trọng số YOLOv8 Nano dùng nhận diện chim
```

---

## 1. Xây dựng bộ dữ liệu ảnh chim

### 1.1. Nguồn dữ liệu và quy mô

| Tiêu chí | Yêu cầu đề bài | Thực tế triển khai |
|---|---|---|
| Số lượng ảnh | ≥ 500 | **1 239 ảnh** |
| Số loài chim khác nhau | Nhiều loài | **122 loài** |
| Kích thước đồng nhất | Cùng kích thước | **224 × 224 pixel** |
| Định dạng | Tùy chọn | **JPEG (.jpg)** |
| Tỉ lệ khung hình đối tượng | Đồng nhất | Chim được cắt theo bounding box YOLO, resize về vuông |
| Tư thế chim | Đang đậu, không bay | Ảnh gốc được lọc thủ công trước khi xử lý tự động |
| Góc chụp | Ngang (side view) | Ảnh nguồn được chọn theo góc chụp ngang |

Dữ liệu được thu thập từ bộ ảnh chim theo chuẩn đặt tên CUB-200 (ví dụ: `Acadian_Flycatcher_0012_795612.jpg`), bao gồm nhiều loài chim Bắc Mỹ với ảnh chụp chim đang đậu, góc ngang.

### 1.2. Quy trình tiền xử lý dữ liệu và lọc tránh biến dạng hình ảnh

Quy trình tiền xử lý dữ liệu được thiết kế nhằm bảo toàn tỷ lệ cơ thể chim (aspect ratio), tránh biến dạng méo hình khi đưa vào trích xuất đặc trưng.

**Giai đoạn 1 — Lọc thủ công ban đầu (`data/process_loc_tay/`)**
- Chọn ảnh thỏa điều kiện: chim **đang đậu**, **không bay**, góc chụp **ngang**.
- Loại bỏ ảnh mờ, chim quá nhỏ, hoặc bị che khuất nhiều.

**Giai đoạn 2 — Tự động cắt vuông bảo toàn tỷ lệ và lọc bằng YOLOv8 (`0_2_yolo_filter_images.py`)**
```
Ảnh gốc → YOLOv8n (class "bird") → Kiểm tra kích thước (>224x224) 
         → Cắt vuông mở rộng 1.35x căn giữa chim (tránh tràn viền) → Resize 224x224
```

*Chi tiết kỹ thuật lọc ảnh:*
1. **Kiểm tra kích thước gốc:** Chỉ chấp nhận các ảnh có chiều cao và chiều rộng lớn hơn 224 pixels.
2. **Nhận diện đối tượng:** Dùng YOLOv8n (`yolov8n.pt`), chỉ lọc `classes=[14]` (nhãn *bird*). Lấy bounding box có độ tin cậy cao nhất.
3. **Cắt hình vuông bảo toàn tỷ lệ:**
   - Trọng tâm vùng cắt được đặt trùng với trọng tâm đối tượng ($cx$, $cy$).
   - Kích thước vùng cắt tối ưu được tính bằng `crop_size = 1.35 * max(bw, bh)` (hệ số mở rộng $1.35\times$ giúp chim chiếm khoảng ~55% diện tích ảnh, tạo bối cảnh tự nhiên vừa vặn).
   - **Ràng buộc quan trọng:** Vùng cắt vuông này phải nằm **hoàn toàn bên trong** ảnh gốc. Nếu bị tràn viền, hệ thống tự động fallback về hệ số $1.0\times$ (chỉ cắt khít bounding box hình vuông). Nếu vẫn bị tràn viền, ảnh bị loại khỏi tập đạt yêu cầu để tránh biến dạng/nhiễu biên.
4. **Phân loại đầu ra:** 
   - Ảnh đạt yêu cầu được cắt, resize về $224 \times 224$ pixels bằng nội suy `cv2.INTER_AREA` và ghi vào `data/process_224` (tương đương `data/process_224_(dts)`).
   - Tên các file được ghi nhận thành hai file danh sách: `data/dat_yeu_cau.txt` và `data/khong_dat_yeu_cau.txt`.

### 1.3. Lý do các ràng buộc dữ liệu

| Ràng buộc | Lý do |
|---|---|
| Cùng kích thước 224×224 | Đồng nhất đầu vào cho các bộ lọc cố định (Gabor, HOG, lưới 4×4) |
| Cắt theo bounding box | Loại bỏ nền thừa, tập trung đặc trưng vào thân chim |
| Cắt vuông căn giữa | Giữ nguyên tỷ lệ chiều rộng/cao của chim, tránh méo mó gây nhiễu thông số kích thước cơ thể |
| Chim đậu, góc ngang | Giảm biến thiên tư thế; đặc trưng hình dáng (Hu Moments, HOG, profile) ổn định hơn |
| ≥ 500 ảnh | Đủ mẫu để PCA học phân phối đặc trưng và đánh giá truy hồi có ý nghĩa thống kê |

---

## 2. Bộ thuộc tính (đặc trưng) nhận diện ảnh chim

Hệ thống trích xuất **vector đặc trưng 1 460 chiều** từ mỗi ảnh, chia thành **4 nhóm** thể hiện cả sự **tương đồng** (cùng loài, cùng màu lông, cùng hình dáng) lẫn sự **khác biệt** (loài khác, vân lông khác, bố cục khác). Toàn bộ logic nằm trong `feature_extractors.py`.

### 2.1. Tiền xử lý: phân tách foreground (GrabCut)

Trước khi trích đặc trưng, mỗi ảnh được phân đoạn để tách chim khỏi nền:
- **GrabCut** với bounding box khởi tạo (margin 10 px, 5 vòng lặp).
- **Morphology** (opening + closing, kernel elip 5×5) làm sạch mask.
- **Fallback Otsu** khi GrabCut thất bại.

Mask nhị phân đảm bảo đặc trưng chỉ tính trên vùng chim, không bị nhiễu bởi nền — yếu tố then chốt khi so sánh ảnh tương đồng.

### 2.2. Bảng tổng hợp các nhóm đặc trưng

| Nhóm | Chiều | Vai trò tương đồng | Vai trò phân biệt |
|---|---:|---|---|
| **Color** (Màu sắc) | 210 | Chim cùng loài thường có phân bố màu lông tương tự | Khác biệt sắc độ lông (vàng, xanh, đen, nâu…) |
| **Texture** (Kết cấu) | 179 | Vân lông, độ mịn da lông giống nhau trong cùng loài | Mẫu vân lông, hướng lông đặc trưng từng loài |
| **Shape** (Hình dáng) | 527 | Tỉ lệ thân–đầu–đuôi tương đồng khi cùng góc chụp | Silhouette, độ tròn, profile đuôi/mỏ khác loài |
| **Spatial** (Không gian) | 544 | Bố cục màu–vân trên thân chim giống nhau | Vị trí đốm màu, vùng lông đặc trưng trên thân |
| **Tổng (raw)** | **1 460** | — | Sau đó **L2-normalize** → **PCA** giảm còn **512 chiều** |

### 2.3. Chi tiết từng nhóm và lý do lựa chọn

#### A. Đặc trưng màu sắc — 210 chiều

| Thành phần | Chiều | Giá trị thông tin |
|---|---:|---|
| Histogram HSV (H, S, V) | 96 | Phân bố màu tổng thể; bắt sắc độ lông đặc trưng loài |
| Histogram CIE-Lab (L, a, b) | 96 | Mô tả màu gần với cảm nhận thị giác người hơn RGB |
| Color Moments (Mean, Std, Skewness) × 6 kênh | 18 | Tóm tắt thống kê màu: độ sáng trung bình, độ tương phản, độ lệch phân phối |

**Lý do chọn:** Màu lông là tín hiệu phân biệt mạnh nhất giữa các loài chim. HSV tách sắc độ khỏi độ sáng; Lab ổn định hơn dưới thay đổi ánh sáng nhẹ.

#### B. Đặc trưng kết cấu — 179 chiều

| Thành phần | Chiều | Giá trị thông tin |
|---|---:|---|
| LBP (Local Binary Pattern) | 10 | Mã hóa mẫu vân cục bộ trên lông |
| EOH (Edge Orientation Histogram) | 9 | Hướng cạnh/lông — bắt hướng lông xếp |
| GLCM / Haralick | 16 | Energy, Contrast, Homogeneity, Correlation — đặc tính vân tổng thể |
| Gabor filter bank (5 tần số × 8 hướng) | 80 | Vân có hướng và tần số — phân biệt lông mịn vs lông xù |
| Stripe FFT (phổ tần số 2D) | 64 | Chu kỳ sọc/vân lông lặp lại theo chiều dọc thân |

**Lý do chọn:** Hai loài có thể trùng màu nhưng khác vân lông. LBP và Gabor bắt chi tiết vi cấu trúc; FFT bắt chu kỳ sọc đặc trưng một số loài.

#### C. Đặc trưng hình dáng — 527 chiều

| Thành phần | Chiều | Giá trị thông tin |
|---|---:|---|
| Hu Moments + scalars (area, perimeter, compactness, circularity) | 11 | Bất biến dưới phép biến đổi hình học — mô tả silhouette tổng thể |
| Grid Mask Density (8×8) | 64 | Phân bố khối lượng chim trên lưới — bắt hình dáng bất đối xứng |
| Width Profile / Contour Profile | 64 | Độ rộng/cao theo từng hàng/cột — phân biệt đuôi dài vs thân tròn |
| Radius Signature | 64 | Khoảng cách tâm → biên theo góc — đặc trưng viền ngoài |
| HOG (ROI 64×64, 9 orientations) | 324 | Gradient hướng trên vùng chim — bắt cấu trúc bộ phận (đầu, cánh, đuôi) |

**Lý do chọn:** Hình dáng là tín hiệu phân biệt khi màu sắc tương tự. Hu Moments ổn định với phép co giãn nhẹ; HOG mô tả cấu trúc cục bộ của thân chim.

#### D. Đặc trưng không gian — 544 chiều

Chia ảnh 224×224 thành lưới **4×4 = 16 ô**. Mỗi ô có foreground chứa:
- HSV histogram cục bộ (24 chiều)
- LBP histogram cục bộ (10 chiều)

→ **34 chiều/ô × 16 ô = 544 chiều**

**Lý do chọn:** Các đặc trưng toàn cục có thể trùng nhau giữa hai loài khác nhau. Đặc trưng không gian giữ thông tin **bố cục** màu và họa tiết trên các bộ phận cơ thể.

### 2.4. Hợp nhất và giảm chiều

```
Ảnh 224×224 → GrabCut mask → [Color | Texture | Shape | Spatial] → Vector 1460D
    → L2-normalize → PCA (fit trên toàn bộ dataset) → Vector 512D
```

- **L2-normalize:** Đưa mọi vector về cùng thang đo, phù hợp so sánh bằng cosine similarity.
- **PCA (tối đa 512 chiều):** Loại nhiễu, giảm chi phí lưu trữ và tính toán, giữ phần lớn phương sai dữ liệu.

---

## 3. Hệ CSDL quản lý siêu dữ liệu và cơ chế tìm kiếm

### 3.1. Kiến trúc lưu trữ lai (Hybrid Storage)

Hệ thống kết hợp cơ sở dữ liệu quan hệ với các cấu trúc file chuyên dụng để tối ưu hiệu năng CBIR:

| Thành phần | File | Vai trò |
|---|---|---|
| Siêu dữ liệu | `database/database.db` (SQLite) | Quản lý danh mục ảnh, ánh xạ `ImageID` với `ImagePath` |
| Vector đặc trưng (Matrix) | `database/features.npy` | Ma trận đặc trưng sau PCA `(N, 512)` kiểu float32 |
| Bảng tra cứu trực quan | `database/features.csv` | File CSV chứa đầy đủ tên file và giá trị 512 chiều đặc trưng |
| Mô hình chiếu | `database/pca_model.pkl` | PCA đã học phân phối — dùng chiếu vector truy vấn |
| Chỉ mục tìm kiếm | `database/faiss.index` | Chỉ mục FAISS IndexFlatIP hỗ trợ so khớp cosine tốc độ cao |

### 3.2. Schema cơ sở dữ liệu SQLite

```sql
CREATE TABLE IF NOT EXISTS Images (
    ImageID   INTEGER PRIMARY KEY AUTOINCREMENT,
    ImagePath TEXT    NOT NULL UNIQUE
);
```

**Quan hệ logic:** Dòng thứ `i` trong ma trận `features.npy` tương ứng với bản ghi thứ `i` khi thực hiện `ORDER BY ImageID` trong bảng `Images`.

### 3.3. Quy trình xây dựng CSDL (Offline)

Thực hiện thông qua: `python 1_extract_offline.py`
1. Đọc dữ liệu từ `dataset/` (1239 ảnh đạt chuẩn).
2. Trích xuất đặc trưng thô 1460D cho từng ảnh, cập nhật đường dẫn vào SQLite.
3. Fit PCA giảm chiều xuống 512D.
4. Ghi nhận ma trận `features.npy`, lưu mô hình `pca_model.pkl`.
5. Tạo và xây dựng chỉ mục `faiss.index` (Cosine).
*Hỗ trợ Resume:* Tiến trình trích xuất tự động bỏ qua các ảnh đã có trong SQLite nếu bị gián đoạn giữa chừng.

### 3.4. Cơ chế tìm kiếm ảnh tương đồng

Cho vector truy vấn sau PCA là **q** và ma trận CSDL **X = {x₁, x₂, …, xₙ}**:

1. **Chiếu đặc trưng:** `q = PCA.transform(raw_query)`
2. **Tính khoảng cách cosine:**
   \[
   d_i = 1 - \frac{q \cdot x_i}{\|q\|_2 \|x_i\|_2}
   \]
3. **Xếp hạng:** Sắp xếp khoảng cách $d_i$ tăng dần, lấy Top-5 kết quả.
4. **Engine tìm kiếm:**
   - **Ưu tiên:** Sử dụng thư viện **FAISS IndexFlatIP** (Inner Product trên vector chuẩn hóa L2 tương đương với Cosine similarity) cho tốc độ tìm kiếm cực nhanh.
   - **Fallback:** Tự động chuyển sang hàm `cdist(metric="cosine")` của SciPy khi hệ thống thiếu thư viện FAISS.

---

## 4. Hệ thống tìm kiếm ảnh chim (Online)

### 4.1. Sơ đồ khối hệ thống

#### Tổng quan hai giai đoạn
```mermaid
flowchart TB
    subgraph OFFLINE["Giai đoạn OFFLINE — Xây dựng CSDL"]
        A1[Ảnh gốc đã lọc] --> A2[YOLOv8 cắt chim]
        A2 --> A3[dataset/ 224×224]
        A3 --> A4[GrabCut + Trích đặc trưng 1460D]
        A4 --> A5[L2-normalize + PCA → 512D]
        A5 --> A6[features.npy + pca_model.pkl]
        A3 --> A7[(SQLite Images)]
        A5 --> A8[faiss.index]
    end

    subgraph ONLINE["Giai đoạn ONLINE — Truy vấn"]
        B1[Ảnh truy vấn upload] --> B2[GrabCut + Trích đặc trưng 1460D]
        B2 --> B3[PCA.transform → 512D]
        B3 --> B4[Cosine search Top-5]
        B4 --> B5[Tra cứu ImagePath]
        B5 --> B6[Hiển thị 5 ảnh + % tương đồng]
    end

    A6 -.->|nạp sẵn| B4
    A7 -.->|nạp sẵn| B5
    A8 -.->|nạp sẵn| B4
```

#### Quy trình chi tiết khi tìm kiếm
```mermaid
flowchart LR
    Q[Ảnh query] --> R[Resize 224×224]
    R --> M[GrabCut → Mask]
    M --> F1[Color 210D]
    M --> F2[Texture 179D]
    M --> F3[Shape 527D]
    M --> F4[Spatial 544D]
    F1 --> V[Vector 1460D]
    F2 --> V
    F3 --> V
    F4 --> V
    V --> N[L2-normalize]
    N --> P[PCA → 512D]
    P --> S[Top-5 cosine]
    S --> O[Kết quả + điểm %]
```

### 4.2. Giao diện Web App Streamlit (`2_app.py`)

Web App cung cấp các tính năng:
- **Tải lên ảnh truy vấn:** Hỗ trợ kéo thả ảnh JPEG/PNG.
- **Hiển thị kết quả trung gian:** Cho phép xem ảnh resize 224x224 và mặt nạ GrabCut phân tách chim.
- **Top-5 kết quả giống nhất:** Hiển thị danh sách ảnh kèm tỷ lệ phần trăm tương đồng:
  \[
  \text{similarity}_i = (1 - d_i) \times 100\%
  \]
- **Bảng so sánh chi tiết chỉ số:** Hỗ trợ phân tích khoảng cách Euclidean (L2), Manhattan (L1), Cosine Distance, Dot Product và các phân phối của vector đặc trưng.

### 4.3. Kết quả đánh giá định lượng thực tế

Hệ thống được đánh giá bằng **weak labels** (suy ra loài từ thư mục cha của tên file) trên **100 truy vấn ngẫu nhiên** (hạt giống seed=42):

> [!NOTE]
> Kết quả đánh giá thực tế dưới đây được trích xuất từ file [results/quantitative_summary.md](file:///c:/Bird-Search-System/bird-search-system/results/quantitative_summary.md) (cấu hình `C0_full` với dữ liệu thực tế 1239 ảnh):

| Chỉ số đánh giá | Giá trị thực tế | Ý nghĩa |
|---|---:|---|
| **Precision@1 (P@1)** | **6.00%** | Tỉ lệ ảnh đầu tiên trùng loài với ảnh truy vấn |
| **Precision@5 (P@5)** | **6.60%** | Tỉ lệ ảnh trùng loài trong Top-5 kết quả |
| **Precision@10 (P@10)** | **5.50%** | Tỉ lệ ảnh trùng loài trong Top-10 kết quả |
| **Precision@20 (P@20)** | **4.45%** | Tỉ lệ ảnh trùng loài trong Top-20 kết quả |
| **mAP@5** | **3.40%** | Độ chính xác trung bình trung bình tại K=5 |
| **mAP@10** | **2.39%** | Độ chính xác trung bình trung bình tại K=10 |
| **Thời gian tìm kiếm (Search Latency)** | **~0.59 ms / query** | Tốc độ so khớp đặc trưng sử dụng chỉ mục |

*Lưu ý: Chỉ số Precision phản ánh độ khớp chính xác về loài (Weak Label). Trên thực tế, do đặc trưng thị giác CV cổ điển (màu sắc, kết cấu toàn cục) có độ tương đồng giữa các loài chim gần nhau, hệ thống luôn trả về các ảnh giống nhất về mặt hình học và thị giác.*

---

## 5. Các Script Tiện Ích & Bổ Trợ

Để hỗ trợ vận hành và quản lý dữ liệu trực quan, hệ thống tích hợp các công cụ bổ trợ nằm trong thư mục `file_support/` và root:

### 5.1. Lọc và chuẩn hóa dữ liệu ảnh (`0_2_yolo_filter_images.py`)
Hỗ trợ tự động hóa khâu tiền xử lý, lọc bỏ các ảnh không đạt kích thước hoặc có nguy cơ bị biến dạng khi crop:
```bash
python 0_2_yolo_filter_images.py
```
*Đầu ra:* Danh sách phân loại lưu tại `data/dat_yeu_cau.txt` và `data/khong_dat_yeu_cau.txt`. Ảnh đạt chuẩn lưu vào `data/process_224`.

### 5.2. Chuyển đổi ma trận đặc trưng sang CSV (`file_support/convert_features_to_csv.py`)
Xuất ma trận đặc trưng `features.npy` dạng binary thành file [database/features.csv](file:///c:/Bird-Search-System/bird-search-system/database/features.csv) trực quan, kết hợp `ImageID` và `ImagePath` tương ứng từ SQLite để dễ dàng kiểm tra các chiều đặc trưng (`Dim_0` đến `Dim_511`):
```bash
python file_support/convert_features_to_csv.py
```

### 5.3. Xây dựng riêng chỉ mục FAISS (`file_support/build_faiss_only.py`)
Khi cần cập nhật hoặc xây dựng riêng file chỉ mục tìm kiếm nhanh `faiss.index` từ file `features.npy` có sẵn mà không cần chạy lại toàn bộ pipeline trích xuất offline:
```bash
python file_support/build_faiss_only.py
```

---

## Hướng dẫn cài đặt và sử dụng

### Yêu cầu hệ thống
- Python 3.8+ (Khuyên dùng Python 3.12 hoặc 3.13)
- Trình quản lý thư viện `pip`

### 1. Cài đặt môi trường

```bash
cd bird-search-system
python -m venv venv

# Kích hoạt môi trường (Windows)
venv\Scripts\activate

# Cài đặt các thư viện cơ bản
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Lưu ý cài đặt thư viện FAISS trên Windows:**
> Đối với môi trường Windows (ví dụ Python 3.13), bạn nên cài đặt thư viện FAISS thông qua lệnh:
> ```bash
> python -m pip install faiss-cpu
> ```

### 2. Sử dụng hệ thống

#### Bước 1: Chuẩn bị dữ liệu và Trích xuất đặc trưng Offline
Đặt các ảnh chim đã chuẩn hóa vào thư mục `dataset/` (hoặc chạy lọc ảnh bằng `0_2_yolo_filter_images.py`). Sau đó chạy script trích xuất để tạo CSDL:
```bash
python 1_extract_offline.py
```
*Kết quả:* Hệ thống sẽ sinh ra các file `database.db`, `features.npy`, `pca_model.pkl` và `faiss.index` bên trong thư mục `database/`.

#### Bước 2: Chạy Web App Tìm kiếm
Khởi chạy ứng dụng Streamlit:
```bash
streamlit run 2_app.py
```
Mở trình duyệt truy cập: `http://localhost:8501` để bắt đầu tải ảnh và tìm kiếm.

#### Bước 3: Đánh giá chất lượng hệ thống (Weak-label Evaluation)
Chạy đánh giá chất lượng truy hồi ngẫu nhiên trên 100 truy vấn:
```bash
python evaluate_weak_labels.py --query-count 100 --seed 42
```
*Kết quả:* Báo cáo hiệu năng chi tiết sẽ được tự động xuất ra thư mục `results/`.
