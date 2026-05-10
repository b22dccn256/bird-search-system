# BÁO CÁO CHI TIẾT HỆ THỐNG TÌM KIẾM CHIM (BIRD CBIR SYSTEM)

## 1. Xây dựng Bộ dữ liệu (Dataset)
*   **Quy mô:** Bộ dữ liệu bao gồm **1.253 hình ảnh** chim được thu thập và chọn lọc.
*   **Tiêu chuẩn hóa dữ liệu (YOLOv8 + Smart Padding):** 
    *   **Kích thước:** Tất cả ảnh được đưa về chuẩn 224 x 244 pixel.
    *   **Tỉ lệ khung hình:** Sử dụng kỹ thuật `Smart Padding` (thêm viền gương - `BORDER_REPLICATE`) để biến các ảnh có hình dáng bất kỳ thành hình vuông mà không làm biến dạng (méo) con chim.
    *   **Góc chụp & Trạng thái:** Dữ liệu tập trung vào các loài chim ở trạng thái đậu, góc chụp ngang (side-view) để tối ưu hóa việc trích xuất đặc trưng hình dáng (Shape Profiles).
    *   **Tập trung đối tượng:** Sử dụng YOLOv8 để tự động cắt (crop) sát con chim, loại bỏ hậu cảnh dư thừa, giúp hệ thống không bị "đánh lừa" bởi môi trường xung quanh.

## 2. Bộ thuộc tính nhận diện (Feature Engineering)
Hệ thống sử dụng tổ hợp **1.460 đặc trưng thô**, đại diện cho 4 khía cạnh thị giác chính:

### 2.1. Đặc trưng Màu sắc (Color - 210 chiều)
*   **Lý do chọn:** Là yếu tố quan trọng nhất để phân biệt các loài chim (ví dụ: chim bói cá màu xanh, chim sẻ màu nâu).
*   **Giá trị thông tin:** 
    *   Sử dụng không gian màu **HSV** và **Lab** thay vì RGB vì chúng tách biệt độ sáng và màu sắc, giúp nhận diện ổn định trong các điều kiện ánh sáng khác nhau.
    *   **Color Moments:** Tính giá trị trung bình, độ lệch chuẩn và độ xiên của màu sắc để nắm bắt phân phối màu tổng thể.

### 2.2. Đặc trưng Kết cấu (Texture - 179 chiều)
*   **Lý do chọn:** Phân biệt các kiểu vân lông (vân sọc, đốm, mịn).
*   **Giá trị thông tin:** 
    *   **LBP (Local Binary Pattern):** Nhận diện các vi cấu trúc bề mặt.
    *   **Gabor Filters:** Mô phỏng cách mắt người nhìn nhận các hướng và tần số, rất hiệu quả để nhận diện cấu trúc lông chim.
    *   **FFT (Fast Fourier Transform):** Phân tích tần số không gian để bắt các dải vân lặp lại.

### 2.3. Đặc trưng Hình dáng (Shape - 527 chiều)
*   **Lý do chọn:** Phân biệt giữa chim mỏ dài/ngắn, đuôi dài/ngắn hoặc dáng người béo/gầy.
*   **Giá trị thông tin:** 
    *   **Hu Moments:** Các thông số hình học không đổi khi ảnh bị xoay hoặc thay đổi kích thước.
    *   **HOG (Histogram of Oriented Gradients):** Nắm bắt cấu trúc biên cạnh và hình dáng tổng quát của con chim.

### 2.4. Đặc trưng Không gian (Spatial - 544 chiều)
*   **Lý do chọn:** Lưu giữ thông tin vị trí (ví dụ: "Đầu đỏ" khác với "Bụng đỏ").
*   **Giá trị thông tin:** Chia ảnh thành lưới **4x4**, trích xuất histogram màu tại từng ô.

## 3. Hệ CSDL và Cơ chế tìm kiếm
### 3.1. Hệ CSDL
*   **Metadata:** Sử dụng **SQLite (`bird.db`)** để quản lý ImageID và ImagePath, đảm bảo truy xuất đường dẫn ảnh cực nhanh từ ID tìm được.
*   **Feature Index:** Lưu trữ ma trận đặc trưng dưới dạng **NumPy (`.npy`)**.
*   **Giảm chiều PCA:** Sử dụng thuật toán **PCA (Principal Component Analysis)** để nén từ 1.460 chiều xuống **512 chiều**, loại bỏ các đặc trưng thừa và nhiễu, giúp tốc độ tìm kiếm nhanh gấp 3 lần.

### 3.2. Cơ chế tìm kiếm
Hệ thống sử dụng **Khoảng cách Cosine (Cosine Similarity)**:
$$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
*   **Tại sao dùng Cosine?** Vì nó đo lường sự tương đồng về "hướng" của vector đặc trưng, không bị ảnh hưởng bởi cường độ sáng của ảnh (độ dài vector).

## 4. Quy trình hệ thống (Workflow)
### 4.1. Sơ đồ khối
1.  **Input:** Ảnh truy vấn (Query Image).
2.  **Tiền xử lý:** Resize 224x224 -> Tách nền GrabCut.
3.  **Trích xuất:** Chạy bộ 4 thuật toán đặc trưng -> Vector 1.460 chiều.
4.  **Chiếu PCA:** Chuyển vector mới về không gian 512 chiều (dùng model đã train offline).
5.  **So khớp:** Tính Cosine Similarity với toàn bộ 1.253 vector trong DB.
6.  **Sắp xếp:** Lấy 5 ảnh có điểm số cao nhất (gần 1.0 nhất).
7.  **Output:** Hiển thị 5 kết quả theo thứ tự giảm dần.

### 4.2. Kết quả trung gian
*   **Mask (Mặt nạ):** Hình ảnh đen trắng chỉ chứa bóng của con chim (chứng minh việc tách nền thành công).
*   **Vector đặc trưng:** Một chuỗi 512 số thực đại diện cho "DNA" của bức ảnh.

## 5. Demo và Đánh giá
*   **Giao diện:** Streamlit Dashboard chuyên nghiệp, hỗ trợ kéo thả ảnh.
*   **Đánh giá:** 
    *   Hệ thống cho kết quả rất tốt với các ảnh có loài đã có trong DB (Top 1 thường là chính nó hoặc ảnh cùng loài).
    *   Với loài chưa có, hệ thống trả về các loài có **hình dáng (silhouette)** và **màu sắc (plumage)** tương đồng nhất (Ví dụ: Upload ảnh chim màu xanh lạ sẽ trả về các loài chim xanh trong DB).
    *   **Tốc độ:** < 0.5 giây cho một lần tìm kiếm trên tập 1.250 ảnh.

---
**Nhóm thực hiện:** Nguyen Duy Ha - Tran Trong Thai - Nguyen Manh Tuan