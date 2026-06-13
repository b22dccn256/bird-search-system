import os
import sys
import io
import numpy as np
import faiss

# Đảm bảo in được tiếng Việt trên console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FAISS_PATH = os.path.join(PARENT_DIR, "database", "faiss.index")

print("--- KIỂM TRA FILE faiss.index ---")
if not os.path.exists(FAISS_PATH):
    print(f"Lỗi: Không tìm thấy file tại {FAISS_PATH}!")
    exit(1)

# 1. Đọc chỉ mục FAISS
index = faiss.read_index(FAISS_PATH)

# 2. Xem siêu dữ liệu (Metadata)
print(f"\n[1] Thông tin chỉ mục (Metadata):")
print(f"  - Kiểu chỉ mục (Index Type)  : {type(index)}")
print(f"  - Tổng số lượng vector (N)   : {index.ntotal}")
print(f"  - Số chiều của mỗi vector (d): {index.d}")
print(f"  - Trạng thái huấn luyện      : {'Đã huấn luyện (Trained)' if index.is_trained else 'Chưa huấn luyện'}")

# 3. Lấy thử vector đầu tiên (Index 0)
# (Phương thức reconstruct chỉ hỗ trợ cho các chỉ mục dạng Flat như IndexFlatIP)
if hasattr(index, "reconstruct"):
    print(f"\n[2] Dữ liệu vector đầu tiên (Index 0) - Trích xuất thử:")
    vec_0 = index.reconstruct(0)
    print(f"  - Kích thước vector: {vec_0.shape}")
    print(f"  - 10 chiều đầu tiên: {vec_0[:10]}")
    print(f"  - 10 chiều cuối cùng: {vec_0[-10:]}")
else:
    print("\n[2] Chỉ mục này không hỗ trợ trích xuất ngược lại vector trực tiếp (Không phải dạng Flat).")

# 4. Trích xuất thử ma trận đầy đủ (ví dụ 5 dòng đầu)
print(f"\n[3] Xem trước ma trận vector (5 dòng đầu tiên):")
preview_matrix = np.array([index.reconstruct(i) for i in range(min(5, index.ntotal))])
for idx, vec in enumerate(preview_matrix):
    print(f"  - Vector #{idx} (5 chiều đầu): {vec[:5]}... (chuẩn hóa L2)")

print("\n--- HOÀN TẤT ---")
