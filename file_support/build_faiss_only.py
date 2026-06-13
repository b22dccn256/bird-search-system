import os
import numpy as np
import faiss

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(BASE_DIR, "database", "features.npy")
FAISS_PATH = os.path.join(BASE_DIR, "database", "faiss.index")

print("--- BAT DAU DUNG CHI MUC FAISS ---")
if not os.path.exists(FEATURES_PATH):
    print(f"Loi: Khong tim thay file dac trung tai {FEATURES_PATH}!")
    exit(1)

# 1. Load các vector đặc trưng đã qua PCA (shape: N x 512)
features_pca = np.load(FEATURES_PATH)
print(f"Da load features.npy voi shape: {features_pca.shape}")

# 2. Chuyển kiểu dữ liệu sang float32 (yêu cầu bắt buộc của FAISS)
x = features_pca.astype(np.float32).copy()

# 3. Chuẩn hóa L2 trên từng vector
faiss.normalize_L2(x)
print("Da chuan hoa L2 cho cac vector dac trung.")

# 4. Tạo index IndexFlatIP (Inner Product) cho so khớp Cosine chính xác
dim = x.shape[1]
index = faiss.IndexFlatIP(dim)

# 5. Thêm các vector đặc trưng vào index
index.add(x)
print(f"Da them {index.ntotal} vector vao chi muc FAISS.")

# 6. Ghi index ra file faiss.index
faiss.write_index(index, FAISS_PATH)
print(f"Da luu chi muc FAISS thanh cong tai: {FAISS_PATH}")
print("--- HOAN TAT ---")
