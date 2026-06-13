import os
import sqlite3
import numpy as np
import pandas as pd

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "database.db")
FEATURES_PATH = os.path.join(BASE_DIR, "database", "features.npy")
CSV_PATH = os.path.join(BASE_DIR, "database", "features.csv")

print("--- BAT DAU CHUYEN DOI NPY SANG CSV ---")

if not os.path.exists(FEATURES_PATH):
    print(f"Loi: Khong tim thay file features.npy tai {FEATURES_PATH}!")
    exit(1)

if not os.path.exists(DB_PATH):
    print(f"Loi: Khong tim thay database.db tai {DB_PATH}!")
    exit(1)

# 1. Đọc danh sách ảnh từ SQLite
print("Dang doc thong tin anh tu SQLite database...")
conn = sqlite3.connect(DB_PATH)
rows = conn.execute("SELECT ImageID, ImagePath FROM Images ORDER BY ImageID").fetchall()
conn.close()

# 2. Đọc đặc trưng npy
print("Dang load features.npy...")
features = np.load(FEATURES_PATH)
print(f"Kich thuoc dac trung: {features.shape}")

if len(rows) != features.shape[0]:
    print("[Canh bao] So luong anh trong SQLite va features.npy khong khop!")
    print(f"SQLite: {len(rows)} anh vs features.npy: {features.shape[0]} anh.")
    # Lay gia tri nho hon lam gioi han
    limit = min(len(rows), features.shape[0])
    rows = rows[:limit]
    features = features[:limit]

# 3. Tạo tiêu đề cột (ImageID, ImagePath, Dim_0, Dim_1, ..., Dim_511)
cols = ["ImageID", "ImagePath"] + [f"Dim_{i}" for i in range(features.shape[1])]

# 4. Gộp dữ liệu
data = []
for i, (img_id, img_path) in enumerate(rows):
    row_data = [img_id, img_path] + list(features[i])
    data.append(row_data)

# 5. Tạo DataFrame và xuất ra file CSV
print("Dang ghi du lieu ra file CSV...")
df = pd.DataFrame(data, columns=cols)
df.to_csv(CSV_PATH, index=False, encoding="utf-8")

print(f"\n--- HOAN TAT ---")
print(f"Da tao thanh cong file CSV tai: {CSV_PATH}")
print(f"Kich thuoc CSV: {df.shape[0]} dong, {df.shape[1]} cot.")
