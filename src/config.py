import os

# Đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
TEST_INPUT = os.path.join(BASE_DIR, "data", "test_input")
DB_DIR = os.path.join(BASE_DIR, "database")

# File lưu trữ
DB_SQLITE = os.path.join(DB_DIR, "birds.db")
FEATURES_RAW = os.path.join(DB_DIR, "features.npy")
SCALER_PATH = os.path.join(DB_DIR, "scaler.pkl")
FAISS_INDEX = os.path.join(DB_DIR, "faiss.index")
FILENAMES_TXT = os.path.join(DB_DIR, "filenames.txt")

# Tham số xử lý
TARGET_SIZE = (256, 256)
GRID_SIZE = 4
HSV_BINS = 32
HOG_ORIENTATIONS = 9
HOG_PPC = (8, 8)
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
PCA_COMPONENTS = 256  # Giảm chiều về 256