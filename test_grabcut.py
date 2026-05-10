import os
import cv2
import numpy as np

# 1. THIẾT LẬP ĐƯỜNG DẪN (Sửa lại theo máy của anh/chị)
INPUT_DIR = "C:/Users/ADMIN/bird-search-system/bird-search-system/dataset"
OUTPUT_DIR = "C:/Users/ADMIN/bird-search-system/bird-search-system/data/result_grabcut"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lấy thử 20 ảnh đầu tiên để kiểm tra (đỡ tốn thời gian chạy toàn bộ)
MAX_TEST_IMAGES = 20

def debug_grabcut(img: np.ndarray, margin=10) -> np.ndarray:
    h, w = img.shape[:2]
    mask_gc = np.zeros((h, w), dtype=np.uint8)

    try:
        # Khởi tạo model nền và model vật thể cho GrabCut
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)

        # Bounding Box thụt lùi vào trong theo margin
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

        # Chạy GrabCut (lặp 5 lần)
        cv2.grabCut(img, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

        # Lọc lấy Foreground (1) và Probable Foreground (3)
        binary = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    except Exception as e:
        print(f"GrabCut lỗi: {e}. Chuyển sang Otsu Fallback.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Xử lý hình thái học (Morphology) để làm mịn viền
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Bọc lót nếu cắt mất sạch ảnh
    if binary.sum() == 0:
        binary[:] = 255

    return binary


# ================= CHƯƠNG TRÌNH CHÍNH =================
count = 0
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Bắt đầu chạy kiểm thử GrabCut trên {min(MAX_TEST_IMAGES, len(image_files))} ảnh...")

for filename in image_files:
    if count >= MAX_TEST_IMAGES:
        break

    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)
    if img is None: continue

    # 1. Chạy hàm tạo Mask (giống hệt trong feature_extractors.py)
    mask_binary = debug_grabcut(img, margin=10)

    # 2. Tạo ảnh Mask 3 kênh màu (để lát nữa ghép nối với ảnh BGR cho khớp chiều)
    mask_3d = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)

    # 3. Ép Mask vào ảnh gốc để lấy kết quả cắt (Foreground)
    # Những chỗ mask=0 sẽ thành màu đen, mask=255 sẽ giữ nguyên màu chim
    mask_normalized = (mask_binary / 255.0).astype(np.uint8)
    foreground = img * mask_normalized[:, :, np.newaxis]

    # 4. TRỰC QUAN HÓA: Ghép 3 ảnh theo chiều ngang (Original | Mask | Result)
    # Vẽ thêm cái khung Rect màu đỏ lên ảnh gốc để xem GrabCut lấy khung ở đâu
    img_with_rect = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img_with_rect, (20, 20), (w - 20, h - 20), (0, 0, 255), 2)

    # Nối 3 bức ảnh lại với nhau
    combined_display = np.hstack((img_with_rect, mask_3d, foreground))

    # Lưu ra file để kiểm tra
    out_path = os.path.join(OUTPUT_DIR, f"10_debug_{filename}")
    cv2.imwrite(out_path, combined_display)

    count += 1

print(f"Đã xuất file kiểm tra tại thư mục: {OUTPUT_DIR}")
print("Hãy mở thư mục đó ra để tận mắt đánh giá chất lượng thuật toán!")