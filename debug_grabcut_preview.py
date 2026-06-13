import os
import cv2
import numpy as np


# 1. THIET LAP DUONG DAN (tu dong theo thu muc du an hien tai)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "result_grabcut")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lay thu 20 anh dau tien de kiem tra (do ton thoi gian)
MAX_TEST_IMAGES = 1239


def debug_grabcut(img: np.ndarray, margin: int = 10) -> np.ndarray:
    h, w = img.shape[:2]
    mask_gc = np.zeros((h, w), dtype=np.uint8)

    try:
        # Khoi tao model nen va model vat the cho GrabCut
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)

        # Bounding Box thut lui vao trong theo margin
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

        # Chay GrabCut (lap 5 lan)
        cv2.grabCut(img, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

        # Loc lay Foreground (1) va Probable Foreground (3)
        binary = np.where(
            (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

    except Exception as e:
        print(f"GrabCut loi: {e}. Chuyen sang Otsu Fallback.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Xu ly hinh thai hoc (Morphology) de lam min vien
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Boc lot neu cat mat sach anh
    if binary.sum() == 0:
        binary[:] = 255

    return binary


def collect_images_recursive(root_dir: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg")
    image_paths: list[str] = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(root, f))
    image_paths.sort()
    return image_paths


# ================= CHUONG TRINH CHINH =================
count = 0
image_paths = collect_images_recursive(INPUT_DIR)

print(f"Bat dau chay kiem thu GrabCut tren {min(MAX_TEST_IMAGES, len(image_paths))} anh...")

for img_path in image_paths:
    if count >= MAX_TEST_IMAGES:
        break

    img = cv2.imread(img_path)
    if img is None:
        continue

    # 1. Chay ham tao Mask (giong trong feature_extractors.py)
    mask_binary = debug_grabcut(img, margin=5)

    # 2. Tao anh Mask 3 kenh mau
    mask_3d = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)

    # 3. Ep Mask vao anh goc de lay foreground
    mask_normalized = (mask_binary / 255.0).astype(np.uint8)
    foreground = img * mask_normalized[:, :, np.newaxis]

    # 4. Truc quan hoa: Original | Mask | Result
    img_with_rect = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img_with_rect, (20, 20), (w - 20, h - 20), (0, 0, 255), 2)
    combined_display = np.hstack((img_with_rect, mask_3d, foreground))

    # Luu ra file de kiem tra
    rel_name = os.path.basename(img_path)
    out_path = os.path.join(OUTPUT_DIR, f"10_debug_{count+1:02d}_{rel_name}")
    cv2.imwrite(out_path, combined_display)

    count += 1

print(f"Da xuat {count} file kiem tra tai thu muc: {OUTPUT_DIR}")
print("Hay mo thu muc do de danh gia chat luong thuat toan!")
