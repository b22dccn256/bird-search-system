import os
import cv2
from ultralytics import YOLO

# Tải mô hình YOLOv8 thu nhỏ (tự động tải về, chạy rất nhẹ)
print("Đang tải mô hình YOLOv8...")
model = YOLO('yolov8n.pt')
RAW_DIR = "C:/Users/ADMIN/bird-search-system/bird-search-system/data/process_loc_tay"  # Thư mục chứa 1200 ảnh bạn đã lọc
PROCESSED_DIR = "C:/Users/ADMIN/bird-search-system/bird-search-system/dataset"  # Thư mục lưu ảnh đã cắt
os.makedirs(PROCESSED_DIR, exist_ok=True)
TARGET_SIZE = (224, 224)
processed_count = 0
print("Bắt đầu tự động tìm và cắt chim...")
for filename in os.listdir(RAW_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(RAW_DIR, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        # Nhận diện đối tượng trong ảnh bằng YOLO
        results = model(img, classes=[14], verbose=False) # class 14 là 'bird'
        # Nếu tìm thấy ít nhất 1 con chim
        if len(results[0].boxes) > 0:
            # Lấy tọa độ của con chim đầu tiên (độ tin cậy cao nhất)
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Cắt ảnh theo tọa độ YOLO tìm được
            cropped_img = img[y1:y2, x1:x2]
            # Resize về 224x224
            if cropped_img.size > 0:
                resized_img = cv2.resize(cropped_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                # Lưu vào thư mục đích
                out_path = os.path.join(PROCESSED_DIR, filename)
                cv2.imwrite(out_path, resized_img)
                processed_count += 1
        if processed_count % 100 == 0 and processed_count > 0:
            print(f"Đã tự động cắt thành công {processed_count} ảnh...")
print(f"Hoàn tất! Tổng cộng đã cắt được {processed_count} ảnh có chứa chim.")