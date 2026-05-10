import os
import cv2
import numpy as np
from ultralytics import YOLO

# 1. TẢI MÔ HÌNH NHẬN DIỆN TĨNH (YOLOv8n)
print("Đang tải mô hình YOLOv8 Object Detection...")
model = YOLO('yolov8n.pt') 

# THAY ĐỔI ĐƯỜNG DẪN THEO MÁY CỦA BẠN
RAW_DIR = "C:/Users/ADMIN/bird-search-system/bird-search-system/data/process_loc_tay"
PROCESSED_DIR = "C:/Users/ADMIN/bird-search-system/bird-search-system/data/processed_224"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Kích thước chuẩn
TARGET_SIZE = (224, 224) 
processed_count = 0

print("Bắt đầu tự động cắt ảnh (Chuẩn bị dữ liệu cho GrabCut)...")

for filename in os.listdir(RAW_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(RAW_DIR, filename)
        img = cv2.imread(img_path)
        
        if img is None: continue

        # 2. CHẠY NHẬN DIỆN KHUNG (OBJECT DETECTION)
        results = model(img, classes=[14], verbose=False) # class 14 là 'bird'
        
        # Nếu tìm thấy ít nhất 1 con chim
        if len(results[0].boxes) > 0:
            
            # --- BƯỚC A: LẤY TỌA ĐỘ VÀ CẮT KHUNG GỐC ---
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            cropped_img = img[y1:y2, x1:x2]
            
            if cropped_img.size == 0: continue
            
            # Lấy kích thước khung chữ nhật gốc
            h, w = cropped_img.shape[:2]
            
            # --- BƯỚC B: SMART PADDING (HÌNH VUÔNG + BẢO TOÀN TỶ LỆ) ---
            max_dim = max(h, w)
            
            # Bù viền để thành hình vuông
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            
            # CHÚ Ý: Dùng BORDER_REPLICATE để viền không bị đen, hỗ trợ GrabCut
            square_img = cv2.copyMakeBorder(
                cropped_img, top_pad, bottom_pad, left_pad, right_pad, 
                cv2.BORDER_REPLICATE
            )
            
            # --- BƯỚC C: TẠO LỀ (MARGIN) CHO GRABCUT HỌC HẬU CẢNH ---
            # Thiết lập con chim chiếm khoảng 65% khung hình
            # Phần 35% lề xung quanh chính là HẬU CẢNH để mồi cho GrabCut
            target_dim = int(max_dim * 1.5)
            pad_margin = (target_dim - max_dim) // 2
            
            final_padded_img = cv2.copyMakeBorder(
                square_img, pad_margin, pad_margin, pad_margin, pad_margin, 
                cv2.BORDER_REPLICATE
            )
            
            # --- BƯỚC D: RESIZE CHUẨN VÀ LƯU ---
            resized_img = cv2.resize(final_padded_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            out_path = os.path.join(PROCESSED_DIR, filename)
            cv2.imwrite(out_path, resized_img)
            processed_count += 1
                
        if processed_count % 100 == 0 and processed_count > 0:
            print(f"Đã xử lý cắt thành công {processed_count} ảnh...")

print(f"Hoàn tất! Tổng cộng đã cắt chuẩn bị {processed_count} ảnh cho GrabCut.")