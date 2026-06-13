import os
import cv2
from ultralytics import YOLO

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "process_loc_tay_2")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "process_224")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Tạo thư mục đích nếu chưa có
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Tải mô hình YOLOv8n
print("Đang tải mô hình YOLOv8...")
model = YOLO(os.path.join(BASE_DIR, 'yolov8n.pt'))

# Các danh sách lưu kết quả
dat_yeu_cau = []
khong_dat_yeu_cau = []

print("Bắt đầu xử lý và lọc ảnh...")
all_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total_images = len(all_files)

for i, filename in enumerate(all_files, 1):
    img_path = os.path.join(RAW_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        khong_dat_yeu_cau.append(f"{filename} - Không đọc được ảnh")
        continue
    
    h, w = img.shape[:2]
    
    # 1. Kiểm tra kích thước ảnh gốc lớn hơn 224x224
    if h <= 224 or w <= 224:
        khong_dat_yeu_cau.append(f"{filename} - Kích thước ảnh gốc quá nhỏ ({w}x{h} <= 224x224)")
        continue
        
    # 2. Nhận diện đối tượng bằng YOLO (class 14 là 'bird')
    results = model(img, classes=[14], verbose=False)
    
    if len(results[0].boxes) == 0:
        khong_dat_yeu_cau.append(f"{filename} - Không phát hiện thấy chim bằng YOLOv8")
        continue
        
    # Lấy thông tin con chim có độ tin cậy cao nhất
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # Kích thước bounding box của chim
    bw = x2 - x1
    bh = y2 - y1
    
    # Trọng tâm của con chim
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # Tỷ lệ mở rộng bối cảnh tối ưu (1.35x theo báo cáo để chim chiếm ~55% khung hình)
    # Đây là hệ số giúp chim có kích thước vừa vặn trong khung hình 224x224 mà không bị quá to hay méo
    expansion_factor = 1.35
    crop_size = int(expansion_factor * max(bw, bh))
    
    # Xác định tọa độ vùng cắt hình vuông căn giữa con chim
    x_start = int(cx - crop_size / 2)
    x_end = int(cx + crop_size / 2)
    y_start = int(cy - crop_size / 2)
    y_end = int(cy + crop_size / 2)
    
    # 3. Kiểm tra xem vùng cắt hình vuông có nằm hoàn toàn trong ảnh gốc không
    # Nếu bị tràn viền, ta không thể căn giữa và cắt vuông trực tiếp mà không dùng padding gây nhiễu/biến dạng
    if x_start >= 0 and x_end <= w and y_start >= 0 and y_end <= h:
        # Cắt ảnh vuông căn giữa chim
        cropped_img = img[y_start:y_end, x_start:x_end]
        
        # Resize về 224x224 bằng phép nội suy bảo toàn tỷ lệ INTER_AREA
        resized_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Lưu vào thư mục process_224
        out_path = os.path.join(PROCESSED_DIR, filename)
        cv2.imwrite(out_path, resized_img)
        
        dat_yeu_cau.append(f"{filename} - Thành công (Cắt vuông và căn giữa)")
    else:
        # Nếu bị tràn viền khi mở rộng 1.35x, thử kiểm tra với hệ số 1.0 (chỉ cắt khít bounding box hình vuông)
        crop_size_min = max(bw, bh)
        x_start_min = int(cx - crop_size_min / 2)
        x_end_min = int(cx + crop_size_min / 2)
        y_start_min = int(cy - crop_size_min / 2)
        y_end_min = int(cy + crop_size_min / 2)
        
        if x_start_min >= 0 and x_end_min <= w and y_start_min >= 0 and y_end_min <= h:
            cropped_img = img[y_start_min:y_end_min, x_start_min:x_end_min]
            resized_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
            out_path = os.path.join(PROCESSED_DIR, filename)
            cv2.imwrite(out_path, resized_img)
            
            dat_yeu_cau.append(f"{filename} - Thành công (Cắt vuông căn giữa khít đối tượng)")
        else:
            # Không thể cắt vuông căn giữa chim mà không bị tràn viền ảnh gốc (phải dịch tâm hoặc dùng padding)
            khong_dat_yeu_cau.append(
                f"{filename} - Tràn viền khi cắt vuông căn giữa (Kích thước yêu cầu {crop_size}x{crop_size} vượt ngoài ảnh {w}x{h})"
            )

    if i % 100 == 0 or i == total_images:
        print(f"Đã xử lý {i}/{total_images} ảnh...")

# Ghi kết quả vào 2 file trong folder data
dat_file_path = os.path.join(DATA_DIR, "dat_yeu_cau.txt")
khong_dat_file_path = os.path.join(DATA_DIR, "khong_dat_yeu_cau.txt")

with open(dat_file_path, "w", encoding="utf-8") as f:
    f.write("\n".join(dat_yeu_cau))

with open(khong_dat_file_path, "w", encoding="utf-8") as f:
    f.write("\n".join(khong_dat_yeu_cau))

print("\n--- HOÀN TẤT ---")
print(f"Tổng số ảnh xử lý: {total_images}")
print(f"Số ảnh ĐẠT yêu cầu (đã chuyển vào process_224): {len(dat_yeu_cau)}")
print(f"Số ảnh KHÔNG ĐẠT yêu cầu: {len(khong_dat_yeu_cau)}")
print(f"Danh sách đạt yêu cầu lưu tại: {dat_file_path}")
print(f"Danh sách không đạt yêu cầu lưu tại: {khong_dat_file_path}")
