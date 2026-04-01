"""
Tạo dữ liệu mẫu cho dự án
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_sample_bird_images(output_dir="data/raw", num_images=15):
    """
    Tạo các ảnh chim mẫu (hình tròn màu giả lập)
    
    Args:
        output_dir: thư mục lưu ảnh
        num_images: số lượng ảnh cần tạo
    """
    os.makedirs(output_dir, exist_ok=True)
    
    bird_names = [
        "eagle", "sparrow", "owl", "parrot", "penguin",
        "flamingo", "peacock", "raven", "swan", "hummingbird",
        "pigeon", "crow", "duck", "hawk", "dove"
    ]
    
    colors = [
        ((200, 100, 50), (100, 50, 20)),    # Brown
        ((100, 150, 200), (50, 100, 150)),  # Blue
        ((150, 200, 100), (100, 150, 50)),  # Green
        ((200, 150, 100), (150, 100, 50)),  # Beige
        ((100, 100, 200), (50, 50, 150)),   # Purple
        ((200, 100, 100), (150, 50, 50)),   # Red
    ]
    
    for i in range(min(num_images, len(bird_names))):
        # Tạo ảnh cơ bản
        size = (224, 224)
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Chọn màu ngẫu nhiên
        color_idx = i % len(colors)
        main_color, dark_color = colors[color_idx]
        
        # Vẽ hình chim đơn giản (vòng tròn + chi tiết)
        center_x, center_y = 112, 112
        radius = 60
        
        # Thân
        draw.ellipse(
            [center_x - radius, center_y - radius, 
             center_x + radius, center_y + radius],
            fill=main_color, outline=dark_color, width=2
        )
        
        # Cánh
        draw.ellipse(
            [center_x - radius - 30, center_y - 20,
             center_x - radius + 10, center_y + 20],
            fill=main_color, outline=dark_color, width=2
        )
        
        # Mắt
        eye_x = center_x + 20
        eye_y = center_y - 15
        draw.ellipse(
            [eye_x - 8, eye_y - 8, eye_x + 8, eye_y + 8],
            fill='black'
        )
        draw.ellipse(
            [eye_x - 4, eye_y - 4, eye_x + 4, eye_y + 4],
            fill='white'
        )
        
        # Mỏ
        draw.polygon(
            [(center_x + radius, center_y), 
             (center_x + radius + 20, center_y - 10),
             (center_x + radius + 20, center_y + 10)],
            fill=dark_color, outline=dark_color
        )
        
        # Lưu ảnh
        filename = f"{bird_names[i]}_{i+1:02d}.jpg"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        print(f"✓ Tạo ảnh mẫu: {filename}")
    
    print(f"\n✅ Đã tạo {min(num_images, len(bird_names))} ảnh mẫu trong {output_dir}/")

if __name__ == "__main__":
    create_sample_bird_images()
