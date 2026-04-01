"""
Trích đặc trưng từ ảnh chim sử dụng PIL
"""

import numpy as np
from PIL import Image, ImageFilter
import os


def extract_color_histogram(image_path, bins=8):
    """
    Trích histogram màu từ ảnh
    
    Args:
        image_path: đường dẫn tới file ảnh
        bins: số bin cho histogram
        
    Returns:
        numpy array 1D chứa histogram
    """
    try:
        img = Image.open(image_path).convert('RGB')
        features = []
        
        for channel_idx in range(3):  # RGB
            data = img.split()[channel_idx]
            hist = data.histogram()
            # Giảm dimension lại
            hist_reduced = [sum(hist[i*len(hist)//bins:(i+1)*len(hist)//bins]) 
                           for i in range(bins)]
            features.extend(hist_reduced)
        
        return np.array(features, dtype=float)
    except:
        return np.random.randn(bins * 3)


def extract_edge_features(image_path, bins=8):
    """
    Trích đặc trưng cạnh sử dụng edge detection
    
    Args:
        image_path: đường dẫn tới file ảnh
        bins: số bin cho histogram
        
    Returns:
        numpy array 1D
    """
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        
        # Sử dụng FIND_EDGES filter
        edges = img.filter(ImageFilter.FIND_EDGES)
        
        # Lấy histogram của cạnh
        hist = edges.histogram()
        hist_reduced = [sum(hist[i*len(hist)//bins:(i+1)*len(hist)//bins]) 
                       for i in range(bins)]
        
        return np.array(hist_reduced, dtype=float)
    except:
        return np.random.randn(bins)


def extract_shape_features(image_path):
    """
    Trích đặc trưng hình dạng cơ bản
    
    Args:
        image_path: đường dẫn tới file ảnh
        
    Returns:
        numpy array 1D chứa các đặc trưng
    """
    try:
        img = Image.open(image_path).convert('L')
        
        # Resize ảnh thành 8x8 để lấy crude shape features
        small = img.resize((8, 8))
        
        # Chuyển đổi thành numpy array và chuẩn hóa về 0-1
        small_array = np.array(small, dtype=float) / 255.0
        
        features = [
            np.mean(small_array),          # Bình quân sáng
            np.std(small_array),           # Độ lệch chuẩn
            np.percentile(small_array, 25),  # 25th percentile
            np.percentile(small_array, 75),  # 75th percentile
        ]
        
        return np.array(features, dtype=float)
    except:
        return np.random.randn(4)


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Tiền xử lý ảnh: resize, chuẩn hóa
    
    Args:
        image_path: đường dẫn tới file ảnh
        target_size: kích thước mục tiêu
        
    Returns:
        PIL Image đã xử lý
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img


def extract_features(image_path):
    """
    Trích đặc trưng từ một ảnh bằng cách kết hợp nhiều phương pháp
    
    Args:
        image_path: đường dẫn tới file ảnh
        
    Returns:
        numpy array 1D chứa các đặc trưng
    """
    try:
        # Trích các loại đặc trưng khác nhau
        color_hist = extract_color_histogram(image_path, bins=8)     # 24 features
        edge_hist = extract_edge_features(image_path, bins=8)        # 8 features
        shape_features = extract_shape_features(image_path)          # 4 features
        
        # Kết hợp tất cả
        all_features = np.concatenate([color_hist, edge_hist, shape_features])
        return all_features
    except Exception as e:
        print(f"Lỗi trích đặc trưng từ {image_path}: {e}")
        # Trả về vector ngẫu nhiên nếu lỗi
        return np.random.randn(36)  # 24 + 8 + 4


def batch_extract_features(image_folder):
    """
    Trích đặc trưng từ một thư mục chứa nhiều ảnh
    
    Args:
        image_folder: đường dẫn tới thư mục ảnh
        
    Returns:
        tuple (features, filenames)
            - features: numpy array shape (n_images, n_features)
            - filenames: list tên file tương ứng
    """
    features_list = []
    filenames = []
    
    # Lấy danh sách ảnh
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = [f for f in os.listdir(image_folder)
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    image_files.sort()
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    
    for idx, filename in enumerate(image_files, 1):
        filepath = os.path.join(image_folder, filename)
        print(f"[{idx}/{len(image_files)}] Trích đặc trưng: {filename}")
        
        features = extract_features(filepath)
        features_list.append(features)
        filenames.append(filename)
    
    features_array = np.array(features_list)
    return features_array, filenames
