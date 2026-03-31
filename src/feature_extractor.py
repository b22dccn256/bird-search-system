"""
Trích đặc trưng từ ảnh chim
"""

import numpy as np
from PIL import Image
import cv2


def extract_features(image_path):
    """
    Trích đặc trưng từ một ảnh
    
    Args:
        image_path: đường dẫn tới file ảnh
        
    Returns:
        numpy array chứa các đặc trưng
    """
    pass


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Tiền xử lý ảnh: resize, chuẩn hóa
    
    Args:
        image_path: đường dẫn tới file ảnh
        target_size: kích thước mục tiêu
        
    Returns:
        ảnh đã xử lý
    """
    pass


def batch_extract_features(image_folder):
    """
    Trích đặc trưng từ một thư mục chứa nhiều ảnh
    
    Args:
        image_folder: đường dẫn tới thư mục ảnh
        
    Returns:
        numpy array chứa các đặc trưng
    """
    pass
