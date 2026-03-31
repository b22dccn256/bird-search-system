"""
Tìm kiếm chim tương tự dựa trên ảnh đầu vào
"""

import numpy as np
import faiss
import pickle


def load_database(db_dir):
    """
    Load database từ disk
    
    Args:
        db_dir: đường dẫn tới thư mục database
        
    Returns:
        dict chứa FAISS index, scaler, filenames
    """
    pass


def search_similar_birds(query_image, db_dict, k=5):
    """
    Tìm kiếm k ảnh chim tương tự nhất với ảnh cung cấp
    
    Args:
        query_image: ảnh đầu vào (đường dẫn hoặc numpy array)
        db_dict: database chứa index và scaler
        k: số kết quả cần trả về
        
    Returns:
        list chứa tên file và độ tương tự
    """
    pass


def get_bird_info(filename):
    """
    Lấy thông tin chi tiết về một loài chim từ database
    
    Args:
        filename: tên file trong database
        
    Returns:
        dict chứa thông tin chi tiết
    """
    pass
