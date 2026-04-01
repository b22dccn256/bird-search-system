"""
Tìm kiếm chim tương tự dựa trên ảnh đầu vào
"""

import numpy as np
import pickle
import os
from .feature_extractor import extract_features


def load_database(db_dir="database"):
    """
    Load database từ disk
    
    Args:
        db_dir: đường dẫn tới thư mục database
        
    Returns:
        dict chứa search index, scaler, filenames
    """
    # Load search index
    index_path = os.path.join(db_dir, "search_index.pkl")
    with open(index_path, 'rb') as f:
        nbrs = pickle.load(f)
    
    # Load scaler
    scaler_path = os.path.join(db_dir, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load filenames
    filenames_path = os.path.join(db_dir, "filenames.txt")
    with open(filenames_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]
    
    # Load features
    features_path = os.path.join(db_dir, "features_scaled.npy")
    features = np.load(features_path)
    
    return {
        'nbrs': nbrs,
        'scaler': scaler,
        'filenames': filenames,
        'features': features
    }


def search_similar_birds(query_image, db_dict, k=5):
    """
    Tìm kiếm k ảnh chim tương tự nhất với ảnh cung cấp
    
    Args:
        query_image: ảnh đầu vào (đường dẫn file hoặc PIL Image)
        db_dict: database chứa index và scaler
        k: số kết quả cần trả về
        
    Returns:
        list tuple (filename, similarity_score)
    """
    # Xử lý query image
    if isinstance(query_image, str):
        # Nếu là đường dẫn file
        query_features = extract_features(query_image)
    else:
        # Nếu là PIL Image hoặc object khác
        # Lưu tạm thời và trích đặc trưng
        from PIL import Image
        import tempfile
        
        if not isinstance(query_image, Image.Image):
            query_image = Image.open(query_image)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            query_image.save(tmp.name)
            query_features = extract_features(tmp.name)
            os.unlink(tmp.name)
    
    # Chuẩn hóa query features
    query_features = query_features.reshape(1, -1)
    query_features_scaled = db_dict['scaler'].transform(query_features)
    
    # Tìm kiếm với sklearn NearestNeighbors
    nbrs = db_dict['nbrs']
    
    # Đảm bảo k không vượt quá số lượng neighbors của model
    k = min(k, nbrs.n_neighbors)
    
    # Tìm k neighbors gần nhất
    distances, indices = nbrs.kneighbors(query_features_scaled, n_neighbors=k)
    
    # Chuyển đổi distances thành similarity scores
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        filename = db_dict['filenames'][idx]
        # Chuyển Euclidean distance thành similarity (0-1)
        similarity = 1 / (1 + dist)
        results.append((filename, similarity))
    
    return results


def get_bird_info(filename, data_dir="data/raw"):
    """
    Lấy thông tin chi tiết về một loài chim từ database
    
    Args:
        filename: tên file trong database
        data_dir: thư mục chứa ảnh
        
    Returns:
        dict chứa thông tin chi tiết
    """
    filepath = os.path.join(data_dir, filename)
    
    info = {
        'filename': filename,
        'filepath': filepath,
        'exists': os.path.exists(filepath),
        'size': None,
        'bird_name': filename.split('_')[0]
    }
    
    if os.path.exists(filepath):
        info['size'] = os.path.getsize(filepath)
        from PIL import Image
        try:
            img = Image.open(filepath)
            info['image_size'] = img.size
        except:
            pass
    
    return info
