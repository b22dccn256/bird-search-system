"""
Xây dựng database sử dụng sklearn nearest neighbors
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from .feature_extractor import batch_extract_features


def scale_features(features):
    """
    Chuẩn hóa các vector đặc trưng
    
    Args:
        features: numpy array chứa các vector đặc trưng
        
    Returns:
        tuple (features_scaled, scaler)
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, scaler


def build_search_index(features):
    """
    Xây dựng nearest neighbors index từ các vector đặc trưng
    
    Args:
        features: numpy array chứa các vector đặc trưng đã chuẩn hóa
        
    Returns:
        NearestNeighbors model đã fit
    """
    n_neighbors = min(10, features.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(features)
    return nbrs


def build_database(image_folder, output_dir="database"):
    """
    Xây dựng hoàn chỉnh database: trích đặc trưng, chuẩn hóa, tạo index
    
    Args:
        image_folder: đường dẫn tới thư mục ảnh
        output_dir: đường dẫn lưu trữ kết quả
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("BƯỚC 1: Trích đặc trưng từ ảnh")
    print("=" * 50)
    features, filenames = batch_extract_features(image_folder)
    print(f"\n✓ Đã trích đặc trưng: {features.shape[0]} ảnh, {features.shape[1]} chiều")
    
    # Lưu features gốc
    features_path = os.path.join(output_dir, "features.npy")
    np.save(features_path, features)
    print(f"✓ Lưu features gốc: {features_path}")
    
    print("\n" + "=" * 50)
    print("BƯỚC 2: Chuẩn hóa đặc trưng")
    print("=" * 50)
    features_scaled, scaler = scale_features(features)
    
    # Lưu features chuẩn hóa
    features_scaled_path = os.path.join(output_dir, "features_scaled.npy")
    np.save(features_scaled_path, features_scaled)
    print(f"✓ Lưu features chuẩn hóa: {features_scaled_path}")
    
    # Lưu scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Lưu scaler: {scaler_path}")
    
    print("\n" + "=" * 50)
    print("BƯỚC 3: Xây dựng search index")
    print("=" * 50)
    nbrs = build_search_index(features_scaled)
    
    # Lưu index
    index_path = os.path.join(output_dir, "search_index.pkl")
    with open(index_path, 'wb') as f:
        pickle.dump(nbrs, f)
    print(f"✓ Lưu search index: {index_path}")
    
    # Lưu danh sách filenames
    filenames_path = os.path.join(output_dir, "filenames.txt")
    with open(filenames_path, 'w') as f:
        for fn in filenames:
            f.write(fn + '\n')
    print(f"✓ Lưu danh sách filenames: {filenames_path}")
    
    print("\n" + "=" * 50)
    print("✅ HOÀN THÀNH XÂY DỰNG DATABASE")
    print("=" * 50)
    print(f"- Tổng ảnh: {len(filenames)}")
    print(f"- Thư mục output: {output_dir}")
    
    return nbrs, features_scaled, filenames
