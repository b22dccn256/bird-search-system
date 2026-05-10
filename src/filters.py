"""
Các hàm lọc ảnh theo tiêu chí chất lượng và góc chụp
"""

import pandas as pd
import numpy as np


def filter_horizontal_angle(df: pd.DataFrame, 
                           aspect_min: float = 1.3,
                           aspect_max: float = 4.0,
                           margin_min: float = 0.15) -> pd.Series:
    """
    Tạo boolean mask cho ảnh có góc chụp ngang
    
    Parameters:
    - df: DataFrame với columns [x, y, w, h, width, height]
    - aspect_min/max: khoảng aspect ratio chấp nhận được
    - margin_min: margin tối thiểu từ mép ảnh (loại top-down/bottom-up)
    
    Returns:
    - Series boolean: True nếu đạt tiêu chí góc ngang
    """
    # Aspect ratio check
    aspect_ratio = df['w'] / df['h'].replace(0, np.nan)
    aspect_ok = (aspect_ratio >= aspect_min) & (aspect_ratio <= aspect_max)
    
    # Vertical margin check (loại góc từ trên xuống/dưới lên)
    top_margin = df['y'] / df['height'].replace(0, np.nan)
    bottom_margin = (df['height'] - df['y'] - df['h']) / df['height'].replace(0, np.nan)
    vertical_ok = (top_margin >= margin_min) & (bottom_margin >= margin_min)
    
    # Horizontal position check (loại góc chéo quá lệch)
    left_margin = df['x'] / df['width'].replace(0, np.nan)
    right_margin = (df['width'] - df['x'] - df['w']) / df['width'].replace(0, np.nan)
    horizontal_ok = (left_margin <= 0.8) & (right_margin <= 0.8)
    
    return aspect_ok & vertical_ok & horizontal_ok


def filter_chim_ratio(df: pd.DataFrame, 
                     min_ratio: float = 0.40,
                     max_ratio: float = 0.70) -> pd.Series:
    """Lọc theo tỷ lệ chim chiếm trong ảnh"""
    bbox_area = df['w'] * df['h']
    img_area = df['width'] * df['height']
    ratio = bbox_area / img_area.replace(0, np.nan)
    
    return (ratio >= min_ratio) & (ratio <= max_ratio)


def filter_bbox_size(df: pd.DataFrame, min_size: int = 50) -> pd.Series:
    """Lọc bbox quá nhỏ (chất lượng thấp)"""
    return (df['w'] >= min_size) & (df['h'] >= min_size)


def apply_all_filters(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Áp dụng tất cả filter và trả về DataFrame đã lọc"""
    mask = pd.Series(True, index=df.index)
    
    # Horizontal angle
    mask &= filter_horizontal_angle(df, 
                                   aspect_min=kwargs.get('aspect_min', 1.3),
                                   aspect_max=kwargs.get('aspect_max', 4.0),
                                   margin_min=kwargs.get('margin_min', 0.15))
    
    # Chim ratio
    mask &= filter_chim_ratio(df,
                             min_ratio=kwargs.get('chim_min', 0.40),
                             max_ratio=kwargs.get('chim_max', 0.70))
    
    # Bbox size
    mask &= filter_bbox_size(df, min_size=kwargs.get('bbox_min', 50))
    
    return df[mask].copy()