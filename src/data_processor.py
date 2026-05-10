"""
Xử lý dataset CUB-200 theo các tiêu chí đã đề ra:
- Lọc ảnh có góc chụp ngang (aspect ratio, vị trí vertical, kích thước bbox)
- Lọc ảnh có tỷ lệ chim trong ảnh hợp lý (40%-70%)
- Crop ảnh theo bbox với padding thông minh
- Resize về 224×224 giữ nguyên aspect ratio, padding nền xám nhạt
- Lưu ảnh đã xử lý vào thư mục processed
- Export một tập test riêng từ processed images
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import shutil

class CUB200Processor:
    """Xử lý ảnh chim từ dataset CUB-200"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
        # Đường dẫn theo cấu trúc project
        self.raw_dir = self.project_root / 'data' / 'raw'
        self.processed_dir = self.project_root / 'data' / 'processed'
        self.test_input_dir = self.project_root / 'data' / 'test_input'
        self.metadata_dir = self.project_root / 'database'
        
        # Tham số xử lý
        self.target_size = (224, 224)
        self.padding_ratio = 0.25  # 25% padding xung quanh bbox
        
        # Tiêu chí lọc góc ngang
        self.aspect_ratio_min = 1.3   # width/height >= 1.3
        self.aspect_ratio_max = 4.0   # width/height <= 4.0
        self.vertical_margin_min = 0.15  # bbox cách mép trên/dưới >= 15%
        
        # Tiêu chí tỷ lệ chim trong ảnh
        self.chim_ratio_min = 0.40
        self.chim_ratio_max = 0.70
        self.min_bbox_size = 50
        
        # Load metadata (sẽ gọi trong load_cub_metadata)
        self.df = None
        
    def setup_directories(self):
        """Tạo cấu trúc thư mục nếu chưa tồn tại"""
        for dir_path in [self.raw_dir, self.processed_dir, 
                        self.test_input_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directories ready at {self.project_root}")
        
    def load_cub_metadata(self, cub_root: str):
        """Load metadata từ dataset CUB-200 gốc"""
        cub_path = Path(cub_root)
        
        # Đọc các file metadata
        self.bboxes = pd.read_csv(
            cub_path / 'bounding_boxes.txt',
            sep=' ',
            names=['img_id', 'x', 'y', 'w', 'h']
        )
        
        self.img_sizes = pd.read_csv(
            cub_path / 'image_sizes.txt',
            sep=' ',
            names=['img_id', 'width', 'height']
        )
        
        self.images = pd.read_csv(
            cub_path / 'images.txt',
            sep=' ',
            names=['img_id', 'filepath']
        )
        
        # (Optional) Load part locations nếu muốn filter góc bằng part positions
        # self.parts = pd.read_csv(cub_path / 'parts' / 'part_locations.txt', ...)
        
        # Merge tất cả vào một dataframe
        self.df = self.images.merge(self.bboxes, on='img_id')
        self.df = self.df.merge(self.img_sizes, on='img_id')
        
        # Thêm cột species từ filepath
        self.df['species'] = self.df['filepath'].apply(
            lambda x: x.split('/')[0].split('.')[-1]
        )
        
        print(f"📊 Loaded {len(self.df)} images from CUB-200")
        return self.df
    
    # ========== HÀM FILTER GÓC NGANG ==========
    
    def calculate_aspect_ratio(self, row: pd.Series) -> float:
        """Tính aspect ratio của bounding box"""
        return row['w'] / row['h'] if row['h'] > 0 else 0
    
    def calculate_chim_ratio(self, row: pd.Series) -> float:
        """Tính tỷ lệ diện tích chim trong ảnh"""
        bbox_area = row['w'] * row['h']
        img_area = row['width'] * row['height']
        return bbox_area / img_area if img_area > 0 else 0
    
    def is_horizontal_angle(self, row: pd.Series) -> bool:
        """
        Kiểm tra ảnh có phải góc chụp ngang không
        Kết hợp 3 tiêu chí: aspect ratio, vị trí vertical, kích thước bbox
        """
        # 1. Aspect ratio: width > height (chim nhìn nghiêng)
        aspect_ratio = self.calculate_aspect_ratio(row)
        if not (self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max):
            return False
        
        # 2. Vị trí vertical: bbox không sát mép trên/dưới (loại góc top-down/bottom-up)
        top_margin = row['y'] / row['height'] if row['height'] > 0 else 0
        bottom_margin = (row['height'] - (row['y'] + row['h'])) / row['height'] if row['height'] > 0 else 0
        
        if top_margin < self.vertical_margin_min or bottom_margin < self.vertical_margin_min:
            return False
        
        # 3. Bbox không quá lệch ngang (loại góc chéo mạnh)
        left_margin = row['x'] / row['width'] if row['width'] > 0 else 0
        right_margin = (row['width'] - (row['x'] + row['w'])) / row['width'] if row['width'] > 0 else 0
        
        # Cho phép lệch một chút nhưng không quá 80% về một phía
        if left_margin > 0.8 or right_margin > 0.8:
            return False
            
        return True
    
    def filter_by_criteria(self) -> pd.DataFrame:
        """Lọc ảnh theo tất cả tiêu chí"""
        if self.df is None:
            raise ValueError("Call load_cub_metadata() first")
        
        # Tính các metrics
        self.df['aspect_ratio'] = self.df.apply(self.calculate_aspect_ratio, axis=1)
        self.df['chim_ratio'] = self.df.apply(self.calculate_chim_ratio, axis=1)
        
        # Apply filters
        filtered = self.df[
            # Tỷ lệ chim trong ảnh
            (self.df['chim_ratio'] >= self.chim_ratio_min) & 
            (self.df['chim_ratio'] <= self.chim_ratio_max) &
            
            # Kích thước bbox tối thiểu
            (self.df['w'] >= self.min_bbox_size) & 
            (self.df['h'] >= self.min_bbox_size) &
            
            # Góc chụp ngang
            (self.df.apply(self.is_horizontal_angle, axis=1))
        ].copy()
        
        print(f"🔍 Filtered: {len(self.df)} → {len(filtered)} images")
        return filtered
    
    # ========== HÀM XỬ LÝ ẢNH ==========
    
    def crop_with_smart_padding(self, img: np.ndarray, row: pd.Series) -> np.ndarray:
        """
        Crop ảnh theo bbox với padding thông minh
        Giữ chim ở trung tâm, thêm không gian xung quanh tự nhiên
        """
        h, w = img.shape[:2]
        x, y, bw, bh = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        
        # Tính padding dựa trên kích thước bbox
        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)
        
        # Tính tọa độ crop với clamp để không vượt biên ảnh
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        
        # Crop
        cropped = img[y1:y2, x1:x2]
        
        return cropped
    
    def resize_keep_aspect(self, img: np.ndarray) -> np.ndarray:
        """
        Resize ảnh giữ nguyên aspect ratio, padding về 224×224
        Tránh làm méo hình dáng chim
        """
        h, w = img.shape[:2]
        target_w, target_h = self.target_size
        
        # Tính tỷ lệ resize
        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # Resize với interpolation chất lượng cao
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Tạo canvas 224×224 với nền trắng (hoặc xám nhạt)
        canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)  # RGB: 245 = xám rất nhạt
        
        # Đặt ảnh resize vào giữa canvas
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return canvas
    
    def process_single_image(self, row: pd.Series, cub_root: str) -> Optional[np.ndarray]:
        """Xử lý một ảnh: đọc → crop → resize"""
        try:
            # Đọc ảnh gốc
            img_path = Path(cub_root) / 'images' / row['filepath']
            if not img_path.exists():
                return None
                
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            # Crop với padding
            cropped = self.crop_with_smart_padding(img, row)
            
            # Kiểm tra kích thước tối thiểu sau crop
            if cropped.shape[0] < 40 or cropped.shape[1] < 40:
                return None
            
            # Resize về 224×224
            final = self.resize_keep_aspect(cropped)
            
            return final
            
        except Exception as e:
            print(f"⚠️  Error processing {row['filepath']}: {e}")
            return None
    
    def process_batch(self, filtered_df: pd.DataFrame, cub_root: str, 
                     output_dir: Optional[Path] = None) -> dict:
        """Xử lý hàng loạt ảnh và lưu vào thư mục"""
        if output_dir is None:
            output_dir = self.processed_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total': len(filtered_df),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'by_species': {}
        }
        
        for idx, row in filtered_df.iterrows():
            result = self.process_single_image(row, cub_root)
            
            if result is None:
                stats['skipped'] += 1
                continue
            
            # Tạo tên file: species_id_original.jpg
            species = row['species']
            original_name = Path(row['filepath']).stem
            output_name = f"{species}_{row['img_id']:03d}_{original_name}.jpg"
            output_path = output_dir / output_name
            
            # Lưu với chất lượng cao
            cv2.imwrite(str(output_path), result, 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            stats['success'] += 1
            stats['by_species'][species] = stats['by_species'].get(species, 0) + 1
            
            # Progress log
            if stats['success'] % 100 == 0:
                print(f"  ✓ Processed {stats['success']}/{stats['total']}...")
        
        # Save processing log
        log_path = self.metadata_dir / 'processing_log.csv'
        filtered_df.to_csv(log_path, index=False)
        
        return stats
    
    # ========== HÀM KIỂM TRA & EXPORT ==========
    
    def quick_quality_check(self, sample_size: int = 12) -> List[str]:
        """Kiểm tra nhanh chất lượng ảnh đã xử lý"""
        processed_images = list(self.processed_dir.glob('*.jpg'))
        
        if len(processed_images) == 0:
            return ["⚠️  No processed images found!"]
        
        # Check kích thước đồng nhất
        sizes = []
        for img_path in processed_images[:min(sample_size, len(processed_images))]:
            img = cv2.imread(str(img_path))
            if img is not None:
                sizes.append(img.shape[:2])
        
        unique_sizes = set(sizes)
        messages = []
        
        if len(unique_sizes) == 1:
            messages.append(f"✅ All images: {sizes[0]}")
        else:
            messages.append(f"⚠️  Found {len(unique_sizes)} different sizes: {unique_sizes}")
        
        # Check số lượng
        total = len(processed_images)
        messages.append(f"📦 Total processed: {total} images")
        
        if total >= 500:
            messages.append(f"✅ Met requirement: {total} >= 500")
        else:
            messages.append(f"⚠️  Need more: {total} < 500")
        
        # Check species diversity
        species_count = len(set(f.stem.split('_')[0] for f in processed_images))
        messages.append(f"🦜 Species diversity: {species_count} species")
        
        return messages
    
    def export_test_set(self, num_samples: int = 25, seed: int = 42):
        """Export một tập test riêng từ processed images"""
        processed_images = list(self.processed_dir.glob('*.jpg'))
        
        if len(processed_images) < num_samples:
            print(f"⚠️  Not enough images for test set: {len(processed_images)} < {num_samples}")
            return
        
        # Chọn ngẫu nhiên, đảm bảo đa dạng species
        np.random.seed(seed)
        
        # Group by species
        by_species = {}
        for img_path in processed_images:
            species = img_path.stem.split('_')[0]
            if species not in by_species:
                by_species[species] = []
            by_species[species].append(img_path)
        
        # Chọn đều từ mỗi species
        selected = []
        species_list = list(by_species.keys())
        np.random.shuffle(species_list)
        
        per_species = max(1, num_samples // len(species_list))
        
        for species in species_list:
            candidates = by_species[species]
            n = min(per_species, len(candidates))
            selected.extend(np.random.choice(candidates, n, replace=False))
            
            if len(selected) >= num_samples:
                selected = selected[:num_samples]
                break
        
        # Copy vào test_input
        self.test_input_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in selected:
            shutil.copy2(img_path, self.test_input_dir / img_path.name)
        
        print(f"✅ Exported {len(selected)} test images to {self.test_input_dir}")
    
    # ========== MAIN PIPELINE ==========
    
    def run_pipeline(self, cub_root: str, target_count: int = 500) -> dict:
        """Chạy toàn bộ pipeline xử lý dữ liệu"""
        print("🚀 Starting CUB-200 Processing Pipeline")
        print(f"📁 Project root: {self.project_root}")
        print(f"📦 CUB-200 source: {cub_root}\n")
        
        # Step 0: Setup
        self.setup_directories()
        
        # Step 1: Load metadata
        print("📥 Step 1: Loading CUB-200 metadata...")
        self.load_cub_metadata(cub_root)
        
        # Step 2: Filter by criteria
        print("\n🔍 Step 2: Filtering images by criteria...")
        print(f"   • Aspect ratio: {self.aspect_ratio_min}-{self.aspect_ratio_max}")
        print(f"   • Chim ratio: {self.chim_ratio_min}-{self.chim_ratio_max}")
        print(f"   • Vertical margin: >{self.vertical_margin_min*100:.0f}%")
        
        filtered_df = self.filter_by_criteria()
        
        if len(filtered_df) < target_count:
            print(f"\n⚠️  Warning: Only {len(filtered_df)} images passed filters")
            print("   Consider relaxing criteria if you need more images.\n")
        
        # Step 3: Process and save
        print(f"\n⚙️  Step 3: Processing {len(filtered_df)} images...")
        stats = self.process_batch(filtered_df, cub_root)
        
        # Step 4: Quality check
        print(f"\n🔎 Step 4: Quality check...")
        for msg in self.quick_quality_check():
            print(f"   {msg}")
        
        # Step 5: Export test set
        print(f"\n🧪 Step 5: Exporting test set...")
        self.export_test_set(num_samples=25)
        
        # Summary
        print(f"\n{'='*50}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Input images:     {stats['total']}")
        print(f"Successfully:     {stats['success']}")
        print(f"Failed/Skipped:   {stats['failed'] + stats['skipped']}")
        print(f"Output directory: {self.processed_dir}")
        print(f"Test set:         {self.test_input_dir}")
        
        if stats['success'] >= target_count:
            print(f"\n✅ Requirement MET: {stats['success']} >= {target_count} images")
        else:
            print(f"\n⚠️  Requirement NOT MET: {stats['success']} < {target_count}")
            print("   Tip: Relax filter criteria in __init__ and re-run")
        
        return stats