"""
Ứng dụng Streamlit để tìm kiếm chim
"""

import streamlit as st
from PIL import Image
import numpy as np
import os
from src.search import load_database, search_similar_birds, get_bird_info

st.set_page_config(page_title="Bird Search System", layout="wide")

st.title("🐦 Bird Search System")
st.write("Tải lên ảnh chim để tìm kiếm những loài chim tương tự")

# Kiểm tra xem database đã được xây dựng chưa
db_dir = "./database"
if not os.path.exists(os.path.join(db_dir, "search_index.pkl")):
    st.error("❌ Database chưa được xây dựng!")
    st.info("""
    Vui lòng chạy lệnh sau để xây dựng database:
    ```bash
    python build_database_script.py
    ```
    """)
    st.stop()

# Load database
try:
    db_dict = load_database(db_dir)
    st.success(f"✅ Đã load database thành công ({len(db_dict['filenames'])} ảnh)")
except Exception as e:
    st.error(f"❌ Lỗi khi load database: {str(e)}")
    st.stop()

# Upload ảnh
uploaded_file = st.file_uploader("Chọn ảnh chim:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh đầu vào")
        image = Image.open(uploaded_file)
        st.image(image)
    
    with col2:
        st.subheader("Kết quả tìm kiếm tương tự")
        k = st.slider("Số kết quả cần hiển thị:", 1, 20, 5)
        
        # Tìm kiếm
        try:
            results = search_similar_birds(uploaded_file, db_dict, k=k)
            
            st.write("**Top kết quả:**")
            for i, (filename, similarity) in enumerate(results, 1):
                # Lấy thông tin chi tiết
                info = get_bird_info(filename)
                
                col_rank, col_name, col_score = st.columns([0.5, 2, 1])
                with col_rank:
                    st.write(f"**{i}.**")
                with col_name:
                    st.write(f"**{filename}**")
                    st.caption(f"Loài: {info['bird_name']}")
                with col_score:
                    st.metric("Độ tương tự", f"{similarity:.1%}")
                
                st.divider()
        
        except Exception as e:
            st.error(f"❌ Lỗi tìm kiếm: {str(e)}")

st.sidebar.title("ℹ️ Thông tin")
st.sidebar.write("""
### Hướng dẫn sử dụng:
1. Tải lên ảnh chim muốn tìm kiếm
2. Chọn số lượng kết quả cần hiển thị
3. Xem danh sách những chim tương tự

### Công nghệ sử dụng:
- **ResNet50**: Trích đặc trưng ảnh
- **FAISS**: Tìm kiếm vector kNN hiệu năng cao
- **Streamlit**: Giao diện web

### Chuẩn bị dữ liệu:
1. Đặt ảnh trong `data/raw/`
2. Chạy: `python build_database_script.py`
3. Sau đó chạy: `streamlit run app.py`
""")
