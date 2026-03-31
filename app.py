"""
Ứng dụng Streamlit để tìm kiếm chim
"""

import streamlit as st
from PIL import Image
import numpy as np
from src.search import load_database, search_similar_birds


st.set_page_config(page_title="Bird Search System", layout="wide")

st.title("🐦 Bird Search System")
st.write("Tải lên ảnh chim để tìm kiếm những loài chim tương tự")

# Load database
db_dict = load_database("./database")

# Upload ảnh
uploaded_file = st.file_uploader("Chọn ảnh chim:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh đầu vào")
        st.image(uploaded_file)
    
    with col2:
        st.subheader("Kết quả tìm kiếm tương tự")
        k = st.slider("Số kết quả cần hiển thị:", 1, 20, 5)
        
        # Tìm kiếm
        results = search_similar_birds(uploaded_file, db_dict, k=k)
        
        for i, (filename, similarity) in enumerate(results, 1):
            st.write(f"**{i}. {filename}** - Độ tương tự: {similarity:.2%}")

st.sidebar.title("Thông tin")
st.sidebar.write("""
### Hướng dẫn sử dụng:
1. Tải lên ảnh chim muốn tìm kiếm
2. Chọn số lượng kết quả cần hiển thị
3. Xem danh sách những chim tương tự
""")
