"""
2_app.py
========
Web demo Streamlit — Hệ thống CBIR tìm kiếm ảnh chim.

Workflow:
  1. Load ``features.npy``, ``pca_model.pkl``, ``database.db``.
  2. Người dùng tải ảnh truy vấn.
  3. Trích xuất đặc trưng -> PCA transform -> cosine distance.
  4. Hiển thị Top-K kết quả tương tự.

Usage::

    streamlit run 2_app.py
"""

from __future__ import annotations

import os
import pickle
import sqlite3
import tempfile

import numpy as np
import streamlit as st
from PIL import Image
from scipy.spatial.distance import cdist

from feature_extractors import extract_raw_features

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "database.db")
FEATURES_PATH = os.path.join(BASE_DIR, "database", "features.npy")
PCA_PATH = os.path.join(BASE_DIR, "database", "pca_model.pkl")


# -----------------------------------------------------------------------
# Load resources (cached)
# -----------------------------------------------------------------------
@st.cache_resource
def load_resources():
    """Load feature matrix, PCA model, và danh sách image paths."""
    features = np.load(FEATURES_PATH).astype(np.float64)

    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT ImageID, ImagePath FROM Images ORDER BY ImageID"
    ).fetchall()
    conn.close()

    paths = [row[1] for row in rows]
    return features, pca, paths


# -----------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------
def search_topk(
    query_vec: np.ndarray,
    db_matrix: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distance -> top-k nearest indices & distances."""
    dists = cdist(query_vec.reshape(1, -1), db_matrix, metric="cosine").flatten()
    idx = np.argsort(dists)[:top_k]
    return idx, dists[idx]


# -----------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------
st.set_page_config(page_title="Bird CBIR System", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>Bird Image Search (CBIR)</h1>"
    "<p style='text-align:center;color:gray;'>"
    "Content-Based Image Retrieval &mdash; Classical Computer Vision features"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# --- Pre-flight check --------------------------------------------------
required = [DB_PATH, FEATURES_PATH, PCA_PATH]
if not all(os.path.exists(p) for p in required):
    st.error(
        "Database chưa được xây dựng!  "
        "Hãy chạy **`python 1_extract_offline.py`** trước."
    )
    st.stop()

features, pca, image_paths = load_resources()

# --- Sidebar -----------------------------------------------------------
st.sidebar.header("Cài đặt")
top_k = st.sidebar.slider("Số kết quả hiển thị", min_value=1, max_value=20, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Database:** {len(image_paths)} ảnh\n\n"
    f"**Feature dim:** {features.shape[1]}"
)
st.sidebar.markdown(
    "### Hướng dẫn\n"
    "1. Tải lên ảnh chim cần tìm kiếm.\n"
    "2. Chọn số kết quả mong muốn.\n"
    "3. Xem kết quả tương tự bên dưới."
)

# --- Upload ------------------------------------------------------------
uploaded = st.file_uploader(
    "Tải ảnh chim truy vấn",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is not None:
    pil_img = Image.open(uploaded)

    col_q, col_r = st.columns([1, 3])
    with col_q:
        st.subheader("Ảnh truy vấn")
        st.image(pil_img, use_container_width=True)

    # Save to temp for OpenCV
    suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getvalue())

        with st.spinner("Đang trích xuất đặc trưng & tìm kiếm..."):
            raw = extract_raw_features(tmp_path)
            if raw is None:
                st.error("Không thể đọc ảnh. Vui lòng thử ảnh khác.")
                st.stop()

            query_pca = pca.transform(raw.reshape(1, -1)).astype(np.float64)
            top_idx, top_dist = search_topk(query_pca, features, top_k=top_k)
    except Exception as exc:
        st.error(f"Lỗi xử lý: {exc}")
        st.stop()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # --- Display results -----------------------------------------------
    with col_r:
        st.subheader(f"Top {top_k} kết quả tương tự")

    n_cols = min(top_k, 5)
    for row_start in range(0, len(top_idx), n_cols):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            i = row_start + j
            if i >= len(top_idx):
                break
            idx = top_idx[i]
            dist = top_dist[i]
            sim = max(0.0, (1.0 - dist)) * 100

            rel_path = image_paths[idx]
            abs_path = os.path.join(BASE_DIR, rel_path)

            with cols[j]:
                if os.path.exists(abs_path):
                    st.image(abs_path, use_container_width=True)
                else:
                    st.warning(f"Không tìm thấy:\n{rel_path}")
                st.caption(f"**#{i + 1}**  —  {sim:.1f}% tương tự")
                st.caption(os.path.basename(rel_path))
