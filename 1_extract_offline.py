"""
1_extract_offline.py
====================
Script offline xây dựng CSDL cho hệ thống CBIR.

Hỗ trợ RESUME: nếu bị gián đoạn, chạy lại sẽ tiếp tục từ ảnh chưa xử lý.

Các bước:
  1. Tạo / mở SQLite database (bảng Images).
  2. Lặp qua ``dataset/``, bỏ qua ảnh đã extract, trích xuất raw features.
  3. Fit PCA giảm chiều -> 512 dim.
  4. Lưu: ``database/features.npy``, ``database/pca_model.pkl``,
          ``database/database.db``.

Usage::

    python 1_extract_offline.py

"""

from __future__ import annotations

import os
import pickle
import sqlite3
import sys
import time

import numpy as np
from sklearn.decomposition import PCA

from feature_extractors import extract_raw_features

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(OUTPUT_DIR, "database.db")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "features.npy")
RAW_FEATURES_PATH = os.path.join(OUTPUT_DIR, "raw_features.npy")
PCA_PATH = os.path.join(OUTPUT_DIR, "pca_model.pkl")
PCA_COMPONENTS = 512
CHECKPOINT_EVERY = 20

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def open_database(db_path: str) -> sqlite3.Connection:
    """Mở (hoặc tạo) SQLite database với bảng Images."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Images (
            ImageID   INTEGER PRIMARY KEY AUTOINCREMENT,
            ImagePath TEXT    NOT NULL UNIQUE
        )
    """)
    conn.commit()
    return conn


def get_existing_paths(conn: sqlite3.Connection) -> set[str]:
    """Lấy danh sách đường dẫn ảnh đã extract."""
    rows = conn.execute("SELECT ImagePath FROM Images").fetchall()
    return {row[0] for row in rows}


def collect_images(folder: str) -> list[str]:
    """Thu thập đường dẫn tuyệt đối tới tất cả ảnh hợp lệ trong folder."""
    paths: list[str] = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
                paths.append(os.path.join(root, f))
    return paths


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("   CBIR  --  Offline Feature Extraction & PCA")
    print("          (Resume supported)")
    print("=" * 60)

    # --- Check dataset ------------------------------------------------
    if not os.path.isdir(DATASET_DIR):
        print(f"\n[ERROR] Dataset folder not found: {DATASET_DIR}")
        print("        Create 'dataset/' and put bird images inside.")
        sys.exit(1)

    image_paths = collect_images(DATASET_DIR)
    n_total = len(image_paths)
    print(f"\n[INFO] Found {n_total} images in '{DATASET_DIR}'")

    if n_total == 0:
        print("[ERROR] No images found. Add images to dataset/.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Open SQLite (resume-safe) ------------------------------------
    conn = open_database(DB_PATH)
    existing = get_existing_paths(conn)
    print(f"[INFO] Database: {DB_PATH}  ({len(existing)} already extracted)")

    # --- Load existing raw features if resuming -----------------------
    if os.path.exists(RAW_FEATURES_PATH) and len(existing) > 0:
        prev_feats = list(np.load(RAW_FEATURES_PATH))
        prev_feats = [np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in prev_feats]
        print(f"[INFO] Resume: loaded {len(prev_feats)} vectors from checkpoint")
    else:
        prev_feats = []

    # Filter out already-processed images
    todo_paths = []
    for p in image_paths:
        rel = os.path.relpath(p, BASE_DIR)
        if rel not in existing:
            todo_paths.append(p)

    n_todo = len(todo_paths)
    if n_todo == 0 and len(existing) > 0:
        print(f"\n[INFO] All {n_total} images already extracted. Skipping to PCA...")
        all_feats = prev_feats
    else:
        # --- Feature extraction ---------------------------------------
        print(f"\n{'-' * 60}")
        print(f"  STEP 1/3 -- Extract features ({n_todo} remaining / {n_total} total)")
        print(f"{'-' * 60}")

        all_feats = prev_feats.copy()
        ok, fail = len(existing), 0
        t0 = time.time()
        width = len(str(n_todo))

        for i, path in enumerate(todo_paths, 1):
            rel = os.path.relpath(path, BASE_DIR)
            try:
                feat = extract_raw_features(path)
                if feat is None:
                    raise ValueError("cv2.imread returned None")
                feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                all_feats.append(feat)
                conn.execute(
                    "INSERT OR IGNORE INTO Images (ImagePath) VALUES (?)", (rel,)
                )
                ok += 1
            except Exception as exc:
                print(f"  [WARN] Skip {rel}: {exc}")
                fail += 1

            # Checkpoint every N images
            if i % CHECKPOINT_EVERY == 0 or i == n_todo:
                conn.commit()
                np.save(RAW_FEATURES_PATH, np.array(all_feats, dtype=np.float64))
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i:>{width}}/{n_todo}]  {rate:5.1f} img/s  "
                    f"|  OK={ok}  FAIL={fail}  (checkpoint saved)"
                )

        elapsed = time.time() - t0
        print(f"\n  => Done: {ok} images in {elapsed:.1f}s  ({fail} failed)")

    if len(all_feats) == 0:
        print("[ERROR] No features extracted.")
        conn.close()
        sys.exit(1)

    # --- PCA ----------------------------------------------------------
    X = np.array(all_feats, dtype=np.float64)
    n_non_finite = np.size(X) - np.count_nonzero(np.isfinite(X))
    if n_non_finite > 0:
        print(f"[WARN] Found {n_non_finite} non-finite values. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    raw_dim = X.shape[1]
    nc = min(PCA_COMPONENTS, X.shape[0], X.shape[1])

    print(f"\n{'-' * 60}")
    print(f"  STEP 2/3 -- PCA: {raw_dim} -> {nc} dims  ({X.shape[0]} samples)")
    print(f"{'-' * 60}")

    pca = PCA(n_components=nc, random_state=42)
    X_pca = pca.fit_transform(X)
    var_kept = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  Variance retained: {var_kept:.1f}%")
    print(f"  Output shape     : {X_pca.shape}")

    # --- Save ---------------------------------------------------------
    print(f"\n{'-' * 60}")
    print(f"  STEP 3/3 -- Save to '{OUTPUT_DIR}'")
    print(f"{'-' * 60}")

    np.save(FEATURES_PATH, X_pca.astype(np.float32))
    print(f"  features.npy   : shape {X_pca.shape}")

    with open(PCA_PATH, "wb") as f:
        pickle.dump(pca, f)
    print(f"  pca_model.pkl  : saved")

    conn.close()
    print(f"  database.db    : {len(all_feats)} records")

    # Cleanup raw checkpoint
    if os.path.exists(RAW_FEATURES_PATH):
        os.remove(RAW_FEATURES_PATH)
        print(f"  raw_features.npy : removed (no longer needed)")

    print(f"\n{'=' * 60}")
    print("   DONE!  Now run:  streamlit run 2_app.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
