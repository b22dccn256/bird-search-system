# Tong quan he thong CBIR

## 1) Muc tieu he thong

He thong CBIR (Content-Based Image Retrieval) duoc xay dung de tim cac anh chim tuong tu dua tren noi dung hinh anh, khong phu thuoc vao nhan text thu cong.

Nguoi dung tai len 1 anh truy van, he thong se:
- Trich xuat vector dac trung tu anh truy van
- Chieu vector qua mo hinh PCA
- So khop voi vector cua toan bo tap du lieu
- Tra ve Top-K anh co do tuong tu cao nhat

> Luu y: He thong su dung **Computer Vision co dien** (khong dung Deep Learning).

---

## 2) Cong nghe va thu vien su dung

- **Python 3.10+**: ngon ngu chinh
- **OpenCV (`opencv-python`)**: doc/resize anh, GrabCut, morphology, Sobel, contour, moments
- **NumPy**: xu ly ma tran, vector, luu `features.npy`
- **SciPy**: FFT (`scipy.fft`), skewness (`scipy.stats`), khoang cach cosine (`scipy.spatial.distance.cdist`)
- **scikit-image**: LBP, GLCM/Haralick, Gabor, HOG
- **scikit-learn**: PCA giam chieu dac trung
- **SQLite**: luu metadata anh (`ImageID`, `ImagePath`)
- **Streamlit**: giao dien web demo tim kiem anh
- **Pickle**: luu/tai mo hinh PCA (`pca_model.pkl`)

---

## 3) Cac mo hinh / bo loc / dac trung duoc su dung

## 3.1 Tien xu ly va tach foreground

- **GrabCut**: tach doi tuong chim khoi nen bang segmentation co dien
- **Morphological Opening/Closing**: lam sach mask (loai nhieu, lap lo)
- Fallback Otsu threshold khi GrabCut loi

Ket qua la mask nhi phan (foreground/background) dung cho cac buoc dac trung sau.

## 3.2 Nhom dac trung mau sac (Color Features)

- Khong gian mau:
  - **HSV**
  - **CIE Lab**
- Dac trung:
  - Global Histogram theo tung kenh
  - Color Moments: **Mean, Std, Skewness** cho moi kenh

Muc tieu: bat thong tin phan bo mau tong quat va tinh chat thong ke mau.

## 3.3 Nhom dac trung ket cau (Texture Features)

- **LBP (Local Binary Pattern)**: ma hoa mau van cuc bo
- **EOH (Edge Orientation Histogram)**:
  - Dung Sobel tinh gradient
  - Lap histogram huong canh
- **GLCM / Haralick**:
  - Energy, Contrast, Homogeneity, Correlation
- **Gabor filter bank**:
  - Nhieu tan so + nhieu huong
  - Bat mau van co huong va tan so khac nhau
- **Stripe FFT (2D FFT)**:
  - Trich profile tu pho tan so
  - Bat tinh chu ky van long/cau truc lap lai

## 3.4 Nhom dac trung hinh dang (Shape Features)

Tinh tren mask foreground:
- **Hu Moments (7 moments)**: dac trung hinh dang bat bien
- Dac trung vo huong:
  - Area, Perimeter, Compactness, Circularity
- **Grid Mask Density**: mat do foreground tren luoi
- **Width Profile / Contour Profile**: profile bien theo truc ngang/doc
- **Radius Signature**: khoang cach tu trong tam den bien theo goc
- **HOG chuyen biet tren ROI**:
  - Cat bounding box cua mask
  - Resize ROI ve 64x64
  - HOG voi cau hinh cell 16x16, block 2x2, 9 bins, L2-Hys

## 3.5 Nhom dac trung khong gian (Spatial Features)

- Chia anh 224x224 thanh luoi **4x4**
- Tren moi o co foreground:
  - HSV histogram cuc bo
  - LBP cuc bo

Muc tieu: giu thong tin bo cuc khong gian, khong chi thong ke toan cuc.

## 3.6 Fusion va giam chieu

- Noi (concatenate) tat ca nhom dac trung thanh vector 1D
- Chuan hoa **L2-Normalization**
- Dung **PCA** de giam chieu ve 512 de:
  - giam nhieu
  - tiet kiem bo nho
  - tang toc so khop

---

## 4) Kien truc luu tru du lieu

- **Anh goc**: `dataset/`
- **SQLite metadata**: `database/database.db`
  - Bang `Images(ImageID, ImagePath)`
- **Ma tran vector sau PCA**: `database/features.npy`
- **Mo hinh PCA**: `database/pca_model.pkl`

---

## 5) Luong nghiep vu tim kiem khi nguoi dung tai anh len

## 5.1 Giai doan Offline (xay dung CSDL)

Script: `1_extract_offline.py`

1. Quet tat ca anh trong `dataset/`
2. Moi anh:
   - Trich vector raw bang `extract_raw_features`
   - Luu `ImagePath` vao SQLite
3. Gom toan bo vector thanh ma tran
4. Fit PCA va transform ve 512 chieu
5. Luu `features.npy` + `pca_model.pkl`

## 5.2 Giai doan Online (truy van tu nguoi dung)

App: `2_app.py` (Streamlit)

1. Nguoi dung upload anh truy van
2. He thong trich vector raw tu anh upload
3. Dua vector qua `pca_model.transform()`
4. Tinh khoang cach cosine den tat ca vector trong `features.npy`
5. Sap xep tang dan theo khoang cach (nho hon = giong hon)
6. Lay Top-K ket qua
7. Tra ve duong dan anh tu bang `Images` va hien thi len giao dien

---

## 6) Diem manh va han che

### Diem manh

- Khong phu thuoc Deep Learning, de trien khai
- Dac trung da dang (mau, ket cau, hinh dang, bo cuc)
- Kien truc tach bach offline/online ro rang
- De mo rong them metric hoac module dac trung moi

### Han che

- Toc do extract offline cham hon deep features pre-trained
- Do chinh xac phu thuoc chat luong segmentation mask (GrabCut)
- PCA mat mot phan thong tin khi giam chieu

---

## 7) Huong nang cap de xuat

- Dung ANN index (FAISS/Annoy) de tang toc truy van tap lon
- Them co che re-ranking cho Top-K
- Luu them metadata loai chim de hien thi giau thong tin hon
- Them dashboard danh gia retrieval (Precision@K, mAP)
- Bo sung logging va monitoring cho pipeline offline

