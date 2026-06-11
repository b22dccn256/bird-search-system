# CHUONG 4: THUC NGHIEM VA DANH GIA

## 4.1. Muc tieu thuc nghiem

Chuong nay trinh bay cach danh gia hieu qua he thong CBIR tim kiem anh chim trong boi canh du lieu khong co nhan chuan duoc gan boi chuyen gia.  
Muc tieu gom:

- Danh gia dinh luong chat luong truy hoi theo `Top-K`.
- Danh gia toc do va kha nang dap ung thuc te cua he thong.
- Phan tich diem manh, diem yeu thong qua cac truong hop thanh cong/that bai.
- So sanh anh huong cua cac nhom dac trung thong qua thuc nghiem ablation.

---

## 4.2. Du lieu va giao thuc danh gia

### 4.2.1. Dac diem du lieu

Bo du lieu trong du an khong kem theo bo nhan ground-truth chuan.  
Thong tin nhan co the suy ra mot phan tu:

- Ten file anh.
- Ten thu muc chua anh (neu cau truc thu muc theo loai).

Do do, chuong nay su dung cach tiep can **weak-label evaluation** ket hop **danh gia dinh tinh**.

### 4.2.2. Dinh nghia nhan yeu (weak label)

Voi moi anh `I`, xac dinh nhan yeu `y(I)` bang mot trong hai cach:

1. **Theo ten thu muc cha:** `dataset/<species_name>/image_xxx.jpg`.
2. **Theo quy tac ten file:** tach token dai dien cho loai chim trong ten file.

Hai anh `I_a`, `I_b` duoc coi la lien quan neu:

\[
\mathrm{rel}(I_a, I_b)=
\begin{cases}
1, & y(I_a)=y(I_b)\\
0, & \text{nguoc lai}
\end{cases}
\]

> Luu y hoc thuat: nhan yeu co the chua nhieu; vi vay can trinh bay ket qua kem phan tich sai so va gioi han.

### 4.2.3. Chia tap truy van

De dam bao tinh lap lai, de xuat protocol:

- Tong so anh: `N`.
- Chon ngau nhien `Q` anh lam tap query (vi du `Q = 100` hoac `Q = 20%N`).
- Phan con lai dung lam tap gallery.
- Co dinh random seed (vi du `seed = 42`).

Neu du lieu nho, co the dung leave-one-out: moi anh lan luot la query, gallery la toan bo anh con lai.

### 4.2.4. Cau hinh he thong trong thuc nghiem

- Anh duoc resize ve `224x224`.
- Vector raw: `1460` chieu.
- PCA toi da `512` chieu (`n_components = min(512, n_samples, 1460)`).
- Do tuong tu: cosine distance.
- Bao cao tren cac nguong `K = {1, 3, 5, 10, 20}` (dieu chinh theo kich thuoc du lieu).

---

## 4.3. Chi so danh gia

### 4.3.1. Precision@K

Voi moi query `q`, goi `TopK(q)` la tap K ket qua dau:

\[
P@K(q)=\frac{1}{K}\sum_{i=1}^{K}\mathrm{rel}(q, r_i)
\]

Trong do `r_i` la anh thu `i` trong danh sach ket qua.  
Gia tri trung binh tren toan bo query:

\[
\overline{P@K}=\frac{1}{|Q|}\sum_{q\in Q}P@K(q)
\]

### 4.3.2. Recall@K

Voi `R(q)` la tong so anh lien quan voi query `q` trong gallery:

\[
R@K(q)=\frac{\sum_{i=1}^{K}\mathrm{rel}(q, r_i)}{|R(q)|}
\]

\[
\overline{R@K}=\frac{1}{|Q|}\sum_{q\in Q}R@K(q)
\]

### 4.3.3. Average Precision va mAP@K

Average Precision cho query `q`:

\[
AP@K(q)=\frac{1}{\min(K, |R(q)|)}
\sum_{i=1}^{K} P@i(q)\cdot \mathrm{rel}(q, r_i)
\]

Mean Average Precision:

\[
mAP@K=\frac{1}{|Q|}\sum_{q\in Q}AP@K(q)
\]

`mAP@K` la chi so quan trong nhat vi danh gia dong thoi chat luong va thu tu xep hang.

### 4.3.4. nDCG@K (tuy chon)

Neu xay dung duoc muc lien quan nhieu cap (vi du 0/1/2), dung:

\[
DCG@K=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}
\]

\[
nDCG@K=\frac{DCG@K}{IDCG@K}
\]

### 4.3.5. Chi so hieu nang he thong

Ngoai do chinh xac truy hoi, can bao cao:

- **Latency/query (ms):**
  - Thoi gian extract dac trung query.
  - Thoi gian PCA transform.
  - Thoi gian tinh cosine + sort.
- **Thong luong (query/s)** khi chay lien tiep.
- **Dung luong luu tru:**
  - `features.npy`
  - `pca_model.pkl`
  - `database.db`

---

## 4.4. Thiet ke thuc nghiem

### 4.4.1. Thi nghiem chinh (Main experiment)

Muc tieu: danh gia hieu qua he thong day du (color + texture + shape + spatial).

Quy trinh:

1. Chon tap query `Q`.
2. Chay truy van cho tung query.
3. Lay Top-K ket qua.
4. So khop voi nhan yeu de tinh `P@K`, `R@K`, `mAP@K`.
5. Tong hop trung binh va do lech chuan.

### 4.4.2. Thi nghiem ablation

Muc tieu: danh gia dong gop cua tung nhom dac trung.  
De xuat cac cau hinh:

- C0: Full feature (co so).
- C1: Bo texture.
- C2: Bo shape.
- C3: Bo spatial.
- C4: Bo color.
- C5: Khong PCA (neu tai nguyen cho phep).

So sanh `mAP@K` va latency giua cac cau hinh.

### 4.4.3. Thi nghiem do ben (robustness)

Kiem tra query sau khi bien doi:

- Tang/giam do sang.
- Them nhieu nhe (Gaussian noise).
- Crop nhe.
- Nen JPEG chat luong thap.

Bao cao muc giam `P@K` va `mAP@K` so voi query goc.

---

## 4.5. Ket qua va phan tich

### 4.5.1. Bang ket qua dinh luong (mau)

| Cau hinh | P@1 | P@5 | P@10 | R@10 | mAP@10 | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| C0 Full |  |  |  |  |  |  |
| C1 -Texture |  |  |  |  |  |  |
| C2 -Shape |  |  |  |  |  |  |
| C3 -Spatial |  |  |  |  |  |  |
| C4 -Color |  |  |  |  |  |  |
| C5 No PCA |  |  |  |  |  |  |

Nhan xet nen tap trung:

- Cau hinh nao dat `mAP@10` cao nhat.
- Danh doi giua do chinh xac va toc do.
- Nhom dac trung nao anh huong manh nhat khi loai bo.

### 4.5.2. Danh gia dinh tinh

Chon 10-20 query dai dien, trinh bay:

- Query image.
- Top-5 ket qua.
- Nhan xet ngan:
  - Trung loai nhung khac pose/goc chup.
  - Sai truong hop do nen phuc tap.
  - Truong hop bi nhieu boi anh sang mau.

Nen chia theo 3 nhom case:

- **Case tot:** Top-5 phan lon dung.
- **Case trung binh:** Top-1 dung nhung Top-5 co nhieu anh lech.
- **Case kho:** query mo/nen roi/mat mot phan doi tuong.

### 4.5.3. Phan tich loi thuong gap

1. **Loi segmentation mask**
   - GrabCut tach sai foreground -> dac trung bi lech.
2. **Loai gan nhau ve mau/texture**
   - Cac loai co long tuong dong gay nham lan.
3. **Bien doi goc nhin manh**
   - Hinh dang va profile thay doi lon theo tu the.
4. **Anh nen chiem uu the**
   - Neu foreground nho, dac trung bi pha boi background.

---

## 4.6. Danh gia khi khong co nhan chuan

De dam bao tinh thuyet phuc, nen ket hop 3 tang danh gia:

### 4.6.1. Tang 1 - Nhan yeu tu dong

- Tinh metric tren nhan suy ra tu ten file/thu muc.
- Uu diem: nhanh, tai lap de dang.
- Nhuoc diem: co nhieu nhan.

### 4.6.2. Tang 2 - Hieu chinh thu cong mot phan

- Chon ngau nhien mot tap query (vi du 30 query).
- Nhan vien/chuyen gia kiem tra tay top-20 cua moi query.
- Tao bo ground-truth nho de doi chieu voi ket qua nhan yeu.

### 4.6.3. Tang 3 - Danh gia nguoi dung

- Moi 3-5 nguoi dung cham diem muc do "hop ly" cua Top-5 (thang 1-5).
- Bao cao diem trung binh va do lech giua nguoi cham.
- Cung cap goc nhin ung dung thuc te ben canh metric ky thuat.

---

## 4.7. Han che va huong cai thien

### 4.7.1. Han che

- Chua co bo ground-truth chuan quy mo lon.
- Nhan yeu gay sai lech trong danh gia dinh luong.
- So khop toan bo ma tran bang brute-force, chua toi uu cho CSDL rat lon.

### 4.7.2. Huong cai thien

- Xay dung bo nhan chuan mot phan (gold set) de danh gia chinh xac hon.
- Bo sung chi muc ANN (FAISS/Annoy) de tang toc do truy van quy mo lon.
- Thu nghiem re-ranking (query expansion, reciprocal neighbors).
- Ket hop metric hoc duoc (metric learning) trong giai doan sau.

---

## 4.8. Tieu ket chuong

Mac du khong co nhan chuan day du, he thong van co the duoc danh gia mot cach khoa hoc thong qua phuong phap nhan yeu ket hop danh gia dinh tinh va kiem chung thu cong co chon loc.  
Bo chi so `P@K`, `R@K`, `mAP@K` cung cac thuc nghiem ablation va hieu nang cho phep danh gia toan dien giua do chinh xac truy hoi va kha nang trien khai thuc te cua he thong CBIR.

Ket qua o chuong nay la co so de dua ra ket luan va de xuat huong phat trien tiep theo trong chuong sau.
