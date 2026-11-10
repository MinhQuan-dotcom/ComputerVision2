# Bộ ảnh Oxford Affine (đã xử lý)

Nguồn: Visual Geometry Group, Đại học Oxford — Affine Covariant Regions Datasets, http://www.robots.ox.ac.uk/~vgg/data/data-aff.html. Bài báo đi kèm: K. Mikolajczyk et al., "A Comparison of Affine Region Detectors," IJCV, vol. 65, no. 1/2, pp. 43–72, 2005. Cần ghi nguồn này vào phần tài liệu tham khảo của notebook.

## Đã làm gì

- Đổi toàn bộ ảnh sang ảnh xám.
- Thu nhỏ để cạnh dài nhất là 512 px (`cv2.INTER_AREA`), lưu dạng PNG.
- Tính lại homography cho ảnh đã thu nhỏ. File `H1to{k}.txt` chiếu điểm từ `img1.png` sang `img{k}.png` **của bản đã thu nhỏ**, không dùng được cho ảnh gốc.
- Đã kiểm tra: chiếu ảnh 1 sang ảnh k bằng homography mới thì sai khác với ảnh k nhỏ hơn hẳn so với khi cố ý dịch lệch 3 px, ở cả 8 bộ.

Script làm việc này: `tools/chuan_bi_oxford.py`.

## Mỗi bộ thay đổi yếu tố gì

| Bộ | Kích thước sau xử lý | Yếu tố thay đổi từ img1 đến img6 | Dùng cho |
| --- | --- | --- | --- |
| `boat` | 512 × 410 | Phóng to + xoay, cảnh có cấu trúc | **A1** (Harris–Laplace vs Harris) |
| `bark` | 512 × 343 | Phóng to + xoay, vân vỏ cây | **A1** |
| `graf` | 512 × 410 | Đổi góc nhìn | Tham khảo |
| `wall` | 512 × 358 (img1), 512 × 396 | Đổi góc nhìn | Tham khảo |
| `bikes` | 512 × 358 | Ảnh mờ dần | Tham khảo |
| `trees` | 512 × 358 | Ảnh mờ dần | Tham khảo |
| `leuven` | 512 × 341 | Ánh sáng tối dần | Có thể dùng cho A2 (tương phản không đều) |
| `ubc` | 512 × 410 | Nén JPEG mạnh dần | Tham khảo |

## Cách dùng

```python
from common import load_oxford
from n1_harris import harris_classic, repeatability

img1, _ = load_oxford("boat", 1)
img4, tf = load_oxford("boat", 4)     # tf(row, col) chiếu điểm từ img1 sang img4
rep = repeatability(harris_classic(img1), harris_classic(img4), tf, img4.shape)
```

## Kết quả nền của Harris cổ điển

Repeatability của `harris_classic` (tham số mặc định, eps = 3 px), từ img1 sang img2…img6:

| Bộ | Số điểm img1 | →2 | →3 | →4 | →5 | →6 |
| --- | --- | --- | --- | --- | --- | --- |
| boat | 595 | 0,76 | 0,61 | 0,46 | 0,38 | 0,27 |
| bark | 709 | 0,58 | 0,39 | 0,33 | 0,23 | 0,20 |
| graf | 462 | 0,79 | 0,59 | 0,51 | 0,43 | 0,31 |
| wall | 1243 | 0,87 | 0,48 | 0,45 | 0,37 | 0,26 |
| bikes | 211 | 0,90 | 0,89 | 0,77 | 0,75 | 0,69 |
| trees | 916 | 0,67 | 0,62 | 0,52 | 0,46 | 0,33 |
| leuven | 176 | 0,56 | 0,47 | 0,40 | 0,42 | 0,39 |
| ubc | 347 | 0,98 | 0,97 | 0,93 | 0,89 | 0,82 |

Ở boat và bark, repeatability của Harris giảm mạnh khi mức phóng tăng. Người 2 có thể dùng bảng này làm đường cơ sở để so với Harris–Laplace.
