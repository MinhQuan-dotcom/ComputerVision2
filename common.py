"""Hàm và hằng số dùng chung cho cả nhóm (Người 1 phụ trách).

Quy ước (xem file Quy_uoc_chung):
- Ảnh: xám, float64, giá trị trong [0, 1], shape (H, W).
- Tọa độ luôn là (row, col).
"""
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter          # chỉ dùng để làm mờ Gauss
from scipy.ndimage import label as nd_label        # chỉ dùng ở phần phân tích

# ---------------------------------------------------------------- hằng số
S_RATIO = 0.7                                   # sigma_d = S_RATIO * sigma_i
NB4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]        # trên, dưới, trái, phải
NB8 = NB4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


# ---------------------------------------------------------------- nạp ảnh
def load_gray(path):
    """Đọc ảnh xám, trả float64 trong [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return img.astype(np.float64) / 255.0


# ---------------------------------------------------------------- phép biến đổi chuẩn
def make_scale(img, factor):
    """Đổi tỉ lệ ảnh bằng cv2.resize.

    -> (ảnh mới, transform). transform(row, col) của ảnh cũ -> (row, col) ở ảnh mới,
    theo quy ước tâm pixel của OpenCV.
    """
    H, W = img.shape
    new_h, new_w = int(round(H * factor)), int(round(W * factor))
    interp = cv2.INTER_LINEAR if factor >= 1 else cv2.INTER_AREA
    out = cv2.resize(img, (new_w, new_h), interpolation=interp)
    fy, fx = new_h / H, new_w / W

    def transform(r, c):
        r = np.asarray(r, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        return (r + 0.5) * fy - 0.5, (c + 0.5) * fx - 0.5

    return out, transform


def make_rotation(img, deg, fill=None):
    """Quay ảnh quanh tâm một góc deg (độ, dương = ngược chiều kim đồng hồ trên màn hình).

    Tự viết bằng numpy (ánh xạ ngược + nội suy song tuyến) vì cv2.warpAffine
    không có trong danh sách thư viện được phép. Ảnh mới giữ nguyên kích thước.
    Pixel rơi ra ngoài ảnh cũ được gán bằng `fill` (mặc định: trung vị của ảnh)
    để mép phần ảnh quay không tạo góc giả.

    -> (ảnh mới, transform). transform(row, col) của ảnh cũ -> (row, col) ở ảnh mới.
    """
    H, W = img.shape
    if fill is None:
        fill = float(np.median(img))
    t = math.radians(deg)
    cs, sn = math.cos(t), math.sin(t)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    # Với mỗi pixel (y', x') của ảnh mới, tìm vị trí (y, x) tương ứng ở ảnh cũ.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    xs = cx + cs * dx - sn * dy
    ys = cy + sn * dx + cs * dy

    inside = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wx = xs - x0
    wy = ys - y0
    val = (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy)
           + img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy)
    out = np.where(inside, val, fill)

    def transform(r, c):
        r = np.asarray(r, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        dxa, dya = c - cx, r - cy
        x_new = cx + cs * dxa + sn * dya
        y_new = cy - sn * dxa + cs * dya
        return y_new, x_new

    return out, transform


# ---------------------------------------------------------------- phần B
def remap_labels(labels, perm):
    """Đổi nhãn i >= 0 thành perm[i]; pixel -1 giữ nguyên.

    Dùng khi so hai kết quả chạy với danh sách hạt giống bị đảo thứ tự.
    """
    perm = np.asarray(perm)
    out = labels.copy()
    m = labels >= 0
    out[m] = perm[labels[m]]
    return out


# ---------------------------------------------------------------- đo đạc
def grid_occupancy(points, img_shape, n_cells=8):
    """BẢN TẠM — thay bằng hàm thầy cho khi có.

    Chia ảnh thành lưới n_cells x n_cells, trả tỉ lệ ô có ít nhất một điểm.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return 0.0
    H, W = img_shape[:2]
    gr = np.clip((pts[:, 0] * n_cells / H).astype(int), 0, n_cells - 1)
    gc = np.clip((pts[:, 1] * n_cells / W).astype(int), 0, n_cells - 1)
    return len(set(zip(gr.tolist(), gc.tolist()))) / float(n_cells * n_cells)


# ---------------------------------------------------------------- vẽ
def show_points(ax, img, pts, title="", color="r", circles=False):
    """Vẽ ảnh xám và các điểm. Cột 1 (col) là trục ngang.

    circles=True: pts có cột sigma (đầu ra harris_laplace), vẽ vòng tròn bán kính 2*sigma.
    """
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    pts = np.asarray(pts)
    if pts.size:
        if circles and pts.shape[1] >= 4:
            for r, c, s in pts[:, :3]:
                ax.add_patch(plt.Circle((c, r), 2 * s, color=color, fill=False, lw=0.8))
        else:
            ax.plot(pts[:, 1], pts[:, 0], "+", color=color, ms=7, mew=1.2)
    ax.set_title(title)
    ax.axis("off")


def show_labels(ax, labels, title=""):
    """Vẽ ảnh nhãn bằng tab20; pixel -1 hiển thị màu đen."""
    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad("black")
    ax.imshow(np.ma.masked_less(labels, 0), cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def time_ms(fn, *args, repeat=3, **kwargs):
    """Chạy fn `repeat` lần, trả (kết quả, thời gian nhỏ nhất tính bằng ms)."""
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, (time.perf_counter() - t0) * 1000)
    return out, best
