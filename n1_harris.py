"""Phần A — Harris cổ điển và hàm đo repeatability (Người 1).

Không gọi cv2.cornerHarris, cv2.goodFeaturesToTrack hay skimage.feature.corner_*
trong file này.
"""
import math

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_blur(img, sigma):
    """Làm mờ Gauss. Được phép gọi scipy.ndimage.gaussian_filter."""
    img = np.asarray(img, dtype=np.float64)
    if sigma <= 0:
        return img.copy()
    return gaussian_filter(img, sigma, mode="reflect")


def structure_tensor(img, sigma_i, sigma_d):
    """Ma trận mô-men bậc hai theo công thức (1), đã nhân sigma_d**2.
    sigma_d: độ mờ trước khi lấy đạo hàm.
    sigma_i: bán kính cửa sổ lấy trung bình các tích đạo hàm.
    -> (Mxx, Mxy, Myy), mỗi phần tử cùng kích thước ảnh.
    """
    L = gaussian_blur(img, sigma_d)
    Ly = np.gradient(L, axis=0)      # đạo hàm theo hàng (hướng y)
    Lx = np.gradient(L, axis=1)      # đạo hàm theo cột (hướng x)
    s2 = sigma_d ** 2                # chuẩn hóa theo tỉ lệ (Lindeberg)
    Mxx = s2 * gaussian_blur(Lx * Lx, sigma_i)
    Mxy = s2 * gaussian_blur(Lx * Ly, sigma_i)
    Myy = s2 * gaussian_blur(Ly * Ly, sigma_i)
    return Mxx, Mxy, Myy


def harris_response(img, sigma_i, sigma_d, k=0.04):
    """Đáp ứng R theo công thức (2). -> mảng float cùng kích thước ảnh."""
    Mxx, Mxy, Myy = structure_tensor(img, sigma_i, sigma_d)
    det = Mxx * Myy - Mxy ** 2
    tr = Mxx + Myy
    return det - k * tr ** 2


def nms_2d(response, window=3):
    """Cực đại địa phương trong cửa sổ window x window. -> mảng bool.

    Phá hòa theo thứ tự quét raster: khi hai pixel trong cùng cửa sổ có R bằng
    nhau tuyệt đối (hay gặp ở ảnh tổng hợp đối xứng như bàn cờ), chỉ pixel đứng
    trước theo thứ tự quét được giữ. Nhờ vậy mỗi đỉnh chỉ cho một điểm.
    """
    if window % 2 == 0 or window < 1:
        raise ValueError("window phải là số lẻ dương")
    R = np.asarray(response, dtype=np.float64)
    p = window // 2
    P = np.pad(R, p, mode="constant", constant_values=-np.inf)
    win = np.lib.stride_tricks.sliding_window_view(P, (window, window))

    # Vị trí trong cửa sổ đứng trước tâm theo thứ tự quét: hàng trên, hoặc cùng hàng bên trái.
    dr, dc = np.mgrid[-p:p + 1, -p:p + 1]
    earlier = (dr < 0) | ((dr == 0) & (dc < 0))
    later = (dr > 0) | ((dr == 0) & (dc > 0))

    max_earlier = np.where(earlier, win, -np.inf).max(axis=(2, 3))
    max_later = np.where(later, win, -np.inf).max(axis=(2, 3))
    return (R > max_earlier) & (R >= max_later)


def harris_classic(img, sigma_i=2.0, sigma_d=1.4, k=0.04, thresh_rel=0.01):
    """Harris một tỉ lệ: R -> nms_2d -> ngưỡng tương đối theo max(R).
    -> mảng (N, 3): (row, col, response), sắp xếp giảm dần theo response.
    """
    R = harris_response(img, sigma_i, sigma_d, k)
    r_max = R.max()
    if not np.isfinite(r_max) or r_max <= 0:
        return np.zeros((0, 3))

    mask = nms_2d(R) & (R > thresh_rel * r_max) & (R > 0)

    # Bỏ vùng sát mép: ở đó đạo hàm bị méo do pad.
    m = int(math.ceil(3 * sigma_i))
    if m > 0:
        mask[:m, :] = False
        mask[-m:, :] = False
        mask[:, :m] = False
        mask[:, -m:] = False

    rows, cols = np.nonzero(mask)
    vals = R[rows, cols]
    order = np.argsort(-vals, kind="stable")
    pts = np.column_stack([rows, cols, vals]).astype(np.float64)
    return pts[order]


def repeatability(pts_a, pts_b, transform, img_shape, eps=3.0):
    """Tỉ lệ điểm của ảnh A tìm lại được trong ảnh B sau phép biến đổi.
    transform: hàm (row, col) của A -> (row, col) tương ứng ở B.
    Chỉ tính trên các điểm mà ảnh chiếu còn nằm trong B.
    -> số thực trong [0, 1].
    """
    A = np.asarray(pts_a, dtype=np.float64)
    B = np.asarray(pts_b, dtype=np.float64)
    if A.size == 0:
        return float("nan")
    A = A.reshape(len(A), -1)[:, :2]          # bỏ cột sigma/response
    r, c = transform(A[:, 0], A[:, 1])
    r = np.asarray(r, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    H, W = img_shape[:2]
    inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    n_valid = int(inside.sum())
    if n_valid == 0:
        return float("nan")
    if B.size == 0:
        return 0.0

    P = np.column_stack([r[inside], c[inside]])
    Q = B.reshape(len(B), -1)[:, :2]
    matched = 0
    for s in range(0, len(P), 1000):          # tính theo khối để không tràn bộ nhớ
        d2 = ((P[s:s + 1000, None, :] - Q[None, :, :]) ** 2).sum(axis=2)
        matched += int((d2.min(axis=1) <= eps ** 2).sum())
    return matched / n_valid
