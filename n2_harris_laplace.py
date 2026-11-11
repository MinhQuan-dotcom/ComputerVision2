"""Harris–Laplace đa tỉ lệ — Người 2 (Biến thể A1).

Quy ước:
- Ảnh xám float64, [0, 1], shape (H, W). Tọa độ (row, col).
- Thư viện: numpy, math, scipy.ndimage.gaussian_filter.
- Tận dụng common.S_RATIO và n1_harris.{harris_response, nms_2d}.
"""
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from common import S_RATIO
from n1_harris import harris_response, nms_2d


def scale_pyramid(sigma0=1.2, xi=1.2, n_levels=10):
    """Tháp tỉ lệ hình học: sigma_n = sigma0 * xi^n, n = 0..n_levels-1."""
    return np.array([sigma0 * (xi ** n) for n in range(n_levels)], dtype=np.float64)


def log_normalized(img, sigma):
    """Laplacian-of-Gaussian chuẩn hóa Lindeberg tại tỉ lệ sigma.

    Trả về sigma^2 * |Lxx + Lyy|, dùng np.gradient bậc hai.
    """
    L = gaussian_filter(img, sigma, mode="reflect")
    Lxx = np.gradient(np.gradient(L, axis=1), axis=1)
    Lyy = np.gradient(np.gradient(L, axis=0), axis=0)
    return (sigma ** 2) * np.abs(Lxx + Lyy)


def harris_laplace(img, sigmas, k=0.04, thresh_rel=0.01, log_thresh=0.0):
    """Harris–Laplace detector đa tỉ lệ.

    Bước 1 — ứng viên Harris: với mỗi tầng sigma_n, tính R, lấy cực đại
    địa phương NMS, lọc theo ngưỡng tương đối thresh_rel * R_max_toàn_cục
    và bỏ vùng mép ceil(3*sigma_n) pixel.

    Bước 2 — chọn tỉ lệ bằng LoG: chỉ giữ ứng viên là cực đại địa phương
    của log_stack theo trục tháp (giữa tầng n-1 và n+1) và vượt log_thresh.

    Trả về ndarray float64 shape (N, 4): [row, col, sigma, response],
    sắp xếp giảm dần theo response. Nếu rỗng trả về zeros((0, 4)).
    """
    n_levels = len(sigmas)

    r_maps = []
    cand_masks = []
    for sigma_n in sigmas:
        R = harris_response(img, sigma_i=sigma_n, sigma_d=S_RATIO * sigma_n, k=k)
        r_maps.append(R)
        cand_masks.append(nms_2d(R, window=3) & (R > 0))

    r_global_max = max(R.max() for R in r_maps)
    if not np.isfinite(r_global_max) or r_global_max <= 0:
        return np.zeros((0, 4), dtype=np.float64)

    threshold = thresh_rel * r_global_max
    step1 = []
    for n, sigma_n in enumerate(sigmas):
        R = r_maps[n]
        mask = cand_masks[n] & (R > threshold)
        m = int(math.ceil(3 * sigma_n))
        mask[:m, :] = False
        mask[-m:, :] = False
        mask[:, :m] = False
        mask[:, -m:] = False
        rows, cols = np.nonzero(mask)
        for r, c in zip(rows, cols):
            step1.append((n, r, c, float(R[r, c])))

    if not step1:
        return np.zeros((0, 4), dtype=np.float64)

    log_stack = np.stack([log_normalized(img, s) for s in sigmas], axis=0)

    final = []
    for n, r, c, resp in step1:
        if n == 0 or n == n_levels - 1:
            continue
        v = log_stack[n, r, c]
        if v > log_stack[n - 1, r, c] and v > log_stack[n + 1, r, c] and v > log_thresh:
            final.append((r, c, sigmas[n], resp))

    if not final:
        return np.zeros((0, 4), dtype=np.float64)

    pts = np.array(final, dtype=np.float64)
    return pts[np.argsort(-pts[:, 3], kind="stable")]


def _harris_laplace_debug(img, sigmas, k=0.04, thresh_rel=0.01, log_thresh=0.0):
    """Như harris_laplace nhưng trả thêm số ứng viên mỗi bước.

    Trả về (pts, n_step1, n_step2) — dùng cho thống kê thí nghiệm.
    """
    n_levels = len(sigmas)

    r_maps = []
    cand_masks = []
    for sigma_n in sigmas:
        R = harris_response(img, sigma_i=sigma_n, sigma_d=S_RATIO * sigma_n, k=k)
        r_maps.append(R)
        cand_masks.append(nms_2d(R, window=3) & (R > 0))

    r_global_max = max(R.max() for R in r_maps)
    if not np.isfinite(r_global_max) or r_global_max <= 0:
        return np.zeros((0, 4), dtype=np.float64), 0, 0

    threshold = thresh_rel * r_global_max
    step1 = []
    for n, sigma_n in enumerate(sigmas):
        R = r_maps[n]
        mask = cand_masks[n] & (R > threshold)
        m = int(math.ceil(3 * sigma_n))
        mask[:m, :] = False
        mask[-m:, :] = False
        mask[:, :m] = False
        mask[:, -m:] = False
        rows, cols = np.nonzero(mask)
        for r, c in zip(rows, cols):
            step1.append((n, r, c, float(R[r, c])))

    n_step1 = len(step1)
    log_stack = np.stack([log_normalized(img, s) for s in sigmas], axis=0)

    final = []
    for n, r, c, resp in step1:
        if n == 0 or n == n_levels - 1:
            continue
        v = log_stack[n, r, c]
        if v > log_stack[n - 1, r, c] and v > log_stack[n + 1, r, c] and v > log_thresh:
            final.append((r, c, sigmas[n], resp))

    n_step2 = len(final)
    if not final:
        return np.zeros((0, 4), dtype=np.float64), n_step1, n_step2

    pts = np.array(final, dtype=np.float64)
    return pts[np.argsort(-pts[:, 3], kind="stable")], n_step1, n_step2
