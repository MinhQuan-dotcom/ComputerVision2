"""Phần A — Adaptive Non-Maximal Suppression (Người 3).

Quy ước: points là mảng (N, 3) gồm row, col, response (đầu ra của harris_classic).
Không gọi hàm phát hiện góc dựng sẵn trong file này.
"""
import numpy as np


def top_n(points, n_keep):
    """Đường cơ sở: N điểm có response lớn nhất.
    -> (n_keep, 3), sắp xếp giảm dần theo response.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 3))
    order = np.argsort(-pts[:, 2], kind="stable")
    return pts[order][:n_keep]


def _anms_radii(points, c_robust=0.9, chunk=1000):
    """Bán kính ức chế r_i theo công thức (4).

    r_i = khoảng cách nhỏ nhất từ điểm i tới các điểm j có f_i < c_robust * f_j.
    Không có j nào như vậy thì r_i = inf (điểm mạnh nhất luôn như vậy).
    Tính theo khối `chunk` điểm để không tạo ma trận N x N quá lớn.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    N = len(pts)
    r = np.full(N, np.inf)
    if N == 0:
        return r
    xy = pts[:, :2]
    f = pts[:, 2]
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d2 = ((xy[s:e, None, :] - xy[None, :, :]) ** 2).sum(axis=2)       # (khối, N)
        manh_hon = f[s:e, None] < c_robust * f[None, :]                   # j mạnh hơn i đáng kể
        d2 = np.where(manh_hon, d2, np.inf)
        r[s:e] = np.sqrt(d2.min(axis=1))
    return r


def anms(points, n_keep, c_robust=0.9):
    """Adaptive non-maximal suppression (Brown, Szeliski & Winder 2005).
    -> (n_keep, 3) gồm row, col, response, sắp xếp giảm dần theo r.
       Khi r bằng nhau, điểm có response lớn hơn đứng trước.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 3))
    pts = pts[np.argsort(-pts[:, 2], kind="stable")]      # sắp theo response trước
    r = _anms_radii(pts, c_robust)
    order = np.argsort(-r, kind="stable")                  # r giảm dần, giữ thứ tự response khi bằng nhau
    return pts[order][:n_keep]
