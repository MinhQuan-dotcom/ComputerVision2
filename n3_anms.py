"""Phần A — Adaptive Non-Maximal Suppression (Người 3)."""
import os
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import cv2

from common import load_gray, grid_occupancy, show_points
from n1_harris import harris_classic, repeatability, make_rotation


def top_n(points, n_keep):
    """Sắp xếp points giảm dần theo cột response (dùng kind="stable"), 

    lấy n_keep dòng đầu. Nếu n_keep lớn hơn số điểm thì trả tất cả.
    Đầu vào: points dạng (N, 3) với các cột [row, col, response].
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts
    sorted_idx = np.argsort(pts[:, 2], kind="stable")[::-1]
    sorted_pts = pts[sorted_idx]
    return sorted_pts[:n_keep]


def _anms_radii(points, c_robust=0.9):
    """Hàm phụ tính bán kính r_i cho từng điểm ứng viên trong ANMS.
    Chia khối để tối ưu tính toán bằng broadcasting tránh tràn bộ nhớ với N lớn.
    """
    pts = np.asarray(points, dtype=np.float64)
    N = pts.shape[0]
    if N == 0:
        return np.array([], dtype=np.float64)
    
    r = np.full(N, np.inf, dtype=np.float64)
    f = pts[:, 2]  # Cột response
    
    chunk_size = 1000
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_pts = pts[start:end, :2]  # Tọa độ (row, col) của khối 
        all_pts = pts[:, :2]            # Tọa độ tất cả các điểm
        
        # Bình phương khoảng cách Euclid bằng broadcasting: (chunk_size, N)
        dist_sq = np.sum((chunk_pts[:, np.newaxis, :] - all_pts[np.newaxis, :, :]) ** 2, axis=2)
        
        # Điều kiện điểm j mạnh hơn i đáng kể: f_i < c_robust * f_j
        mask = f[start:end, np.newaxis] < c_robust * f[np.newaxis, :]
        
        # Đặt inf ở các ô không thỏa mãn điều kiện
        dist_sq_masked = np.where(mask, dist_sq, np.inf)
        
        # Lấy khoảng cách nhỏ nhất theo hàng rồi căn bậc hai
        min_dist_sq = np.min(dist_sq_masked, axis=1)
        r[start:end] = np.sqrt(min_dist_sq)
        
    return r


def anms(points, n_keep, c_robust=0.9):
    """Thuật toán Adaptive Non-Maximal Suppression.

    Đầu ra: (n_keep, 3) với các cột row, col, response, sắp xếp theo r giảm dần.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts
    
    # Sắp xếp giảm dần theo response trước để đảm bảo tính ổn định
    sorted_idx = np.argsort(pts[:, 2], kind="stable")[::-1]
    pts = pts[sorted_idx]
    
    # Tính bán kính r cho từng điểm
    r = _anms_radii(pts, c_robust=c_robust)
    
    # Sắp xếp các điểm theo r giảm dần (ưu tiên điểm ở vùng thưa thớt trước)
    sort_r_idx = np.argsort(r, kind="stable")[::-1]
    sorted_by_r = pts[sort_r_idx]
    
    return sorted_by_r[:n_keep]
