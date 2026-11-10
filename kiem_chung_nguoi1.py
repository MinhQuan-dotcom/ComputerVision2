"""Kiểm chứng phần A do Người 1 phụ trách: A1 (bàn cờ), A2 (quay 30°), A4 (đối chiếu OpenCV).

Chạy:  python kiem_chung_nguoi1.py
Hình lưu vào ket_qua/, ảnh thử lưu vào anh/.
Khi ghép notebook: mỗi mục A1/A2/A4 dưới đây là một cell.
"""
import os

import matplotlib
matplotlib.use("Agg")  # chạy được không cần màn hình; trong notebook thì bỏ dòng này

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import make_rotation, show_points
from n1_harris import harris_classic, harris_response, nms_2d, repeatability

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ket_qua")
ANH = os.path.join(HERE, "anh")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ANH, exist_ok=True)


# ---------------------------------------------------------------- ảnh thử
def tao_ban_co(n_o=8, canh=32):
    """Bàn cờ n_o x n_o ô, mỗi ô `canh` px, giá trị 0 và 1, phủ kín ảnh."""
    idx = np.arange(n_o * canh) // canh
    return ((idx[:, None] + idx[None, :]) % 2).astype(np.float64)


def _da_giac(H, W, dinh):
    """Mặt nạ của đa giác lồi, đỉnh (row, col) theo một chiều."""
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.ones((H, W), dtype=bool)
    n = len(dinh)
    s0 = None
    for i in range(n):
        (r1, c1), (r2, c2) = dinh[i], dinh[(i + 1) % n]
        cross = (c2 - c1) * (yy - r1) - (r2 - r1) * (xx - c1)
        if s0 is None:
            s0 = np.sign((c2 - c1) * (dinh[(i + 2) % n][0] - r1) - (r2 - r1) * (dinh[(i + 2) % n][1] - c1))
        m &= (cross * s0) >= 0
    return m


def tao_canh_tong_hop(H=320, W=320):
    """Nền xám 0.5 với vài hình đặc, đặt gọn trong vòng tròn nội tiếp để quay không bị cắt."""
    img = np.full((H, W), 0.5)
    img[_da_giac(H, W, [(60, 70), (60, 130), (120, 130), (120, 70)])] = 0.1     # hình vuông
    img[_da_giac(H, W, [(180, 60), (180, 150), (220, 150), (220, 60)])] = 0.9   # hình chữ nhật
    img[_da_giac(H, W, [(90, 200), (150, 260), (150, 180)])] = 0.2              # tam giác
    img[_da_giac(H, W, [(195, 220), (230, 255), (265, 220), (230, 185)])] = 0.85  # hình thoi
    return img


def luu_anh(img, ten):
    cv2.imwrite(os.path.join(ANH, ten), np.round(np.clip(img, 0, 1) * 255).astype(np.uint8))


# ---------------------------------------------------------------- A1. Bàn cờ
print("=" * 60)
print("A1. Bàn cờ tổng hợp")
canh, n_o = 32, 8
ban_co = tao_ban_co(n_o, canh)
luu_anh(ban_co, "a0_ban_co.png")

pts = harris_classic(ban_co)
# Giao điểm trong nằm giữa pixel canh*k - 1 và canh*k, tức là tại canh*k - 0.5.
g = np.array([canh * k - 0.5 for k in range(1, n_o)])
truth = np.array([(r, c) for r in g for c in g])

d = np.sqrt(((pts[:, None, :2] - truth[None, :, :]) ** 2).sum(axis=2))
gan_nhat_cua_truth = d.min(axis=0)          # mỗi giao điểm -> điểm phát hiện gần nhất
gan_nhat_cua_pts = d.min(axis=1)            # mỗi điểm phát hiện -> giao điểm gần nhất
n_khop = int((gan_nhat_cua_truth <= 1.0).sum())
n_thua = int((gan_nhat_cua_pts > 1.0).sum())
print(f"Số giao điểm trong (đáp án): {len(truth)}")
print(f"Số điểm harris_classic trả về: {len(pts)}")
print(f"Số giao điểm tìm được với sai số <= 1 px: {n_khop}/{len(truth)}")
print(f"Số điểm thừa (cách mọi giao điểm > 1 px): {n_thua}")
print(f"Sai số lớn nhất: {gan_nhat_cua_truth.max():.3f} px")
print("ĐẠT" if (n_khop == len(truth) and n_thua == 0) else "CHƯA ĐẠT")

fig, ax = plt.subplots(figsize=(5, 5))
show_points(ax, ban_co, pts, f"A1 — harris_classic trên bàn cờ ({len(pts)} điểm)")
ax.plot(truth[:, 1], truth[:, 0], "o", mfc="none", mec="lime", ms=9, mew=1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "a1_ban_co.png"), dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- A2. Quay 30°
print("=" * 60)
print("A2. Bất biến phép quay (30°)")
canh_th = tao_canh_tong_hop()
luu_anh(canh_th, "a0_canh_tong_hop.png")
quay, tf = make_rotation(canh_th, 30)
luu_anh(quay, "a0_canh_tong_hop_quay30.png")

pa = harris_classic(canh_th)
pb = harris_classic(quay)
rep = repeatability(pa, pb, tf, quay.shape, eps=3.0)
print(f"Số điểm ảnh gốc: {len(pa)} | ảnh quay 30°: {len(pb)}")
print(f"Repeatability (gốc -> quay, eps = 3 px): {rep:.3f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_points(axs[0], canh_th, pa, f"Ảnh gốc ({len(pa)} điểm)")
r_proj, c_proj = tf(pa[:, 0], pa[:, 1])
show_points(axs[1], quay, pb, f"Quay 30° ({len(pb)} điểm), repeatability = {rep:.3f}")
axs[1].plot(c_proj, r_proj, "o", mfc="none", mec="lime", ms=9, mew=1)
fig.suptitle("A2 — đỏ: điểm tìm được, xanh: điểm ảnh gốc chiếu sang")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "a2_quay_30.png"), dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- A4. Đối chiếu OpenCV
# đối chiếu thư viện
print("=" * 60)
print("A4. Đối chiếu với cv2.cornerHarris")
BLOCK, KSIZE, K = 5, 3, 0.04   # cửa sổ 5x5 ~ sigma_i = 2; Sobel 3x3; cùng k
print(f"Tham số OpenCV: blockSize={BLOCK}, ksize={KSIZE}, k={K}")
print("Tham số của mình: sigma_i=2.0, sigma_d=1.4, k=0.04, thresh_rel=0.01")


def diem_opencv(img, thresh_rel=0.01, margin=6):
    R = cv2.cornerHarris(img.astype(np.float32), BLOCK, KSIZE, K).astype(np.float64)
    mask = nms_2d(R) & (R > thresh_rel * R.max()) & (R > 0)
    mask[:margin, :] = mask[-margin:, :] = False
    mask[:, :margin] = mask[:, -margin:] = False
    rr, cc = np.nonzero(mask)
    return np.column_stack([rr, cc]).astype(np.float64)


def ti_le_trong_ban_kinh(p_minh, p_cv, rad=3.0):
    if len(p_minh) == 0 or len(p_cv) == 0:
        return float("nan")
    d = np.sqrt(((p_minh[:, None, :2] - p_cv[None, :, :2]) ** 2).sum(axis=2))
    return float((d.min(axis=1) <= rad).mean())


ket_qua_a4 = []
for ten, img in [("bàn cờ", ban_co), ("cảnh tổng hợp", canh_th), ("cảnh quay 30°", quay)]:
    p_minh = harris_classic(img)
    p_cv = diem_opencv(img)
    t3 = ti_le_trong_ban_kinh(p_minh, p_cv, 3.0)
    t2 = ti_le_trong_ban_kinh(p_minh, p_cv, 2.0)
    ket_qua_a4.append((ten, img, p_minh, p_cv))
    print(f"{ten:15s}: của mình {len(p_minh):3d} điểm, OpenCV {len(p_cv):3d} điểm | "
          f"trong 2 px: {t2:.3f} | trong 3 px: {t3:.3f}")

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, (ten, img, p_minh, p_cv) in zip(axs, ket_qua_a4):
    show_points(ax, img, p_minh, f"{ten}: đỏ = của mình, xanh = OpenCV")
    ax.plot(p_cv[:, 1], p_cv[:, 0], "x", color="cyan", ms=6, mew=1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "a4_doi_chieu_opencv.png"), dpi=110)
plt.close(fig)

print("=" * 60)
print("Hình đã lưu trong ket_qua/")
