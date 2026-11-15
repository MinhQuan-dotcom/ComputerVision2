"""Kiểm chứng A5 và thí nghiệm A2 (ANMS vs Harris top-N) — Người 3.

Chạy:  python kiem_chung_nguoi3.py
Hình lưu vào ket_qua/, ảnh tự tạo lưu vào anh/.

"""

# **Prompt AI** — Người 3 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Viết phần import, hằng đường dẫn, hàm tính khoảng cách láng giềng gần nhất
# > kc_lang_gieng, hàm giảm tương phản nửa trái ảnh và khởi tạo tập ứng viên
# > Harris (thresh_rel=0.001) trên ảnh oxford/boat/img1.

import os

import matplotlib
matplotlib.use("Agg")  # trong notebook thì bỏ dòng này
import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import grid_occupancy, load_oxford, show_points
from n1_harris import harris_classic
from n3_anms import _anms_radii, anms, top_n

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ket_qua")
ANH = os.path.join(HERE, "anh")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ANH, exist_ok=True)

THRESH_UNG_VIEN = 0.001   # ngưỡng thấp để có nhiều ứng viên; top_n và anms chọn trên CÙNG tập này


def kc_lang_gieng(pts):
    """Khoảng cách trung bình từ mỗi điểm tới điểm gần nhất (px). Lớn hơn = trải đều hơn."""
    if len(pts) < 2:
        return float("nan")
    d = np.sqrt(((pts[:, None, :2] - pts[None, :, :2]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def giam_tuong_phan_nua_trai(img, he_so):
    """Giữ nguyên nửa phải, nén tương phản nửa trái quanh giá trị trung bình."""
    out = img.copy()
    W = img.shape[1]
    m = img.mean()
    out[:, :W // 2] = m + (img[:, :W // 2] - m) * he_so
    return out


img, _ = load_oxford("boat", 1)
H, W = img.shape
ung_vien = harris_classic(img, thresh_rel=THRESH_UNG_VIEN)


# ============================================================
# KIỂM CHỨNG A5 — Phân bố điểm: top_n vs ANMS
# ============================================================

# **Prompt AI** — Người 3 · công cụ: Antigravity IDE · ngày 11/11/2026
# > So sánh độ phủ lưới (grid_occupancy) và khoảng cách láng giềng gần nhất
# > giữa top_n và anms (c_robust=0.9) với N=100 và N=300 trên cùng tập ứng viên.
# > Vẽ lưới 2×2 và lưu vào ket_qua/a5_topn_vs_anms.png.

print("=" * 64)
print("A5. Phân bố điểm: top_n vs anms (ảnh oxford/boat/img1, c_robust = 0.9)")
print(f"Số ứng viên Harris (thresh_rel = {THRESH_UNG_VIEN}): {len(ung_vien)}")
print(f"{'N':>5} | {'phủ lưới top_n':>15} | {'phủ lưới anms':>14} | {'kc láng giềng top_n':>20} | {'anms':>6}")
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
for i, N in enumerate([100, 300]):
    pt, pa = top_n(ung_vien, N), anms(ung_vien, N)
    ct, ca = grid_occupancy(pt, img.shape), grid_occupancy(pa, img.shape)
    print(f"{N:>5} | {ct:>15.3f} | {ca:>14.3f} | {kc_lang_gieng(pt):>17.1f} px | {kc_lang_gieng(pa):>4.1f} px")
    show_points(axs[i, 0], img, pt, f"top_n, N = {N}, phủ lưới = {ct:.2f}")
    show_points(axs[i, 1], img, pa, f"anms, N = {N}, phủ lưới = {ca:.2f}", color="cyan")
fig.suptitle("A5 — cùng tập ứng viên, cùng N")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "a5_topn_vs_anms.png"), dpi=110)
plt.close(fig)


# ============================================================
# 6.1 — Bằng chứng khác biệt khi giảm tương phản cục bộ
# ============================================================

# **Prompt AI** — Người 3 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Nén tương phản nửa trái ảnh xuống 30%, chạy harris_classic lấy ứng viên
# > rồi chọn N=100 điểm bằng top_n và anms. Tính tỉ lệ điểm rơi vào nửa trái,
# > vẽ hình đối chiếu có vạch chia đôi ảnh lưu vào ket_qua/a2_bang_chung_tuong_phan.png.

print("=" * 64)
print("6.1 Bằng chứng: nửa trái ảnh bị giảm tương phản còn 30%")
HE_SO = 0.3
anh_tp = giam_tuong_phan_nua_trai(img, HE_SO)
cv2.imwrite(os.path.join(ANH, "a2_boat_nua_trai_tp30.png"), np.round(np.clip(anh_tp, 0, 1) * 255).astype(np.uint8))
uv_tp = harris_classic(anh_tp, thresh_rel=THRESH_UNG_VIEN)
N = 100
pt, pa = top_n(uv_tp, N), anms(uv_tp, N)
trai_t = (pt[:, 1] < W / 2).mean()
trai_a = (pa[:, 1] < W / 2).mean()
uv_trai = (uv_tp[:, 1] < W / 2).sum()
print(f"Ứng viên: {len(uv_tp)}, trong đó {uv_trai} ở nửa trái (ứng viên vẫn còn, chỉ là R nhỏ)")
print(f"N = {N}: tỉ lệ điểm ở nửa trái  top_n = {trai_t:.2f} | anms = {trai_a:.2f}")
print(f"        phủ lưới                top_n = {grid_occupancy(pt, img.shape):.3f} | anms = {grid_occupancy(pa, img.shape):.3f}")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
show_points(axs[0], anh_tp, pt, f"top_n: {trai_t:.0%} điểm ở nửa trái")
show_points(axs[1], anh_tp, pa, f"anms: {trai_a:.0%} điểm ở nửa trái", color="cyan")
for ax in axs:
    ax.axvline(W / 2, color="yellow", lw=1, ls="--")
fig.suptitle(f"6.1 — nửa trái giảm tương phản còn {HE_SO:.0%}, N = {N}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "a2_bang_chung_tuong_phan.png"), dpi=110)
plt.close(fig)


# ============================================================
# 6.2 — Quét tham số (tương phản nửa trái, số điểm N, c_robust)
# ============================================================

# **Prompt AI** — Người 3 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Thực hiện 3 phép quét tham số: (6.2a) quét hệ số tương phản nửa trái từ
# > 1.0 xuống 0.1; (6.2b) quét số điểm N từ 25 đến tổng ứng viên; (6.2c) quét
# > c_robust từ 0.5 đến 1.0 đo kc láng giềng và số điểm r=inf. Vẽ 3 biểu đồ
# > ngang hàng và lưu vào ket_qua/a2_quet_tham_so.png.

print("=" * 64)
print("6.2a Quét mức tương phản nửa trái (N = 100): top_n bỏ hẳn nửa trái từ đâu?")
he_so_ds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
trai_top, trai_anms, uv_trai_ds = [], [], []
for hs in he_so_ds:
    uv = harris_classic(giam_tuong_phan_nua_trai(img, hs), thresh_rel=THRESH_UNG_VIEN)
    trai_top.append((top_n(uv, 100)[:, 1] < W / 2).mean())
    trai_anms.append((anms(uv, 100)[:, 1] < W / 2).mean())
    uv_trai_ds.append(int((uv[:, 1] < W / 2).sum()))
    print(f"  hệ số {hs:.1f} (R nửa trái x {hs ** 4:.4f}): nửa trái top_n = {trai_top[-1]:.2f} | anms = {trai_anms[-1]:.2f}"
          f" | ứng viên còn ở nửa trái: {uv_trai_ds[-1]}")

print("6.2b Quét N (ảnh gốc): độ phủ lưới theo số điểm giữ lại")
N_ds = [25, 50, 75, 100, 150, 200, 300, 400, 600, len(ung_vien)]
phu_top = [grid_occupancy(top_n(ung_vien, n), img.shape) for n in N_ds]
phu_anms = [grid_occupancy(anms(ung_vien, n), img.shape) for n in N_ds]
for n, a, b in zip(N_ds, phu_top, phu_anms):
    print(f"  N = {n:4d}: top_n = {a:.3f} | anms = {b:.3f}")

print("6.2c Quét c_robust (ảnh gốc, N = 100)")
c_ds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
phu_c, kc_c, so_inf = [], [], []
for c in c_ds:
    pa = anms(ung_vien, 100, c_robust=c)
    phu_c.append(grid_occupancy(pa, img.shape))
    kc_c.append(kc_lang_gieng(pa))
    so_inf.append(int(np.isinf(_anms_radii(ung_vien, c)).sum()))
    print(f"  c = {c:.2f}: phủ lưới = {phu_c[-1]:.3f} | kc láng giềng = {kc_c[-1]:5.1f} px | số điểm r = inf: {so_inf[-1]}")

fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axs[0]
ax.plot(he_so_ds, trai_top, "o-", label="top_n")
ax.plot(he_so_ds, trai_anms, "s-", label="anms")
ax.axhline(0.5, color="gray", lw=0.8, ls=":")
ax.invert_xaxis()
ax.set_xlabel("Hệ số tương phản nửa trái")
ax.set_ylabel("Tỉ lệ điểm ở nửa trái")
ax.set_title("6.2a — N = 100, c_robust = 0.9")
ax.legend()
ax.grid(alpha=0.3)

ax = axs[1]
ax.plot(N_ds, phu_top, "o-", label="top_n")
ax.plot(N_ds, phu_anms, "s-", label="anms")
ax.set_xscale("log")
ax.set_xlabel("Số điểm giữ lại N (thang log)")
ax.set_ylabel("Độ phủ lưới 8 × 8")
ax.set_title(f"6.2b — ảnh gốc, {len(ung_vien)} ứng viên")
ax.legend()
ax.grid(alpha=0.3)

ax = axs[2]
ax.plot(c_ds, kc_c, "o-", color="C2", label="kc láng giềng trung bình (px)")
ax.set_xlabel("c_robust")
ax.set_ylabel("Khoảng cách tới điểm gần nhất (px)")
ax2 = ax.twinx()
ax2.bar(c_ds, so_inf, width=0.03, alpha=0.3, color="C3", label="số điểm r = inf")
ax2.set_ylabel("Số điểm có r = inf")
ax.set_title("6.2c — N = 100")
ax.legend(loc="upper left")
ax2.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "a2_quet_tham_so.png"), dpi=110)
plt.close(fig)


# ============================================================
# 6.3 — Cơ chế và giải thích hiện tượng (ANMS vs top-N)
# ============================================================

# **Prompt AI** — Người 3 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Viết đoạn giải thích cơ chế của ANMS so với top_n theo 3 ý: đại lượng
# > tính khác nhau (f_i vs bán kính ức chế r_i), bước áp dụng, và giải thích
# > kết quả thực nghiệm ở mục 6.1, 6.2a, 6.2b, 6.2c.

print("=" * 64)
print("""6.3 Giải thích
(1) Đại lượng tính khác: top_n xếp hạng điểm theo response f_i. ANMS xếp hạng theo
    bán kính ức chế r_i = khoảng cách tới điểm gần nhất mạnh hơn đáng kể (f_i < c·f_j).
    r_i là đại lượng hình học: nó chỉ dùng f để so sánh hơn-kém, không dùng độ lớn của f.
(2) Bước: sau harris_classic, ở bước chọn N điểm cuối cùng. Phần phát hiện ứng viên
    giữ nguyên, hai cách dùng chung một tập ứng viên.
(3) Vì sao ra đúng hiện tượng:
    - M chứa tích đạo hàm nên tỉ lệ với bình phương tương phản; R = det − k·tr² tỉ lệ
      với lũy thừa 4. Nửa trái giảm còn 30% thì R ở đó nhân 0.3^4 ≈ 0.008, nên mọi
      điểm nửa trái tụt xuống cuối bảng xếp hạng và top_n bỏ hẳn nửa trái (6.1, 6.2a).
    - Với ANMS, điểm mạnh nhất của nửa trái chỉ bị các điểm nửa phải ức chế, mà
      chúng ở xa, nên r của nó lớn và nó vẫn được chọn sớm. Vì vậy anms giữ được
      khoảng 1/3–1/2 số điểm ở nửa trái tới hệ số 0.3.
    - Từ hệ số 0.2 thì anms cũng mất nửa trái. Lý do không nằm ở ANMS mà ở bước trước:
      R nửa trái chỉ còn khoảng 0.0016 lần, rơi dưới ngưỡng thresh_rel·max(R) của
      harris_classic, nên ứng viên nửa trái gần như không còn để ANMS chọn. ANMS chỉ
      sắp xếp lại các ứng viên, không tạo thêm ứng viên.
    - c_robust nhỏ thì điều kiện f_i < c·f_j khó thỏa hơn, nhiều điểm không có ai ức
      chế (r = inf). Các điểm r = inf được chọn trước bất kể vị trí, nên ANMS lùi dần
      về hành vi của top_n: khoảng cách láng giềng giảm đều từ 32.5 xuống 19.3 px (6.2c).
      Độ phủ lưới cũng giảm theo xu hướng nhưng dao động, vì lưới 8 × 8 khá thô.
    - Khi N tiến tới tổng số ứng viên, hai cách chọn gần như cùng một tập điểm nên
      hai đường độ phủ gặp nhau (6.2b).""")
print("Hình đã lưu trong ket_qua/")