"""
Người 4 — Kiểm chứng B4, srg_step_frames và thí nghiệm SRG vs Region Growing.
Chạy từ thư mục ComputerVision2:  python kiem_chung_nguoi4.py
Hình lưu vào ket_qua/, ảnh thử lưu vào anh/.
"""

# **Prompt AI** — Người 4 · công cụ: Antigravity IDE · ngày 10/11/2026
# > Viết phần import, hằng đường dẫn, hàm tạo ảnh hai vùng chênh delta kèm dải
# > sáng dọc (ramp=0.15) và nhiễu Gauss (noise=0.03), cùng các hàm tính độ chính
# > xác (do_chinh_xac) và tỉ lệ rò vùng (ti_le_ro).

import os

import matplotlib
matplotlib.use("Agg")  # trong notebook thì bỏ dòng này
import matplotlib.pyplot as plt
import numpy as np

from common import show_labels
from n4_rg import region_growing_classic, seed_sensitivity, srg_step_frames
from n5_srg import srg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ket_qua")
ANH = os.path.join(HERE, "anh")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ANH, exist_ok=True)

# ------------------------------------------------------------------ ảnh thử
# Hai vùng trái/phải chênh nhau delta, cộng một dải sáng dần theo chiều dọc (ramp)
# và nhiễu Gauss. Vì tự tạo nên biết đáp án (truth) và đo được độ chính xác.
H = W = 128
TRUTH = np.zeros((H, W), dtype=np.int32)
TRUTH[:, W // 2:] = 1
SEEDS = [(64, 20), (64, 108)]          # hạt giống 0 ở vùng trái, 1 ở vùng phải
RAMP, NOISE = 0.15, 0.03


def tao_anh(delta, ramp=RAMP, noise=NOISE, seed=1):
    rng = np.random.default_rng(seed)
    base = np.where(TRUTH == 0, 0.4, 0.4 + delta)
    doc = ramp * np.linspace(-0.5, 0.5, H)[:, None]
    return np.clip(base + doc + noise * rng.normal(size=(H, W)), 0.0, 1.0)


def do_chinh_xac(labels):
    """Tỉ lệ pixel có nhãn đúng với đáp án (pixel -1 tính là sai)."""
    return float((labels == TRUTH).mean())


def ti_le_ro(labels):
    """Tỉ lệ pixel của vùng phải bị vùng 0 (vùng trái) chiếm."""
    return float(((labels == 0) & (TRUTH == 1)).sum() / (TRUTH == 1).sum())


def luu_anh(img, ten):
    plt.imsave(os.path.join(ANH, ten), img, cmap="gray", vmin=0, vmax=1)


DELTA = 0.2
img = tao_anh(DELTA)
luu_anh(img, "b_hai_vung_delta02.png")


# ============================================================
# KIỂM CHỨNG B4 — Đối chiếu Region Growing (stack/queue) và SRG
# ============================================================

# **Prompt AI** — Người 4 · công cụ: Antigravity IDE · ngày 10/11/2026
# > Chạy region_growing_classic (order="stack" và "queue", T=0.08) cùng srg
# > trên ảnh hai vùng (delta=0.2). Thống kê số pixel khác nhau giữa các cặp,
# > tỉ lệ pixel chưa gán nhãn, và vẽ lưới 1×4 lưu vào ket_qua/b4_rg_vs_srg.png.

print("=" * 66)
print(f"B4. Đối chiếu với Region Growing (ảnh hai vùng, delta = {DELTA}, T = 0.08)")
T_B4 = 0.08
rg_stack = region_growing_classic(img, SEEDS, T_B4, order="stack")
rg_queue = region_growing_classic(img, SEEDS, T_B4, order="queue")
lab_srg, bien = srg(img, SEEDS)
cap = [("stack", "queue", rg_stack, rg_queue),
       ("stack", "srg", rg_stack, lab_srg),
       ("queue", "srg", rg_queue, lab_srg)]
for a, b, la, lb in cap:
    print(f"  {a:5s} vs {b:5s}: {int((la != lb).sum()):5d} pixel khác nhau "
          f"({(la != lb).mean():.1%} ảnh)")
print(f"  Pixel chưa gán nhãn: stack = {(rg_stack < 0).mean():.1%}, "
      f"queue = {(rg_queue < 0).mean():.1%}, srg = {(lab_srg < 0).mean():.1%}")

fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
axs[0].imshow(img, cmap="gray", vmin=0, vmax=1)
axs[0].plot([c for _, c in SEEDS], [r for r, _ in SEEDS], "r*", ms=12)
axs[0].set_title("Ảnh thử + hạt giống")
axs[0].axis("off")
show_labels(axs[1], rg_stack, f"RG stack, T = {T_B4}")
show_labels(axs[2], rg_queue, f"RG queue, T = {T_B4}")
show_labels(axs[3], lab_srg, "SRG (không cần T)")
fig.suptitle("B4 — đen: pixel chưa gán nhãn")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "b4_rg_vs_srg.png"), dpi=110)
plt.close(fig)


# ============================================================
# MINH HỌA TIẾN TRÌNH — srg_step_frames (8 mốc thời gian)
# ============================================================

# **Prompt AI** — Người 4 · công cụ: Antigravity IDE · ngày 10/11/2026
# > Dùng srg_step_frames chụp 8 frame trong quá trình lan vùng của SRG,
# > kiểm tra frame cuối trùng khớp srg và lưu hình ket_qua/b_srg_tien_trinh.png.

print("=" * 66)
frames = srg_step_frames(img, SEEDS, n_frames=8)
print("srg_step_frames: tỉ lệ pixel đã có nhãn ở từng frame:",
      [round(float((f >= 0).mean()), 3) for f in frames])
print("  Frame cuối trùng kết quả srg:", bool((frames[-1] == lab_srg).all()))
fig, axs = plt.subplots(1, 8, figsize=(20, 3))
for k, (ax, f) in enumerate(zip(axs, frames), 1):
    show_labels(ax, f, f"{k}/8")
fig.suptitle("Tiến trình SRG: hai vùng lớn đồng thời, pixel tương phản thấp được nhận trước")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "b_srg_tien_trinh.png"), dpi=100)
plt.close(fig)


# ============================================================
# 6.1 — Bằng chứng: RG phụ thuộc ngưỡng T vs SRG không cần T
# ============================================================

# **Prompt AI** — Người 4 · công cụ: Antigravity IDE · ngày 10/11/2026
# > So sánh RG ở 3 mức ngưỡng T = [0.06, 0.12, 0.18] với SRG trên ảnh hai vùng.
# > In độ chính xác, tỉ lệ chưa gán, tỉ lệ rò vùng và lưu ket_qua/b_bang_chung_rg_vs_srg.png.

print("=" * 66)
print("6.1 Bằng chứng: RG phải dò đúng T, SRG thì không có T")
T_ds_61 = [0.06, 0.12, 0.18]
fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, T in zip(axs, T_ds_61):
    lab = region_growing_classic(img, SEEDS, T, order="stack")
    print(f"  RG stack T = {T:.2f}: chính xác = {do_chinh_xac(lab):.3f} | "
          f"chưa gán = {(lab < 0).mean():.1%} | rò sang vùng phải = {ti_le_ro(lab):.1%}")
    show_labels(ax, lab, f"RG, T = {T}: chính xác {do_chinh_xac(lab):.2f}")
print(f"  SRG              : chính xác = {do_chinh_xac(lab_srg):.3f}")
show_labels(axs[3], lab_srg, f"SRG: chính xác {do_chinh_xac(lab_srg):.2f}")
fig.suptitle(f"6.1 — ảnh hai vùng chênh {DELTA}, có dải sáng dọc {RAMP} và nhiễu {NOISE}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "b_bang_chung_rg_vs_srg.png"), dpi=110)
plt.close(fig)


# ============================================================
# 6.2a — Quét ngưỡng T × độ chênh delta giữa hai vùng
# ============================================================

# **Prompt AI** — Người 4 · công cụ: Antigravity IDE · ngày 10/11/2026
# > Quét lưới 2 chiều giữa độ chênh delta trong [0.10, 0.40] và ngưỡng T trong
# > [0.02, 0.40]. Vẽ heatmap độ chính xác của RG và đồ thị so sánh RG (T tốt nhất)
# > với SRG, lưu vào ket_qua/b_quet_T_delta.png.

print("=" * 66)
print("6.2a Quét ngưỡng T của RG theo độ chênh delta giữa hai vùng")
deltas = np.round(np.arange(0.10, 0.41, 0.05), 2)
Ts = np.round(np.arange(0.02, 0.42, 0.02), 2)
acc = np.zeros((len(deltas), len(Ts)))
acc_srg = []
cua_so = []
for i, d in enumerate(deltas):
    im = tao_anh(d)
    for j, T in enumerate(Ts):
        acc[i, j] = do_chinh_xac(region_growing_classic(im, SEEDS, T, order="stack"))
    acc_srg.append(do_chinh_xac(srg(im, SEEDS)[0]))
    tot = Ts[acc[i] >= 0.95]
    cua_so.append((tot.min(), tot.max()) if len(tot) else None)
    cs = f"[{tot.min():.2f}, {tot.max():.2f}]" if len(tot) else "không có"
    print(f"  delta = {d:.2f}: RG tốt nhất = {acc[i].max():.3f} | "
          f"khoảng T cho chính xác >= 0.95: {cs:14s} | SRG = {acc_srg[-1]:.3f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axs[0]
im_ = ax.imshow(acc, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                extent=[Ts[0] - 0.01, Ts[-1] + 0.01, deltas[0] - 0.025, deltas[-1] + 0.025])
ax.plot(deltas, deltas, "w--", lw=1, label="T = delta")
ax.set_xlabel("Ngưỡng T của RG")
ax.set_ylabel("Độ chênh delta giữa hai vùng")
ax.set_title("6.2a — độ chính xác của RG (stack)")
ax.legend(loc="lower right")
fig.colorbar(im_, ax=ax, label="Độ chính xác")
ax = axs[1]
ax.plot(deltas, acc.max(axis=1), "o-", label="RG, T tốt nhất")
ax.plot(deltas, acc_srg, "s-", label="SRG")
ax.set_xlabel("Độ chênh delta giữa hai vùng")
ax.set_ylabel("Độ chính xác")
ax.set_ylim(0, 1.05)
ax.set_title("RG dù chọn T tốt nhất vs SRG")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "b_quet_T_delta.png"), dpi=110)
plt.close(fig)


# ============================================================
# 6.2b — Độ nhạy vị trí hạt giống trên ảnh thực (oxford/boat)
# ============================================================

# **Prompt AI** — Người 4 · công cụ: Antigravity IDE · ngày 10/11/2026
# > Đánh giá độ nhạy hạt giống bằng seed_sensitivity (20 lần thử, jitter = 2, 5, 10 px)
# > cho RG (T=0.10) và SRG trên ảnh oxford/boat/img1 thu nhỏ (256×205) với 5 hạt giống.
# > Vẽ bản đồ tần suất đổi nhãn lưu vào ket_qua/b_do_nhay_hat_giong.png.

print("=" * 66)
print("6.2b Độ nhạy hạt giống trên ảnh thật (oxford/boat/img1 thu nhỏ 256 px, 20 lần thử)")
import cv2
from common import load_oxford
boat, _ = load_oxford("boat", 1)
boat = cv2.resize(boat, (256, 205), interpolation=cv2.INTER_AREA)
# Hạt giống: trời, đồi tối, mặt nước, thân tàu trắng, cỏ
SEEDS_BOAT = [(8, 60), (40, 60), (75, 40), (95, 140), (190, 120)]
T_tot = 0.10
lab_rg_boat = region_growing_classic(boat, SEEDS_BOAT, T_tot)
lab_srg_boat, _ = srg(boat, SEEDS_BOAT)
print(f"  RG (T = {T_tot}): pixel có nhãn = {(lab_rg_boat >= 0).mean():.1%} | "
      f"kích thước vùng = {[int((lab_rg_boat == i).sum()) for i in range(len(SEEDS_BOAT))]}")
print(f"  SRG          : pixel có nhãn = {(lab_srg_boat >= 0).mean():.1%} | "
      f"kích thước vùng = {[int((lab_srg_boat == i).sum()) for i in range(len(SEEDS_BOAT))]}")
jitters = [2, 5, 10]
nhay_rg, nhay_srg = [], []
for j in jitters:
    f_rg = seed_sensitivity(boat, SEEDS_BOAT, lambda im, s: region_growing_classic(im, s, T_tot), 20, j)
    f_srg = seed_sensitivity(boat, SEEDS_BOAT, srg, 20, j)
    nhay_rg.append(float((f_rg > 0).mean()))
    nhay_srg.append(float((f_srg > 0).mean()))
    print(f"  jitter = {j:2d} px: tỉ lệ pixel từng đổi nhãn  RG = {nhay_rg[-1]:.3f} | SRG = {nhay_srg[-1]:.3f}")
    if j == 10:
        ban_do_rg, ban_do_srg = f_rg, f_srg

fig, axs = plt.subplots(1, 5, figsize=(22, 4.2))
axs[0].imshow(boat, cmap="gray", vmin=0, vmax=1)
axs[0].plot([c for _, c in SEEDS_BOAT], [r for r, _ in SEEDS_BOAT], "r*", ms=12)
axs[0].set_title("Ảnh + 5 hạt giống")
axs[0].axis("off")
show_labels(axs[1], lab_rg_boat, f"RG, T = {T_tot} (đen: chưa gán)")
show_labels(axs[2], lab_srg_boat, "SRG")
for ax, bd, ten in [(axs[3], ban_do_rg, "RG"), (axs[4], ban_do_srg, "SRG")]:
    m = ax.imshow(bd / 20, cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"{ten}: tần suất đổi nhãn, jitter = 10 px")
    ax.axis("off")
    fig.colorbar(m, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "b_do_nhay_hat_giong.png"), dpi=100)
plt.close(fig)


# ============================================================
# 6.2c — So sánh lân cận 4-connectivity vs 8-connectivity
# ============================================================

# **Prompt AI** — Người 4 · công cụ: Antigravity IDE · ngày 10/11/2026
# > Kiểm tra khả năng lan vùng của SRG dọc theo đường chéo mảnh 1 pixel
# > khi dùng connectivity=4 so với connectivity=8.

print("=" * 66)
print("6.2c Liên thông 4 vs 8 của SRG")
nen = tao_anh(0.0, ramp=0.0)               # nền đồng nhất 0.4 + nhiễu
duong = nen.copy()
rr = np.arange(10, 118)
duong[rr, rr] = 0.9                         # đường chéo mảnh 1 pixel
truth_duong = np.zeros_like(TRUTH)
truth_duong[rr, rr] = 1
seeds_c = [(64, 20), (10, 10)]              # hạt giống 1 nằm trên đường chéo
for conn in (4, 8):
    lab, _ = srg(duong, seeds_c, connectivity=conn)
    tren_duong = float((lab[rr, rr] == 1).mean())
    print(f"  connectivity = {conn}: tỉ lệ pixel đường chéo thuộc vùng 1 = {tren_duong:.3f} | "
          f"kích thước vùng 1 = {int((lab == 1).sum())} pixel")


# ============================================================
# 6.3 — Giải thích cơ chế (RG vs SRG)
# ============================================================

# **Prompt AI** — Người 4 · công cụ: Antigravity IDE · ngày 10/11/2026
# > Soạn phần giải thích cơ chế khác biệt giữa RG và SRG theo 3 câu hỏi:
# > đại lượng tính khác nhau, bước thực hiện, và lý giải kết quả thực nghiệm
# > ở các mục B4, 6.1, 6.2a, 6.2b, 6.2c.

print("=" * 66)
print("""6.3 Giải thích
(1) Đại lượng tính khác:
    - RG quyết định nhận hay bỏ một pixel bằng phép so |I(q) − mu| <= T với một ngưỡng T
      cố định, và thứ tự xét do stack/queue quyết định.
    - SRG không có ngưỡng. Nó tính delta(x) = |I(x) − mean(A_i)| cho mọi pixel ở biên của
      mọi vùng, rồi luôn nhận pixel có delta nhỏ nhất toàn cục trước.
(2) Ở bước nào: bước chọn pixel kế tiếp và bước quyết định có nhận pixel hay không.
    Khởi tạo hạt giống và cập nhật trung bình vùng giữ nguyên.
(3) Vì sao ra đúng hiện tượng đã đo:
    - RG cần T đủ lớn để nuốt dải sáng dọc và nhiễu bên trong một vùng, nhưng phải nhỏ
      hơn độ chênh giữa hai vùng để không rò sang vùng kia. Vì vậy chỉ có một khoảng T
      dùng được, và khoảng đó hẹp lại khi delta giảm. Ở delta nhỏ thì khoảng này biến
      mất hẳn: không có T nào cho kết quả tốt (6.2a).
    - Ngoài ra RG chạy vùng 0 xong mới tới vùng 1. Khi T quá lớn, vùng 0 chiếm cả vùng
      phải trước khi vùng 1 kịp lớn, nên độ chính xác rơi về khoảng 0.5 (6.1).
    - Stack và queue cho kết quả khác nhau (B4) vì mu thay đổi trong lúc vùng lớn: duyệt
      theo thứ tự khác thì mu tại lúc xét cùng một pixel khác nhau.
    - SRG cho mọi vùng lớn đồng thời. Pixel ở ranh giới hai vùng có delta lớn với cả hai,
      nên bị xét sau cùng, khi phần lõi mỗi vùng đã được nhận. Do đó ranh giới tự rơi vào
      chỗ tương phản cao nhất mà không cần chọn ngưỡng. SRG chỉ sai khi dải sáng trong
      vùng lớn hơn độ chênh giữa hai vùng (delta = 0.10 < RAMP = 0.15).
    - Ở ảnh thật (6.2b), SRG có tỉ lệ pixel đổi nhãn cao hơn RG. Đây không phải vì RG ổn
      định hơn: RG để khoảng 80% ảnh không có nhãn, và các pixel đó giữ nhãn -1 dù hạt
      giống dịch đi đâu. SRG bắt buộc chia hết ảnh, nên khi một hạt giống trượt sang vùng
      khác thì ranh giới giữa các vùng dịch theo, kéo theo cả mảng pixel đổi nhãn. Đây
      đúng là hệ quả đề nêu: SRG bỏ ngưỡng T, nên toàn bộ độ khó chuyển sang việc đặt
      hạt giống.
    - Cũng ở ảnh này, vùng 1 (đồi tối) của RG rỗng: vùng 0 (trời) chạy trước và chiếm luôn
      hạt giống của vùng 1. Ở SRG vùng 1 vẫn còn (dù chỉ 42 pixel) vì mọi vùng lớn đồng
      thời, không vùng nào chiếm được hạt giống của vùng khác.
    - Liên thông 4 không đi qua được đường chéo mảnh 1 pixel vì hai pixel chéo nhau không
      kề nhau; liên thông 8 thì đi được (6.2c).""")
print("Hình đã lưu trong ket_qua/")
