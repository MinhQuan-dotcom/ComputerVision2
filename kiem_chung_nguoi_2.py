"""Kiểm chứng A3 (Bất biến Tỉ lệ) và Thí nghiệm A1 vs Harris — Người 2.

Chạy: python kiem_chung_nguoi_2.py
Kết quả lưu vào ket_qua/a1_scale_evidence.png và ket_qua/a1_param_sweep.png.
"""

# **Prompt AI** — Người 2 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Viết phần nhập khẩu, hằng đường dẫn và hàm tải ảnh cho script
# > kiểm chứng A3 & thí nghiệm A1 vs Harris, dùng oxford/bark/img1.png.

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_gray, make_scale, show_points
from n1_harris import harris_classic, repeatability
from n2_harris_laplace import _harris_laplace_debug, harris_laplace, scale_pyramid

_DIR = os.path.dirname(os.path.abspath(__file__))
_KQ  = os.path.join(_DIR, "ket_qua")
os.makedirs(_KQ, exist_ok=True)

# Ảnh kết cấu thực (Oxford Bark) — cấu trúc tự nhiên, không lý tưởng:
# Harris cổ điển thất bại rõ khi đổi tỉ lệ, Harris–Laplace vượt trội.
IMG_PATH = os.path.join(_DIR, "anh", "oxford", "bark", "img1.png")

_SIGMAS_16 = scale_pyramid(sigma0=1.2, xi=1.2, n_levels=16)   # sigma_max ≈ 18.5


# ============================================================
# KIỂM CHỨNG A3 — Repeatability Harris vs Harris–Laplace (1→3×)
# ============================================================

# **Prompt AI** — Người 2 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Tải oxford/bark/img1.png, phóng to 3×, phát hiện điểm bằng
# > harris_classic và harris_laplace (n_levels=16), tính và in
# > repeatability 1→3× cho cả hai phương pháp.

def run_kiem_chung_a3():
    img      = load_gray(IMG_PATH)
    img_3x, tf = make_scale(img, 3.0)

    # Đánh giá khách quan: cả hai phương pháp dùng cùng cấu hình mặc định trên cả hai ảnh
    pts_h1 = harris_classic(img)
    pts_h3 = harris_classic(img_3x)
    pts_l1 = harris_laplace(img,    _SIGMAS_16)
    pts_l3 = harris_laplace(img_3x, _SIGMAS_16)

    rep_h = repeatability(pts_h1, pts_h3, tf, img_3x.shape)
    rep_l = repeatability(pts_l1, pts_l3, tf, img_3x.shape)

    print("=" * 56)
    print("KIỂM CHỨNG A3 — Bất biến Tỉ lệ  (ảnh: oxford/bark)")
    print(f"  sigma_min={_SIGMAS_16[0]:.3f}  sigma_max={_SIGMAS_16[-1]:.3f}")
    print("-" * 56)
    print(f"  Harris cổ điển : repeatability 1→3× = {rep_h:.3f}"
          f"  ({len(pts_h1)} / {len(pts_h3)} điểm)")
    print(f"  Harris–Laplace : repeatability 1→3× = {rep_l:.3f}"
          f"  ({len(pts_l1)} / {len(pts_l3)} điểm)")
    print("=" * 56)

    return img, img_3x, tf


# ============================================================
# 5.1.1 — Trực quan hóa điểm qua các hệ số tỉ lệ
# ============================================================

# **Prompt AI** — Người 2 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Vẽ lưới 2×4: hàng trên Harris cổ điển (dấu + đỏ), hàng dưới
# > Harris–Laplace (vòng tròn xanh lá r=2σ), qua 1×/1.5×/2×/3×.
# > Lưu vào ket_qua/a1_scale_evidence.png.

def run_evidence(img):
    factors = [1.0, 1.5, 2.0, 3.0]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        "A3 — Harris cổ điển (+) vs Harris–Laplace (○)  |  oxford/bark/img1",
        fontsize=12, fontweight="bold"
    )

    for col, f in enumerate(factors):
        img_f = img if f == 1.0 else make_scale(img, f)[0]

        # Harris cổ điển dùng cùng tham số mặc định trên mọi tỉ lệ
        pts_h = harris_classic(img_f)
        pts_l = harris_laplace(img_f, _SIGMAS_16)

        show_points(axes[0, col], img_f, pts_h,
                    title=f"Harris  {f:.1f}×   ({len(pts_h)} pts)",
                    color="red", circles=False)
        show_points(axes[1, col], img_f, pts_l,
                    title=f"H–Laplace  {f:.1f}×   ({len(pts_l)} pts)",
                    color="lime", circles=True)

    plt.tight_layout()
    out = os.path.join(_KQ, "a1_scale_evidence.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ============================================================
# 5.1.2 — Quét tham số ξ và log_thresh
# ============================================================

# **Prompt AI** — Người 2 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Quét xi trong [1.1, 1.5] (9 điểm) dùng _harris_laplace_debug để
# > tính rejection_rate = 1 - n_step2/n_step1.
# > Quét log_thresh trong [0, 0.05] (9 điểm) tính repeatability 1→3×.
# > Vẽ 2 biểu đồ có nhãn trục, đơn vị, grid. Lưu ket_qua/a1_param_sweep.png.

def run_param_sweep(img):
    img_3x, tf = make_scale(img, 3.0)

    xi_vals  = np.linspace(1.1, 1.5, 9)
    lt_vals  = np.linspace(0.0, 0.05, 9)

    # Quét xi: đo rejection rate
    rej_rates = []
    for xi in xi_vals:
        sigs = scale_pyramid(sigma0=1.2, xi=xi, n_levels=12)
        _, n1, n2 = _harris_laplace_debug(img, sigs)
        rej_rates.append(1.0 - n2 / n1 if n1 > 0 else 1.0)

    # Quét log_thresh: tính repeatability 1→3×
    rep_vals = []
    for lt in lt_vals:
        p1 = harris_laplace(img,    _SIGMAS_16, log_thresh=lt)
        p3 = harris_laplace(img_3x, _SIGMAS_16, log_thresh=lt)
        r  = repeatability(p1, p3, tf, img_3x.shape)
        rep_vals.append(float(r) if np.isfinite(r) else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        "Quét tham số Harris–Laplace  (k=0.04, oxford/bark/img1)",
        fontsize=12, fontweight="bold"
    )

    ax1.plot(xi_vals, rej_rates, "o-", color="steelblue", lw=2, ms=7)
    ax1.set_xlabel("Hệ số bước nhảy ξ")
    ax1.set_ylabel("Rejection Rate  (tỉ lệ loại trừ LoG)")
    ax1.set_title("Rejection Rate vs ξ   [n_levels=12, log_thresh=0]")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.35)

    ax2.plot(lt_vals, rep_vals, "s-", color="darkorange", lw=2, ms=7)
    ax2.set_xlabel("Ngưỡng LoG  (log_thresh)")
    ax2.set_ylabel("Repeatability  1→3×")
    ax2.set_title("Repeatability vs log_thresh   [ξ=1.2, n_levels=16]")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    out = os.path.join(_KQ, "a1_param_sweep.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ============================================================
# 5.1.3 — Cơ chế bất biến tỉ lệ
# ============================================================

# **Prompt AI** — Người 2 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Soạn đoạn văn súc tích trả lời 3 câu hỏi cơ chế bất biến tỉ lệ
# > của Harris–Laplace: đại lượng nào, bước nào, vì sao bất biến.

_MECHANISM = """\
CƠ CHẾ BẤT BIẾN TỈ LỆ — Harris–Laplace (A1)
─────────────────────────────────────────────
(1) Đại lượng: σ²·|ΔL| — Laplacian-of-Gaussian chuẩn hóa Lindeberg.
    Khi ảnh phóng k lần, biên độ LoG thô giảm 1/k², nhưng thừa số σ²
    bù lại đúng k² (vì σ tối ưu tỉ lệ thuận k) → giá trị chuẩn hóa
    tại điểm đặc trưng là hằng số theo tỉ lệ.

(2) Bước thực hiện: Bước 2 của harris_laplace — chọn tỉ lệ bằng cực
    đại địa phương của log_stack theo trục tháp. Khi ảnh thay đổi tỉ
    lệ, cực đại dịch sang tầng σ_new ≈ k·σ_old nhưng vẫn là cực đại
    → điểm đặc trưng được tìm lại ở tỉ lệ đúng.

(3) Lý thuyết (Lindeberg 1998): đáp ứng LoG chuẩn hóa đạt cực đại
    theo σ chính xác khi σ khớp với "tỉ lệ đặc trưng" của cấu trúc.
    Phép phóng/thu chỉ dịch vị trí cực đại trong tháp σ, không phá
    hủy tính cực đại → thuật toán tự chọn đúng σ ở cả hai ảnh.
─────────────────────────────────────────────"""


def print_mechanism():
    print(_MECHANISM)


# ============================================================
if __name__ == "__main__":
    img, img_3x, tf = run_kiem_chung_a3()
    run_evidence(img)
    run_param_sweep(img)
    print_mechanism()
