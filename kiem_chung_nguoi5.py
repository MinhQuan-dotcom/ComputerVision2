"""
Người 5 — Kiểm chứng và thí nghiệm SRG.
Chạy từ thư mục ComputerVision2 bằng: python kiem_chung_nguoi5.py
"""

# **Prompt AI** — Người 5 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Viết phần import thư viện, cấu hình đường dẫn ket_qua/ và anh/, hàm lưu
# > bản đồ nhãn save_labels và hàm test_snapshot_core để kiểm tra giao diện
# > chụp snapshot của _srg_core.

from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # trong notebook thì bỏ dòng này
import matplotlib.pyplot as plt

from n5_srg import srg, _srg_core, _srg_engine

try:
    from common import NB4, NB8, remap_labels, show_labels
except ImportError:
    NB4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    NB8 = NB4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    def remap_labels(labels, perm):
        out = labels.copy()
        mask = labels >= 0
        out[mask] = np.asarray(perm)[out[mask]]
        return out
    def show_labels(ax, labels, title=""):
        ax.imshow(labels, cmap="tab20", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ket_qua"
IMG_DIR = ROOT / "anh"
OUT.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)


def save_labels(labels, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    show_labels(ax, labels, title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def test_snapshot_core():
    """Kiểm tra giao diện _srg_core mà Người 4 sẽ dùng."""
    img = np.linspace(0.0, 1.0, 40 * 50).reshape(40, 50)
    seeds = [(10, 8), (30, 40)]
    marks = [500, 1000, 2000]
    labels, boundary, frames = _srg_core(
        img, seeds, connectivity=4, stale="lazy", tie="insertion",
        snapshot_at=marks
    )
    assert len(frames) == len(marks)
    assert all(frame.shape == labels.shape for frame in frames)
    print("Giao diện _srg_core — số frame:", len(frames), "| mốc:", marks)


# ============================================================
# KIỂM CHỨNG B1, B2 & B3 — Phân hoạch đầy đủ, số vùng & thứ tự seed
# ============================================================

# **Prompt AI** — Người 5 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Viết hàm kiểm chứng B1 (mọi pixel đều có nhãn >= 0), B2 (số vùng bằng
# > đúng số seed) và B3 (đảo ngược danh sách seed rồi remap_labels để kiểm tra
# > tính bất biến thứ tự hạt giống trên ảnh nhiễu thực và ảnh phẳng).

def test_partition_and_region_count():
    """B1, B2: mọi pixel được gán nhãn và số vùng bằng số seed."""
    rng = np.random.default_rng(7)
    img = rng.random((80, 100), dtype=np.float64)
    seeds = [(10, 10), (10, 85), (65, 50)]
    labels, boundary = srg(img, seeds)

    assert (labels >= 0).all()
    assert len(np.unique(labels)) == len(seeds)
    print("B1 — Phân hoạch đầy đủ:", bool((labels >= 0).all()))
    print("B2 — Số vùng:", len(np.unique(labels)), "| số seed:", len(seeds))
    save_labels(labels, "B1-B2: Phân hoạch SRG", "b_kiem_chung_phan_hoach.png")
    return img, seeds, labels


def test_seed_order(img, seeds):
    """B3: đảo seed rồi remap nhãn về thứ tự ban đầu."""
    n = len(seeds)
    results = {}
    for tie in ("raster", "insertion"):
        labels_a, _ = srg(img, seeds, stale="lazy", tie=tie)
        labels_b, _ = srg(img, seeds[::-1], stale="lazy", tie=tie)
        labels_b_remapped = remap_labels(labels_b, list(range(n - 1, -1, -1)))
        different = int(np.count_nonzero(labels_a != labels_b_remapped))
        results[tie] = different
        print(f"B3 — Số pixel khác sau remap (tie={tie}):", different)

    # Trường hợp nhiều đồng hạng: ảnh phẳng có thể làm insertion phụ thuộc thứ tự seed.
    flat = np.full(img.shape, 0.5, dtype=np.float64)
    flat_a, _ = srg(flat, seeds, stale="frozen", tie="insertion")
    flat_b, _ = srg(flat, seeds[::-1], stale="frozen", tie="insertion")
    flat_b = remap_labels(flat_b, list(range(n - 1, -1, -1)))
    results["insertion_flat"] = int(np.count_nonzero(flat_a != flat_b))
    print("B3 — Số pixel khác sau remap (ảnh phẳng, insertion):",
          results["insertion_flat"])
    return results


# ============================================================
# KIỂM CHỨNG B5 — Tie-breaking trên ảnh phẳng (insertion, raster, random)
# ============================================================

# **Prompt AI** — Người 5 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Chạy srg trên ảnh phẳng 101×101 với 2 hạt giống đối xứng qua 3 quy tắc
# > phá hòa (insertion, raster, random). Thống kê kích thước vùng, số pixel
# > khác nhau và lưu hình so sánh ket_qua/b_so_sanh_tie.png.

def test_flat_ties():
    """B5: ảnh phẳng, mọi delta bằng 0 nên tie-breaking quyết định hình dạng."""
    img = np.full((101, 101), 0.5, dtype=np.float64)
    seeds = [(50, 20), (50, 80)]
    results = {}
    for tie in ("insertion", "raster", "random"):
        labels, boundary = srg(img, seeds, stale="frozen", tie=tie)
        results[tie] = labels
        save_labels(labels, f"Ảnh phẳng — tie={tie}", f"b_flat_tie_{tie}.png")
        print(f"B5 — tie={tie}: kích thước vùng",
              np.bincount(labels.ravel(), minlength=2).tolist())

    comparisons = {}
    for a, b in (("insertion", "raster"),
                 ("insertion", "random"),
                 ("raster", "random")):
        diff = int(np.count_nonzero(results[a] != results[b]))
        comparisons[(a, b)] = diff
        print(f"Khác biệt {a} vs {b}: {diff} pixel")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, tie in zip(axes, ("insertion", "raster", "random")):
        show_labels(ax, results[tie], f"tie={tie}")
    fig.tight_layout()
    fig.savefig(OUT / "b_so_sanh_tie.png", dpi=150)
    plt.close(fig)
    return comparisons


# ============================================================
# THÍ NGHIỆM B1 — Cặp xử lý phần tử cũ: Lazy vs Frozen theo độ dốc sáng
# ============================================================

# **Prompt AI** — Người 5 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Quét độ mạnh dải sáng dọc (ramp) từ 0.0 đến 0.8 trên ảnh hai vùng chênh 0.2.
# > Dùng _srg_engine đo độ chính xác, số lần đẩy lại (requeues) và kích thước
# > heap cực đại giữa stale="lazy" và stale="frozen". Vẽ 2 đồ thị lưu vào
# > ket_qua/b_lazy_frozen_quetch.png.

def test_lazy_frozen():
    """Thí nghiệm lazy vs frozen: hai vùng chênh 0.2, mỗi vùng có dải sáng dần theo chiều dọc.

    Biết đáp án nên đo được độ chính xác. Quét độ mạnh của dải sáng (ramp) từ 0 tới 0.8.
    """
    h, w = 100, 100
    truth = np.zeros((h, w), dtype=np.int32)
    truth[:, w // 2:] = 1
    seeds = [(50, 25), (50, 75)]
    noise = 0.02 * np.random.default_rng(3).normal(size=(h, w))
    ramps = np.round(np.arange(0.0, 0.81, 0.05), 2)
    rows = []
    for ramp in ramps:
        img = np.where(truth == 0, 0.4, 0.6) + ramp * np.linspace(-0.5, 0.5, h)[:, None] + noise
        img = np.clip(img, 0.0, 1.0)
        lazy, _, _, st_l = _srg_engine(img, seeds, stale="lazy", tie="insertion")
        frozen, _, _, st_f = _srg_engine(img, seeds, stale="frozen", tie="insertion")
        rows.append((ramp, float((lazy == truth).mean()), float((frozen == truth).mean()),
                     int((lazy != frozen).sum()), st_l["requeues"],
                     st_l["max_heap_size"], st_f["max_heap_size"]))
        print("ramp={:.2f} | chính xác lazy={:.3f} frozen={:.3f} | pixel khác={:4d} | "
              "lazy đẩy lại={} | heap max lazy={} frozen={}".format(*rows[-1]))
        if abs(ramp - 0.3) < 1e-9:
            plt.imsave(IMG_DIR / "b_hai_vung_ramp03.png", img, cmap="gray", vmin=0, vmax=1)
            save_labels(lazy, f"Ramp 0.3 — Lazy (chính xác {rows[-1][1]:.3f})", "b_gradient_lazy.png")
            save_labels(frozen, f"Ramp 0.3 — Frozen (chính xác {rows[-1][2]:.3f})", "b_gradient_frozen.png")

    r = np.array(rows)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    axs[0].plot(r[:, 0], r[:, 1], "o-", label="lazy")
    axs[0].plot(r[:, 0], r[:, 2], "s-", label="frozen")
    axs[0].axvline(0.2, color="gray", ls=":", lw=1)
    axs[0].text(0.205, 0.62, "ramp = độ chênh\nhai vùng (0.2)", fontsize=9)
    axs[0].set_xlabel("Độ mạnh dải sáng trong vùng (ramp)")
    axs[0].set_ylabel("Độ chính xác so với đáp án")
    axs[0].set_title("Lazy vs Frozen theo ramp (hai vùng chênh 0.2)")
    axs[0].grid(alpha=0.3)
    axs[0].legend()
    axs[1].plot(r[:, 0], r[:, 5], "o-", label="heap lớn nhất — lazy")
    axs[1].plot(r[:, 0], r[:, 6], "s-", label="heap lớn nhất — frozen")
    axs[1].plot(r[:, 0], r[:, 4], "^-", label="số lần lazy đẩy lại")
    axs[1].set_xlabel("Độ mạnh dải sáng trong vùng (ramp)")
    axs[1].set_ylabel("Số phần tử")
    axs[1].set_title("Chi phí của lazy")
    axs[1].grid(alpha=0.3)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "b_lazy_frozen_quetch.png", dpi=130)
    plt.close(fig)
    return rows


# ============================================================
# THÍ NGHIỆM B2 — Ảnh hưởng của lượng tử hóa mức xám tới Tie-breaking
# ============================================================

# **Prompt AI** — Người 5 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Quét số mức lượng tử hóa mức xám L trong [256, 128, 64, 32, 16, 8, 4]
# > (lấy trung bình trên 6 ảnh nhiễu), đo tỉ lệ pixel khác nhãn giữa insertion
# > với raster và random. Vẽ biểu đồ thang log2 lưu vào ket_qua/b_tie_luong_tu.png.

def test_tie_quantization(n_anh=6):
    """Đồng hạng theo số mức xám: ảnh càng ít mức xám thì càng nhiều delta bằng nhau,
    quy tắc phá hòa càng làm đổi kết quả. Lấy trung bình trên n_anh ảnh nhiễu khác nhau."""
    h, w = 100, 100
    seeds = [(50, 25), (50, 75), (20, 50)]
    base = np.where(np.arange(w)[None, :] < w // 2, 0.4, 0.6) * np.ones((h, 1))
    levels = [256, 128, 64, 32, 16, 8, 4]
    rows = []
    for L in levels:
        d_ir, d_ia = [], []
        for sd in range(n_anh):
            goc = np.clip(base + 0.08 * np.random.default_rng(sd).normal(size=(h, w)), 0.0, 1.0)
            img = np.round(goc * (L - 1)) / (L - 1)
            kq = {t: srg(img, seeds, stale="frozen", tie=t)[0] for t in ("insertion", "raster", "random")}
            d_ir.append(float((kq["insertion"] != kq["raster"]).mean()))
            d_ia.append(float((kq["insertion"] != kq["random"]).mean()))
        rows.append((L, float(np.mean(d_ir)), float(np.mean(d_ia))))
        print(f"Mức xám = {L:3d} | tỉ lệ pixel khác (trung bình {n_anh} ảnh): "
              f"insertion vs raster = {rows[-1][1]:.2%} | insertion vs random = {rows[-1][2]:.2%}")
    r = np.array(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r[:, 0], 100 * r[:, 1], "o-", label="insertion vs raster")
    ax.plot(r[:, 0], 100 * r[:, 2], "s-", label="insertion vs random")
    ax.set_xscale("log", base=2)
    ax.invert_xaxis()
    ax.set_xlabel("Số mức xám (thang log, giảm dần)")
    ax.set_ylabel("Tỉ lệ pixel khác nhãn (%)")
    ax.set_title(f"Ảnh hai vùng + nhiễu, lượng tử hóa — frozen, TB {n_anh} ảnh")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "b_tie_luong_tu.png", dpi=130)
    plt.close(fig)
    return rows


# ============================================================
# GIẢI THÍCH CƠ CHẾ — Tie-breaking, Lazy vs Frozen & Lượng tử hóa
# ============================================================

# **Prompt AI** — Người 5 · công cụ: Antigravity IDE · ngày 11/11/2026
# > Soạn văn bản giải thích cơ chế B5 (hình dạng vùng theo quy tắc phá hòa),
# > B3 (phụ thuộc thứ tự seed khi đồng hạng), cơ chế Lazy vs Frozen theo 3 ý
# > và hiện tượng đồng hạng tăng khi giảm số mức xám.

GIAI_THICH = """
===== GIẢI THÍCH =====
B5 — hình dạng trên ảnh phẳng (mọi delta = 0, chỉ quy tắc phá hòa quyết định):
  - insertion: phần tử đẩy vào trước được lấy ra trước, nên cả hai vùng lớn đều nhau như
    duyệt theo chiều rộng. Hai vùng gặp nhau ở giữa hai hạt giống, ranh giới là đường
    thẳng đứng (5151 / 5050 pixel).
  - raster: lấy pixel có row*W + col nhỏ nhất. Vùng 0 có láng giềng ở hàng trên với khóa
    nhỏ hơn nên chiếm hết các hàng phía trên, rồi bao vây luôn hạt giống của vùng 1.
    Vùng 1 chỉ còn đúng hạt giống (10200 / 1 pixel).
  - random: khóa ngẫu nhiên gán lúc đẩy vào heap. Các phần tử còn lại trong heap có khóa
    ngày càng lớn, nên phần tử mới đẩy vào thường nhỏ hơn và được lấy ra ngay. Vùng nào
    đang lớn thì tiếp tục lớn, gần giống duyệt theo chiều sâu, nên một vùng chiếm gần hết.
B3 — trên ảnh nhiễu số thực, đảo thứ tự hạt giống cho 0 pixel khác: thứ tự nhận chỉ do
  delta quyết định. Trên ảnh phẳng thì 52 pixel khác, vì khi delta bằng nhau thì khóa
  insertion phụ thuộc thứ tự hạt giống được đẩy vào. Đây là hạn chế đề nêu ở mục đồng hạng.

Thí nghiệm lazy vs frozen:
(1) Đại lượng tính khác: lazy so I(x) với trung bình HIỆN TẠI của vùng; frozen so với giá
    trị của hạt giống, không bao giờ cập nhật.
(2) Ở bước: tính delta khi đẩy vào heap và khi lấy ra (lazy tính lại, đẩy lại nếu đã cũ).
(3) Vì sao: khi trong vùng có dải sáng, pixel xa hạt giống lệch khỏi giá trị hạt giống một
    khoảng ramp/2. Khi ramp/2 tiến gần nửa độ chênh hai vùng, delta tới hạt giống của
    chính nó bằng delta tới vùng bên kia, nên frozen bắt đầu gán nhầm (độ chính xác rơi từ
    ramp khoảng 0.2). Lazy cập nhật trung bình khi vùng lớn dần vào dải sáng nên theo
    được lâu hơn (ở ramp 0.3: lazy 0.932, frozen 0.821). Đổi lại, lazy phải đẩy lại
    khoảng 9–10 nghìn phần tử vào heap, gần bằng số pixel của ảnh (10 nghìn).
    Khi ramp quá lớn (> 0.5) cả hai đều sai vì trong vùng còn chênh nhiều hơn giữa hai vùng.

Thí nghiệm đồng hạng theo số mức xám:
  Ảnh càng ít mức xám thì càng nhiều pixel có cùng delta, nên khóa phá hòa quyết định
  nhiều lần nhận hơn. Tỉ lệ pixel khác giữa các quy tắc tăng từ khoảng 0.02% (256 mức)
  lên 18–26% (4–8 mức). Đường không tăng đều tuyệt đối vì vị trí đồng hạng phụ thuộc
  cách làm tròn, nên đã lấy trung bình trên nhiều ảnh nhiễu.
"""


def main():
    print("===== KIỂM CHỨNG NGƯỜI 5 — SRG =====")
    test_snapshot_core()
    img, seeds, _ = test_partition_and_region_count()
    test_seed_order(img, seeds)
    print("\n===== B5 — ẢNH PHẲNG / TIE =====")
    test_flat_ties()
    print("\n===== THÍ NGHIỆM LAZY VS FROZEN =====")
    test_lazy_frozen()
    print("\n===== THÍ NGHIỆM ĐỒNG HẠNG THEO SỐ MỨC XÁM =====")
    test_tie_quantization()
    print(GIAI_THICH)
    print("\nĐã lưu hình trong:", OUT)


if __name__ == "__main__":
    main()