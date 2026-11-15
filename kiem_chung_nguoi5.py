"""
Người 5 — Kiểm chứng và thí nghiệm SRG.
Chạy từ thư mục ComputerVision2 bằng: python kiem_chung_nguoi5.py
"""
from pathlib import Path
import sys
import time
import numpy as np
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


def test_lazy_frozen():
    """Thí nghiệm gradient: quét độ dốc và đo pixel khác nhau, heap, requeue."""
    h, w = 100, 140
    seeds = [(50, 5), (50, 135)]
    slopes = [0.05, 0.15, 0.30, 0.50]
    rows = []
    yy, xx = np.mgrid[0:h, 0:w]
    noise = 0.03 * np.random.default_rng(42).normal(size=(h, w))
    for slope in slopes:
        # Gradient chéo mượt, có nhiễu nhỏ cố định để tránh hình đối xứng suy biến.
        img = np.clip(0.5 + 0.5 * (xx / w - 0.5)
                      + slope * (yy / h - 0.5) + noise, 0.0, 1.0)
        lazy, _, _, stat_l = _srg_engine(img, seeds, stale="lazy", tie="insertion")
        frozen, _, _, stat_f = _srg_engine(img, seeds, stale="frozen", tie="insertion")
        diff = int(np.count_nonzero(lazy != frozen))
        rows.append((slope, diff, stat_l["requeues"],
                     stat_l["max_heap_size"], stat_f["max_heap_size"]))
        print("Gradient slope={:.2f} | pixel khác={} | lazy requeue={} | "
              "heap max lazy={} frozen={}".format(*rows[-1]))
        if slope == 0.15:
            save_labels(lazy, "Gradient — Lazy", "b_gradient_lazy.png")
            save_labels(frozen, "Gradient — Frozen", "b_gradient_frozen.png")
            plt.imsave(IMG_DIR / "b_gradient_ngang.png", img, cmap="gray", vmin=0, vmax=1)

    slopes_a = [r[0] for r in rows]
    diffs = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(slopes_a, diffs, marker="o")
    ax.set_title("Lazy vs Frozen trên ảnh gradient")
    ax.set_xlabel("Hệ số độ dốc")
    ax.set_ylabel("Số pixel khác nhãn (pixel)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "b_lazy_frozen_quetch.png", dpi=150)
    plt.close(fig)
    return rows


def main():
    print("===== KIỂM CHỨNG NGƯỜI 5 — SRG =====")
    test_snapshot_core()
    img, seeds, _ = test_partition_and_region_count()
    test_seed_order(img, seeds)
    print("\n===== B5 — ẢNH PHẲNG / TIE =====")
    test_flat_ties()
    print("\n===== THÍ NGHIỆM LAZY VS FROZEN =====")
    test_lazy_frozen()
    print("\nĐã lưu hình trong:", OUT)


if __name__ == "__main__":
    main()
