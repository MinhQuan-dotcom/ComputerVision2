"""
Người 5 — Seeded Region Growing (SRG).

Quy ước:
- Ảnh xám float64, giá trị [0, 1], shape (H, W).
- seeds là list các tuple (row, col), nhãn theo thứ tự seed.
- Pixel chưa gán nhãn có nhãn -1.
"""
import heapq
import numpy as np

try:
    from common import NB4, NB8
except ImportError:
    NB4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    NB8 = NB4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _validate_srg_inputs(img, seeds, connectivity, stale, tie):
    """Kiểm tra đầu vào và chuẩn hóa danh sách hạt giống."""
    image = np.asarray(img)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("img phải là ảnh xám 2 chiều và không rỗng.")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("img phải có kiểu dữ liệu số.")
    if not np.isfinite(image).all():
        raise ValueError("img không được chứa NaN hoặc vô cực.")

    h, w = image.shape
    if connectivity not in (4, 8):
        raise ValueError("connectivity chỉ nhận 4 hoặc 8.")
    if stale not in ("lazy", "frozen"):
        raise ValueError("stale chỉ nhận 'lazy' hoặc 'frozen'.")
    if tie not in ("insertion", "raster", "random"):
        raise ValueError("tie chỉ nhận 'insertion', 'raster' hoặc 'random'.")
    if seeds is None or len(seeds) == 0:
        raise ValueError("Cần ít nhất một hạt giống.")

    normalized = []
    seen = set()
    for seed in seeds:
        if len(seed) != 2:
            raise ValueError("Mỗi hạt giống phải có dạng (row, col).")
        r, c = seed
        if not isinstance(r, (int, np.integer)) or not isinstance(c, (int, np.integer)):
            raise ValueError("Tọa độ hạt giống phải là số nguyên.")
        r, c = int(r), int(c)
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError(f"Hạt giống {(r, c)} nằm ngoài ảnh.")
        if (r, c) in seen:
            raise ValueError(f"Hạt giống {(r, c)} bị trùng.")
        seen.add((r, c))
        normalized.append((r, c))
    return image.astype(np.float64, copy=False), normalized


def _srg_engine(img, seeds, connectivity=4, stale="lazy", tie="insertion",
                snapshot_at=None):
    """Lõi SRG, trả thêm thống kê để phục vụ thí nghiệm."""
    image, seeds = _validate_srg_inputs(img, seeds, connectivity, stale, tie)
    h, w = image.shape
    n_regions = len(seeds)
    neighbors = NB4 if connectivity == 4 else NB8

    labels = np.full((h, w), -1, dtype=np.int32)
    boundary = np.zeros((h, w), dtype=bool)

    sums = np.zeros(n_regions, dtype=np.float64)
    counts = np.zeros(n_regions, dtype=np.int64)
    seed_values = np.empty(n_regions, dtype=np.float64)

    for i, (r, c) in enumerate(seeds):
        labels[r, c] = i
        sums[i] = image[r, c]
        counts[i] = 1
        seed_values[i] = image[r, c]

    rng = np.random.default_rng(0) if tie == "random" else None
    insertion_counter = 0
    heap = []
    in_heap = set()
    requeues = 0
    pushes = 0
    max_heap_size = 0

    requested = set()
    if snapshot_at is not None:
        requested = {int(x) for x in snapshot_at}
        if any(x < n_regions or x > h * w for x in requested):
            raise ValueError("snapshot_at phải nằm trong [số seed, số pixel ảnh].")
    frames = {}
    if n_regions in requested:
        frames[n_regions] = labels.copy()

    def current_delta(r, c, region):
        if stale == "frozen":
            reference = seed_values[region]
        else:
            reference = sums[region] / counts[region]
        return abs(float(image[r, c]) - float(reference))

    def push(r, c, region, is_requeue=False):
        nonlocal insertion_counter, pushes, requeues, max_heap_size
        pair = (r, c, region)
        if labels[r, c] >= 0 or pair in in_heap:
            return

        delta = current_delta(r, c, region)
        if tie == "insertion":
            tie_key = insertion_counter
            insertion_counter += 1
        elif tie == "raster":
            tie_key = r * w + c
        else:
            tie_key = float(rng.random())

        heapq.heappush(heap, (delta, tie_key, region, r, c))
        in_heap.add(pair)
        pushes += 1
        if is_requeue:
            requeues += 1
        max_heap_size = max(max_heap_size, len(heap))

    # Đưa tất cả láng giềng chưa gán nhãn của các hạt giống vào heap.
    for i, (r, c) in enumerate(seeds):
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] < 0:
                push(rr, cc, i)

    assigned = n_regions
    while heap:
        delta_saved, tie_key, region, r, c = heapq.heappop(heap)
        in_heap.discard((r, c, region))

        # Pixel đã được một vùng khác kết nạp thì phần tử này hết hiệu lực.
        if labels[r, c] >= 0:
            continue

        # Lazy: δ lưu trong heap có thể đã cũ vì trung bình vùng thay đổi.
        if stale == "lazy":
            delta_now = current_delta(r, c, region)
            if abs(delta_now - delta_saved) > 1e-12:
                push(r, c, region, is_requeue=True)
                continue

        # Pixel biên nếu các láng giềng đã gán nhãn thuộc từ hai vùng trở lên.
        adjacent_labels = set()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] >= 0:
                adjacent_labels.add(int(labels[rr, cc]))
        # Pixel hiện tại thuộc region; nếu có láng giềng thuộc vùng khác thì là biên.
        if any(x != region for x in adjacent_labels):
            boundary[r, c] = True

        labels[r, c] = region
        sums[region] += image[r, c]
        counts[region] += 1
        assigned += 1

        if assigned in requested:
            frames[assigned] = labels.copy()

        # Chỉ vùng vừa kết nạp pixel mới được mở rộng từ pixel đó.
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] < 0:
                push(rr, cc, region)

    if not (labels >= 0).all():
        raise AssertionError("SRG không tạo được phân hoạch đầy đủ.")

    # Trả frame theo đúng thứ tự snapshot_at mà người gọi yêu cầu.
    ordered_frames = []
    if snapshot_at is not None:
        for k in snapshot_at:
            k = int(k)
            if k in frames:
                ordered_frames.append(frames[k])
    stats = {
        "pushes": pushes,
        "requeues": requeues,
        "max_heap_size": max_heap_size,
        "region_sizes": counts.copy(),
    }
    return labels, boundary, ordered_frames, stats


def _srg_core(img, seeds, connectivity=4, stale="lazy", tie="insertion",
              snapshot_at=None):
    """Lõi SRG dùng chung; trả (labels, boundary_mask, frames).

    Khi cả delta và tie_key bằng nhau, region có chỉ số nhỏ hơn thắng.
    snapshot_at là danh sách số pixel đã gán nhãn cần lưu ảnh nhãn.
    """
    labels, boundary, frames, _ = _srg_engine(
        img, seeds, connectivity, stale, tie, snapshot_at
    )
    return labels, boundary, frames


def srg(img, seeds, connectivity=4, stale="lazy", tie="insertion"):
    """Seeded Region Growing theo Adams & Bischof (1994).

    stale:
        'lazy'   — tính lại delta khi pop, đẩy lại nếu giá trị đã cũ.
        'frozen' — delta luôn đo với giá trị cường độ tại seed.
    tie:
        'insertion' — thứ tự được đưa vào heap.
        'raster'    — thứ tự row * W + col.
        'random'    — số ngẫu nhiên với default_rng(0), tất định.
    Nếu delta và tie_key cùng bằng nhau, vùng có chỉ số nhỏ hơn thắng.

    Trả về (labels, boundary_mask).
    """
    labels, boundary, _ = _srg_core(
        img, seeds, connectivity=connectivity, stale=stale, tie=tie,
        snapshot_at=None
    )
    return labels, boundary