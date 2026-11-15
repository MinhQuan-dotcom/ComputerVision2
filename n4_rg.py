"""
Người 4 — Region Growing cổ điển và công cụ đo cho phần B.

Quy ước:
- Ảnh xám float64, giá trị [0, 1], shape (H, W).
- seeds là list các tuple (row, col); hạt giống thứ i cho nhãn i.
- Pixel chưa gán nhãn có nhãn -1.
"""
import math

import numpy as np

from common import NB4
from n5_srg import _srg_core


def _kiem_tra_hat_giong(seeds, h, w):
    """Chuẩn hóa hạt giống thành list (row, col) nguyên; báo lỗi nếu trùng hoặc ra ngoài ảnh."""
    out, seen = [], set()
    for seed in seeds:
        r, c = int(seed[0]), int(seed[1])
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError(f"Hạt giống {(r, c)} nằm ngoài ảnh.")
        if (r, c) in seen:
            raise ValueError(f"Hạt giống {(r, c)} bị trùng.")
        seen.add((r, c))
        out.append((r, c))
    return out


def region_growing_classic(img, seeds, T, order="stack"):
    """Region growing cổ điển, một hoặc nhiều hạt giống.

    order: "stack" (LIFO) hoặc "queue" (FIFO).
    Các hạt giống chạy lần lượt theo thứ tự trong danh sách: vùng 0 lớn xong mới tới vùng 1.
    Pixel được gán nhãn ngay khi đẩy vào stack/queue (đúng mã giả của đề).
    Điều kiện kết nạp: |I[q] - mu| <= T, mu là trung bình hiện tại của vùng.
    -> mảng nhãn int32, -1 cho pixel không thuộc vùng nào.
    """
    if order not in ("stack", "queue"):
        raise ValueError("order chỉ nhận 'stack' hoặc 'queue'.")
    image = np.asarray(img, dtype=np.float64)
    h, w = image.shape
    seeds = _kiem_tra_hat_giong(seeds, h, w)
    labels = np.full((h, w), -1, dtype=np.int32)

    for i, (sr, sc) in enumerate(seeds):
        if labels[sr, sc] >= 0:          # hạt giống đã bị vùng trước chiếm -> vùng i rỗng
            continue
        labels[sr, sc] = i
        tong = image[sr, sc]
        n = 1
        mu = tong
        chua = [(sr, sc)]                # list dùng làm stack hoặc queue
        head = 0                         # chỉ số đầu hàng đợi khi order = "queue"

        while (len(chua) > 0) if order == "stack" else (head < len(chua)):
            if order == "stack":
                r, c = chua.pop()
            else:
                r, c = chua[head]
                head += 1
            for dr, dc in NB4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] < 0:
                    if abs(image[rr, cc] - mu) <= T:
                        labels[rr, cc] = i
                        tong += image[rr, cc]
                        n += 1
                        mu = tong / n
                        chua.append((rr, cc))
    return labels


def srg_step_frames(img, seeds, n_frames=8, **kwargs):
    """Ảnh nhãn của SRG tại các mốc tiến trình (12%, 25%, ...).

    Mốc k (k = 1..n_frames) là lúc đã gán nhãn ceil(k / n_frames) số pixel cần gán
    (không tính hạt giống). Frame cuối là kết quả cuối cùng của srg.
    kwargs được chuyển thẳng cho srg (connectivity, stale, tie).
    -> danh sách n_frames mảng nhãn.
    """
    image = np.asarray(img, dtype=np.float64)
    h, w = image.shape
    n_seeds = len(seeds)
    can_gan = h * w - n_seeds
    # _srg_core đếm tổng số pixel đã có nhãn, kể cả hạt giống.
    moc = [n_seeds + math.ceil(k * can_gan / n_frames) for k in range(1, n_frames + 1)]
    _, _, frames = _srg_core(image, seeds, snapshot_at=moc, **kwargs)
    return frames


def seed_sensitivity(img, seeds, segment_fn, n_trials=20, jitter=3):
    """Chạy lại segment_fn với hạt giống bị nhiễu +-jitter px.

    Kết quả gốc: segment_fn(img, seeds) với hạt giống chưa nhiễu.
    Mỗi lần thử: dịch từng hạt giống một số nguyên ngẫu nhiên trong [-jitter, jitter]
    theo mỗi trục, cắt vào trong ảnh, rút lại nếu trùng nhau. rng = default_rng(0).
    Nếu segment_fn trả tuple thì lấy phần tử đầu (ảnh nhãn).
    -> bản đồ tần suất: mỗi pixel đổi nhãn bao nhiêu lần trên n_trials.
    """
    image = np.asarray(img, dtype=np.float64)
    h, w = image.shape

    def _chay(s):
        out = segment_fn(image, s)
        return out[0] if isinstance(out, tuple) else out

    goc = _chay(list(seeds))
    dem = np.zeros((h, w), dtype=np.int32)
    rng = np.random.default_rng(0)
    for _ in range(n_trials):
        while True:
            moi = []
            for r, c in seeds:
                dr, dc = rng.integers(-jitter, jitter + 1, size=2)
                moi.append((int(np.clip(r + dr, 0, h - 1)), int(np.clip(c + dc, 0, w - 1))))
            if len(set(moi)) == len(moi):
                break
        dem += (_chay(moi) != goc).astype(np.int32)
    return dem
