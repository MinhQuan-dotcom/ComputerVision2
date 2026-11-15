"""Ghép các file .py của nhóm thành một notebook bt1 duy nhất.

Dùng (chạy ở thư mục repo):  python tools/ghep_notebook.py . bt1_Nhom.ipynb
Chạy lại mỗi khi có người sửa file .py, rồi mở notebook và Restart & Run All.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])

INTERNAL = ("common", "n1_harris", "n2_harris_laplace", "n3_anms", "n4_rg", "n5_srg")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    text = text.strip("\n")
    if text.strip():
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text})


def strip_docstring(src):
    return re.sub(r'\A\s*(?:"""|\'\'\')[\s\S]*?(?:"""|\'\'\')\s*\n', "", src, count=1)


def strip_try_import(src):
    """Bỏ khối try: from <nội bộ> import ... except ImportError: ... (thân thụt lề)."""
    lines = src.split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "try:" and i + 1 < len(lines) and re.match(
                r"\s+from (%s) import" % "|".join(INTERNAL), lines[i + 1]):
            i += 1
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or lines[i].strip() == ""):
                i += 1
            if i < len(lines) and lines[i].startswith("except"):
                i += 1
                while i < len(lines) and (lines[i].startswith((" ", "\t")) or lines[i].strip() == ""):
                    i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def clean(src):
    src = strip_docstring(src)
    src = strip_try_import(src)
    keep = []
    for ln in src.split("\n"):
        s = ln.strip()
        if re.match(r"(from (%s) import)" % "|".join(INTERNAL), s):
            continue
        if s == "import matplotlib" or s.startswith("matplotlib.use("):
            continue
        keep.append(ln)
    src = "\n".join(keep)
    src = src.replace("os.path.dirname(os.path.abspath(__file__))", "os.getcwd()")
    src = src.replace("Path(__file__).resolve().parent", "Path.cwd()")
    src = re.sub(r"plt\.close\(fig\)", "plt.show()", src)
    assert "__file__" not in src, "còn __file__"
    return src


def split_sections(src):
    """Tách script theo khối tiêu đề 3 dòng '# ====' / '# TIÊU ĐỀ' / '# ===='.
    -> list (tiêu đề hoặc None, mã nguồn)."""
    pat = re.compile(r"^# ={10,}\n# (.+)\n# ={10,}\n", re.M)
    parts, last, title = [], 0, None
    for m in pat.finditer(src):
        parts.append((title, src[last:m.start()]))
        title, last = m.group(1).strip(), m.end()
    parts.append((title, src[last:]))
    return parts


def emit(body, extra_call=""):
    """Tách các khối '# **Prompt AI** ...' thành cell markdown đặt ngay trên phần code theo sau."""
    lines = body.split("\n")
    buf, i = [], 0
    while i < len(lines):
        if lines[i].startswith("# **Prompt AI**"):
            code("\n".join(buf))
            buf = []
            pr = []
            while i < len(lines) and lines[i].startswith("#"):
                pr.append(lines[i][2:] if lines[i].startswith("# ") else lines[i][1:])
                i += 1
            md("\n".join(pr))
            continue
        buf.append(lines[i])
        i += 1
    code("\n".join(buf) + ("\n\n" + extra_call if extra_call else ""))


def add_script(fname, calls=None, drop_from=None, level="####"):
    src = clean((REPO / fname).read_text(encoding="utf-8"))
    if drop_from:
        k = src.find(drop_from)
        if k >= 0:
            src = src[:k]
    src = re.sub(r"\n# ={10,}\n(?=\s*\Z|\nif __name__)", "\n", src)
    for title, body in split_sections(src):
        if title:
            md(f"{level} {title}")
        emit(body, (calls or {}).get(title or "", ""))


def add_module(fname):
    emit(clean((REPO / fname).read_text(encoding="utf-8")))


# ======================================================================== nội dung
md("""
# Bài tập 1 — Đặc trưng góc & Phân vùng ảnh

Môn Thị giác máy tính (121036) — Trường Đại học Giao thông Vận tải TP.HCM

**Nhóm:** ........

| Người | Họ tên | MSSV | Phụ trách |
| --- | --- | --- | --- |
| 1 | | | Harris cổ điển, đo repeatability, kiểm chứng A1, A2, A4, ghép notebook |
| 2 | | | Harris–Laplace (A1), kiểm chứng A3, thí nghiệm A1 vs Harris |
| 3 | | | ANMS (A2), kiểm chứng A5, thí nghiệm A2 vs Harris |
| 4 | | | Region Growing cổ điển, `srg_step_frames`, `seed_sensitivity`, kiểm chứng B4, thí nghiệm SRG vs RG |
| 5 | | | Seeded Region Growing, kiểm chứng B1, B2, B3, B5, thí nghiệm lazy/frozen và đồng hạng |

**Cách chạy:** đặt notebook cùng cấp với thư mục `anh/` (có sẵn `anh/oxford/`), rồi chọn Restart & Run All. Toàn bộ notebook chạy khoảng 5 phút; phần lâu nhất là kiểm chứng A3 của Người 2 (khoảng 4 phút).
""")

md("## 0. Thư viện, hằng số và hàm dùng chung")
code("""
%matplotlib inline
import math
import os
import sys
import time
import heapq
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter          # chỉ dùng để làm mờ Gauss
from scipy.ndimage import label as nd_label        # chỉ dùng ở phần phân tích

plt.rcParams["figure.max_open_warning"] = 0
""")
add_module("common.py")

md("# Phần A — Harris và hai biến thể")
md("## A.1 Harris cổ điển (Người 1)")
add_module("n1_harris.py")
md("## A.2 Biến thể A1 — Harris–Laplace (Người 2)")
add_module("n2_harris_laplace.py")
md("## A.3 Biến thể A2 — Adaptive non-maximal suppression (Người 3)")
add_module("n3_anms.py")

md("## A.4 Kiểm chứng A1, A2, A4 (Người 1)")
add_script("kiem_chung_nguoi1.py")

md("## A.5 Kiểm chứng A3 và thí nghiệm A1 vs Harris (Người 2)")
add_script("kiem_chung_nguoi2.py", calls={
    "KIỂM CHỨNG A3 — Repeatability Harris vs Harris–Laplace (1→3×)": "img, img_3x, tf = run_kiem_chung_a3()",
    "5.1.1 — Trực quan hóa điểm qua các hệ số tỉ lệ": "run_evidence(img)",
    "5.1.2 — Quét tham số ξ và log_thresh": "run_param_sweep(img)",
    "5.1.3 — Cơ chế bất biến tỉ lệ": "print_mechanism()",
}, drop_from="if __name__ ==")

md("## A.6 Kiểm chứng A5 và thí nghiệm A2 vs Harris (Người 3)")
add_script("kiem_chung_nguoi3.py")

md("# Phần B — Region Growing và Seeded Region Growing")
md("## B.1 Region Growing cổ điển và công cụ đo (Người 4)")
add_module("n4_rg.py")
md("## B.2 Seeded Region Growing (Người 5)")
add_module("n5_srg.py")

md("## B.3 Kiểm chứng B1, B2, B3, B5 và thí nghiệm lazy/frozen, đồng hạng (Người 5)")
add_script("kiem_chung_nguoi5.py", calls={
    "": "test_snapshot_core()",
    "KIỂM CHỨNG B1, B2 & B3 — Phân hoạch đầy đủ, số vùng & thứ tự seed":
        "img_b, seeds_b, _ = test_partition_and_region_count()\ntest_seed_order(img_b, seeds_b)",
    "KIỂM CHỨNG B5 — Tie-breaking trên ảnh phẳng (insertion, raster, random)": "test_flat_ties()",
    "THÍ NGHIỆM B1 — Cặp xử lý phần tử cũ: Lazy vs Frozen theo độ dốc sáng": "test_lazy_frozen()",
    "THÍ NGHIỆM B2 — Ảnh hưởng của lượng tử hóa mức xám tới Tie-breaking": "test_tie_quantization()",
    "GIẢI THÍCH CƠ CHẾ — Tie-breaking, Lazy vs Frozen & Lượng tử hóa": "print(GIAI_THICH)",
}, drop_from="def main():")

md("## B.4 Kiểm chứng B4 và thí nghiệm SRG vs Region Growing (Người 4)")
add_script("kiem_chung_nguoi4.py")

md("""
# Tài liệu tham khảo

1. C. Harris and M. Stephens, "A combined corner and edge detector," in *Proc. 4th Alvey Vision Conf.*, Manchester, UK, 1988, pp. 147–151.
2. T. Lindeberg, "Feature detection with automatic scale selection," *Int. J. Comput. Vis.*, vol. 30, no. 2, pp. 79–116, 1998.
3. K. Mikolajczyk and C. Schmid, "Scale & affine invariant interest point detectors," *Int. J. Comput. Vis.*, vol. 60, no. 1, pp. 63–86, 2004. doi: 10.1023/B:VISI.0000027790.02288.f2
4. M. Brown, R. Szeliski, and S. Winder, "Multi-image matching using multi-scale oriented patches," in *Proc. IEEE CVPR*, San Diego, CA, USA, 2005, vol. 1, pp. 510–517. doi: 10.1109/CVPR.2005.235
5. R. Adams and L. Bischof, "Seeded region growing," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 16, no. 6, pp. 641–647, Jun. 1994. doi: 10.1109/34.295913
6. A. Mehnert and P. Jackway, "An improved seeded region growing algorithm," *Pattern Recognit. Lett.*, vol. 18, no. 10, pp. 1065–1071, 1997.
7. J. Shi and C. Tomasi, "Good features to track," in *Proc. IEEE CVPR*, Seattle, WA, USA, 1994, pp. 593–600.
8. K. Mikolajczyk, T. Tuytelaars, C. Schmid, A. Zisserman, J. Matas, F. Schaffalitzky, T. Kadir, and L. Van Gool, "A comparison of affine region detectors," *Int. J. Comput. Vis.*, vol. 65, no. 1/2, pp. 43–72, 2005. Bộ ảnh Oxford Affine Covariant Regions dùng trong bài: http://www.robots.ox.ac.uk/~vgg/data/data-aff.html
""")

# ------------------------------------------------------------------ ghi file
for c in cells:
    src = c["source"]
    lines = src.split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for k, c in enumerate(nb["cells"]):
    c["id"] = f"c{k:03d}"
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Đã ghi {OUT} — {len(cells)} cell ({sum(c['cell_type'] == 'code' for c in cells)} code)")
