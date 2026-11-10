# BT1 — Đặc trưng góc & Phân vùng ảnh

Môn Thị giác máy tính (121036), ĐH Giao thông Vận tải TP.HCM. Bài nộp cuối cùng là một notebook `bt1_<Nhom>.ipynb` kèm thư mục `anh/`. Trong lúc làm, mỗi người viết code trong file `.py` riêng ở thư mục này. Người 1 ghép tất cả thành notebook ở bước cuối.

Quy ước chung của nhóm (kiểu ảnh, tọa độ, định dạng đầu ra, cách đặt tên) nằm trong file `Quy_uoc_chung_BT_Harris_SRG.docx`. Hãy đọc file này trước khi viết code.

## Cấu trúc

| File | Người | Trạng thái |
| --- | --- | --- |
| `common.py` | 1 | Xong |
| `n1_harris.py` | 1 | Xong |
| `kiem_chung_nguoi1.py` | 1 | Xong (A1, A2, A4) |
| `n2_harris_laplace.py` | 2 | Chưa có |
| `n3_anms.py` | 3 | Chưa có |
| `n4_rg.py` | 4 | Chưa có |
| `n5_srg.py` | 5 | Chưa có |
| `anh/` | cả nhóm | Ảnh dùng trong bài |
| `ket_qua/` | cả nhóm | Hình xuất ra từ các script kiểm chứng |

## Cài đặt và chạy

```
pip install -r requirements.txt
python kiem_chung_nguoi1.py
```

## Dùng hàm của Người 1

```python
from common import load_gray, make_scale, make_rotation, show_points, S_RATIO, NB4, NB8
from n1_harris import gaussian_blur, harris_response, nms_2d, harris_classic, repeatability

img = load_gray("anh/ten_anh.png")
pts = harris_classic(img)                      # (N, 3): row, col, response
img2, tf = make_scale(img, 3)                  # tf(row, col) ảnh cũ -> ảnh mới
rep = repeatability(pts, harris_classic(img2), tf, img2.shape)
```

## Kết quả kiểm chứng của Người 1

| Kiểm chứng | Kết quả |
| --- | --- |
| A1. Bàn cờ 8 × 8 ô, mỗi ô 32 px | Tìm đủ 49/49 giao điểm trong, 0 điểm thừa, sai số lớn nhất 0,707 px |
| A2. Quay 30° (ảnh tổng hợp 4 hình) | 15 điểm ở cả hai ảnh, repeatability = 1,000 |
| A4. So với `cv2.cornerHarris` (blockSize 5, ksize 3, k 0,04) | 100% điểm của mình nằm trong 2 px của một điểm OpenCV, trên cả 3 ảnh thử |

Ảnh thử ở A2 là ảnh tổng hợp, sạch, không nhiễu, nên repeatability bằng 1 là bình thường. Ảnh thật sẽ thấp hơn. Phần thí nghiệm nên dùng thêm ảnh thật.

## Những điểm đã quyết khi cài đặt

- **`nms_2d` có phá hòa theo thứ tự quét.** Ở ảnh đối xứng như bàn cờ, đỉnh R nằm giữa 4 pixel có giá trị bằng nhau tuyệt đối. Nếu chỉ dùng "R ≥ max của cửa sổ" thì mỗi giao điểm cho 4 điểm. Cách làm: so lớn hơn hẳn với các pixel đứng trước trong cửa sổ, và lớn hơn hoặc bằng với các pixel đứng sau. File quy ước chung đã cập nhật theo điểm này.
- **`make_rotation` tự viết bằng numpy** (ánh xạ ngược, nội suy song tuyến) vì `cv2.warpAffine` không có trong danh sách thư viện được phép. Phần ảnh nằm ngoài được tô bằng trung vị của ảnh để không tạo góc giả ở mép.
- **`grid_occupancy` trong `common.py` là bản tạm** (lưới 8 × 8). Khi thầy gửi hàm chính thức thì thay vào.
- **`cv2.cornerHarris` chỉ xuất hiện trong `kiem_chung_nguoi1.py`**, ở mục đối chiếu A4, không nằm trong hàm phải nộp.

## Khai báo AI

Code của Người 1 trong thư mục này được viết với sự hỗ trợ của Claude (Anthropic). Theo quy định của đề, cell nào dùng AI phải có cell markdown phía trên chép nguyên văn prompt. Khi ghép notebook, đặt cell sau phía trên các cell code của Người 1:

```
**Prompt AI** — Người 1 · công cụ: Claude · ngày 10/11/2026

> làm của t trước đi, người 1, trên một thư mục để t đưa lên git
```

Đề chưa nói rõ phần cài đặt thuật toán có được dùng AI hay không (dòng quy định trong file đề bị mất chữ). Cần hỏi lại thầy trước khi nộp.
