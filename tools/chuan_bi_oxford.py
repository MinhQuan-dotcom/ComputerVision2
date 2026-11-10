"""Chuẩn bị bộ ảnh Oxford Affine cho bài tập (đã chạy sẵn, kết quả nằm trong anh/oxford/).

Chạy lại khi cần:  python tools/chuan_bi_oxford.py <thư mục đã giải nén> anh/oxford
<thư mục đã giải nén> chứa các thư mục con bark/, boat/, ... (mỗi thư mục có img1..img6 và H1to2p..H1to6p).

Việc làm: đổi ảnh sang xám, thu nhỏ để cạnh dài <= 512 px (INTER_AREA), lưu PNG,
và tính lại homography cho ảnh đã thu nhỏ: H' = S_k · H · S_1^-1,
với S là ma trận đổi tỉ lệ theo quy ước tâm pixel của OpenCV.
"""
import sys, os, glob
import cv2, numpy as np
src, dst = sys.argv[1], sys.argv[2]
MAX = 512
def S(s):
    o = 0.5*s - 0.5
    return np.array([[s,0,o],[0,s,o],[0,0,1.0]])
for d in sorted(os.listdir(src)):
    files = sorted(glob.glob(os.path.join(src, d, 'img*')))
    if not files: continue
    out = os.path.join(dst, d); os.makedirs(out, exist_ok=True)
    scales = []
    for k, f in enumerate(files, 1):
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3: im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        h, w = im.shape
        s = min(1.0, MAX / max(h, w))
        nh, nw = int(round(h*s)), int(round(w*s))
        sy, sx = nh/h, nw/w
        scales.append((sx, sy))
        if s < 1: im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, f'img{k}.png'), im)
    S1 = np.array([[scales[0][0],0,0.5*scales[0][0]-0.5],[0,scales[0][1],0.5*scales[0][1]-0.5],[0,0,1]])
    for k in range(2, len(files)+1):
        H = np.loadtxt(os.path.join(src, d, f'H1to{k}p'))
        sx, sy = scales[k-1]
        Sk = np.array([[sx,0,0.5*sx-0.5],[0,sy,0.5*sy-0.5],[0,0,1]])
        Hn = Sk @ H @ np.linalg.inv(S1)
        Hn /= Hn[2,2]
        np.savetxt(os.path.join(out, f'H1to{k}.txt'), Hn, fmt='%.10e')
    print(d, [ (round(a,4),round(b,4)) for a,b in scales[:2] ])
