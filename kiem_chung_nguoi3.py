# Thiết lập thư mục lưu trữ
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ket_qua")
ANH_DIR = os.path.join(HERE, "anh")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ANH_DIR, exist_ok=True)


def chay_kiem_chung_va_thich_nghi():
    print("=" * 60)
    print("KIỂM CHỨNG A5 & THÍ NGHIỆM A2 (ANMS vs Top-N)")

    # 1. Tạo hoặc tải ảnh thử nghiệm có vùng nhiều chi tiết và vùng trơn
    img_path = os.path.join(ANH_DIR, "a0_canh_tong_hop.png")
    if not os.path.exists(img_path):
        # Nếu chưa có ảnh từ Người 1, tạo ảnh tổng hợp mẫu
        from kiem_chung_nguoi1 import tao_canh_tong_hop, luu_anh
        img = tao_canh_tong_hop()
        luu_anh(img, "a0_canh_tong_hop.png")
    else:
        img = load_gray(img_path)

    # 2. Trích xuất ứng viên Harris với ngưỡng thấp (thresh_rel=0.001) để thu thập vài nghìn điểm
    # Lưu ý: Cả top_n và anms bắt buộc phải dùng chung một tập ứng viên này để đảm bảo công bằng.
    candidates = harris_classic(img, thresh_rel=0.001)
    print(f"Tổng số ứng viên Harris trích xuất (thresh_rel=0.001): {len(candidates)}")

    # ------------------
    # A5. Kiểm chứng phân bố điểm (Grid Occupancy)
    # ------------------
    n_list = [100, 300]
    print("\n--- BẢNG KIỂM CHỨNG ĐỘ PHỦ LƯỚI (A5) ---")
    print(f"{'N':<10} | {'Độ phủ Top-N':<15} | {'Độ phủ ANMS':<15}")
    print("-" * 46)

    fig, axes = plt.subplots(len(n_list), 2, figsize=(10, 4 * len(n_list)))

    for idx, n_val in enumerate(n_list):
        pts_topn = top_n(candidates, n_val)
        pts_anms = anms(candidates, n_val, c_robust=0.9)

        cov_topn = grid_occupancy(pts_topn, img.shape, n_cells=8)
        cov_anms = grid_occupancy(pts_anms, img.shape, n_cells=8)

        print(f"{n_val:<10} | {cov_topn * 100:.2f}%          | {cov_anms * 100:.2f}%")

        # Vẽ hình minh họa so sánh
        ax_top = axes[idx, 0] if len(n_list) > 1 else axes[0]
        ax_ans = axes[idx, 1] if len(n_list) > 1 else axes[1]

        show_points(ax_top, img, pts_topn, title=f"Top-N (N={n_val}, Cov={cov_topn*100:.1f}%)", color="r")
        show_points(ax_ans, img, pts_anms, title=f"ANMS (N={n_val}, Cov={cov_anms*100:.1f}%)", color="b")

    plt.tight_layout()
    fig_path = os.path.join(ANH_DIR, "a2_comparison_a5.png")
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Đã lưu ảnh kiểm chứng A5 tại: {fig_path}")

    # -------------------
    # Thí nghiệm A2: Quét tham số c_robust và đánh giá Repeatability dưới phép quay
    # -------------------
    print("\n--- THÍ NGHIỆM A2: QUÉT THAM SỐ C_ROBUST ---")
    quay, tf = make_rotation(img, 30)
    candidates_quay = harris_classic(quay, thresh_rel=0.001)

    c_vals = [0.5, 0.7, 0.9, 0.95, 0.99]
    n_eval = 200
    print(f"{'c_robust':<10} | {'Độ phủ ANMS (Ảnh gốc)':<22} | {'Repeatability (N=200)':<20}")
    print("-" * 58)

    for c in c_vals:
        pts_a = anms(candidates, n_eval, c_robust=c)
        pts_b = anms(candidates_quay, n_eval, c_robust=c)
        
        cov = grid_occupancy(pts_a, img.shape, n_cells=8)
        rep = repeatability(pts_a, pts_b, tf, quay.shape, eps=3.0)
        print(f"{c:<10.2f} | {cov * 100:.2f}%                | {rep:.3f}")

    # Lưu hình ảnh minh họa kết quả ANMS trên ảnh quay
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    pa_anms = anms(candidates, 200, c_robust=0.9)
    pb_anms = anms(candidates_quay, 200, c_robust=0.9)
    rep_final = repeatability(pa_anms, pb_anms, tf, quay.shape, eps=3.0)

    show_points(axs[0], img, pa_anms, f"ANMS Gốc (N=200)")
    r_proj, c_proj = tf(pa_anms[:, 0], pa_anms[:, 1])
    show_points(axs[1], quay, pb_anms, f"ANMS Quay 30° (Rep = {rep_final:.3f})")
    axs[1].plot(c_proj, r_proj, "o", mfc="none", mec="lime", ms=8, mew=1)
    
    fig.suptitle("Thí nghiệm A2 — ANMS: đỏ = tìm được, xanh = chiếu từ ảnh gốc sang")
    fig.tight_layout()
    fig_exp = os.path.join(OUT, "a2_anms_quay_30.png")
    plt.savefig(fig_exp, dpi=120)
    plt.close(fig)
    print(f"Đã lưu kết quả thí nghiệm A2 tại: {fig_exp}")


if __name__ == "__main__":
    chay_kiem_chung_va_thich_nghi()
