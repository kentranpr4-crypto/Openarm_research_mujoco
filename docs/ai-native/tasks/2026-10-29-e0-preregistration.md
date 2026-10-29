# E0 — đa seed huấn luyện C0 vs S1: pre-registration
Ngày 2026-10-29 · đóng băng **trước khi** train/đánh giá bất kỳ seed mới nào · cấu hình máy đọc được: `outputs/e0/e0_config.json` (bản sao `assets/e0_config.json`), **sha256 `25f20ea4c887534ddc80c8aedee01e93da05790dac182d1b0a86f4217d545e9d`**, mã nguồn commit `89d19e1`.
Phạm vi được duyệt: chỉ E0. Không S3, không E1/E4/E5, không adopt S1/S2, không dùng test seed 31–33, không chạy thí nghiệm nào khác.

## 1. Benchmark hiệu năng (bước 1 của yêu cầu)
Máy vừa khởi động lại lần nữa (kernel fc-v185; trước đó fc-v165 rồi fc-v180). Đo ngay trên máy hiện tại, không có tiến trình thừa, không giới hạn cgroup, steal = 0:

| Phép đo | Kết quả |
|---|---|
| CPU | Xeon 2.1 GHz, 4 lõi, AVX-512; matmul fp32 478 GFLOPS (4 luồng), 120 GFLOPS (1 luồng) — phần cứng khỏe |
| Bước train, `threads=4, workers=1` (cấu hình Stage 1) | **1.28 s/bước** (bản gốc 0.93; giai đoạn chậm sau lần khởi động lại trước 2.0–2.3) |
| `threads=4, workers=2` / `workers=0` | 1.28 / 1.23 s/bước |
| **`threads=3, workers=1`** | **0.79 s/bước** (nhanh hơn 1.6×) |
| Đánh giá, 4 worker | 8 episode / 144 s (gồm ~25 s khởi động) ⇒ ~15 s/episode hiệu dụng |
| Tái lập chéo máy | đánh giá lại 8 episode của C0 seed 0 trên máy mới: **8/8 cùng kết quả, chênh khoảng cách hộp–tray 0.0 m** |

Chẩn đoán: máy hiện tại bình thường về phần cứng; với `threads=4` thì 4 luồng torch cộng 1 worker nạp dữ liệu làm quá tải 4 lõi (5 tác vụ nóng), còn `threads=3` thì vừa đủ. Nguyên nhân đúng của giai đoạn chậm 2.1× trước đó **không thể đo lại** vì nó thuộc về một phiên máy đã mất; bằng chứng còn lại: độ trễ suy luận lúc đó cũng cao 2.1× (≈108 ms so với ≈50 ms), tức tốc độ CPU hiệu dụng của máy đó thấp, không phải lỗi mã. Kết luận: runtime hiện tại **hợp lý**, có cấu hình tốt hơn bản gốc.
Hệ quả phương pháp: số luồng đổi thứ tự cộng dấu phẩy động, nên chạy `threads=3` cho quỹ đạo khác `threads=4` (chênh loss 8e-2 sau 60 bước; các cấu hình cùng số luồng cho loss giống hệt 0.0). Đây là **sai lệch D1** đã ghi trong cấu hình.

## 2. Dùng lại run Stage 1 (bước 2): được, với điều kiện
Dùng lại **seed 0** của C0 và S1 (`outputs/checkpoints/e6b_{c0,s1}_seed0`, đánh giá ở `outputs/eval/e6b/{c0,s1}`), thêm **seed 1 và 2** mới cho cả hai nhánh ⇒ 3 seed/nhánh, 4 lần train mới.
| Kiểm tra tính độc lập | Kết quả |
|---|---|
| Seed khác nhau thật (khởi tạo mô hình + thứ tự dữ liệu) | seed 0, 1, 2 khác nhau; dataset và split cố định (54/6, `split_sha256` đã ghi) |
| Seed 0 không chọn theo kết quả | seed 0 là seed của V1, chọn trước mọi kết quả; không có lần chọn lại |
| Checkpoint seed 0 nguyên vẹn | sha256 `model.safetensors` và `norm_contract.json` đã ghi vào cấu hình; báo cáo cuối sẽ kiểm lại |
| Cùng tập đánh giá | cùng dev seed 21/22, cùng id; kết quả đánh giá tái lập bit-bit trên máy mới |
| Khác biệt cấu hình | D1 (số luồng); mã nguồn giống nhau (Stage 1 chạy đúng nội dung của commit `89d19e1`); phân tích độ nhạy loại seed 0 |
| Rủi ro còn lại | Stage 1 đã định hình giả thuyết (không định hình seed); các số dev đã được xem; test seed 31–33 vẫn nguyên |
Không dùng lại V1 (stats toàn dataset) trong phân tích suy luận; V1 chỉ còn là tham chiếu lịch sử.

## 3. Thiết kế cố định (bước 3, chi tiết ở JSON)
- **Nhánh:** C0 = stats train-only, không floor; S1 = stats train-only + floor state 0.02 rad / gripper 0.002 m, action không floor.
- **Huấn luyện:** `pickplace_v1`, 5400 bước, batch 16, lr 1e-4, ResNet18 từ đầu, dim 256, 2 encoder layer, chunk 20 (exec 20), 2 camera, `workers=1`, `threads=3` cho run mới; cờ chuẩn hóa như Stage 1.
- **Đánh giá (dev seed 21–22 duy nhất):** NOM seed 21 n=30; BOUND seed 21 n=30 và seed 22 n=30; SHORTCUT trên NOM seed 21 id 0–19 với `right_mean` và `right_jitter_small` (đúng đơn vị). Bỏ so với Stage 1: nhiễu lớn và hai phép thử độc lập (đã có kết quả cho seed 0 nhưng không lặp lại). 130 episode/checkpoint.
- **Metric:** thành công nhiệm vụ/pick, shortcut index (SI), độ cao TCP lúc đóng, jerk (tổng/interior/biên), tracking, độ trễ mỗi truy vấn, lệch lệnh tay phải, phân loại lỗi, động học vận chuyển (đỉnh tốc độ/gia tốc TCP), chi phí train, val L1 và MAE vật lý.

## 4. Kế hoạch phân tích (bước 5)
- **Đơn vị lặp lại = seed huấn luyện.** Mỗi (nhánh, seed, tập đánh giá) cho một tỉ lệ thành công.
- **Báo cáo A — biến thiên giữa các seed train:** trung bình nhánh, SD mẫu (n=3), khoảng min–max; từng kịch bản riêng.
- **Báo cáo B — biến thiên giữa các kịch bản đánh giá:** bảng riêng NOM / BOUND21 / BOUND22 / từng chế độ shortcut, ma trận kịch bản × seed, và tỉ số SD giữa các seed so với sai số nhị thức trong một seed (Wilson 95% cho từng seed).
- **Suy luận (cố định):** Welch t trên tỉ lệ theo seed (df Welch–Satterthwaite); kiểm định hoán vị chính xác trên nhãn seed (p hai phía nhỏ nhất 0.1 cho 3 so 3) là phụ; hierarchical bootstrap (lấy mẫu seed rồi episode) chỉ để kiểm độ nhạy. Tất cả lặp lại khi loại seed 0.
- **Luật quyết định (không sửa sau khi có kết quả):**
  - **R1** — hỗ trợ "độ bền shortcut nhân bản" nếu S1 có SI ≥ 0.90 ở **cả** `right_mean` và `jitter_small` trong ≥2/3 seed **và** C0 có SI ≤ 0.25 ở cả 3 seed.
  - **R2** — chi phí năng lực: hiệu số = trung bình(S1) − trung bình(C0) của tỉ lệ theo seed. Không kém (non-inferior) nếu cận dưới CI 95% Welch ≥ −10 điểm (NOM) và ≥ −15 điểm (BOUND, gộp seed 21+22); "có chi phí" nếu cận trên < 0; còn lại là "chưa kết luận".
  - **R3** — khuyến nghị: S1 chỉ được **đề xuất để nghiên cứu tiếp** làm ứng viên baseline V2 nếu R1 hỗ trợ và R2 không kém ở cả NOM và BOUND. R1 hỗ trợ mà R2 chưa kết luận ⇒ "cần thêm seed" kèm số seed cần. R2 có chi phí ⇒ khuyến nghị không dùng floor này nguyên trạng. **Không tự chọn policy.**
- **Lưu ý công suất (nêu trước):** 3 seed/nhánh với SD giữa các seed giả định ~8 điểm cho nửa độ rộng CI của hiệu số ≈ 18 điểm, nên chỉ phân giải được hiệu ứng lớn. "Chưa kết luận" là kết quả có khả năng cao và chấp nhận được; mở rộng thêm seed cần duyệt riêng.
- **Phân tích lỗi:** cùng phân loại như Stage 1 (`audit_v1.classify`), kèm động học vận chuyển và độ cao đóng, theo từng seed; kiểm tra xem kiểu lỗi của S1 ở Stage 1 (rơi khi vận chuyển với động học bình thường) có lặp lại không.

## 5. Ngân sách và điều kiện dừng (bước 4–5)
Đo được: train 0.79 s/bước ⇒ **71 phút/run**; đánh giá ~15 s/episode ⇒ **~34 phút/checkpoint** (gồm khởi động). Ước tính: train 4 × 71 phút ≈ 4.7 giờ + đánh giá 4 × 34 phút ≈ 2.3 giờ ⇒ **≈ 7.0 giờ**, trong mức ≈ 8 giờ của kế hoạch gốc cho E0/giai đoạn 2. Thứ tự chạy: C0 seed 1 → S1 seed 1 → C0 seed 2 → S1 seed 2, đánh giá ngay sau mỗi lần train (có dữ liệu cặp nếu phải dừng).
**Dừng và báo cáo** nếu: tốc độ trung bình 200 bước đầu của một run mới > 1.10 s/bước; ước tính tổng > 8.0 giờ; lệnh đánh giá có seed 31–33; run lỗi (chỉ chạy lại một lần, đúng đối số cũ). Không đổi tham số/ngưỡng/tập đánh giá sau khi đóng băng.

## 6. Sản phẩm
Báo cáo riêng cho (A) biến thiên giữa seed train và (B) biến thiên giữa kịch bản đánh giá, bảng đa seed kèm CI, biểu đồ, phân tích lỗi, chi phí tính toán, khuyến nghị C0 so với S1 (không chọn policy cuối nếu bằng chứng chưa đủ). File/script: `eval/e0_run.sh` (hàng đợi có thể tiếp tục), `eval/e0_report.py`.
