# SuperMimic-OpenArm: đóng P0 và sẵn sàng P1 (Block Pushing) — báo cáo

Ngày: 2026-11-11 · Nhánh `main-f9g0uh` (HEAD `fcabee9`) · Trạng thái: **PARTIAL** (tự kiểm; chưa có kiểm độc lập)
> **CẬP NHẬT**: các mục môi trường P1, expert và World Model trong báo cáo này đã được sửa chữa và đo lại; xem `2026-11-11-supermimic-p1-env-repair-report.md` (kết quả mới thay thế bảng ở mục 3, 4, 6, 7). Mục P0 bên dưới đã được **hiệu chỉnh**: có bằng chứng đánh giá trực quan trên checkpoint thật (xem mục 2).

Ràng buộc giữ nguyên: không retrain P0; không đụng G2a/G2b, sealed poses, OpenArm-WAM; không ghi đè checkpoint/dataset; không commit/push; dừng chờ duyệt trước khi train P1 nhiều giờ.

> Nhãn bằng chứng: **[đã đo]** = chạy trong phiên này, có file kết quả; **[từ mã]** = đọc code, chưa chạy; **[chưa kiểm]** = cần máy của chủ việc.
> Môi trường phiên này: cloud sandbox 4 CPU, **không có GPU**, đĩa trống 3,4 GB. Mọi số đo GPU/VRAM/throughput GPU là **chưa kiểm**.

## 1. Kết luận ngắn

| Hạng mục | Kết luận |
|---|---|
| P0 đóng khoa học? | **Chưa đóng hoàn toàn, nhưng có bằng chứng thật.** Đã có **30 episode đánh giá trực quan trên `best.pt` thật** (3 seed × 10 goal held-out; cả 30 thành công, mỗi seed 10/10). Đánh giá **độc lập 100 goal/seed** (CI, jerk, best vs latest) có công cụ đã test nhưng **chưa chạy trên checkpoint thật**. 10/10 trên 10 goal cho Wilson 95% dưới ≈ 0,72 mỗi seed, nên chưa đủ cho kết luận chặt về độ tin cậy. |
| Môi trường P1 | **Có 2 lỗi thật** (ma sát, khối lượng) + 1 điểm yếu (≈2% goal nằm trong khay) → cần quyết định của chủ việc trước khi train. |
| Pushing expert | Đặt được hộp (≤3 cm, đứng yên) **41/48 = 85%**, nhưng đạt tiêu chí chặt (hộp không nhấc >1 cm) chỉ **20/48 = 42%**. Dưới ngưỡng đăng ký trước (80%). |
| World Model (proxy, dữ liệu nhỏ) | Căn chỉnh hành động đúng; **tốt hơn baseline đứng yên khi chạm lần đầu** (tỉ lệ 0,61–0,72), **kém hơn** khi tự do (1,20–1,42); **vùng chạm kéo dài chưa đo được** (n=0). Có dấu hiệu khai thác mô hình (xem 4). |
| **GO / NO-GO pilot P1** | **NO-GO ở trạng thái hiện tại.** Chuyển GO khi đạt 4 điều kiện ở mục 7. |

## 2. Phase A — P0 (kiểm toán)

- **A.1 viewer / đánh giá trực quan**: 30 episode trực quan trên `best.pt` thật đã chạy (bằng `visualize`) **[đã đo, trên máy bạn]**. `view_live` (cửa sổ MuJoCo trực tiếp) đã test trong Xvfb (thoát mã 0, đóng giữa episode, loop) **[đã đo]**; việc chạy `view_live` với `best.pt` thật **[chưa kiểm]**.
- **A.2–A.4**: `p0_final_eval.py` chạy end-to-end trên checkpoint debug (3 goal; kết quả FAIL đúng kỳ vọng vì chưa huấn luyện) **[đã đo]**. Chạy thật **[chưa kiểm]**.
- **A.5 kiểm toán từ mã [từ mã]**: reward/loss/kiến trúc/siêu tham số đã ghi trong kế hoạch `2026-11-11-supermimic-p0-independent-eval-plan.md`. Phát hiện giữ nguyên:
  1. Ba run dùng chung `train_seed_base` → cùng chuỗi goal huấn luyện, **không phải bản sao độc lập**.
  2. `best.pt` chọn theo return trên 32 goal; return không phân biệt "dừng ổn định" với "rung lắc".
  3. Chỉ đánh giá trong phân phối; chuyển động bám sát giới hạn slew (bang-bang).
- **Lệnh chạy trên máy bạn** (thư mục mới, không ghi đè):
  `python -m research.supermimic_openarm.p0_final_eval --run-root <ROOT> --seeds 1000 1001 1002 --episodes 100 --seed-base 9900000 --out outputs/p0_final_eval_<ngày>`
  Ngưỡng đăng ký trước: mỗi `best.pt` ≥95/100, gộp ≥0,97, sai số cuối trung vị ≤5 mm, ≥95% episode dwell ≥50 bước, ở lại ≥0,95.

## 3. Phase B — kiểm định môi trường P1 (`outputs/p1_validation/env_validation_v2.json`, 8 PASS / 2 FAIL)

| Mã | Kiểm | Kết quả | Evidence |
|---|---|---|---|
| V1a | Ngẫu nhiên hoá nằm trong đặc tả (300 reset) | PASS | 0 vi phạm; KS đều p=0,33/0,12/0,77 |
| V1b | Goal nằm trong khay | PASS (có cảnh báo) | 2% (6/300) goal cần vượt gờ 12 mm; ngưỡng 2% |
| V2 | Ngẫu nhiên hoá khối lượng | **FAIL** | khối lượng cố định 0,05 kg; chỉ ma sát được ngẫu nhiên |
| V3a | Hộp nằm yên giữ yên | PASS | drift 2,4 µm, tốc độ 1,5e-5 m/s |
| V3b | Ma sát đổi vật lý trên toàn dải | **FAIL** | scale 0,7/0,85/1,0 cho cùng quãng trượt 0,02342 m |
| V3c | Xuyên thấu hộp–robot nhỏ | PASS | ngón tay −1,1 mm (ngưỡng −2 mm); desk −4,5 mm và tray −3,2 mm là tiếp xúc nghỉ, báo riêng |
| V4 | Ngữ nghĩa hành động (clip→slew) | PASS | sai khác với tham chiếu độc lập 1,1e-7; 0 vi phạm; tay phải giữ nguyên |
| V5 | Tiêu chí thành công tính lại = cờ simulator | PASS | 0/48 lệch |
| V6 | Reset tất định + replay snapshot | PASS | sai khác replay 0,0 |
| V7 | Pushing expert, phân phối phát triển | PASS ở mức "đặt được" (85% ≥ 80%) | strict 20/48 (42%) |

**Nguyên nhân V3b [đã đo + từ mã]**: MuJoCo lấy hệ số ma sát **lớn hơn** của hai geom (bàn 0,8 so với hộp 0,8×scale), nên scale ≤1,0 không có tác dụng; chỉ >1,0 mới đổi (1,15 → 0,02071 m; 1,3 → 0,01983 m). Vậy khoảng 60% dải ngẫu nhiên hoá hiện **không làm gì**.

**Nguyên nhân thất bại expert (28 ca)**: hộp bị nhấc >1 cm 19, không chạm 4, dừng thiếu 4, chưa nghỉ 1. Tiêu chí chặt (≤1 cm) giữ nguyên, **không nới**.

## 4. Phase C — chẩn đoán World Model (`wm_diagnostics_v2.json`)

Giới hạn: dữ liệu proxy (expert/nhiễu/ngẫu nhiên), 9000 chuyển tiếp train, 20 episode val tách rời, 1500 cập nhật WM, CPU. Đây là **cận dưới** về độ chín của mô hình, không phải WM của lần train thật.

| Kiểm | Kết quả |
|---|---|
| Căn chỉnh hành động thực thi | PASS: `prev_ctrl[t+1]` = hành động thực thi t, sai lệch tối đa 0,0 |
| Sai số vị trí hộp, WM (mm) / baseline đứng yên (mm), horizon 1·5·10·16 | Tự do: 1,4/1,0 · 5,0/3,9 · 8,2/6,6 · 10,7/8,9 → **WM kém hơn** (tỉ lệ 1,2–1,4). Chạm lần đầu: 4,6/7,5 · 17,9/25,7 · 26,6/39,2 · 33,0/46,1 → **WM tốt hơn** (0,61–0,72) |
| Vùng chạm kéo dài | **NOT_CHECKED** (n=0: dữ liệu proxy chưa có mẫu) |
| Baseline vận tốc không đổi | Tệ hơn WM khi chạm (62–82 mm); lợi thế khi tự do ở h1 |
| Gradient actor qua động lực | Hữu hạn, khác 0 (norm 0,056; BPTT 0,011) |
| Độ nhạy hộp theo hành động | Khác 0 trên cả 8 chiều (~2e-4–5e-4), nhưng nhỏ |
| Sai số reward tưởng tượng | MAE 0,79/bước so với thang 1,09; tương quan 0,58 → **kém** |
| Khai thác mô hình | Chênh (tưởng tượng − thực) trung bình **+7,4** (trung vị +4,4); return thực của chuỗi tối ưu 22,9 so với expert 25,7; 42% trạng thái vượt biên độ expert |

Diễn giải [suy luận, chưa kiểm bằng thực nghiệm khác]: mô hình đoán hộp "trôi" khi lẽ ra đứng yên (rò rỉ vào vùng tự do), và tối ưu hoá thắng bằng cách khai thác sai số đó. Hướng sửa ứng viên: nhiều dữ liệu chạm kéo dài hơn, trọng số loss riêng cho hộp, và đánh giá lại sau khi WM được fit trên dữ liệu thật của pilot. Đây là đề xuất, chưa thử.

## 5. Phase D — chuẩn bị huấn luyện có giới hạn

- Cấu hình `configs/pushing_p1_pilot.yaml`: 1 seed, 24 env CPU, WM/actor/critic cùng kích thước với P0, 8000 iteration (ước lượng thô, **chưa đo** nên chưa biết có vừa 2 giờ không), eval mỗi 400, checkpoint mỗi 200, `train_seed_base` 1.100.000 (khác P0), `max_vram_gb` 20.
- `scripts/p1_pilot.sh`: **mặc định dry-run**, `--execute` mới chạy; trần cứng 2 giờ (`timeout`), `nice`, ghim 1 GPU, thư mục output riêng, `--resume`. Preflight chỉ đọc: **từ chối** nếu GPU có tiến trình khác (bảo vệ OpenArm-WAM), GPU bận >30%, <8 GiB trống, thiếu lõi CPU, <10 GB đĩa. Không kill/chỉnh tiến trình khác.
  Test trong sandbox: trả `RESOURCE_BLOCKED (exit 3)` đúng thiết kế (không GPU, 3 lõi, 3,4 GB) **[đã đo]**. Lỗi `PY` chưa định nghĩa đã sửa.
- `scripts/p1_benchmark.sh --device cuda:N`: đo VRAM đỉnh, throughput env, thời gian update và tải GPU (lấy mẫu 1 Hz). **Chưa chạy** (cần GPU). GPU thực tế (RTX 5090 theo máy bạn) và các số đo **[chưa kiểm]**.
- Ước tính tài nguyên dựa trên đo: **chưa có**. Không đưa số ước lượng thay thế.

**Lệnh khi được duyệt (dry-run trước):**
```
bash research/supermimic_openarm/scripts/p1_benchmark.sh --device cuda:0
bash research/supermimic_openarm/scripts/p1_pilot.sh --device cuda:0            # dry-run
bash research/supermimic_openarm/scripts/p1_pilot.sh --device cuda:0 --execute  # chỉ sau khi duyệt riêng
```
Chọn chỉ số GPU **không** phải GPU đang chạy WAM (WAM chạy ở GPU 1 theo lần kiểm trước; xác nhận bằng `nvidia-smi`).

## 6. Bảng AC

| AC | Nội dung | Trạng thái | Evidence |
|---|---|---|---|
| A1 | Đánh giá trực quan trên best.pt thật / viewer trực tiếp | PASS (30/30 trực quan, 3 seed) / NOT_CHECKED (`view_live` với best.pt thật) | `visualize` trên máy bạn; `view_live` mới test Xvfb |
| A2–A4 | 100 goal/seed, CI, TCP/vel/acc/jerk, best vs latest | NOT_CHECKED | công cụ đã test trên dữ liệu debug |
| A5 | Kiểm toán reward/loss/WM/siêu tham số | PASS (từ mã) | plan doc |
| A6 | Giữ nguyên checkpoint/kết quả | PASS | chỉ đọc; output mới ở `outputs/p1_validation/` |
| B | Môi trường P1 hợp lệ | **FAIL** | V2, V3b; V1b cảnh báo |
| B-expert | Expert chặt ≥ ngưỡng | **FAIL** | strict 42% |
| C | WM sẵn sàng | PARTIAL | vùng chạm kéo dài NOT_CHECKED; khai thác mô hình, reward kém |
| D | Đo GPU, benchmark | NOT_CHECKED | cần GPU |
| D-safe | Launcher dry-run + preflight | PASS (tự kiểm) | RESOURCE_BLOCKED đúng |
| E | Báo cáo | PASS | tài liệu này |

## 7. GO / NO-GO và việc cần chủ việc

**NO-GO** cho pilot P1. Chuyển sang GO khi cả bốn điều kiện sau đạt:
1. **Chạy P0 final eval thật** (mục 2) và đạt ngưỡng đăng ký trước.
2. **Quyết định sửa môi trường** (cần chủ việc, vì đổi đặc tả task): (a) ma sát: dải scale >1,0 hoặc hạ ma sát bàn để scale có hiệu lực; (b) thêm ngẫu nhiên hoá khối lượng; (c) loại goal trong khay. Sau sửa, chạy lại `p1_validate_env` và V2/V3b phải PASS.
3. **Expert**: nâng tỉ lệ strict (giảm nhấc hộp) mà không nới tiêu chí; ngưỡng chấp nhận ≥80% trên seed phát triển mới.
4. **Benchmark GPU** bằng `p1_benchmark.sh` trên máy bạn, báo VRAM/throughput/tải GPU, để chốt số iteration vừa trần 2 giờ.

Rủi ro đã biết cho pilot: khai thác mô hình (mục 4) có thể làm actor học trong WM nhưng tệ trên simulator. Khi chạy pilot cần theo dõi khoảng cách imagined vs real, không chỉ return.

## 8. Thay đổi git (chưa commit, chưa push)

File mới chưa theo dõi: `baselines/push_expert.py`, `configs/pushing_p1_pilot.yaml`, `p0_final_eval.py`, `p1_validate_env.py`, `p1_wm_diagnostics.py`, `scripts/p1_pilot.sh`, `scripts/p1_benchmark.sh`, hai file kế hoạch/aggregator P0 và báo cáo này. `research/openarm_wam/` vẫn chưa theo dõi và **không đụng tới**. Không có file đã theo dõi nào bị sửa. `outputs/` nằm ngoài git.

## 9. Quyết định đã đưa ra

- Không nới bất kỳ ngưỡng nào sau khi thấy kết quả (V4 chỉ đổi dung sai so sánh float32 1e-9 → 1e-6, không đổi giới hạn vật lý).
- Chạy pilot không bắt đầu; launcher giữ dry-run mặc định.
