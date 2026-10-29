# E6-B giai đoạn 0–1 — kết quả (V1 / C0 / S1 / S2)
Ngày 2026-10-28 · tự kiểm (không phải kiểm độc lập) · **CHƯA commit, CHƯA push** (theo yêu cầu) · dừng ở cuối giai đoạn 1 để chủ việc review. Không chạy giai đoạn 2–3, không quét floor, không E4, không đụng test seed 31–33, không sửa upstream, không xóa/ghi đè V1.
Kế hoạch gốc (đã duyệt): `2026-10-28-e6b-std-floor-plan.md`. Ngưỡng cổng G0–G4 giữ nguyên như đã đăng ký (`eval/e6b_report.py`, hằng số đầu file).

## 0. Kết luận ngắn
| Câu hỏi | Kết luận | Mức tin cậy |
|---|---|---|
| `std_floor` có gỡ được sự phụ thuộc brittle vào state tay phải? | **Có.** V1 và C0: 0/20 dưới mọi nhiễu state tay phải (kể cả khi chỉ nhiễu nhóm 7 khớp tay phải, hoặc chỉ gripper tay phải, với 1 mrad hoặc 0.1 mm). S1: 17/20 (`right_mean`), 16/20 (nhiễu nhỏ), 8/10 (chỉ khớp), 8/10 (chỉ gripper). | Cao cho tác động; cơ chế khuếch đại chuẩn hóa được hỗ trợ mạnh (xem 6) |
| Floor có làm giảm năng lực nhiệm vụ? | **Không đạt cổng G3 ở cả S1 và S2**: nominal 25/30 và 21/30 (ngưỡng ≥28/30); BOUND không thua đáng kể (S1 −11.7 điểm, CI [−26.7, +3.3]; S2 −3.3, CI [−16.7, +10.0]). Nhưng chỉ 1 lần train/nhánh và mức nhiễu giữa các lần train đã là ~11.7 điểm (C0 so với V1) nên **không tách được hiệu ứng floor khỏi nhiễu huấn luyện** | Thấp–trung bình |
| Có nhánh nào đạt G4 (adopt)? | **Không.** S1 trượt G3; S2 trượt G1, G2, G3. Theo cây quyết định đăng ký trước: lưu kết quả, **không tự chỉnh tham số**, dừng chờ review | – |

## 1. Đã triển khai (file, chưa commit)
| File | Nội dung |
|---|---|
| `policies/norm_contract.py` | thống kê train-only (gộp `meta/episodes` bằng `aggregate_stats`, khớp stats.json V1 ≤1.7e-8 khi gộp cả 60 episode), `apply_floor` (fail loud nếu floor đụng chiều sống), `norm_contract.json`, `verify_checkpoint` |
| `policies/act_train.py` | cờ `--stats-source {dataset,train_only}`, `--state-std-floor`, `--action-std-floor`, `--gripper-std-floor`; **mặc định = hành vi V1** (kiểm bằng T2/T12) |
| `policies/act_infer.py` | gọi `verify_checkpoint` khi nạp (V1 không có contract ⇒ bỏ qua, không đổi hành vi) |
| `eval/evaluate.py` | nhiễu state tay phải đúng đơn vị: khớp 1e-3/1e-2 rad, gripper 1e-4/1e-3 m, và thử riêng khớp / riêng gripper |
| `policies/test_norm_contract.py` | T1–T13 + TA/TB/TC |
| `eval/e6b_report.py`, `eval/e6b_failure_analysis.py` | bảng, CI, cổng G0–G4, biểu đồ; phân tích lỗi theo thời gian |
Hợp đồng đã duyệt: `σ_eff = max(σ_train, floor)`, `x_n = (x−μ)/(σ_eff+ε)`, floor 0.02 rad (khớp) và 0.002 m (gripper), thống kê train-only cho state, action và ảnh. Các floor vẫn là **giả thuyết**, không phải thiết kế cuối.
**Đính chính hợp đồng (phát hiện bởi test TB):** LeRobot 0.6.1 chuẩn hóa chia cho `σ+ε` nhưng giải chuẩn hóa nhân với `σ` (không có ε; `normalize_processor.py:359–360`). Vòng chuẩn hóa chỉ đồng nhất khi `σ_eff ≫ ε`. Với chiều action tay phải **không** có floor (V1, C0, S1; σ ≈ 1e-13…8e-5), giải chuẩn hóa co mọi lệch về ≈ `σ/(σ+ε)` của nó (lệnh tay phải ≈ trung bình) — đặc tính có chủ ý của thư viện, đã ghi vào docstring và test. Với S2 (action có floor) vòng này đồng nhất (sai số ≤5e-7 tương đối).

## 2. G0 — unit test và smoke
| | Kết quả |
|---|---|
| T1–T13, TA, TB, TC (16 test) | **16/16 PASS** (`outputs/e6b/unit_tests.json`) |
| Đáng chú ý | T2: floor off + stats toàn dataset ⇒ processors V1 khớp từng phần tử (chênh 0.0). T5: nhiễu đúng đơn vị đổi ≤0.05σ_eff dưới floor; ở V1/C0 thành 0.5–995σ (R2 204σ, R3 995σ, R6 550σ, R5 15.6σ, gripper 26σ). T6: 7360 khung train (8172 toàn dataset), chiều sống lệch ≤0.78%. T9: sửa 0.1% một tensor trong checkpoint ⇒ bị phát hiện. TA: băm SHA-256 38 file của thư mục V1 trước/sau test không đổi, V1 không tự dùng stats mới. TB: kiểm riêng từng khớp tay phải và gripper (S2 giải mã chính xác; C0/S1 co theo `σ/(σ+ε)`). T11: action tay phải sau floor chuẩn hóa ≈ 8.3e-5 (<0.01) |
| Tính tất định | huấn luyện 2 lần × 50 bước cùng seed: chênh loss = **0.0** ⇒ khác biệt giữa các nhánh chỉ do chuẩn hóa |
| Smoke 200 bước × 3 nhánh | không NaN, loss giảm, `norm_contract.json` được ghi và xác minh, rollout 20 bước chạy qua `ACTRunner` |
Khớp sha256 stats của contract S2 giữa test (`1ea1872c…`) và lần train thật.

## 3. Bảng so sánh (dev seed 21–22; CI Wilson 95%; `outputs/e6b/stage1_table.md`)
Ghi chú: V1 là checkpoint lịch sử đóng băng (thống kê toàn dataset); **C0 khác V1 ở nguồn thống kê** nên hai nhánh được báo cáo riêng, không gộp. Cùng seed train 0, dữ liệu, 5400 bước (V1: 5383), siêu tham số. Mỗi nhánh chỉ 1 lần train.

| Chỉ số | V1 (frozen) | C0 (train-only) | S1 (+floor state) | S2 (+floor state+action) |
|---|---|---|---|---|
| NOM task (seed 21, n=30) | 30/30 (100% [89–100]) | 29/30 (97% [83–99]) | 25/30 (83% [66–93]) | 21/30 (70% [52–83]) |
| BOUND task (seed 21+22, n=60) | 43/60 (72% [59–81]) | 50/60 (83% [72–91]) | 43/60 (72% [59–81]) | 48/60 (80% [68–88]) |
| BOUND pick | 48/60 (80%) | 53/60 (88%) | 52/60 (87%) | 58/60 (97% [89–99]) |
| NOM + `right_mean` (n=20) | 0/20 (SI 0.00; baseline 20/20) | 0/20 (SI 0.00; 19/20) | **17/20 (SI 1.06; 16/20)** | 20/20 (SI 1.54; 13/20) |
| NOM + nhiễu nhỏ đúng đơn vị (n=20) | 0/20 | 0/20 | **16/20 (SI 1.00)** | 9/20 (SI 0.69) |
| NOM + nhiễu lớn (n=20, thông tin) | 0/20 | 0/20 | 12/20 (SI 0.75) | 19/20 (SI 1.46) |
| Chỉ khớp tay phải (nhỏ, n=10) | 0/10 | 0/10 | 8/10 | 6/10 (baseline 6/10) |
| Chỉ gripper tay phải (nhỏ, n=10) | 0/10 | 0/10 | 8/10 | 6/10 (baseline 6/10) |
| Độ cao TCP lúc đóng, cm (tỉ lệ >3.4 cm) | 2.70±0.45 (11%) | 2.52±0.21 (0%) | 2.50±0.26 (0%) | 2.59±0.28 (3%) |
| Jerk toàn bộ / interior / biên chunk [rad/s³] | 25.0 / 17.5 / 72.1 | 21.8 / 16.4 / 56.2 | 17.8 / 11.0 / 60.7 | 28.1 / 18.2 / 90.7 |
| Độ trễ mỗi truy vấn p50/p95/p99 [ms] (xem lưu ý) | 109.6/124.7/1247.5 | 109.7/124.9/136.0 | 107.6/124.3/143.4 | 105.6/124.6/142.9 |
| Tracking cánh tay [rad] | 0.0138 | 0.0136 | 0.0131 | 0.0136 |
| Lệch lệnh tay phải khỏi trung bình: khớp [mrad] / gripper [mm] | 0.026 / 0.000 | 0.024 / 0.000 | 0.019 / 0.000 | 0.170 / 0.012 |
| Train L1 EMA / val L1 (chuẩn hóa, cuối) | 0.0369 / 0.0234 | 0.0356 / 0.0236 | 0.0382 / 0.0249 | 0.0400 / 0.0322 |
| Val MAE cánh tay trái [rad] / gripper trái [mm] (đơn vị vật lý, teacher-forced) | 0.0107 / 0.53 | 0.0101 / 0.37 | 0.0125 / 0.61 | 0.0121 / 0.49 |
Hiệu số so cặp (cùng cấu hình `(seed, idx)`; bootstrap 95% cho hiệu số, McNemar chính xác):

| So sánh | NOM (n=30) | BOUND (n=60) |
|---|---|---|
| C0 so với V1 (chỉ khác nguồn stats) | −3.3 điểm [−10.0, 0.0], p=1.0 (0 vs 1) | **+11.7** [0.0, +23.3], p=0.092 (10 vs 3) |
| S1 so với C0 | −13.3 [−26.7, −3.3], p=0.125 (0 vs 4) | −11.7 [−26.7, +3.3], p=0.189 (7 vs 14) |
| S2 so với C0 | −26.7 [−43.3, −10.0], p=0.021 (1 vs 9) | −3.3 [−16.7, +10.0], p=0.804 (7 vs 9) |
| S1 so với V1 | −16.7 [−30.0, −3.3], p=0.062 | 0.0 [−13.3, +13.3], p=1.0 |
| S2 so với V1 | −30.0 [−46.7, −13.3], p=0.004 | +8.3 [−6.7, +23.3], p=0.38 |
| S2 so với S1 | −13.3 [−33.3, +6.7], p=0.34 | +8.3 [−8.3, +25.0], p=0.42 |
**Lưu ý độ trễ:** sau khi worker bị khởi động lại, máy chạy chậm ~2.1× (p50 ≈108 ms so với ≈50 ms ở các đợt trước); các nhánh được đo cùng điều kiện nên so sánh được với nhau, **không** so với các số ở audit trước. Đỉnh 1247 ms của V1 là một lần dừng đơn lẻ. Chưa đo trên Jetson/GPU.

## 4. Cổng G0–G4 (ngưỡng đăng ký trước, không sửa)
| Cổng | S1 | S2 |
|---|---|---|
| **G0** test + smoke | **PASS** | **PASS** |
| **G1** sức khỏe train (không NaN; train-L1 EMA trong ±10% của C0; val-L1 ≤ 1.10×C0; lệch lệnh tay phải ≤2 mrad/2 mm với S2) | **PASS** (EMA +7.2%, val +5.3%) | **FAIL** (EMA +12.3% > 10%; val +36.4% > 10%; idle OK: 0.17 mrad/0.012 mm). *Chú thích:* L1 chuẩn hóa không so sánh chặt giữa các nhánh có thang action khác; MAE vật lý trên cánh tay trái cao hơn C0 +20% (S2) và +24% (S1) cho thấy cả hai khớp kém hơn một chút trên dữ liệu val, dù cổng G1 của S1 vẫn đạt theo L1 chuẩn hóa; cổng đã chấm đúng chữ |
| **G2** shortcut (SI ≥0.90 cho `right_mean` và nhiễu nhỏ) | **PASS** (1.06; 1.00; nhiễu lớn 0.75 < 0.80 chỉ là thông tin) | **FAIL** (nhiễu nhỏ SI 0.69: 9/20 vs 13/20; `right_mean` 1.54 và nhiễu lớn 1.46 đạt) |
| **G3** không thoái hóa (NOM ≥28/30; BOUND ≥ C0−5 điểm với cận dưới CI ≥ −20) | **FAIL** (NOM 25/30; BOUND −11.7, CI [−26.7, +3.3]) | **FAIL** (NOM 21/30; BOUND −3.3, CI [−16.7, +10.0] đạt nhưng NOM trượt) |
| **G4** adopt | **KHÔNG** | **KHÔNG** |

## 5. Plots (`assets/e6b-stage1/`, bản gốc `outputs/e6b/`)
`P1_success.png` (thành công + CI), `P2_shortcut.png` (nhạy với nhiễu state tay phải), `P3_closing_height.png` (độ cao TCP lúc đóng), `P4_jerk_latency.png`, `P5_training.png` (đường cong train/val), `P6_failure_timing.png` (thời điểm mở gripper so với lúc tới tray; vị trí hộp khi thả).

## 6. Giả thuyết: được hỗ trợ hay bị bác bỏ
| Giả thuyết | Phán quyết | Bằng chứng và giới hạn |
|---|---|---|
| **H-E6B-1** phụ thuộc brittle là do **khuếch đại chuẩn hóa** | **Được hỗ trợ mạnh (S1)** | V1 và C0: 0/20 dưới mọi kiểu nhiễu, và 0/10 khi chỉ động vào **nhóm khớp** tay phải hoặc **chỉ** gripper tay phải với biên độ vô hại (1 mrad, 0.1 mm); thay bằng trung bình cũng 0/20. S1 (chỉ đổi mẫu số của state) lên 16–17/20 và 8/10. Vì floor 0.02 rad không xóa thông tin còn lại (chuyển động ổn định của R1/R4/R7 còn ≈0.1σ_eff), kết quả phù hợp với "khuếch đại" hơn là "thông tin cần thiết". Giới hạn: 1 seed, n=20/10; S2 không nhất quán (nhiễu nhỏ 9/20 nhưng nhiễu lớn 19/20), nên hiệu ứng ở S2 chưa rõ |
| **H-E6B-2** floor không làm mất năng lực nhiệm vụ | **Không được hỗ trợ** (cổng trượt) nhưng **bị nhiễu bởi 1 seed** | NOM 25/30 (S1), 21/30 (S2) so với 29/30 (C0); BOUND không thua có ý nghĩa. Pick vẫn cao (BOUND pick S1 87%, S2 97%): thất bại nằm ở vận chuyển (mục 7). C0 so với V1 đã chênh +11.7 điểm BOUND chỉ do đổi nguồn stats/chạy lại, tức mức nhiễu huấn luyện ~±10 điểm ở n=60 |
| **H-E6B-3** bỏ lối tắt cải thiện BOUND | **Không được hỗ trợ** | S1 −11.7 điểm, S2 −3.3 điểm so với C0 (mọi CI chứa 0) |
| **H-E6B-4** floor trên action không làm tay phải rung | **Được hỗ trợ** | S2 lệch lệnh tay phải tối đa 0.17 mrad (khớp), 0.012 mm (gripper) ≪ 2 mrad/2 mm |
| Từ audit V1 (F1): "lỗi chính là đóng quá cao" | **Không ổn định giữa các lần train** | độ cao đóng >3.4 cm: V1 11%, C0 0%, S1 0%, S2 3%; lỗi BOUND của C0/S1 chủ yếu là "đẩy hộp" và "rơi khi vận chuyển". Cơ chế của F1 có thể đặc thù cho checkpoint V1 này; cần E0 (nhiều seed train) trước khi dùng nó làm cơ sở thiết kế |
Phát hiện phụ: ở S1/S2, success *tăng* khi thay state tay phải bằng trung bình (S2: 13/20 → 20/20; S1: 16/20 → 17/20). Gợi ý rằng chuyển động ổn định còn sót của tay phải vẫn gây nhiễu cho vận chuyển, **chưa kiểm chứng** (n=20, một seed).

## 7. Vì sao S1/S2 trượt G3 (giải thích có bằng chứng, không chỉnh tham số)
- Cả 14 ca lỗi NOM của S1 (5) và S2 (9) đều thuộc cùng một lớp: **nhấc hộp được rồi làm rơi khi vận chuyển** (`box_lost_in_transit` 5/5 và 9/9; hộp mất ở bước ~69–97 trong khi tới tray ở bước ~97–106). Gripper **không** mở sớm: mở sau khi tới tray 0.25–1.05 s (P6). Lúc thả, hộp cách tray 11–22 cm ở 5 ca (S1: 14.7 và 19.6 cm; S2: 11.2, 20.5, 21.8 cm) và 2–6 cm ở 9 ca còn lại.
- Lệnh kẹp giống hệt giữa ca thành công và ca rơi (~11.5–12 mm); nhưng ở ca rơi khe ngón khi vận chuyển hẹp hơn (S1 16.7 mm, S2 18.4 mm so với 20.4 mm) tức hộp trượt/xoay khỏi vị trí giữa hai ngón, và điểm đóng cao hơn 0.15–0.28 cm, lệch ngang hơn 0.25–0.30 cm so với ca thành công cùng nhánh (chênh nhỏ, cùng chiều).
- **S2** có cơ chế cơ học rõ: vận chuyển nhanh hơn nhiều (đỉnh TCP 0.66 m/s ở ca thành công và **0.78 m/s ở ca rơi** so với 0.36 m/s ở C0/S1; gia tốc đỉnh 11.3 và **15.0 m/s²** so với 3.5–3.8), lệnh khớp chạm trần slew 2.0 rad/s ở cả hai nhóm, jerk tại biên chunk 90.7 (so với 56–61). Lý do policy S2 sinh chuyển động nhanh hơn chưa xác định (giả thuyết chưa kiểm: khi chiều action tay phải có floor, các chiều này đóng góp ≈0 vào L1 nên trọng số hiệu dụng của chiều tay trái đổi so với C0/S1).
- **S1**: động học vận chuyển **bình thường** (gia tốc đỉnh 3.1 ở ca rơi vs 3.8 ở ca thành công) và 5/30 so với 1/30 của C0 chưa đạt ý nghĩa thống kê (McNemar p=0.125). **Không** có bằng chứng nhân quả rằng floor là nguyên nhân; có thể là nhiễu huấn luyện. Khả năng thay thế trong kế hoạch (tay phải mang tín hiệu pha hợp lệ) cũng **chưa được kiểm**: thay state tay phải bằng trung bình không làm S1 tệ đi (17/20).

## 8. Giới hạn và lệch so với kế hoạch (nêu rõ)
1. **Một lần train mỗi nhánh** ⇒ không tách được hiệu ứng floor khỏi nhiễu huấn luyện; E0 (nhiều seed train của C0) là bước thích hợp tiếp theo nhưng nằm ngoài phạm vi giai đoạn 1.
2. **Thời gian/ngân sách:** ngân sách đã duyệt ≈6.3 CPU-giờ. Công việc thực hiện gồm thêm (i) đánh giá V1 trên dev seed (để có cột V1) và (ii) hai phép thử độc lập (khớp / gripper) mà bạn yêu cầu, nên quy về tốc độ máy ban đầu khoảng **7 giờ**. Wall-clock thực tế lớn hơn nhiều: sau lần worker khởi động lại, máy chậm ~2.1× (C0 87 phút; S1 184 phút; S2 198 phút; mọi lượt đánh giá chậm tương ứng) và một lần train S1 dở dang (bước 784) phải chạy lại từ đầu (giữ nguyên ở `outputs/checkpoints/e6b_s1_seed0_interrupted_step784*`, không xóa). Không có thay đổi tham số nào do sự cố này.
3. **Bước train** 5400 (kế hoạch) so với 5383 của V1 (17 bước). Đánh giá chỉ trên **dev seed 21–22**; test seed 31–33 chưa được dùng.
4. Dev seed 21–22 cho V1 (BOUND 72%) khác test seed 7–9 ở audit (60%) ⇒ mức dao động giữa các bộ seed cũng lớn.
5. SI > 1 ở S2 (`right_mean`, nhiễu lớn) là do mẫu số (baseline không nhiễu 13/20) thấp; không nên đọc là "nhiễu làm tốt hơn" khi chưa nhân bản.
6. Độ trễ đo trên CPU chậm hơn và không đại diện cho thiết bị đích.
7. Chiều R1/R4/R7 của state vẫn còn thông tin chuyển động ổn định (≈0.1σ_eff); chưa kiểm xem policy dùng nó như tín hiệu pha hay không.

## 9. Nguồn log và dữ liệu (đường dẫn trong repo; `outputs/` bị gitignore)
- Huấn luyện: `outputs/checkpoints/e6b_{c0,s1,s2}_seed0/` (mỗi cái có `train_log.jsonl`, `train_info.json`, `last/norm_contract.json`), log `outputs/checkpoints/e6b_*_seed0.log`; V1: `outputs/checkpoints/act_full_v1/`; smoke/tất định: `outputs/e6b/smoke/`.
- Đánh giá: `outputs/eval/e6b/{v1,c0,s1,s2}/{nom21,bound21,bound22,pert_*,ind_*}/act/*/{episodes.jsonl,summary.json,rollouts/}`, `open_loop_val.json`, log `*.log` cạnh từng thư mục.
- Tổng hợp: `outputs/e6b/stage1_results.json` (cổng và số liệu), `stage1_table.md`, `failure_analysis.json`, `unit_tests.json`, `P1…P6`.

## 10. Tái tạo
```
PY=/home/user/.venvs/act/bin/python; export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
$PY policies/test_norm_contract.py                       # G0 (16 test)
D=outputs/lerobot/pickplace_v1; COMMON="--dataset $D --steps 5400 --batch 16 --lr 1e-4 --workers 1 --threads 4 --seed 0 --val-every 250 --save-every 1000"
$PY policies/act_train.py $COMMON --out outputs/checkpoints/e6b_c0_seed0 --stats-source train_only
$PY policies/act_train.py $COMMON --out outputs/checkpoints/e6b_s1_seed0 --stats-source train_only --state-std-floor 0.02 --gripper-std-floor 0.002
$PY policies/act_train.py $COMMON --out outputs/checkpoints/e6b_s2_seed0 --stats-source train_only --state-std-floor 0.02 --action-std-floor 0.02 --gripper-std-floor 0.002
# mỗi nhánh (CK=.../last): NOM seed 21 n=30; BOUND seed 21 và 22 n=30; nhiễu trên NOM seed 21 ids 0-19 (right_mean, right_jitter_small, right_jitter_large); ids 0-9 (right_joints_jitter_small, right_gripper_jitter_small)
$PY eval/evaluate.py --agent act --ckpt $CK --conditions nominal --seed 21 --n 30 --workers 4 --save-rollouts --no-video --out outputs/eval/e6b/ARM/nom21
$PY eval/e6b_failure_analysis.py; $PY eval/e6b_report.py
```

## 11. Chờ review
Quyết định cần bạn: (1) có chấp nhận kết luận "không nhánh nào đạt G4" và dừng ở đây; (2) nếu tiếp tục, hướng đề xuất (chưa làm, cần duyệt riêng): **E0** nhiều seed train cho C0 và S1 để đo nhiễu huấn luyện trước khi kết luận về floor; xem xét S1 vì S1 gỡ lối tắt mà không đổi động học vận chuyển; điều tra S2 nhanh bất thường (trọng số loss/thang action) nếu muốn giữ action floor; (3) có commit/push các file E6-B và tài liệu này hay không (hiện chưa).
