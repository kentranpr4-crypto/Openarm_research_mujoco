# E6-B — Kế hoạch `std_floor` cho chuẩn hóa proprioception của ACT
Ngày 2026-10-28 · **CHỈ LÀ KẾ HOẠCH — chưa triển khai, chưa train, chưa commit.** Chờ chủ việc phê duyệt (mục 10) trước khi viết mã hoặc chạy training.
Nền: `docs/ai-native/tasks/2026-10-28-act-v1-audit-v2-plan.md` (commit `eac7100`), phát hiện F3. Quy ước seed: tuning dùng dev seed 21–22; test seed 31–33 chỉ dùng cho đánh giá cuối.

## 1. Vấn đề cần giải (bằng chứng hiện có)
- LeRobot 0.6.1 chuẩn hóa MEAN_STD: `x_n = (x − μ) / (σ + 1e-8)` (`lerobot/processor/normalize_processor.py:359`); thống kê gộp toàn dataset.
- Các chiều tay phải (đứng yên) có σ rất nhỏ: state 1.0e-6…2.2e-3, action 9e-13…8.1e-5 (audit A4). Với V1, **nhiễu 1 mrad trên các chiều này thành 15–1000σ** (R2 204σ, R3 995σ, R5 15.6σ, R6 550σ; R1/R4/R7 ≈0.5σ), và thay bằng giá trị trung bình (0σ) vẫn làm ACT 0/10 ⇒ policy phụ thuộc vào chi tiết siêu nhỏ của tay phải.
- **Hạn chế của phép thử cũ (đính chính):** `right_jitter` của V1 cộng 1e-3 vào *cả gripper tay phải (đơn vị mét ⇒ 1 mm)*, không phải 1 mrad thuần. Kết luận F3 chủ yếu dựa vào `right_mean` (không bị lỗi đơn vị) nhưng E6-B dùng nhiễu đúng đơn vị (mục 7).
- Chưa tách được: (a) tay phải mang **thông tin pha/thời gian có thật** (chuyển động ổn định lúc đầu episode, std≈2e-3 ở R1/R4/R7) hay (b) chỉ là **khuếch đại chuẩn hóa** của nhiễu số học.

## 2. Hợp đồng chuẩn hóa (normalization contract) E6-B
**Công thức** (mọi chiều d của `observation.state` và `action`; ảnh giữ nguyên MEAN_STD theo kênh):

`σ_eff[d] = max(σ_train[d], floor[d])`,  `x_n[d] = (x[d] − μ_train[d]) / (σ_eff[d] + ε)`,  `x[d] = x_n[d]·(σ_eff[d] + ε) + μ_train[d]` (giải chuẩn hóa), ε = 1e-8.

| Mục | Quy định |
|---|---|
| Nguồn thống kê | **chỉ các episode train** (sửa rò rỉ val): gộp `stats/*` theo episode bằng `lerobot.datasets.compute_stats.aggregate_stats` trên danh sách episode train từ `openarm_splits.json`; assert `count == 7360` khung, assert không chứa episode val |
| Floor theo đơn vị | khớp (rad): **0.02 rad** (≈1.1°; ≥20× nhiễu 1 mrad; < ¼ std nhỏ nhất của chiều sống 0.113 rad); gripper v1 (m): **0.002 m** (< std sống 0.0114 m (state) / 0.0152 m (action)). Floor **vector theo chiều**, ghi đủ 16 giá trị, không dùng một số chung |
| Phạm vi | S1: floor chỉ cho `observation.state`; S2: floor cho cả `state` và `action` (mục 4) |
| Chiều "bị ràng buộc" mong đợi | đúng tập {state 8–15} (S1) hoặc {state 8–15, action 8–15} (S2). Nếu dữ liệu khác khiến một chiều sống (0–7) bị ràng buộc ⇒ **fail to load** (không ngầm đổi chuẩn hóa) |
| Lưu trữ | `norm_contract.json` trong thư mục checkpoint: floors, σ_train, μ_train, nguồn episode, số khung, sha256 của stats, phiên bản mã; và chính các std đã nâng sàn được ghi vào safetensors của processor (nên suy luận dùng đúng đại lượng huấn luyện, không cần mã riêng) |
| Tương thích | cờ mặc định giữ **hành vi V1** (`--stats-source dataset`, không floor); V1 checkpoint đọc/đánh giá không đổi |
| Ảnh | MEAN_STD theo kênh, thống kê gộp từ cùng danh sách episode train (cũng sửa rò rỉ nhẹ) |

**Bất biến cần kiểm (I1–I8):**
I1 mọi chiều có mẫu số ≥ floor; I2 chiều sống (σ_train ≥ floor) **không đổi** so với công thức V1 tính trên cùng thống kê (sai khác tương đối <1e-6); I3 round-trip chuẩn hóa→giải chuẩn hóa là đồng nhất (<1e-6 tuyệt đối) kể cả chiều tay phải; I4 nhiễu đúng đơn vị ≤1e-3 rad (khớp) / 1e-4 m (gripper) làm chiều tay phải đổi ≤0.05σ_eff / 0.05σ_eff; I5 thống kê train-only khác thống kê toàn dataset **đúng** ở mức đã đo (≤1% trên chiều sống) và count đúng; I6 tập chiều bị ràng buộc đúng như mong đợi; I7 checkpoint nạp lại cho cùng giá trị chuẩn hóa như lúc train (cùng code path, không phân kỳ train/infer); I8 chuẩn hóa tất định.

Ảnh hưởng định lượng của floor 0.02 rad lên 1 mrad: **0.05σ_eff** ở mọi chiều khớp tay phải (V1: 0.5–995σ); lên gripper 0.1 mm→0.05σ; thay bằng giá trị trung bình vẫn 0σ nhưng biến thiên thực sự của chiều R1/R4/R7 (std 2e-3) co từ ~1σ xuống **~0.1σ_eff**.

## 3. Giả thuyết và tiên đoán (đăng ký trước)
| ID | Giả thuyết | Tiên đoán nếu đúng | Nếu sai nghĩa là |
|---|---|---|---|
| H-E6B-1 | Phụ thuộc brittle của V1 là do **khuếch đại chuẩn hóa** | S1/S2 giữ ≥90% thành công NOM dưới `right_mean` và nhiễu nhỏ (V1 và C0: ≈0%) | cơ chế không phải khuếch đại; chuyển sang E6-A (bỏ chiều) |
| H-E6B-2 | Floor **không** làm mất năng lực nhiệm vụ | NOM ≥ 28/30; BOUND ≥ C0 − 5 điểm | tay phải mang thông tin pha cần thiết (kênh "đồng hồ" hợp lệ) ⇒ cần một tín hiệu pha hợp lệ khác (ví dụ bộ đếm bước) thay vì bỏ/ép |
| H-E6B-3 (phụ) | Bỏ lối tắt có thể cải thiện BOUND | BOUND(S) > BOUND(C0) | không kết luận; chỉ báo cáo |
| H-E6B-4 (S2) | Floor trên **action** không làm tay phải rung | độ lệch lệnh tay phải khỏi home ≤2 mrad trong mọi episode | chọn S1 hoặc kèm cơ chế giữ lệnh tay phải |

## 4. Nhánh thí nghiệm và đối chứng
| Nhánh | Thống kê | state floor | action floor | Mục đích |
|---|---|---|---|---|
| B0 (tham chiếu, đã có) | toàn dataset | không | không | checkpoint V1, **không** dùng làm đối chứng nhân quả |
| **C0 (đối chứng)** | **train-only** | không | không | công thức V1 + sửa rò rỉ ⇒ cô lập hiệu ứng của floor; đồng thời là baseline của E0 |
| **S1** | train-only | có | không | chỉ cắt khuếch đại phía đầu vào |
| **S2** | train-only | có | có | hợp đồng đầy đủ; kiểm rung lệnh tay phải |
| S3 (có điều kiện) | train-only | quét floor {0.005, 0.02, 0.08} rad | theo S1/S2 | chỉ chạy nếu S1/S2 qua G3 nhưng nhạy với floor; ≤3 giá trị (ngân sách tuning) |
| Kiểm tra tất định (không train đầy đủ) | – | – | – | train 2 lần 50 bước cùng seed, so chuỗi loss ⇒ biết nhiễu so sánh C0 vs B0 là do seed hay do tất định |

**Biến cố định cho mọi nhánh:** dữ liệu `outputs/lerobot/pickplace_v1` (54 train / 6 val như V1), kiến trúc và siêu tham số V1 (ResNet18 từ đầu, dim 256, 2 enc, chunk 20, exec 20, lr 1e-4, batch 16, 5400 bước, dropout 0.1, 2 camera), seed train 0 (giống V1), thứ tự dữ liệu, scene V1, render profile `fast`, thread torch. Chỉ thay: nguồn thống kê và floor.

## 5. Kế hoạch triển khai (khi được duyệt)
Files mới: `policies/norm_contract.py` (tính thống kê train-only, áp floor, ghi/đọc `norm_contract.json`), `policies/test_norm_contract.py`, `eval/e6b_gates.py` (tính cổng chấp nhận từ kết quả eval). Sửa nhỏ: `policies/act_train.py` thêm cờ `--stats-source {dataset,train_only}`, `--state-std-floor`, `--action-std-floor` (mặc định = hành vi V1); `eval/audit_policy_inputs.py` thêm kiểm tra contract; `eval/evaluate.py` thêm kiểu nhiễu đúng đơn vị (mục 7). Không sửa upstream, scene, expert, dataset hiện có.
Cách áp dụng: truyền `dataset_stats` đã nâng sàn vào `make_pre_post_processors` (không patch thư viện); processor lưu std đã nâng sàn trong checkpoint.

## 6. Unit tests (plain asserts, chạy < 1 phút, không cần GPU)
| Test | Kiểm |
|---|---|
| T1 | công thức + I1: dữ liệu tổng hợp có chiều std 1e-9, 1e-3, 0.3 ⇒ mẫu số = max(σ, floor)+ε |
| T2 | **hồi quy V1**: floor=0, `stats_source=dataset` ⇒ std tensor trong processor bằng V1 `…normalizer_processor.safetensors` (khớp từng phần tử) và đầu ra chuẩn hóa trùng V1 trên 100 khung thật |
| T3 | I2: chiều sống không đổi giữa floor=0 và floor đặt; tập chiều bị ràng buộc đúng {8–15} |
| T4 | I3: round-trip state/action trên 1000 vector ngẫu nhiên + 100 khung thật, kể cả chiều tay phải |
| T5 | I4: nhiễu ±1e-3 rad / ±1e-4 m trên chiều tay phải làm đầu vào chuẩn hóa đổi ≤0.05σ_eff (đối chứng: V1 đổi >10σ ở ≥4 chiều) |
| T6 | I5: stats train-only: `count==7360`, khác stats toàn dataset đúng mức (≤1% chiều sống), loại đúng 6 episode val; hoán đổi tập val ⇒ stats đổi |
| T7 | chiều sống bất ngờ bị ràng buộc (σ giả < floor) ⇒ `NormContractError` (fail loud) |
| T8 | ghi/đọc `norm_contract.json`: sha256 stats khớp, floors khớp; sửa 1 byte ⇒ báo lỗi |
| T9 | I7: dựng ACT nhỏ khởi tạo ngẫu nhiên, lưu checkpoint, nạp lại ⇒ cùng đầu ra chuẩn hóa/giải chuẩn hóa như trước khi lưu |
| T10 | I8: hai lần chuẩn hóa cùng đầu vào cho cùng bit; dtype float32 |
| T11 | action floor (S2): target train chuẩn hóa của chiều tay phải ≈0 (|x_n|<0.01); giải chuẩn hóa đầu ra 0 ⇒ đúng μ |
| T12 | mặc định CLI không đổi: chạy `act_train.py` với cờ mặc định tạo cấu hình giống V1 (so config.json, trừ đường dẫn) |
| T13 | `eval/audit_policy_inputs.py` vẫn PASS (không lộ ground truth) |
Smoke (tích hợp): train **200 bước** mỗi nhánh (≈3 phút) ⇒ không NaN, checkpoint có `norm_contract.json`, nạp lại và chạy 1 rollout 20 bước, loss giảm; kiểm tất định (2 lần 50 bước).

## 7. Giao thức đánh giá (dev seed 21–22)
- **NOM**: seed 21, n=30 (vùng train). **BOUND**: seed 21 + 22, mỗi seed 30 (≥60 cặp). **SHORTCUT** (n=20 trên NOM seed 22, ba kiểu can thiệp lên state tay phải, đúng đơn vị): `right_mean`; `jitter_small` (±1e-3 rad khớp, ±1e-4 m gripper); `jitter_large` (±1e-2 rad, ±1e-3 m). Chỉ số **SI = thành công(bị nhiễu)/thành công(không nhiễu, cùng cấu hình)**.
- Thêm cho mọi nhánh: pick/place, độ cao đóng gripper (dz, tỉ lệ >3.4 cm), jerk (interior/biên), tracking, **độ lệch lệnh tay phải khỏi home** (max |a−a_home| trên mọi episode), latency/truy vấn, ablation ảnh `swap`.
- So cặp cùng `(seed, idx)`; McNemar chính xác; Wilson 95%; báo cáo hiệu số cặp kèm CI. Test seed 31–33 **không** động tới trong E6-B giai đoạn 1.

## 8. Tiêu chí chấp nhận (đăng ký trước, không sửa sau khi thấy kết quả)
| Cổng | Điều kiện đạt |
|---|---|
| **G0 Code** | T1–T13 đều PASS, smoke 200 bước PASS cho C0/S1/S2 |
| **G1 Sức khỏe train** | không NaN; train-L1 EMA cuối trong ±10% của C0; val-L1 không tệ hơn C0 quá 10%; độ lệch lệnh tay phải ≤2 mrad (S2) |
| **G2 Shortcut** | SI ≥ 0.90 cho `right_mean` **và** `jitter_small` (≥18/20 nếu NOM không nhiễu 20/20); `jitter_large` SI ≥ 0.80 (thông tin) |
| **G3 Không thoái hóa** | NOM ≥ 28/30 **và** BOUND(S) ≥ BOUND(C0) − 5 điểm (điểm ước lượng) với CI cặp 95% có cận dưới ≥ −20 điểm |
| **G4 Kết luận** | S đạt G0–G3 ⇒ chấp nhận làm **nền V2** (ưu tiên S1 nếu S1 và S2 cùng đạt, vì ít thay đổi hơn) |

**Cây quyết định:** (i) G2 đạt, G3 đạt ⇒ adopt, chuyển sang E1/E5/E4 trên nền này; (ii) G2 đạt nhưng NOM tụt ⇒ chiều tay phải mang tín hiệu pha hợp lệ ⇒ **dừng, báo chủ việc** và đề xuất bộ đếm bước/pha minh bạch thay vì ép; (iii) G2 không đạt dù đã nâng sàn ⇒ cơ chế không phải khuếch đại ⇒ chuyển E6-A (bỏ chiều) cùng đối chứng; (iv) S2 gây rung tay phải ⇒ dùng S1. Mọi kết quả (kể cả âm) được ghi vào báo cáo.

## 9. Ngân sách (đo trên CPU 4 lõi: train 0.93 s/bước, eval ≈14 s/episode hiệu dụng)
| Giai đoạn | Việc | Ước tính |
|---|---|---|
| Phát triển | viết `norm_contract.py` + test, CLI | ~0.5 ngày công, không tính CPU dài |
| 0 | T1–T13, smoke 200 bước ×3, kiểm tất định | ≈20 phút |
| 1 (screening) | train C0, S1, S2 (seed 0), mỗi nhánh 5400 bước ≈84 phút ⇒ **4.2 giờ**; eval mỗi nhánh NOM30 + BOUND60 + 3×20 SHORTCUT = 150 ep ≈35 phút ⇒ **1.75 giờ** | **≈6 giờ** tuần tự (không chồng train và eval để tránh tranh CPU) |
| 2 (có điều kiện, nếu qua G2) | seed 1, 2 cho C0 (đây chính là **E0**) và nhánh thắng: 4 train ≈5.6 giờ + eval ≈2.4 giờ | ≈8 giờ |
| 3 (có điều kiện) | S3 quét floor (tối đa 2 train thêm) | ≈3.5 giờ |
| Cuối | test seed 31–33 một lần cho C0 vs nhánh chọn (n=90 mỗi nhánh + SHORTCUT) | ≈1.2 giờ |
Tổng xấu nhất ≈ 19 giờ CPU; chắc chắn cần chỉ là giai đoạn 0–1 (≈6.3 giờ). Dừng sớm nếu G0/G1 hỏng. Khối lượng này tăng ~10× nếu thêm seed hoặc đổi sang dữ liệu 200–300 demo (không thuộc E6-B).

## 10. Rủi ro và câu hỏi chờ duyệt
Rủi ro: (R1) 1 seed/nhánh ở giai đoạn 1 ⇒ chỉ dùng để **sàng lọc**, kết luận thuộc giai đoạn 2; (R2) floor 0.02 rad là lựa chọn từ phân phối std hiện tại, không phải đặc tả phần cứng; (R3) tính tất định CPU có thể khác nhau giữa lần chạy ⇒ có bước kiểm tra; (R4) giai đoạn 1 dùng 54 demo nên BOUND vẫn có thể ≈60% — E6-B không nhằm cải thiện BOUND; (R5) mọi so sánh dùng dev seed, tránh tinh chỉnh quá khớp.
Cần bạn duyệt: (1) hợp đồng ở mục 2, nhất là floor 0.02 rad / 0.002 m và việc sửa nguồn thống kê thành train-only (kể cả ảnh); (2) nhánh C0/S1/S2 và kiểm tra tất định; (3) cổng G0–G4 (đặc biệt ngưỡng SI ≥0.90, NOM ≥28/30, biên BOUND −5 điểm); (4) ngân sách giai đoạn 1 (≈6.3 giờ CPU); (5) chạy giai đoạn 2/3 tự động nếu qua G2, hay dừng chờ duyệt lại sau giai đoạn 1.
