# OpenArm ACT V1 — Research Audit & V2 Experiment Plan
Ngày 2026-10-28 · tự kiểm (không phải kiểm độc lập) · **không retrain, không đổi kiến trúc, không sửa upstream, không xóa dữ liệu, không commit/push.**
Trạng thái: **Phase A = DONE · Phase B = DONE (kế hoạch) · Phase C = DONE (chỉ diagnostics inference-only)**. Thí nghiệm E4 (pretrained backbone) **BLOCKED** về hạ tầng (mục 5).

Đầu vào đã đọc: `outputs/eval/full_v1*/report.md` + `summary_*.json`, checkpoint `outputs/checkpoints/act_full_v1/last` (config, train_info, normalizer), dataset `outputs/datasets/demo_train_clipped` + `outputs/lerobot/pickplace_v1`, rollouts lỗi, và mã `eval/metrics.py`, `eval/evaluate.py`, `policies/{baseline,act_train,act_infer}.py`, `data/collect_demos.py`.
Hình trong báo cáo nằm ở `assets/act-v1-audit/` (cạnh file này); dữ liệu thô ở `outputs/audit_v1/audit_results.json` và `outputs/eval/audit_v1/` (thư mục `outputs/` bị gitignore).

## 0. Kết luận ngắn (xếp theo mức bằng chứng)

| # | Phát hiện | Mức bằng chứng |
|---|---|---|
| F1 | Lỗi `unseen_pose` chủ yếu là **sai độ cao khi đóng gripper**: TCP cao hơn expert **+1.39 cm** (dz 3.89 cm ở ca lỗi vs 2.50 cm ở ca thành công vs 2.50 cm expert); 10/12 ca lỗi có finger–box overlap < 1.7 cm. Sai số do **lệnh của policy** (khớp 4 ≈ +2.5 cm), không phải tracking (sai số bám tại thời điểm đóng: 0.0098 rad lỗi vs 0.0099 rad thành công) và không phải timing | **Đã đo** (60 rollout có GT, A2/A6b/A9) |
| F2 | Policy **chạy kịch bản theo thời gian, không hồi phục**: 12/12 ca lỗi vẫn đóng gripper ở 2.40–2.55 s rồi mở lại ở 4.15–5.95 s và đưa TCP tới tray (<1 cm) dù hộp gần như không di chuyển; 0 lần gắp lại (cả 12 ca lỗi lẫn 48 ca thành công) | **Đã đo** (A2, timeline) |
| F3 | **Phụ thuộc cực mạnh vào proprioception của tay phải đứng yên**: thay state tay phải bằng trung bình train (0σ, nằm đúng phân bố) hoặc cộng nhiễu 1 mrad → ACT **0/10 ở cả `nominal` lẫn `unseen_pose`** (baseline 10/10 và 3/10). Nguyên nhân gốc rất có thể là chuẩn hóa MEAN_STD chia cho std 1e-6…2e-3 của các chiều gần hằng số, biến chuyển động ổn định lúc đầu episode thành "đồng hồ" | **Can thiệp inference-only đã đo**; cơ chế (clock vs khuếch đại chuẩn hóa) **chưa tách được** |
| F4 | Policy **có dùng ảnh**: ảnh đen/ảnh trung bình → 0/10; ảnh của demo khác cùng thời điểm → 4/10 (nominal), 1/10 (unseen) | **Đã đo** (nhưng zero/mean là OOD nên chỉ chứng minh "có phụ thuộc", chưa chứng minh "dùng đúng thông tin vị trí hộp") |
| F5 | Jerk cao hơn expert 3.7× (25.2 vs 6.7 rad/s³): **42%** phần dư do cửa sổ chạm biên chunk (73.6 vs 17.5 interior), **58%** do nhiễu bên trong chunk (interior 17.5 vs expert 6.65). Temporal ensembling (coeff 0.01) → **3.7**, thấp hơn expert | **Đã đo** (rollouts + pilot) |
| F6 | Pilot inference-only trên 10 cấu hình `unseen_pose` seed 7: exec20 **3/10**, exec10 **7/10**, exec5 **7/10**, TE0.01 **5/10**. Cùng cấu hình, 4 ca được sửa và 0 ca hỏng (McNemar chính xác p=0.125) | **Gợi ý, chưa có ý nghĩa thống kê** (n=10) |
| F7 | Không rò rỉ giữa train và test: seed RNG rời nhau, 0 pose trùng, split val/train rời nhau, checkpoint `last` chọn trước. **Một đính chính:** thống kê chuẩn hóa tính trên cả 60 episode (gồm val), ảnh hưởng ≤0.8% std và ≤0.011σ ở các chiều có phương sai | **Đã đo** (A4) |
| F8 | `unseen_pose` thật sự xa train (khoảng cách tới pose gần nhất gấp **2.4×** nominal, p=1.5e-9) nhưng **trong nhóm unseen khoảng cách không dự đoán được thành công** (ρ=0.01); x<0.38: 4/12 thành công, x≥0.38: 14/18 | **Đã đo**, nhưng nguyên nhân sâu **chưa giải thích** (mục 3, A5) |

## 1. Phase A — Audit chi tiết

### A1. Metric, split và số liệu thô (`eval/metrics.py:4-15`, `eval/evaluate.py:32`)
- **pick** = tồn tại một bước mà hộp nâng ≥5 cm **và** cách TCP <6 cm; **place** = tâm hộp cuối episode cách tâm tray <3 cm (xy), |z − (đáy tray + 2 cm)| <6 mm, tốc độ <0.02 m/s; **task** = pick ∧ place. Horizon 200 bước (10 s) ở 20 Hz. Expert và ACT dùng **cùng** cấu hình `(seed, idx)` (so cặp).
- Vùng: `TRAIN_REGION` x∈[0.38,0.45], y∈[0.07,0.13], |yaw|≤0.20; `unseen_pose` = lấy mẫu đều trong x∈[0.36,0.47], y∈[0.05,0.15], |yaw|≤0.30 rồi **loại** các điểm nằm trong hộp train (`policies/baseline.py:71-80`). Vậy "unseen" ở đây nghĩa là **sát biên, lệch 0.5–2 cm**, chưa phải vật lạ hay vùng xa.
- Tái tính từ rollout thô (`eval/audit_v1.py::a1`): **120 rollout, 0 nhãn lệch** so với `episodes.jsonl`. Tái chạy ACT cả 3 seed × 2 điều kiện: **60/60 cùng outcome, chênh khoảng cách hộp–tray = 0.0** ⇒ mô phỏng + suy luận tất định, số liệu V1 tái lập được. (Lưu ý: rollout cũ chưa ghi `box_speed`, audit dùng tốc độ bước cuối/dt làm xấp xỉ; lần ghi mới đã sửa trong `eval/evaluate.py`.)
- Bảng gộp tái tính: expert 30/30 và 30/30; ACT nominal **30/30**, ACT unseen **18/30** (pick 19/30, place 18/30), theo seed 3, 7, 8.
- Cảnh báo thống kê: n=10/seed nên Wilson 95% rộng (60% gộp: [42–75]); các điều kiện dùng chung cấu hình nên **không độc lập**; hai điều kiện `nominal`/`unseen` ở seed 7–9 đã bị dùng để chẩn đoán trong audit này ⇒ **không còn là test sạch** cho V2 (xem protocol mục 4).

### A2. Failed rollouts theo thời gian — `assets/act-v1-audit/A2_failure_timelines.png`
12 lỗi, tất cả ở `unseen_pose` (nominal 0/30 lỗi). Phân loại tự động theo luật (`eval/audit_v1.py::classify`, ngưỡng từ hình học ngón: nửa khe 4.3 cm vs nửa bề rộng hộp 2.0 cm; ngón cao ±3.1 cm):

| Lớp (chính, ưu tiên: pushed > lệch ngang > quá cao) | n |
|---|---|
| approach/localization: **đóng quá cao** (overlap < 1.7 cm) | 5 |
| approach: **hộp bị đẩy** >3 cm trước khi đóng | 4 |
| approach/localization: lệch ngang/dọc khi đóng | 2 |
| release: đã nhấc nhưng hộp không nằm lại trong tray | 1 |
| perception thuần (không đóng gripper), grasp trượt sau khi đúng chỗ, transport rơi giữa đường | 0 |

Cờ đa nhãn trong 12 ca: **too_high 10**, pushed 4, lateral_miss 4, long_miss 0. Các lớp chồng nhau vì "bị đẩy" thường là hệ quả của ngón chạm hộp khi hạ không đúng độ cao/lệch ngang. Đặc trưng thời gian (cùng hình): gripper đóng ở 2.40–2.55 s (bước 46–51, expert 46–53 nên timing **không** lệch), mở lại ở 4.15–5.95 s (đã kiểm cả 12/12 ca), TCP tới tray (<1 cm) ở 12/12 ca; không có lần gắp lại nào. Camera đầu vào: hộp luôn thấy trong `front` (diện tích ≥53 px ở train, ≥102 px ở unseen, ảnh 160×120, ~3 px/cm); trong `wrist_left` ở t=0 chỉ 85% train và 60% unseen có hộp trong khung (`A7_image_space.png`).

### A3. Biên action chunk và jerk — `A3_chunk_jerk.png`
- Bước thay đổi action ở **vị trí đầu của chunk mới** gấp **2.42×** trung vị các vị trí bên trong chunk (9.5 mrad vs ~4 mrad).
- Jerk |Δ³a|/dt³ (cánh tay): ACT exec20 **25.16**, trong đó cửa sổ qua biên chunk **73.55** (chiếm 39.9% tổng jerk dù chỉ ~15% số cửa sổ), cửa sổ interior **17.47**; expert **6.73** (interior 6.65). Khớp 4, 7, 1 jerk nhiều nhất cả ở ACT lẫn expert.
- Pilot cùng 10 cấu hình (seed 7, unseen): exec20 jerk 29.5 (interior 20.2) · exec10 24.8 (12.1) · exec5 32.8 (12.3) · TE0.01 **3.66**. ⇒ horizon ngắn **giảm** nhiễu interior nhưng **thêm** biên nên jerk tổng không giảm; chỉ temporal ensembling làm mịn thật sự.
- Tracking (|ctrl−state| trung bình): ACT 0.01385 rad vs expert 0.01500 rad ⇒ jerk cao **không** làm giảm khả năng bám. Số lần clip ~1300–1550/điều kiện chủ yếu là lệnh mở gripper đúng bằng giới hạn trên 0.044 (làm tròn), slew-limit chỉ kích hoạt 3–18 lần ⇒ lớp an toàn không tạo ra jerk.

### A4. Rò rỉ và độc lập (`eval/audit_v1.py::a4`)
| Kiểm tra | Kết quả |
|---|---|
| Seed RNG | train demo 1 000 000–1 000 059; eval 700 000–900 009; **giao = ∅** |
| Pose trùng | 0; khoảng cách tối thiểu tới pose train (z-score) 0.041 |
| Split | 54 train / 6 val rời nhau (theo episode); `train_info.json` khớp `openarm_splits.json` |
| Chọn checkpoint | `last` chọn trước, không early-stopping, không chọn theo val/test |
| Ánh sáng | train `light_scale∈[0.85,1.15]`, nominal=1.0 (trong dải); `light_dim 0.55`, `light_bright 1.6`, `light_shift`, `cam_shift` chưa thấy lúc train |
| **Thống kê chuẩn hóa** | checkpoint chứa `action.count=8172` = 7360 (train) + 812 (val) ⇒ **gồm cả episode val**; ảnh hưởng trên chiều có phương sai: Δstd ≤0.78%, Δmean ≤0.011σ ⇒ không đáng kể, không chạm test. Báo cáo trước và comment mã từng ghi "chỉ train" là **sai**, đã sửa (`policies/act_train.py:8,60`, doc 10-27) |
| Phụ thuộc giữa các điều kiện | các điều kiện dùng chung `(seed, idx)`; chỉ seed là bản sao độc lập |
| **Chiều gần hằng số** | tay phải: action std 1e-13…8e-5, state std 1e-6…2e-3 (chiều 8, 11, 14 ≈2e-3 do tay ổn định lúc đầu episode); chuẩn hóa chia cho các std này (xem F3) |

### A5. Phân phối pose — `A5_pose_distribution.png`
- Train (60 demo): x 0.383–0.450, y 0.070–0.130, yaw −0.193…0.196, trung bình (0.415, 0.101, −0.018), std (0.020, 0.017, 0.116).
- Nominal eval: 0/30 ngoài hộp train, khoảng cách tới pose train gần nhất trung vị 0.155; unseen: 30/30 ngoài hộp, trung vị **0.370** (Mann–Whitney p=1.5e-9).
- Trong nhóm unseen: thành công **không** tương quan với khoảng cách tới pose gần nhất (ρ=0.01, p=0.97), tương quan yếu với độ vượt biên (ρ=−0.20, p=0.30), x (ρ=+0.35, p=0.055); **tương quan dương với |yaw|** (ρ=+0.57, p=0.001). Theo nhóm: xy-trong/yaw-ngoài **3/3**, xy-ngoài/yaw-ngoài **9/10**, xy-ngoài/yaw-trong **6/17**; x<0.38: **4/12**, x≥0.38: **14/18**; Fisher xy-ngoài vs xy-trong p=0.26 (n nhỏ). ⇒ lệch **yaw** không gây lỗi; lệch vị trí gây lỗi nhưng phản ứng **không đơn điệu theo khoảng cách** — nguyên nhân chưa giải thích (có thể là cấu trúc mẫu rejection: nhóm yaw-trong luôn lệch xy, nhóm yaw-ngoài có thể nằm trong xy).

### A6. Định vị điểm gắp — `A6_localisation.png`, `A6b_closing_height.png`, `eval/audit_grasp_pose_error.py`
- Điểm TCP lúc bắt đầu đóng so với vị trí hộp thật: độ dốc x **0.974** (expert 1.002), y **0.766** (expert 0.940); R² 0.96/0.84. Lệch ngang tuyệt đối trung bình 0.43 cm (thành công) vs 1.38 cm (lỗi); dọc trung bình +1.38±0.97 cm (expert +1.00±0.05). ⇒ xy được định vị **khá tốt** (co về trung bình nhẹ ở y), sai chính là **độ cao**.
- Độ cao đóng: thành công 2.50±0.20 cm = expert 2.50±0.04; lỗi 3.89±0.74 cm (2.80–5.86); dz tăng khi x giảm (ρ=−0.37, p=0.004) và y tăng (ρ=+0.28, p=0.03); hồi quy tuyến tính theo (x, y, yaw) chỉ giải thích R²=0.18.
- So với lệnh expert cùng cấu hình tại thời điểm đóng: ở ca lỗi TCP lệch (x,y,z)=(+0.45, +0.69, **+1.06**) cm (thành công: 0.75, 0.01, 0.02); phân rã FK theo khớp: khớp 4 **+2.47 cm** lên độ cao, khớp 7 −1.13 cm, các khớp khác ≤0.5 cm. Sai số xoay bàn tay ≈6° ở **cả** ca thành công (6.1°) và lỗi (6.7°) ⇒ không phải yếu tố phân biệt.

### A7–A8. Vision, state và pilot inference-only (`A7_image_space.png`, `A8_vision_ablation.png`)
Tất cả trên cùng cấu hình seed 7, n=10, ACT `last`, không retrain.

| Can thiệp | nominal | unseen_pose |
|---|---|---|
| Ảnh thật (baseline) | 10/10 | 3/10 |
| Ảnh đen | 0/10 | 0/10 |
| Ảnh trung bình train | 0/10 | 0/10 |
| Ảnh của demo khác, cùng thời điểm | 4/10 (pick 7/10) | 1/10 (pick 4/10) |
| State tay phải → trung bình train | **0/10** | **0/10** |
| State tay phải + nhiễu 1 mrad | **0/10** | **0/10** |
| exec10 / exec5 / TE 0.01 | – | 7/10 / 7/10 / 5/10 |

Truy vấn mỗi chunk: p50 ≈ 49–51 ms, p95 57–63 ms, p99 61–77 ms (CPU 1 luồng); chưa đo Jetson.

## 2. Bảng chứng cứ: đã kiểm vs giả thuyết chưa kiểm

| Giả thuyết | Trạng thái | Căn cứ / cách kiểm tiếp |
|---|---|---|
| H1 Sai số chính là **độ cao đóng gripper** (policy output) | **Hỗ trợ mạnh** | A6b, A9; chưa biết **vì sao** (hội tụ tới trung bình trong không gian khớp? thiếu độ phân giải ảnh? dữ liệu ít?) |
| H2 Policy chạy kịch bản mở vòng, không phản ứng khi gắp hụt | **Hỗ trợ mạnh** (quan sát) | A2; kiểm nhân quả cần demo hồi phục (E7 backlog) |
| H3 Policy dựa vào chiều tay phải gần hằng số như tín hiệu thời gian/định danh | **Hỗ trợ bằng can thiệp** | F3; tách "clock" khỏi "khuếch đại chuẩn hóa" cần retrain (E6) |
| H4 Policy ignore ảnh | **Bác bỏ** | zero/mean/swap |
| H5 Lỗi do perception/localization xy | **Phần lớn bác bỏ** cho xy, **chưa kiểm** cho z | A6: xy dốc 0.97/0.77, lỗi là z |
| H6 Lỗi do tracking/servo | **Bác bỏ** | sai số bám tại thời điểm đóng bằng nhau |
| H7 Lỗi do timing đóng gripper | **Bác bỏ** | t_close ACT 46–51 vs expert 46–53 |
| H8 Jerk do biên chunk | **Một phần (42%)** | A3; phần còn lại là nhiễu interior |
| H9 Rò rỉ train→test | **Không có bằng chứng** | A4 |
| H10 Thiếu dữ liệu/độ phủ gây lỗi unseen | **Chưa kiểm** | E1 |
| H11 Backbone pretrained giúp | **Chưa kiểm, BLOCKED** | E4 |
| H12 Augmentation ánh sáng/camera giúp | **Chưa kiểm** | E5; V1 đã 80–100% nên có hiệu ứng trần |
| H13 Yaw chưa thấy làm hỏng | **Bác bỏ** (xy-trong/yaw-ngoài 3/3) | A5 |
| H14 Exec ngắn/ensembling nâng thành công | **Gợi ý** (pilot n=10) | E2/E3 |
| H15 Độ phân giải ảnh 160×120 giới hạn việc suy ra độ cao | **Chưa kiểm** | thêm biến độ phân giải ở V2 backlog |

## 3. Việc chưa giải thích (ghi rõ)
1. Vì sao đóng cao hơn ở x<0.375 / y>0.135 (A6b) nhưng không tương quan đơn điệu với khoảng cách tới pose train (A5).
2. Vì sao |yaw| lớn đi cùng thành công nhiều hơn trong nhóm unseen (có thể là hiệu ứng cấu trúc mẫu).
3. Cơ chế chính xác của F3 (phụ thuộc có thông tin vs khuếch đại chuẩn hóa).
4. Sai số huấn luyện 1 lần/1 seed: **chưa biết phương sai giữa các lần train** ⇒ hiệu ứng 60% vs 100% có thể có thành phần ngẫu nhiên train (E0).

## 4. Phase B — Kế hoạch thí nghiệm V2

### 4.1 Giao thức chung
- **Baseline B0** = checkpoint V1 `act_full_v1/last` (không sửa). Công thức: ResNet18 từ đầu, dim 256, 2 enc, chunk 20, exec 20, 54 demo (60 gồm val), 5383 bước batch 16, lr 1e-4, CPU 4 luồng.
- **Seed**: seed 7/8/9 đã bị dùng để chẩn đoán ⇒ dùng làm **dev-reference**. Dev mới: seed 21, 22; **test cuối chỉ chạy một lần**: seed 31, 32, 33. Pose sinh bằng `default_rng(seed*100000+idx)` như V1 để so sánh trực tiếp.
- **Eval set**: NOM (vùng train, n=30), BOUND (như V1 `unseen_pose`, n=60 = 2 seed×30), STRESS (ánh sáng ×0.3/×2.5, dời đèn, camera ±4 cm/6°; n=20 mỗi loại; V1 ở mức 80–100% nên thang V1 bị trần), SHORTCUT (`--state-perturb right_mean` và `right_jitter` trên NOM, n=10). Với nhánh dữ liệu mở rộng, thêm **EXT**: pose lệch 0.5–1.5 cm ngoài vùng train mới (cần expert xác minh trước).
- **Metric chính** = task success BOUND. **Phụ**: pick, place, NOM, STRESS, SHORTCUT, closing-height dz (trung bình, tỉ lệ dz>3.4 cm), jerk (interior/biên), tracking, latency mỗi truy vấn p50/p95/p99, số lần gắp lại.
- **Thống kê**: so cặp cùng `(seed, idx)`, McNemar chính xác hai phía, hiệu chỉnh Holm cho các so sánh cùng họ, Wilson 95%. Cỡ mẫu: để phát hiện +20 điểm % (60→80%) với power 0.8, α=0.05 cần **≈82 cặp** (hai tỉ lệ độc lập) hoặc **≈57 cặp** (McNemar, giả định 30% cặp bất đồng) ⇒ dùng **n≥60 cặp**. Pilot n=10 ở V1 (4 sửa/0 hỏng, p=0.125) chưa đủ.
- **Biến cố định (mọi thí nghiệm, trừ biến đang thử)**: scene V1, expert, vùng và seed pose, chunk 20 (trừ E2/E3 là inference), số bước train 5400 (ghi thêm biến thể "epoch-matched" khi dữ liệu tăng), batch 16, lr 1e-4, render profile `fast`, 2 camera (front, wrist_left), 20 Hz, horizon 200.
- **Ngưỡng chấp nhận chung**: cải thiện BOUND ≥ +15 điểm % (so cặp, p<0.05 sau Holm) **và** NOM ≥ 95% (không tụt >3 điểm) **và** SHORTCUT không tệ hơn V1 **và** jerk không vượt 1.5× baseline trừ khi thí nghiệm nhằm giảm jerk.
- **Throughput đo được** (CPU 4 lõi, dùng để ước ngân sách): thu thập **14.6 s/demo** (4 worker; ~46 s/demo/worker), train **0.93 s/bước** (5400 bước ≈ 84 phút), eval ACT **≈14 s/episode** hiệu dụng (đo: 10 episode trong ~138 s, 4 worker; ~45 s/episode/worker).

### 4.2 Ma trận thí nghiệm
Thứ tự và cổng quyết định: **E2/E3 (không train) → E0 → E6 → rồi E1, E5, E4 như độ lệch một yếu tố từ nền V2 (V1 + E6 nếu E6 đạt)**. Lý do: F3 cho thấy công thức V1 có lối tắt proprioception, nếu không xử lý trước thì mọi so sánh sau bị nhiễu.

| ID | Giả thuyết | Baseline | Biến thay đổi | Biến cố định | Metric (chính / phụ) | Ngân sách (CPU 4 lõi) | Tiêu chí chấp nhận / dừng |
|---|---|---|---|---|---|---|---|
| **E2** Execute horizon (inference-only) | Thực thi ngắn hơn cho phép policy sửa độ cao đóng (H1/H2), giảm nhiễu interior | B0, exec20 | `n_action_steps` ∈ {20, 10, 5} | checkpoint B0, dev seed 21–22, chunk 20 | BOUND success / jerk interior+biên, latency, tần suất truy vấn | 3 arm × 60 ep ≈ 42 phút eval (pilot n=10 đã có) | Chấp nhận nếu một arm ≥ +15 điểm % so cặp (Holm). Cảnh báo: pilot cho jerk exec5 **tăng** (32.8 vs 29.5) → kết hợp E3 hoặc làm mịn. Dừng nếu exec10≈exec5≈exec20 (±5) |
| **E3** Temporal ensembling (inference-only) | TE làm mịn lệnh (H8) với giá phải trả là trễ | B0, exec20, không TE | TE coeff ∈ {off, 0.1, 0.01, 0.001}, exec=1 | như E2 | jerk (mục tiêu ≤1.5× expert ≈ ≤10), BOUND, latency/bước (mỗi bước phải truy vấn) | 3 arm × 60 ep ≈ 42 phút | Chấp nhận nếu jerk ≤10 **và** BOUND không thua exec10 quá 5 điểm; pilot: jerk 3.66, success 5/10. Rủi ro: truy vấn mỗi bước ≈50 ms CPU = bằng chu kỳ điều khiển ⇒ cần đo trên Jetson/GPU trước khi kết luận triển khai |
| **E0** Sàn nhiễu huấn luyện | Chênh 60% vs 100% có một phần do ngẫu nhiên train | B0 (seed train 0) | seed train ∈ {1, 2}, cùng dữ liệu/công thức | dữ liệu, bước, siêu tham số | BOUND (60 ep) mỗi seed, độ lệch chuẩn | 2 train × 84 phút + 2 × 17 phút eval ≈ **3.4 giờ** | Xác định SD; mọi hiệu ứng V2 phải > 2×SD hoặc ≥ +20 điểm. Nếu SD ≥ 10 điểm: bắt buộc ≥3 seed train/arm |
| **E6** Vệ sinh proprioception (bổ sung từ F3) | Loại chiều tay phải (policy chỉ tay trái, state 8-D/action 8-D) hoặc sàn std ≥1e-2 loại bỏ lối tắt "đồng hồ"; SHORTCUT sẽ ≈ baseline | B0 | state/action chỉ chiều tay trái (+ tay phải giữ hằng bởi controller), hoặc std-floor | dữ liệu 54 ep, công thức | SHORTCUT (right_mean/jitter), BOUND, NOM | 1–2 train ≈ 1.4–2.8 giờ + eval 25 phút | Chấp nhận nếu SHORTCUT ≥ 90% nominal và NOM ≥95% và BOUND ≥ B0−5. Nếu BOUND tụt mạnh ⇒ lối tắt đang gánh nhiệm vụ: ghi nhận và quay lại E1 trước |
| **E1** Độ phủ dữ liệu | Độ phủ (không chỉ số lượng) quyết định BOUND/EXT (H10) | nền V2 với 60 demo vùng train | (a) 200 demo vùng train; (b) 200 demo vùng rộng x∈[0.36,0.47], y∈[0.05,0.15], \|yaw\|≤0.20; (c) 300 demo vùng rộng | công thức, 5400 bước (+ biến thể 16200 bước cho (c) để tách số bước vs dữ liệu) | BOUND, EXT, NOM, dz>3.4 cm, vùng lỗi theo (x,y) | thu thập 49 + 49 + 73 phút ≈ **2.9 giờ**; 3–4 train ≈ 4.2–5.6 giờ; eval 4 × 25 phút | Chấp nhận (b)/(c) nếu BOUND ≥80% (+20) và EXT ≥ 60%; (a)≈(b) (±5) ⇒ độ phủ không quan trọng, dừng nhánh. Rủi ro: vùng rộng đổi định nghĩa "unseen" ⇒ bắt buộc dùng EXT, không dùng BOUND cũ làm thước duy nhất |
| **E4** Backbone khởi tạo | Trọng số ImageNet cải thiện học định vị với 60–200 demo (H11) | nền V2 random init | `random` vs `ImageNet ResNet18` (tùy chọn: lr backbone ×0.1) | dữ liệu (60 và 200-rộng), công thức | BOUND, STRESS, NOM | 2–3 train ≈ 2.8–4.2 giờ + eval | **BLOCKED**: HuggingFace và `download.pytorch.org` bị proxy từ chối trong sandbox (chỉ PyPI thông). Cần (i) mở allowlist hoặc (ii) tải trước `resnet18-f37072fd.pth` + kiểm sha256. Chấp nhận: pre ≥ +10 điểm BOUND ở cùng dữ liệu; nếu chỉ lợi ở 60 demo, giá trị hạn chế |
| **E5** Augmentation camera/ánh sáng | Domain randomization và aug quang học tăng độ bền STRESS mà không đổi BOUND (H12) | nền V2, không aug | (a) DR lúc thu thập: ánh sáng U(0.5,1.8), dời đèn, camera ±2 cm/±3°, 200 demo; (b) aug quang học lúc train (brightness/contrast/saturation/noise); (c) đối chứng âm: random-shift/crop của camera front | dữ liệu pose, công thức | STRESS, BOUND, NOM | thu thập có render thêm ≈ 1 giờ + 3 train ≈ 4.2 giờ + eval 3 × 40 phút | Chấp nhận (a)/(b) nếu STRESS +20 điểm, NOM ≥ −3, BOUND không tệ hơn. Dự đoán (c) làm hại định vị (vị trí pixel mã hóa vị trí hộp): nếu sai, ghi lại |
| (backlog) E7 Hồi phục | Demo gắp-lại (nhiễu expert) giảm lỗi "chạy kịch bản" (H2) | – | thêm demo có sai lệch + sửa | – | số lần gắp lại, BOUND | chưa ước | ngoài phạm vi yêu cầu, ghi nhận từ F2 |

Tổng ước tính nếu chạy hết trên máy này: train ≈ 11 lần × 1.4 giờ ≈ 15.6 giờ, thu thập ≈ 4.7 giờ, eval ≈ 9 giờ ⇒ **≈ 30 giờ wall-clock CPU**; có GPU thì train/eval nhanh hơn nhiều nhưng cần chạy lại phép đo throughput.

## 5. Phase C — những gì đã chạy (không retrain)
Thay đổi mã (nhỏ, mặc định không đổi hành vi): `policies/act_infer.py` thêm tùy chọn `temporal_ensemble` và ghi action thô; `eval/evaluate.py` thêm `--temporal-ensemble`, `--state-perturb`, ghi `box_speed`/`raw_action` vào rollout; `eval/audit_policy_inputs.py` vẫn PASS. Script mới: `eval/audit_v1.py`, `eval/audit_vision_ablation.py`, `eval/audit_pose_visibility.py`, `eval/audit_grasp_pose_error.py`.
Giữ nguyên: scene V1, checkpoint, dataset, log cũ (chỉ ghi thêm vào `outputs/eval/audit_v1/`, `outputs/audit_v1/`).
Hạn chế: pilot và vision/state ablation chỉ n=10 và một seed; rollout seed 8–9 (chạy lại) thiếu `box_speed`; AC "hộp thấy trong ảnh" đo ở t=0 (tư thế home) chứ không đo suốt episode; `grasp_pose_error` dùng kế hoạch expert (thông tin đặc quyền) chỉ để chẩn đoán.

## 6. Hướng dẫn reproduce
```
# (1) rollouts có GT và kiểm tái lập: sim/ACT venv, 4 worker, ~45 phút
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa; PY=/home/user/.venvs/act/bin/python; CK=outputs/checkpoints/act_full_v1/last; O=outputs/eval/audit_v1
for s in 7 8 9; do
  $PY eval/evaluate.py --agent expert --conditions nominal unseen_pose --n 10 --seed $s --save-rollouts --out $O/seed$s
  $PY eval/evaluate.py --agent act --ckpt $CK --conditions nominal unseen_pose --n 10 --seed $s --workers 4 --save-rollouts --no-video --out $O/seed$s
done
# (2) vision ablation (không render, ~5 phút)
$PY eval/audit_vision_ablation.py --ckpt $CK --demos outputs/datasets/demo_train_clipped --seed 7 --n 10 --out $O/vision_ablation_seed7.json
# (3) pilots inference-only (unseen, seed 7)
$PY eval/evaluate.py --agent act --ckpt $CK --conditions unseen_pose --n 10 --seed 7 --n-action-steps 10 --workers 4 --save-rollouts --no-video --out $O/pilot_exec10
$PY eval/evaluate.py --agent act --ckpt $CK --conditions unseen_pose --n 10 --seed 7 --n-action-steps 5  --workers 4 --save-rollouts --no-video --out $O/pilot_exec5
$PY eval/evaluate.py --agent act --ckpt $CK --conditions unseen_pose --n 10 --seed 7 --temporal-ensemble 0.01 --workers 4 --save-rollouts --no-video --out $O/pilot_te001
# (4) can thiệp state tay phải
for m in right_mean right_jitter; do $PY eval/evaluate.py --agent act --ckpt $CK --conditions nominal unseen_pose --n 10 --seed 7 --state-perturb $m --workers 4 --save-rollouts --no-video --out $O/diag_$m; done
# (5) vị trí hộp trong ảnh + sai số tư thế gắp so với expert + tổng hợp, biểu đồ
sim/.venv/bin/python eval/audit_pose_visibility.py --out $O/pose_visibility.json
$PY eval/audit_grasp_pose_error.py ; $PY eval/audit_v1.py     # -> outputs/audit_v1/audit_results.json + *.png
```
Mọi lượt chạy tất định: chạy lại cho cùng kết quả (đã kiểm 60/60).

## 7. Việc cần chủ việc quyết định
1. Duyệt thứ tự E2/E3 → E0 → E6 → E1/E5/E4 và việc thêm **E6** (không nằm trong danh sách ban đầu nhưng là hệ quả trực tiếp của F3).
2. E4 cần: mở allowlist `download.pytorch.org`/`huggingface.co` hoặc tải trước file trọng số vào repo/scratch (kèm sha256).
3. Có chấp nhận dùng **dev seed 21–22** và **test seed 31–33** (seed 7–9 đã dùng để chẩn đoán) không; có cần GPU cho nhánh train không.
4. Ngay bây giờ chưa có gì được commit/push; các file mới: `eval/audit_*.py`, thay đổi nhỏ ở `eval/evaluate.py`, `policies/act_infer.py`, `policies/act_train.py` (chỉ comment), tài liệu này và `assets/act-v1-audit/*.png`.
