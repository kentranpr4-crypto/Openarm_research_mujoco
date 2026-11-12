# SuperMimic-OpenArm P1 Pushing: đóng gói, benchmark GPU, giảm thiểu khai thác mô hình, cổng giám sát, P0, PickPlace

Ngày: 2026-11-11 · Nhánh `main-f9g0uh` (HEAD `fcabee9`) · **Trạng thái: DONE cho phạm vi chuẩn bị; chưa chạy GPU, chưa train, chưa commit/push.** Tự kiểm (một người chạy), CPU sandbox 4 lõi, không có GPU.
Giữ nguyên: không train actor dài, không commit/push, không đụng OpenArm-WAM, G2a/G2b, `sim/` upstream, checkpoint/dataset hiện có, kiến trúc huấn luyện chính (mọi ablation là nhánh riêng).

> Nhãn: **[đã đo]** = chạy trong phiên này, file trong `outputs/p1_validation/`; **[suy luận]** = diễn giải chưa kiểm riêng; **[chưa kiểm]** = cần máy/GPU của chủ việc.

## 1. Rà soát thay đổi P1 và danh sách staged (Phase A)

**Rà soát [đã đo].** Tracked bị sửa: 2 file (`envs/openarm_env.py`, `envs/tasks.py`, +40/−7). `git grep` cho thấy **không file đã theo dõi nào nhập** mô-đun mới. Không đường dẫn `sim/`, `control/`, `eval/g2*`, `data/`, `checkpoints` nào đổi. `research/openarm_wam/` vẫn chưa theo dõi và không đụng tới. Ảnh hưởng P0: khoá `friction_mode`/`mass_scale` chỉ có trong cấu hình lấy mẫu của pushing, nên reaching/pick_place không đổi; test `test_p0_compat_...` băm quỹ đạo reaching + pick_place và **bằng bit với `git archive HEAD`**. Checkpoint và dataset: không file nào bị ghi (kiểm bằng `git status`; `outputs/` gitignored, chỉ thêm file mới).
**Test [đã đo]:** bộ test đầy đủ chạy lại ở cuối báo cáo (mục 9).

**Danh sách staged đề xuất cho commit chỉ-P1 (chưa stage; xin duyệt riêng):** file đầy đủ `outputs/p1_validation/p1_commit_manifest.txt`.
```
research/supermimic_openarm/envs/openarm_env.py        (M)
research/supermimic_openarm/envs/tasks.py              (M)
research/supermimic_openarm/baselines/push_expert.py
research/supermimic_openarm/configs/pushing_p1_pilot.yaml
research/supermimic_openarm/gates/p1_pilot_gates.yaml
research/supermimic_openarm/p1_validate_env.py
research/supermimic_openarm/p1_expert_eval.py
research/supermimic_openarm/p1_wm_eval.py
research/supermimic_openarm/p1_wm_ablate.py
research/supermimic_openarm/p1_actor_eval.py
research/supermimic_openarm/p1_gate_check.py
research/supermimic_openarm/p1_benchmark.py
research/supermimic_openarm/p1_pickplace_audit.py
research/supermimic_openarm/scripts/p1_pilot.sh
research/supermimic_openarm/scripts/p1_benchmark.sh
research/supermimic_openarm/tests/test_p1_env.py
docs/ai-native/tasks/2026-11-11-supermimic-p1-env-repair-report.md
docs/ai-native/tasks/2026-11-11-supermimic-p1-wm-mitigation-report.md
```
**Loại trừ có chủ đích:** `research/openarm_wam/`; `p0_final_eval.py`, `p0_seed_audit.py` và 3 tài liệu P0 (kế hoạch, aggregator, báo cáo readiness hỗn hợp P0/P1), để commit P0 riêng; `p1_wm_diagnostics.py` (đã thay bằng `p1_wm_eval.py`, giữ untracked hoặc xoá sau review); mọi `outputs/`, checkpoint, log, dataset.

## 2. Benchmark GPU có giới hạn (Phase B)

**Địa hình GPU thực tế [chưa kiểm]**: sandbox này không có `nvidia-smi`, nên mình không thể xem GPU/workload thật trên máy bạn. Hãy chạy (chỉ đọc) và dán kết quả:
```
cd <repo> && source .venv-supermimic/bin/activate     # hoặc venv của bạn
python -m research.supermimic_openarm.p1_benchmark --list
```
Lệnh in từng GPU (chỉ số, **UUID**, tên, bộ nhớ, tải, bus PCI), tiến trình compute trên mỗi GPU (pid, dòng lệnh, VRAM) và ma trận topo `nvidia-smi topo -m`. Chọn GPU **không** có tiến trình nào (không phải GPU đang chạy WAM).
**Lệnh benchmark (chọn theo UUID, không theo chỉ số):**
```
python -m research.supermimic_openarm.p1_benchmark --gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
    --budget-s 300 --num-envs 8 --steps 800 --updates 5 --max-vram-gb 8 \
    --out outputs/p1_validation/benchmark_<tên>.json
```
**Giới hạn tài nguyên (cứng):** ngân sách thời gian 300 s mặc định (trần 900 s; SIGTERM rồi SIGKILL **chỉ nhóm tiến trình của chính nó**); `nice -n 10`; trần VRAM mỗi tiến trình 8 GiB (4 GiB nếu `--allow-shared`); 8 env CPU; preflight chỉ đọc **từ chối** (exit 3, `RESOURCE_BLOCKED`) nếu GPU đó đã có tiến trình compute khác, tải > 30%, thiếu VRAM trống hoặc thiếu lõi CPU rảnh; ghi ra file mới, không ghi đè. **Không khởi chạy huấn luyện, không ghi checkpoint.**
**Đo được:** tốc độ mô phỏng thuần (1 env, expert, không mạng); env transitions/s kèm suy luận chính sách; độ trễ cập nhật WM và actor/critic (bước huấn luyện); VRAM đỉnh (allocator torch và `nvidia-smi` gồm ngữ cảnh CUDA); tải GPU (1 Hz); CPU (số lõi bận, từ tick `/proc` của cây tiến trình) và RSS đỉnh (tổng RSS cộng trùng bộ nhớ chia sẻ, đọc như cận trên).
**Kiểm tra đường đo [đã đo, KHÔNG phải bằng chứng GPU]:** selftest CPU (`--cpu-selftest`): mô phỏng thuần **201 bước điều khiển/s/lõi** (10.060 bước vật lý/s; 4,97 ms/bước); cập nhật WM 0,17 s và actor/critic 0,30 s với mạng thu nhỏ; 2,3 lõi bận trung bình. Các mã lỗi: UUID giả → exit 3; thiếu `--gpu-uuid` → exit 2; không tự chọn thiết bị.
**Launcher pilot cũng chọn theo UUID** (`p1_pilot.sh --gpu-uuid ...`, vẫn mặc định dry-run, trần 2 h). Lý do [đã đo từ mã]: thứ tự CUDA mặc định khác thứ tự `nvidia-smi`, nên `CUDA_VISIBLE_DEVICES=<chỉ số>` có thể đặt nhầm lên GPU của WAM. Sandbox: chặn đúng (`RESOURCE_BLOCKED`).

## 3. Phân tích lỗi World Model (Phase C)

Cấu hình: 8 nhánh huấn luyện trên **cùng** bộ dữ liệu proxy (54.000 chuyển tiếp, `wm_eval_dataset_v1.npz`), cùng 6000 cập nhật, mỗi lần đổi **một** yếu tố; baseline `base_s0/s1/s2` cho độ phân tán giữa các seed. Đánh giá trên 120 episode validation tách rời; tối ưu/gradient trên 24 trạng thái. File: `wm_ablation_v1.json`.

### 3.1 Vì sao hộp "trôi" trong vùng tự do [đã đo + suy luận]
Trên 3597 chuyển tiếp tự do (baseline):
- WM dự đoán hộp dịch **0,39 mm/bước** trong khi thực tế **0,12 mm/bước**; vận tốc hộp dự đoán 5,4 mm/s so với 1,9 mm/s.
- Độ dịch dự đoán **không phụ thuộc vào khoảng cách TCP–hộp** (0,394 mm khi TCP trong 15 cm; 0,345 mm khi xa hơn) và chỉ tương quan yếu với chuyển động khớp (r = 0,17). Nên **không phải rò rỉ từ cơ cấu chấp hành sang hộp**.
- Hướng dịch khá nhất quán (hệ số nhất quán hướng 0,72 trên thang 0–1): sai lệch có hệ thống, cộng dồn gần tuyến tính qua 16 bước → 4,3 mm ở h=16 (thực tế đứng yên: 0,45 mm).
- **Suy luận [chưa kiểm riêng]:** đầu ra phần dư của hộp có "sàn" sai số cố định; phương sai chuẩn hoá dùng chung cho cả vùng chạm (delta lớn) và vùng tự do (delta ≈ 0) nên sai số ~1% độ lệch chuẩn của vùng chạm đã là cỡ 0,4 mm/bước. Cách duy nhất đã kiểm là buộc hộp đứng yên khi không có tương tác (3.2).

### 3.2 Các ablation có kiểm soát (mm sai số vị trí hộp tại h=16; baseline đứng yên: free 0,45, contact 36–39)

| Nhánh | free | first_contact | sustained | contact_loss | orientation free (°) | Nhận xét |
|---|---|---|---|---|---|---|
| base s0 / s1 / s2 | 4,31 / 4,27 / 4,65 | 12,8 / 13,3 / 12,8 | 13,7 / 13,7 / 15,4 | 14,3 / 14,5 / 14,1 | 3,06 / 1,83 / 3,70 | phân tán seed ≈ ±0,2 (free), ±1,0 (sustained) |
| **gate** (hộp chỉ đổi khi pusher gần hoặc hộp đang chuyển động) | **0,79** | 14,5 | 14,2 | 15,8 | **0,69** | trôi tự do −82%; chạm kém hơn ~1–1,5 mm (nhỏ so với 36–39) |
| gate + bal | 0,95 | 13,6 | 14,9 | 15,8 | 0,74 | tương tự gate |
| K1 (loss 1 bước) | 5,80 | 18,6 | 16,6 | 18,9 | 2,75 | tệ hơn ở mọi chế độ |
| K16 (cửa sổ 16) | 4,72 | 15,3 | 14,6 | 17,7 | 2,47 | không tốt hơn K=8, chậm gấp 2,2 |
| bal (replay cân bằng chạm 50%) | 5,90 | 12,3 | 13,3 | 14,9 | 3,65 | chạm cải thiện nhẹ trong phân tán seed, tự do tệ hơn |

- **Cổng hộp theo chạm** là biện pháp duy nhất làm hết ghost motion và giảm orientation ở vùng tự do; chi phí chạm nhỏ. Đây là **đề xuất**: cần biến thể `WorldModel` tuỳ chọn (mặc định tắt) nếu được duyệt; **chưa** sửa kiến trúc chính.
- **Huấn luyện đa bước:** K=8 (hiện tại) là hợp lý; K=1 tệ hơn rõ, K=16 không có lợi.
- **Replay cân bằng chạm:** không cải thiện dự đoán chạm ngoài phân tán seed và làm xấu vùng tự do → **không đề xuất**.
- Khoảng lạc quan trên hành động ghi lại ≈ 0 ở mọi nhánh (|gap| ≤ 0,14, tương quan ≥ 0,986): **vấn đề không nằm ở dự đoán trên dữ liệu đã thấy**.

### 3.3 Khai thác mô hình khi tối ưu hành động và biện pháp giảm thiểu [đã đo, 24 trạng thái, so cặp]
Khoảng lạc quan = return tưởng tượng − return thật (16 bước, cùng snapshot). Tối ưu 16 hành động mở bằng Adam, khởi tạo từ hành động ghi lại.

| Biến thể (base s0) | Khoảng lạc quan | Lợi return thật vs ghi lại [CI95] | Tỉ lệ thật tệ hơn | Hành động bão hoà | Nằm ngoài phân phối hành động huấn luyện* | Bất đồng ensemble (ghi lại 3,19) |
|---|---|---|---|---|---|---|
| baseline (40 bước, lr 0,05) | **+8,43** [4,7; 12,0] | +2,71 [−1,7; +7,0] | 38% | 15% | 6% | 11,5 |
| horizon ngắn 8 | +11,41 | −1,27 | 58% | 24% | 17% | 12,5 |
| horizon ngắn 4 | +7,07 | −3,46 | 62% | 43% | 51% | 11,0 |
| vùng tin cậy ±0,1 / bước | +2,78 [1,2; 4,6] | **+1,70 [+0,6; +3,0]** | **17%** | 0% | 0% | 4,3 |
| bảo thủ (10 bước, lr 0,02, ±0,1) | +2,43 [0,8; 4,5] | +1,82 [+0,6; +3,2] | 17% | 0% | 0% | 4,2 |
| phạt bất đồng ensemble, λ=1 | +4,93 | +3,56 [−0,6; +7,7] | 29% | 4% | 1% | 1,4 |
| phạt bất đồng ensemble, λ=3 | +2,36 [1,1; 3,7] | **+3,15 [+1,5; +4,8]** | 38% | 1% | 0% | 0,7 |
| kết hợp (h8, 20 bước, ±0,1, λ=1) | **+1,19** [−0,2; +2,7] | +1,28 [+0,4; +2,2] | 25% | 0% | 0% | 2,3 |

*Khoảng Mahalanobis 99% của phân phối hành động huấn luyện (ngưỡng 7,68).
- **Horizon ngắn không giúp** (còn tệ hơn: hành động bão hoà và ra ngoài phân phối nhiều hơn).
- **Vùng tin cậy hành động và phạt bất đồng ensemble giảm khoảng lạc quan ~3–7 lần** và biến lợi ích thật thành dương có ý nghĩa (CI loại 0). Kết hợp đưa khoảng lạc quan về +1,2.
- **Hành động tối ưu có nằm trong phân phối huấn luyện không?** Theo biên độ từng bước thì **có** (chỉ 6% bước vượt ngưỡng 99%), nhưng **bất đồng ensemble tăng 3,6 lần** (3,19 → 11,5). Khai thác diễn ra ở các tổ hợp trạng thái–hành động chưa thấy, không phải ở hành động cực đoan. **[suy luận]**: kiểm tra phân phối hành động đơn lẻ là không đủ; dùng bất đồng ensemble.
- **Cổng hộp không giảm khai thác** (khoảng lạc quan +11,2 ở nhánh gate; vùng tin cậy → +4,1): sửa trôi tự do và giảm khai thác là hai vấn đề tách biệt.
- Hạn chế: 24 trạng thái, CI rộng, một seed mỗi nhánh; tối ưu mở loop là kiểm tra căng thẳng mạnh hơn huấn luyện actor thật.

### 3.4 Gradient qua động lực có khớp sai phân hữu hạn trên return thật MuJoCo không [đã đo]
Nhiễu loạn 4 bước × 7 khớp (chuẩn 0,5): 6 hướng ngẫu nhiên + ±hướng gradient WM, 24 trạng thái, 192 phép đo/nhánh, return thật 16 bước.
- 127–138 phép đo có thay đổi thật đo được (> 0,05).
- Dấu đồng ý (khi đo được): **0,71–0,78** (phi tuyến), 0,67–0,74 (gradient tuyến tính). Tương quan: 0,28–0,41.
- **Đi theo +gradient cho return thật cao hơn −gradient ở 83–88% số trạng thái** (mọi nhánh; ngẫu nhiên là 50%). Hướng của gradient có thông tin.
- **Độ lớn bị phóng đại**: WM dự đoán +2,3 đến +3,2, return thật chỉ +0,3 đến +1,0 (nhánh gate: −0,8). Đây chính là cơ chế của khoảng lạc quan (độ dốc quá lớn).
- **Hiệu chỉnh kết luận cũ:** báo cáo trước nêu "đồng ý dấu 50%, n=10". Đó là đo **dịch chuyển hộp**, chỉ 10 ca đo được, công suất thấp; phép đo mới (return thật, 135 ca) đáng tin hơn. Không nhánh nào cải thiện độ trung thực gradient vượt phân tán seed.

### 3.5 Khuyến nghị kỹ thuật (chưa thực hiện, cần duyệt)
1. Pilot đầu: **giữ kiến trúc hiện tại** (không đổi chính), chạy cổng giám sát (mục 4) để biết khai thác có xảy ra trong huấn luyện thật hay không.
2. Nếu G1 hoặc G3 gọi STOP: thử, như nhánh A/B riêng: (a) cổng hộp theo chạm (biến thể tuỳ chọn), (b) giới hạn thay đổi actor mỗi cập nhật và/hoặc phạt bất đồng ensemble 3 thành viên (chi phí WM ×3).
3. Chưa thử: sửa kiến trúc chính, cập nhật actor trong trainer. Cả hai cần mã mới và duyệt.

## 4. Cổng giám sát đăng ký trước (Phase D)

File: `research/supermimic_openarm/gates/p1_pilot_gates.yaml` (**sha256 `a7dc91ca66f6350d4798602aa86258a4c01777d9d123f2e724519499d24a8dca`**, ghi lại trước khi pilot, không sửa sau khi có kết quả). Công cụ: `p1_gate_check.py` (đọc JSON đo, in PASS/WARN/STOP/NOT_AVAILABLE và OVERALL; không sửa ngưỡng). Tần suất: mỗi 400 iteration trên **bản sao** checkpoint mới nhất (CPU, ~5 phút) và một lần cuối. STOP = dừng pilot, giữ mọi sản phẩm, báo cáo; không tự khởi động lại, không đổi ngưỡng. "Liên tiếp" = k lần đo cuối đều vượt ngưỡng STOP, nếu không chỉ là WARN.

| Cổng | Định nghĩa chỉ số | Dữ liệu kiểm định | WARN / STOP | Chính sách | Căn cứ |
|---|---|---|---|---|---|
| G1 tự do | sai số vị trí hộp trung bình (mm) sau 16 bước mở, cửa sổ không chạm | val cố định 120 episode (`wm_eval_dataset_v1.npz`, seed 25.100.000+) | > 5,0 / > 8,0, 2 lần liên tiếp | STOP | 8 mm = 27% dung sai 30 mm; baseline chưa cổng 4,3–4,7 mm; có cổng 0,79 |
| G2 chạm | max tỉ lệ (sai số WM / đứng yên) ở h=8, first_contact và sustained | cùng val | > 0,6 / > 0,8, 2 lần | STOP | đo được 0,39 và 0,37; ô n < 30 bị bỏ |
| G3 khoảng return | trung bình (tưởng tượng − thật), 16 bước, trên **quỹ đạo của chính actor** | 32 goal held-out 27.300.000+ | > 1,0 / > 3,0, 2 lần | STOP | ghi lại ≈ 0,05; khai thác 8–11; 3,0 ≈ 1/3 mức khai thác, ~12% return expert 16 bước |
| G4 thành công | strict success trên goal held-out | 64 goal 27.000.000+ (`p1_actor_eval`) | giữa chừng (≥ 50% ngân sách): STOP nếu < 5%; cuối: đề xuất chạy dài chỉ nếu ≥ 40% và Wilson dưới ≥ 25% | STOP / đề xuất | expert 86,7%; chính sách vô dụng ≈ 0% |
| G5 chất lượng gradient | tỉ lệ trạng thái có return thật cao hơn khi đi +gradient so với −gradient | ≥ 24 trạng thái val (`--grad-states 24`) | < 0,65 / < 0,45, 2 lần | STOP | ngẫu nhiên 0,5; đo được 0,83–0,88 |
| G6 an toàn hành động | vi phạm cứng = ngoài ctrl range + slew vượt + không hữu hạn + tay phải đổi (phải = 0); tỉ lệ episode hộp rơi bàn; xuyên thấu robot–hộp nhỏ nhất | 64 episode của G4 | hộp rơi > 10%; xuyên < −4 mm (cảnh báo −2 mm); vi phạm cứng > 0 | STOP ngay lần đầu | giới hạn do adapter gốc thực thi nên vi phạm = lỗi phần mềm; −2 mm là ngưỡng V3c |
| G7 GPU | VRAM đỉnh torch (GiB), tiến trình khác xuất hiện trên GPU, thời gian, số iteration vừa trong 2 h | `p1_benchmark` trên đúng UUID, ngay trước pilot | VRAM > 16 STOP; < 4000 iteration trong 2 h → giảm kế hoạch **trước** khi chạy | STOP / giảm kế hoạch | trần cấu hình 20 GiB; chừa dư trên thẻ 32 GB; tiến trình khác trên GPU là điều kiện chặn khởi chạy |

Lệnh đo cho cổng (mẫu, trên bản sao checkpoint `CK`):
```
python -m research.supermimic_openarm.p1_wm_eval --out outputs/p1_gates/wm_it<N>.json --checkpoint CK --dataset outputs/p1_validation/wm_eval_dataset_v1.npz --actor-episodes 32 --grad-states 24
python -m research.supermimic_openarm.p1_actor_eval --checkpoint CK --episodes 64 --out outputs/p1_gates/actor_it<N>.json
python -m research.supermimic_openarm.p1_gate_check --wm-eval outputs/p1_gates/wm_it*.json --actor-eval outputs/p1_gates/actor_it<N>.json --benchmark outputs/p1_validation/benchmark_<tên>.json --iteration <N> --budget-iterations 8000
```
Đã chạy thử đường này trên dữ liệu proxy và checkpoint smoke: các cổng cho kết quả đúng định dạng; `G7` coi file selftest CPU là NOT_AVAILABLE (không phải bằng chứng GPU).

## 5. Đánh giá cuối P0 (Phase E)

Mình **không retrain P0**, chỉ chuẩn bị lệnh; checkpoint chỉ đọc. Hãy chạy trên máy có checkpoint thật (`ROOT` chứa `reaching_long_s1000/seed1000` ...), tốt nhất trên bản sao:
```
python -m research.supermimic_openarm.p0_seed_audit \
    --run-dir $ROOT/reaching_long_s1000/seed1000 --run-dir $ROOT/reaching_long_s1001/seed1001 --run-dir $ROOT/reaching_long_s1002/seed1002 \
    --out outputs/p0_final/seed_audit.json
python -m research.supermimic_openarm.p0_final_eval --run-root $ROOT --seeds 1000 1001 1002 --episodes 100 --seed-base 9900000 \
    --ckpts best.pt latest.pt --out outputs/p0_final/eval_<ngày>        # thêm --save-traj nếu cần quỹ đạo
```
`p0_final_eval` tự áp dụng ngưỡng đăng ký trước và ghi `results.json` + `summary.md` (verdict PASS/PARTIAL/FAIL); không cần aggregator riêng (aggregator của kế hoạch P0 dành cho đầu ra của `research.supermimic_openarm.evaluate`).
Goal mới: seed 9.900.000–9.900.099, **cùng 100 goal** cho mọi checkpoint (so cặp best vs latest, McNemar chính xác); không trùng chọn `best.pt` (9.000.000+), đánh giá trực quan (9.800.000+) hay huấn luyện. Ngưỡng đăng ký trước đã cố định trong kế hoạch P0 (mỗi `best.pt` ≥ 95/100; gộp ≥ 0,97; sai số cuối trung vị ≤ 5 mm; ...).

**Kiểm toán độc lập giữa 3 seed huấn luyện [đã đo từ mã; xác nhận trên config thật bằng `p0_seed_audit`]:**
- Seed episode huấn luyện = `episode_seed(train_seed_base, env_idx, ep_idx)` (`envs/vector.py`), **không phụ thuộc `experiment.seed`**. Nếu ba run dùng cùng `train_seed_base` (như kết luận trước từ cấu hình của bạn) thì **chuỗi vị trí hộp/goal huấn luyện giống hệt nhau** theo từng cặp (env, episode): 100% seed trùng (đã kiểm trên hai run debug cùng cấu hình: 40/40).
- `experiment.seed` (1000/1001/1002) chỉ điều khiển khởi tạo mạng, nhiễu thăm dò, **hành động ngẫu nhiên khởi động** (hành động khác nhưng điểm bắt đầu/goal giống nhau) và thứ tự lấy mẫu replay (`Replay(seed=experiment.seed)`).
- Dữ liệu replay khởi đầu: không có bộ dữ liệu bên ngoài; bộ đệm khởi động bằng thu thập ngẫu nhiên trong run, nên **giống nhau về trạng thái bắt đầu, khác nhau về hành động**.
- Chọn `best.pt` dùng chung `eval_seed_base`.
- **Kết luận:** ba seed **độc lập một phần** (độc lập về tối ưu hoá, **không** độc lập về phân phối huấn luyện và chọn checkpoint). Chúng ước lượng phương sai tối ưu hoá, không phải độ nhạy với phân phối huấn luyện. Khi báo cáo tổng, không gộp như ba bản sao độc lập; khẳng định tổng quát chỉ dựa vào 100 goal held-out. Kiểm tra xác nhận: `grep train_seed_base $ROOT/*/seed*/config.resolved.json` (hoặc chạy `p0_seed_audit`).

## 6. Kiểm toán PickPlace (Phase F, chỉ đọc, không áp dụng sửa lỗi pushing)

Công cụ `p1_pickplace_audit.py`; kết quả `pickplace_audit.json`. Hệ số ma sát hiệu dụng đọc từ **contact thật** (`d.contact[i].friction`) và quy tắc kết hợp của MuJoCo.
- **Box–desk và box–tray: khiếm khuyết được xác nhận.** Cả hai cặp có `priority` bằng nhau nên dùng `max(0,8; 0,8×scale)`. Với dải scale của PickPlace [0,8; 1,2]: μ hiệu dụng 0,8–0,96 thay vì 0,64–0,96 dự định; **scale ≤ 1,0 (một nửa dải) không có tác dụng** (đo trên contact: 0,8 ở scale 0,8 và 1,0; 0,96 ở 1,2).
- **Ma sát nắm (ngón tay–hộp): không hề được ngẫu nhiên hoá.** Ngón tay có `priority=1` > hộp → luôn dùng μ = 1,0 của ngón (quy tắc "geom2"); phổ biến thiên trên toàn dải scale = 0,0. Đây là ma sát quan trọng nhất cho gắp-đặt.
- Hộp–trụ đỏ / khối xanh: cùng quy tắc `max`.
- **Đề xuất (chưa áp dụng, cần quyết định của bạn vì đổi đặc tả nhiệm vụ):** (1) phần trượt: dùng `priority` như pushing; (2) phần nắm: chỉ có hiệu lực nếu ma sát ngón (hoặc `priority`) được ngẫu nhiên hoá có chủ đích. Đặt `priority` hộp = 2 sẽ làm ma sát nắm thành 0,64–0,96 (**thấp hơn** mức 1,0 hiện tại) và có thể giảm tỉ lệ gắp thành công, nên cần đo lại expert trước. PickPlace **không bị sửa** trong thay đổi này.

## 7. Bảng AC

| AC | Nội dung | Trạng thái | Evidence |
|---|---|---|---|
| A1–A2 | rà soát diff; P0/checkpoint/G2/upstream/WAM không bị ảnh hưởng | PASS | `git grep`, `git status`, test P0 bit-identical |
| A3 | env và expert qua test hồi quy | PASS | mục 9 |
| A4–A6 | danh sách staged P1-only; không commit; loại WAM/outputs | PASS | manifest; chưa stage |
| B1 | địa hình + workload GPU thực | NOT_CHECKED | cần `p1_benchmark --list` trên máy bạn |
| B2–B4 | lệnh theo UUID; giới hạn; đo các đại lượng | PASS (công cụ) / NOT_CHECKED (số đo GPU) | selftest CPU |
| B5–B6 | không can thiệp job khác; không tự khởi chạy pilot | PASS | preflight từ chối; không gọi trainer |
| C1 | nguyên nhân trôi tự do | PASS | 3.1 (suy luận phần cơ chế) |
| C2 | cổng hộp theo chạm | PASS | 3.2: −82% trôi tự do |
| C3 | huấn luyện đa bước / replay cân bằng | PASS | K1/K16/bal; không có lợi |
| C4 | actor bảo thủ / phạt bất định / horizon ngắn | PASS | 3.3 |
| C5 | hành động tối ưu có nằm trong phân phối huấn luyện | PASS | 6% ngoài; bất đồng ensemble ×3,6 |
| C6 | gradient vs sai phân hữu hạn trên return thật | PASS | 3.4 |
| D | cổng giám sát đăng ký trước | PASS | YAML + sha256 + checker |
| E | lệnh P0 + kiểm toán độc lập seed | PASS (lệnh, từ mã) / NOT_CHECKED (chạy P0 thật) | mục 5 |
| F | kiểm toán PickPlace độc lập | PASS | mục 6 |

## 8. GO / NO-GO cho pilot actor P1 ngắn

**GO CÓ ĐIỀU KIỆN, chỉ như một thí nghiệm đo (kiến trúc hiện tại, không giảm thiểu), 1 seed, ≤ 2 h.** Lý do: môi trường (11/11) và expert (86,7%) đã sẵn sàng; WM tốt khi chạm và gradient có hướng đúng (83–88%); khai thác mô hình là rủi ro **đã được định lượng** và có các biện pháp đã thử, nên pilot là cách đúng để biết nó xảy ra thật hay không, với cổng STOP đăng ký trước. **NO-GO** cho multi-seed hoặc nhiều giờ.
Điều kiện (đều cần chủ việc):
1. Chạy `p1_benchmark --list` rồi `p1_benchmark --gpu-uuid ...` trên máy bạn; GPU đã chọn không có tiến trình nào; số iteration vừa trần 2 h (≥ 4000).
2. Duyệt file cổng (sha256 ở mục 4) và ghi lại hash trước khi chạy.
3. Duyệt riêng việc chạy GPU; `p1_pilot.sh --gpu-uuid ... --execute` chỉ sau đó.
4. Quyết định có làm nhánh giảm thiểu (cổng hộp, ràng buộc cập nhật, phạt ensemble) hay không; chúng cần mã mới trong WM/trainer (chưa làm).
Rủi ro còn lại: WM đánh giá là proxy; khai thác có thể xảy ra trong pilot (gate sẽ STOP); orientation khi chạm chưa tốt hơn baseline.

## 9. Hồi quy cuối và giới hạn

**Test cuối [đã đo]: 54/54 pass** (44 cũ + 10 mới). Một lần chạy trước đó có 1 test hỏng (`test_all_shipped_configs_validate`) vì file cổng YAML nằm trong `configs/` bị quét như một cấu hình huấn luyện; đã chuyển sang thư mục riêng `gates/` (nội dung và hash không đổi) và test qua. Hạn chế: không có GPU; WM proxy; 24 trạng thái cho tối ưu và gradient; một seed mỗi nhánh ablation (dùng phân tán 3 seed baseline làm thước đo); tự kiểm.
