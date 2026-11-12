# SuperMimic-OpenArm P1 trên Colab Tesla T4: bộ chạy tự động và kết quả benchmark

Ngày: 2026-11-12 · Trạng thái: **bộ chạy đã được chủ dự án chạy trên Colab T4 tới hết bước benchmark; kết quả NO-GO, pilot B0 KHÔNG được chạy.** Số đo ở mục 0 (lấy từ log terminal do chủ dự án cung cấp; các file `results_*` không nằm trong kho).

## 0. Kết quả thực tế trên Tesla T4 (chạy bởi chủ dự án, 2026-11-12) [đo trên VM thật; nguồn: log terminal]
Lệnh: `python3 scripts/colab_t4_run.py --benchmark-only`. Xác minh cục bộ OK (hash đầy đủ của gate `a7dc91ca…`, giao thức `126266d5…`, dataset `92b9e518…` khớp); phiên `openarm-test` cũ đã hết hạn nên driver tạo phiên mới (`colab new --gpu T4`); `nvidia-smi` xác nhận Tesla T4, 15.360 MiB; tải lên và băm lại trên VM đúng; hash gate/giao thức trên VM đúng.
| Chỉ số | Giá trị đo | Điều kiện | Đạt? |
|---|---|---|---|
| RAM VM | 12.975 MB tổng, 11.825 MB khả dụng | — | — |
| RAM mỗi worker env (PSS, thăm dò 2 worker) | 1.732,7 MiB | dự đoán cho 24 worker: **41,88 GiB** so với 11,27 GiB khả dụng (dự phòng 15%) | **Không** |
| VRAM đỉnh (`nvidia-smi`, gồm context CUDA, smoke chẩn đoán 2 env sync, batch/mạng/horizon đăng ký) | **811 MiB** / 15.360 MiB | ≤ 80% = 12.288 MiB | Đạt |
| Giây/iteration (smoke chẩn đoán) | **2,1375 s** | dự báo iteration trong 7200 s, đã tính checkpoint + eval + gate: **2.441** | **Không** (cần ≥ 4000) |
| Smoke | `smoke_ok: false` là chủ ý: chế độ chẩn đoán không bao giờ đạt điều kiện pilot | — | — |
Quyết định: **NO-GO ở benchmark**; benchmark 24 env bị bỏ qua có chủ ý (nguy cơ OOM làm sập runtime Colab); pilot không chạy; artifact xác minh đạt và phiên Colab đã được giải phóng. Config đăng ký không bị đổi. **Kết quả chỉ có giá trị cho T4, không xác nhận benchmark RTX 5090.** Hai nút thắt độc lập: RAM host (config đăng ký 24 worker) và tốc độ (≈2,1 s/iteration trên T4).

## 1. Bối cảnh: vì sao bộ chạy phải do chủ dự án chạy (lịch sử)
- Mình chạy trong sandbox cloud. Colab CLI **không** được cài/xác thực ở đây (cấu hình thăm dò trước đó mình đã xoá), và phiên Colab nằm trong tài khoản Google của chủ dự án trên máy của họ.
- Kiểm tra kết nối tới máy của chủ dự án: `list_computer_folders` trả về `{"folders":[],"cannot_be_used":[]}`, tức là chưa có thư mục nào được chia sẻ qua Remote Control cho session này.
- Mã P1 **chưa commit**, nên bản clone trên máy của chủ dự án không có nó; cách duy nhất để đưa nó lên Colab là đóng gói từ working tree (làm ở đây).
=> Mình đã dựng bộ chạy tự động để **một lệnh trên máy chạy Colab CLI** thực hiện đúng quy trình bạn mô tả. Nếu muốn trợ lý tự thao tác trên máy đó, cần chia sẻ một thư mục của máy cho session (Settings → Claude Code → Remote Control); khi đó mình có thể chạy cùng bộ này và theo dõi trực tiếp.

## 2. Gói và cách chạy
Gói: `outputs/p1_validation/colab/p1_t4_package.tar.gz` (15,3 MB; sha256 `fbfd3b102c39fedceff51cb52ec4001ce744331892817e9721866849577c6c92`), giải nén ra `p1_t4_package/`:
`p1_bundle.tar.gz` (mã, 1,6 MB, 115 file + manifest băm từng file; sha256 `5da8ce5055d570d26e087ec6f68765f0264b4d19f71ee458892a0edb76bfb4de`), `wm_eval_dataset_v1.npz` (13,7 MB, dataset kiểm định), `EXPECTED_HASHES.json`, `scripts/colab_t4_run.py`, `README.txt`.
```
tar xzf p1_t4_package.tar.gz && cd p1_t4_package
python3 scripts/colab_t4_run.py --verify-only        # chỉ kiểm tra cục bộ, không đụng Colab
python3 scripts/colab_t4_run.py --benchmark-only     # tới bước quyết định, không train
python3 scripts/colab_t4_run.py                      # đầy đủ: nếu đạt điều kiện thì chạy pilot B0 ≤ 2 h
```
Driver chỉ dùng thư viện chuẩn của Python và lệnh `colab` đã xác thực (phiên mặc định `openarm-test`; nếu phiên không còn thì tự `colab new -s openarm-test --gpu T4`).

### Những gì driver làm theo thứ tự (mỗi bước dừng ngay nếu sai)
1. **Kiểm tra cục bộ trước khi đụng Colab:** hash đầy đủ của gate `a7dc91ca66f6350d4798602aa86258a4c01777d9d123f2e724519499d24a8dca`, giao thức A/B `126266d5f4235249a1cfde2070fee2b86d89f1efd09c0d332abc1b335d5611f2` (lấy từ trong bundle) và dataset `92b9e5185a313e1095e49729c5a974bb6cf45fba4e6be975bad11d2561bf23c6` so với `EXPECTED_HASHES.json`; thiếu giá trị đối chiếu hoặc lệch → dừng. Liệt kê toàn bộ bundle (`bundle_listing.txt`), chặn `.venv`, `.pt/.npz`, `outputs/`, `openarm_wam`, `.git`, credential/oauth/token/khoá; kiểm băm từng file trong bundle.
   *Lưu ý:* hash đầy đủ của dataset chỉ được ghi trong báo cáo (không nằm trong file đăng ký nào); đã đối chiếu khớp hôm nay.
2. Phiên Colab: `colab sessions`; nếu thiếu thì tạo mới; `nvidia-smi` phải là Tesla T4.
3. Tải lên **chỉ hai file** (bundle + dataset) rồi băm lại trên VM; giải nén, băm lại gate/giao thức trên VM.
4. **Thăm dò RAM an toàn (2 worker)** trước khi chạy bất cứ thứ gì cần 24 tiến trình env (xem mục 3).
5. Smoke: CUDA thật (matmul 4096², UUID, PyTorch/CUDA, VRAM), 3 iteration huấn luyện thật với config và batch đã đăng ký, lưu và nạp lại checkpoint (so sánh trọng số), rollout eval MuJoCo, và **một lần đánh giá gate đầy đủ** để đo chi phí thật.
6. Benchmark `p1_benchmark` chọn theo UUID (trần 12 GiB VRAM, 300 s), chỉ khi bước 4 dự đoán đủ RAM.
7. **Quyết định** (tất cả phải đạt): smoke OK; hash khớp; VRAM đỉnh (`nvidia-smi`, gồm context CUDA) ≤ 80% của 15.360 MiB = 12.288 MiB; RAM host dự đoán đủ (dự phòng 15%) và đo thật không xuống dưới 1,5 GiB; số iteration dự báo trong 7200 s **≥ 4000**, mô hình chi phí `N·t_iter + ⌊N/200⌋·t_checkpoint + ⌊N/400⌋·(t_eval_trainer + t_gate) + khởi động ≤ 7200`. Không đạt → dừng ở benchmark, ghi `decision.json`.
8. Nếu GO: B0, 1 seed, **config đăng ký không override**. Trình giám sát trên VM tạm dừng huấn luyện (SIGSTOP) mỗi 400 iteration, sao bản `latest.pt` (kiểm sha256 với manifest), chạy eval gate trên CPU rồi `p1_gate_check`; STOP (1), INCOMPLETE/INCONSISTENT (2), lỗi eval hoặc hết ngân sách 7200 s tổng (gồm thời gian tạm dừng) → dừng. Sau khi dừng chạy một lần đánh giá cuối trên checkpoint đã lưu cuối cùng. Không đụng tập final 28.000.000+, không chạy B1–B3.
9. Driver theo dõi mỗi 60 s, **đồng bộ về máy chạy Colab CLI** (tar của metrics, log, checkpoint, kết quả gate) theo gate mới hoặc mỗi 10 phút, và đồng bộ đầy đủ khi kết thúc.
10. **Xác minh artifact trước khi giải phóng:** giải nén, kiểm sha256 checkpoint với manifest, đọc metrics, ghi `SHA256SUMS`; chỉ khi đạt mới `colab stop`. Lỗi bất kỳ → không giải phóng phiên, ghi `run_summary.json` có trường `error`.
Kết quả ở `results_<thời gian>/` (`run_summary.json`, `decision.json`, `smoke.json`, `memprobe.json`, `benchmark.json`, `vm_resources.txt`, `sync_*.tgz`, `SHA256SUMS`).

## 3. Phát hiện quan trọng: config đã đăng ký rất có thể không vừa RAM của VM T4
`configs/pushing_p1_pilot.yaml` có `env.num_envs: 24`, backend `subproc`, mà backend này dùng `spawn` (mỗi worker là một interpreter mới, nạp cả torch; các worker không chia sẻ bộ nhớ). **Đo trong sandbox này [đã đo]:** mỗi worker ≈ **1,68–1,75 GiB** bộ nhớ thực (PSS) → 24 worker + tiến trình chính ≈ **41–42 GiB**. Sandbox 15 GB RAM đã **bị kill (mã 137)** khi benchmark 24 env. VM Colab T4 thường có ≈ 12–13 GB RAM [giá trị điển hình; driver sẽ đo thật bằng `vm_resources.txt` và `memprobe.json`]. Con số này phụ thuộc phiên bản PyTorch/Python trên VM nên có thể thấp hơn nhưng không có lý do để tin sẽ giảm 3 lần.
Hệ quả: nếu thăm dò RAM trên VM cho kết quả tương tự, driver **không** chạy 24 env (để không OOM làm sập runtime Colab và mất phiên bạn đã tải file), chạy một **smoke chẩn đoán** (2 env sync; batch/mạng/horizon vẫn đăng ký) để vẫn đo phần GPU (VRAM đỉnh, thời gian cập nhật WM/actor), và kết thúc ở benchmark với **NO-GO và số đo**. Đây đúng quy tắc của bạn: không tự đổi config. Smoke chẩn đoán không bao giờ thoả điều kiện pilot.
**Quyết định cần chủ việc nếu muốn có pilot trên T4:** (a) VM Colab nhiều RAM hơn (G4/A100/H100 theo quota); hoặc (b) một **phiên bản giao thức mới** có `num_envs` nhỏ hơn / backend khác, với gate mới và hash mới (không phải sửa lén config hiện tại); hoặc (c) chạy pilot trên RTX 5090 của bạn (máy có RAM lớn, P0 của bạn đã chạy 64 env). Mình không thực hiện (b) khi chưa được duyệt.

## 4. Kiểm thử đã làm [đã đo] và giới hạn
- 7 test logic quyết định/xác minh (`test_t4_driver.py`) + 11 test hợp đồng giám sát (một test mới cho file thiếu/hỏng).
- **Self-test end-to-end với `colab` giả** (`tests/run_t4_driver_selftest.py`; máy hiện tại đóng vai VM, CPU, mini-dataset, trainer giả cho pilot): đi hết xác minh → tải lên + băm → cài đặt → thăm dò RAM (dự đoán ≈ 41,6 GiB, `fits=false`) → smoke chẩn đoán (đủ 5 bước OK trên CPU, gồm nạp lại checkpoint bằng nhau và một lần đánh giá gate) → bỏ qua benchmark 24 env → NO-GO → (nhánh pilot ép bằng cờ chỉ dành cho self-test) → gate tại iteration 400 → INCOMPLETE → dừng huấn luyện → đồng bộ cuối → xác minh artifact (19 file băm) → giải phóng phiên.
- **Bốn lỗi thật do kiểm thử phát hiện và đã sửa:** (1) khởi chạy pilot bằng `cd … && lệnh &` giữ kết nối SSH đến khi pilot xong (đã dùng subshell); (2) `p1_gate_check` văng exception khi thiếu file benchmark với mã 1 trùng mã STOP (nay là INCOMPLETE mã 2, và giám sát chỉ coi mã 1 là STOP khi có file kết quả); (3) driver không ghi lại lỗi khi văng exception (nay ghi `error`, không giải phóng phiên); (4) self-test ban đầu có thể xoá `/content` (nay chỉ xoá đúng các đường dẫn của nó và từ chối chạy nếu chúng đã tồn tại).
- **Chưa kiểm [chưa kiểm]:** mọi thứ chạy trên Colab thật: `colab ssh -- …` với lệnh nền dài, `pip install` mujoco/pyyaml trên Python của VM, tên GPU T4 và dung lượng RAM thực, thời gian eval gate trên 2 vCPU (rất có thể lớn: một lần gate ≈ 2 phút ở mini-dataset trên 4 lõi; với dataset đầy đủ và 2 lõi sẽ lâu hơn, và đã được tính trong quyết định), cơ chế timeout 30 s của CLI khi không có output (các bước dài chạy nền và thăm dò bằng file mã thoát).
- Kết quả trên T4 chỉ có giá trị cho pilot T4 và không xác nhận benchmark RTX 5090.

## 5. Việc còn lại (quyết định của chủ dự án)
1. Sao chép gói sang máy chạy Colab CLI và chạy `--verify-only`, rồi `--benchmark-only` (hoặc bản đầy đủ); gửi lại `results_*/decision.json`, `memprobe.json`, `vm_resources.txt` nếu muốn mình phân tích.
3. Quyết định hướng đi nếu thăm dò RAM chặn (mục 3).
4. Duyệt manifest v3 (37 đường dẫn) nếu muốn commit sau; vẫn chưa có commit/push nào.
