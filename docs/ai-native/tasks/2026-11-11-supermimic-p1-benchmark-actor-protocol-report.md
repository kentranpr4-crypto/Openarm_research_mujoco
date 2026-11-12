# SuperMimic-OpenArm P1: đóng gói nguồn, hợp đồng giám sát, benchmark GPU, thiết kế thí nghiệm actor B0–B3

Ngày: 2026-11-11 · Nhánh `main-f9g0uh` (HEAD `fcabee9`) · **Trạng thái: DONE cho phạm vi chuẩn bị; chưa chạy GPU, chưa huấn luyện actor, chưa commit/push.** Tự kiểm (một người chạy). Sandbox cloud 4 CPU, không GPU.
Giữ nguyên: không train actor, không commit/push, không đụng `research/openarm_wam/`, G2a/G2b, ACT, `sim/` upstream, checkpoint/dataset hiện có, vật lý PickPlace. B0 vẫn là trainer gốc.

> Nhãn: **[đã đo]** = chạy trong phiên này; **[từ mã]** = đọc code; **[chưa kiểm]** = cần máy/GPU/tài khoản của bạn.

## 1. Rà soát nguồn và manifest commit P1 (STEP 1)

**Manifest** (`outputs/p1_validation/p1_commit_manifest.txt`, phiên bản 2): **27 đường dẫn**, chưa stage (`git diff --cached` rỗng), không dùng `git add -A`.
```
M  research/supermimic_openarm/envs/openarm_env.py
M  research/supermimic_openarm/envs/tasks.py
   research/supermimic_openarm/baselines/push_expert.py
   research/supermimic_openarm/configs/pushing_p1_pilot.yaml
   research/supermimic_openarm/gates/p1_pilot_gates.yaml
   research/supermimic_openarm/experiments/p1_actor_ab_protocol.yaml
   research/supermimic_openarm/p1_validate_env.py          p1_expert_eval.py     p1_wm_eval.py      p1_wm_ablate.py
   research/supermimic_openarm/p1_actor_eval.py            p1_gate_check.py      p1_run_summary.py  p1_pickplace_audit.py
   research/supermimic_openarm/p1_benchmark.py             p1_variants.py        p1_train_variant.py
   research/supermimic_openarm/scripts/p1_benchmark.sh     p1_pilot.sh           make_colab_bundle.py   colab_p1_bootstrap.py
   research/supermimic_openarm/tests/test_p1_env.py        test_p1_gate_check.py test_p1_variants.py
   docs/ai-native/tasks/2026-11-11-supermimic-p1-env-repair-report.md
   docs/ai-native/tasks/2026-11-11-supermimic-p1-wm-mitigation-report.md
   docs/ai-native/tasks/2026-11-11-supermimic-p1-benchmark-actor-protocol-report.md
```
Loại trừ có chủ đích: `research/openarm_wam/`; `p0_final_eval.py`, `p0_seed_audit.py` và 3 tài liệu P0; `p1_wm_diagnostics.py` (đã bị `p1_wm_eval.py` thay); `docs/.../pickplace-friction-issue.md` (issue riêng); mọi `outputs/`, checkpoint, log, dataset, bundle.

**Diff đã theo dõi [đã đo]:** 2 file, +40/−7 (`tasks.py` +32/−4, `openarm_env.py` +8/−3; `git diff --numstat`). File mới: 25 trong manifest, ≈1.615 dòng nguồn/test (không kể 3 báo cáo). Không có tệp nào khác bị sửa.

**Kiểm tra bắt buộc:**
| Kiểm tra | Kết quả | Evidence |
|---|---|---|
| Không hồi quy hành vi P0 | PASS | test `test_p0_compat...`: quỹ đạo reaching + pick_place bằng bit với `git archive HEAD`; khoá `friction_mode`/`mass_scale` chỉ có trong lấy mẫu pushing |
| Không đổi ACT, G2a/G2b, dữ liệu được bảo vệ | PASS | `git status`/`git diff --name-only`: không đường dẫn `sim/`, `control/`, `eval/g2*`, `data/`, ACT nào đổi; không file đã theo dõi nào nhập mã mới (`git grep`) |
| Không đổi `research/openarm_wam/` | PASS | vẫn `??` (chưa theo dõi), không diff |
| Không có checkpoint/dataset/output trong manifest | PASS | quét manifest: không `outputs/`, `.pt`, `.npz`, `__pycache__`, `.venv`; `outputs/` được gitignore |
| Không có thông tin cá nhân/bí mật/đường dẫn tuyệt đối | PASS | quét email, `/home/user`, `/root`, token/password/khoá: không khớp |
| Tĩnh (pyflakes) | PASS | không tên chưa định nghĩa; chỉ còn `import os` thừa có từ trước trong `openarm_env.py` (không phải của thay đổi này) |
| Tái hiện 54/54 | PASS | 72/72 đạt (54 + 18 mới), mục 2 |

Hai lỗi thật tìm thấy khi rà soát và đã sửa trong file mới: (a) `make_colab_bundle.py` ban đầu kéo theo `sim/.venv` (84 MB, 4785 file) → loại mọi `.venv*`, còn 1,6 MB/101 file; (b) trường `config` của test cũ quét `configs/*.yaml` làm hỏng khi file cổng nằm đó (đã chuyển sang `gates/` ở lượt trước).

## 2. Kết quả test

**[đã đo] `pytest research/supermimic_openarm/tests` = 72 passed, 0 failed (97,9 s)** — tái hiện đủ 54/54 trước đó và thêm 18 test mới. Gồm: 44 test gốc (core + pipeline), 10 test môi trường P1, 10 test hợp đồng giám sát, 8 test biến thể (B0 = `train.py` byte-giống nhau, B1/B2/B3, checkpoint, giao thức A/B). Lưu ý môi trường: sandbox dùng Python 3.13 / torch 2.11 (cu130); lock của dự án là Python 3.10 / torch 2.9.1, máy bạn Python 3.12 / torch 2.9.1. **[chưa kiểm]** trên môi trường của bạn: chạy lại `python -m pytest research/supermimic_openarm/tests -q` ở đó.

## 3. Hợp đồng giám sát (STEP 2)

**SHA256 của `gates/p1_pilot_gates.yaml` = `a7dc91ca66f6350d4798602aa86258a4c01777d9d123f2e724519499d24a8dca` — KHỚP giá trị đăng ký** (tính bằng `sha256sum` và trong test). Không đổi ngưỡng nào; file YAML không bị sửa trong lượt này.

| Gate | Chỉ số / dữ liệu | WARN / STOP | Tần suất |
|---|---|---|---|
| G1 tự do | sai số hộp WM (mm), h=16, cửa sổ không chạm; val cố định 120 episode (sha dataset `92b9e5185a313e1095e49729c5a974bb6cf45fba4e6be975bad11d2561bf23c6`) | > 5,0 / > 8,0 (2 lần liên tiếp) | mỗi 400 iter + cuối |
| G2 chạm | max tỉ lệ WM/đứng yên, h=8, first_contact & sustained | > 0,6 / > 0,8 (2 lần) | như trên |
| G3 khoảng return | imagined − real (16 bước) trên quỹ đạo của chính actor, 32 goal 27.300.000+ | > 1,0 / > 3,0 (2 lần) | như trên |
| G4 thành công | strict success, 64 goal 27.000.000+ | giữa chừng (≥ 50%) STOP nếu < 5%; cuối: PASS nếu ≥ 40% và Wilson dưới ≥ 25% | như trên |
| G5 gradient | tỉ lệ trạng thái có return thật cao hơn theo +grad so với −grad, ≥ 24 trạng thái | < 0,65 / < 0,45 (2 lần) | như trên |
| G6 an toàn | vi phạm cứng = 0; tỉ lệ episode terminated; xuyên thấu | hộp rơi > 10% hoặc xuyên < −4 mm STOP (cảnh báo −2 mm); vi phạm cứng STOP ngay | như trên |
| G7 GPU | VRAM đỉnh, thời gian, số iteration vừa 2 h (benchmark đúng UUID) | VRAM > 16 GiB STOP; < 4000 iteration/2 h → giảm kế hoạch trước khi chạy | trước pilot |

**Hành vi khi thiếu/NaN/ít mẫu — phát hiện lỗi và sửa (chỉ ở bộ kiểm tra, không đổi ngưỡng):**
- Trước: một giá trị **NaN bị coi là PASS** (so sánh với NaN luôn sai) và thiếu dữ liệu chỉ in "CONTINUE (some gates NOT_AVAILABLE)".
- Sau (`p1_gate_check.py`, protocol v1.0 + làm rõ ở mức công cụ): NaN/inf → **STOP ngay** (không có "2 lần liên tiếp"); thiếu chỉ số hoặc ít mẫu (G1 < 100 cửa sổ, G2 < 30/ô, G3/G4/G5 dưới `min_*` của YAML) → `NOT_AVAILABLE` và toàn cục **INCOMPLETE (exit 2)**, không được tiếp tục trừ khi nêu tên `--allow-missing`; file benchmark selftest CPU không tính là bằng chứng GPU. Hash giao thức khác giá trị đăng ký → từ chối (exit 2) trừ khi có `--protocol-ack <sha>` (buộc phải ghi tài liệu phiên bản mới); thiếu/thừa gate → `MALFORMED`. 10 test bao phủ các nhánh này **[đã đo]**.
- **Chọn checkpoint:** gate chạy trên **bản sao `latest.pt`** tại iteration 400, 800, …; `best.pt` (chọn theo return trên goal 9.000.000+) chỉ mô tả. Mọi file đo phải cùng sha256 checkpoint và cùng dataset (dataset hash giờ được ghi trong `p1_wm_eval`), và `--iteration` phải khớp; sai → INCONSISTENT → INCOMPLETE. (File `wm_eval_v1.json` cũ không có hash dataset nên bị coi là không nhất quán; hãy chạy lại baseline bằng bản mới.)
- **terminated vs truncated:** `p1_actor_eval` nay ghi `episode_ends` (terminated do hộp rơi bàn/không hữu hạn, truncated tại 150 bước) và `success_reached_but_not_held_at_the_end` (success của môi trường là cờ "đã giữ 5 bước liên tiếp", không bị hủy nếu sau đó hộp rời đích). G6 đếm mọi episode terminated (không chỉ khi chưa thành công). Trong G3 chỉ dùng cửa sổ 16 bước nằm trọn trong episode, nên không bootstrap qua điểm kết thúc; `p1_wm_eval` ghi số episode terminated/truncated của actor.
- Hạn chế: tần suất "mỗi 400 iteration" chỉ nằm trong chú thích đầu YAML (không máy đọc được); bộ kiểm tra có hằng `INTERVAL = 400` và in cảnh báo lệch lịch, nhưng việc chạy đúng lịch là quy trình của người vận hành.

## 4. Gói benchmark GPU (STEP 3)

Mọi lệnh đều chỉ benchmark: **không** khởi chạy actor pilot (`starts_training: false` trong JSON). Giới hạn cứng: tối đa 300 s (trần công cụ 900 s), VRAM 8 GiB, `nice -n 10`, từ chối GPU đã có tiến trình khác, SIGTERM→SIGKILL chỉ nhóm tiến trình của chính nó, ghi ra file mới.

**A. Trên máy RTX 5090 của bạn** (chỉ đọc cho đến lệnh benchmark):
```
cd <repo> && source <venv-có-torch-và-mujoco>/bin/activate
nvidia-smi -L                                              # 1. liệt kê GPU và UUID vật lý
python -m research.supermimic_openarm.p1_benchmark --list  # 2. UUID, tải, bộ nhớ, tiến trình compute (pid, dòng lệnh), ma trận topo
python -m research.supermimic_openarm.p1_benchmark --pick-idle "RTX 5090"   # 3. in UUID của GPU 5090 RẢNH (không tiến trình, tải <= 30%, đủ VRAM); thoát 3 nếu không có
UUID=$(python -m research.supermimic_openarm.p1_benchmark --pick-idle "RTX 5090" | head -1 | cut -f1)    # chỉ dùng nếu đúng MỘT dòng hợp lý; nếu nhiều thẻ, tự chọn UUID từ bước 2
python -m research.supermimic_openarm.p1_benchmark --gpu-uuid "$UUID" --budget-s 300 --num-envs 8 --steps 800 --updates 5 --max-vram-gb 8 \
       --out outputs/p1_validation/benchmark_rtx5090_$(date +%Y%m%dT%H%M).json
```
Đầu ra gồm: tốc độ mô phỏng thuần (bước điều khiển/s, bước vật lý/s), env transitions/s kèm suy luận chính sách, độ trễ cập nhật WM và actor/critic (bước huấn luyện), VRAM đỉnh (allocator torch và `nvidia-smi`), tải GPU (1 Hz), số lõi CPU bận và RSS đỉnh, thời gian tường, và **`provenance`** (commit, số file chưa commit, sha256 của config/cổng/công cụ/trainer, phiên bản python/torch/mujoco/numpy, bản build CUDA, driver/tên/UUID/compute capability). **Chọn theo UUID, không theo chỉ số**, vì thứ tự CUDA mặc định khác `nvidia-smi` (rủi ro đặt nhầm lên GPU của WAM).
Kiểm tra đã làm **[đã đo, KHÔNG phải bằng chứng GPU]**: `--list`/`--pick-idle`/UUID giả → thoát 3 đúng; selftest CPU: mô phỏng thuần ≈190–201 bước điều khiển/s/lõi, 5,3 ms/bước, cập nhật WM 0,18 s, actor/critic 0,30 s (mạng thu nhỏ). **Số đo trên 5090: [chưa kiểm].**

**B. Route Colab (`github.com/googlecolab/google-colab-cli`) — GPU của Colab, không phải 5090.** Những gì mình xác minh được **[đã đo]**: README của repo đọc được; CLI cài được bằng `uv tool install --python 3.13 google-colab-cli` (bản 0.8.0; **cần Python ≥ 3.12**; bản 0.7.3 trên PyPI bị yanked "bad version pin" nên cài bằng Python 3.10 sẽ lỗi); các lệnh `colab new --gpu {T4,L4,G4,H100,A100}`, `upload`, `exec -f … --env K=V --timeout`, `download`, `stop` tồn tại. **Mình KHÔNG xác thực và KHÔNG tải gì lên**: CLI bắt đầu luồng OAuth bằng tài khoản Google của bạn (nó in ra một URL xác thực mà mình không mở), và tải lên nghĩa là gửi mã dự án tới VM Google Colab dưới tài khoản của bạn — việc đó do bạn quyết định và thực hiện. Mình đã đóng gói và thử cục bộ (CPU selftest, không mạng): bundle chỉ chứa mã (1,6 MB, 101 file, có manifest băm từng file), script bootstrap giải nén, kiểm băm, clone upstream ở SHA ghim, cài gói thiếu, chọn GPU rảnh theo UUID, chạy `p1_benchmark` với trần 300 s / 8 GiB.
```
uv tool install --python 3.13 google-colab-cli            # một lần; lần đầu dùng sẽ mở OAuth trong trình duyệt CỦA BẠN
python research/supermimic_openarm/scripts/make_colab_bundle.py --out outputs/p1_validation/colab/p1_bundle.tar.gz
colab new -s p1bench --gpu G4                              # hoặc L4 / T4 / A100 / H100 (theo quota); `colab usage` xem đơn vị tính toán
colab upload -s p1bench outputs/p1_validation/colab/p1_bundle.tar.gz /content/p1_bundle.tar.gz
colab exec  -s p1bench -f research/supermimic_openarm/scripts/colab_p1_bootstrap.py --timeout 0 --env P1_BUDGET_S=300 --env P1_NUM_ENVS=2 --env P1_MAX_VRAM_GB=8
colab download -s p1bench /content/p1/outputs/p1_validation/colab_benchmark.json ./colab_benchmark.json
colab download -s p1bench /content/p1/outputs/p1_validation/colab_run_info.json  ./colab_run_info.json
colab stop -s p1bench                                      # giải phóng VM
```
Giới hạn của route Colab: (1) GPU liệt kê là T4/L4/G4/H100/A100, **không có RTX 5090**, nên kết quả **không dùng cho gate G7 hay chốt số iteration của pilot trên 5090** (checker coi file này không phải bằng chứng nếu nhãn là selftest; với Colab bạn phải chủ động không đưa nó vào `--benchmark` của gate); (2) VM Colab ít lõi CPU nên tốc độ env không đại diện; (3) giá trị là xác nhận chức năng đường CUDA và bậc độ lớn VRAM/thời gian bước huấn luyện; (4) `colab exec` mặc định timeout 30 s không output, nên bắt buộc `--timeout 0`; (5) các bước cài gói/EGL/`pip` trên VM thật **[chưa kiểm]**.

## 5. Thiết kế thí nghiệm actor B0–B3 (STEP 4)

**Hiện thực (opt-in, B0 không đổi) [đã đo bằng test đơn vị, không chạy vòng huấn luyện]:** `p1_variants.py` (lớp con `P1VariantTrainer`) và `p1_train_variant.py` (`--variant b0` chuyển nguyên xi sang `train.py`; test xác nhận đầu ra dry-run **giống từng byte**). Trainer gốc không biết đến biến thể; không file đã theo dõi nào bị sửa cho việc này.

| | B0 | B1 | B2 | B3 |
|---|---|---|---|---|
| Thay đổi duy nhất | trainer gốc | vùng tin cậy hành động ±ρ quanh bản EMA của actor (chỉ trong imagination), ρ=0,1, EMA 0,02 | ensemble 3 WM; reward tưởng tượng = trung bình thành viên − λ·100·‖std vị trí hộp‖, λ=3 | box dynamics có cổng theo khoảng cách pusher/vận tốc hộp |
| Lệnh | `p1_train_variant --variant b0` | `--variant b1 --variant-arg rho=0.1 --variant-arg ema=0.02` | `--variant b2 --variant-arg lam=3.0 --variant-arg members=3` | `--variant b3` |
| Bằng chứng proxy | cơ sở | lạc quan +8,4 → +2,8; lợi return thật +1,7 [0,6; 3,0] | lạc quan → +2,4; lợi +3,2 [1,5; 4,8]; bất đồng 11,5 → 0,7 | trôi tự do 4,3 → 0,79 mm; **không** giảm khai thác |
| Chi phí | 1× | ≈1× | WM ×3 (huấn luyện và imagination) | ≈1× |

Khác biệt với nhánh ablation proxy (cần ghi rõ): vùng tin cậy proxy giới hạn **hành động mở so với hành động ghi lại**; B1 là tương tự vòng kín (chặn lệch khỏi bản EMA của chính actor), chưa được kiểm chứng riêng. B2 dùng hành động của thành viên chính cho mọi thành viên. Hai điều này là giả thuyết cần pilot xác nhận.

**Cấu hình chung (một file):** `configs/pushing_p1_pilot.yaml`, cùng `experiment_seed=1000`, `train_seed_base=1100000`, cùng ngân sách (số iteration, tính từ benchmark: `N = min(8000, ⌊0,9·7200/iteration_estimate_s⌋)`), cùng GPU, chạy **tuần tự**. Giao thức đầy đủ: `experiments/p1_actor_ab_protocol.yaml` (sha256 `126266d5f4235249a1cfde2070fee2b86d89f1efd09c0d332abc1b335d5611f2`), đăng ký hash trước khi chạy.

**Tách dữ liệu chọn mô hình khỏi đánh giá cuối:**
| Tập | Seed | Vai trò |
|---|---|---|
| huấn luyện | `episode_seed(1100000, env, ep)` | thu thập/huấn luyện |
| chọn checkpoint | 9.000.000+ (32 goal) | chọn `best.pt` theo return; không bao giờ là kết quả |
| validation WM | 25.100.000+ (120 episode, file cố định) | G1, G2, G5 |
| giám sát | 27.000.000+ (64 goal), 27.300.000+ (32 goal G3) | G3, G4, G6 trong lúc chạy |
| **đánh giá cuối** | **28.000.000–28.000.199 (200 goal)** | **điểm cuối chính duy nhất; chỉ chạy một lần mỗi biến thể sau huấn luyện; không dùng để chỉnh gì** |

**Cách đo (mỗi mục có công cụ đã test):** return MuJoCo thật (`p1_actor_eval`, `p1_run_summary`); thành công held-out chặt (`p1_actor_eval`, Wilson 95%, McNemar chính xác so với B0 trên **cùng** 200 goal); return dự đoán vs thực (`p1_wm_eval --actor-episodes 32`, G3); sai số tự do/chạm (`p1_wm_eval`, G1/G2); khai thác mô hình (`p1_wm_eval --opt-states 24`: khoảng lạc quan sau tối ưu hành động vs lợi thật); hướng gradient (`--grad-states 24`, G5) và độ lớn cập nhật (`p1_run_summary`: `actor_grad_norm`, bước tham số actor giữa các bản sao checkpoint 400 iteration, `b1_*`, `b2_*`); chi phí (transitions, giờ GPU đặt trước = giờ tường × 1 GPU, giây/iteration, VRAM đỉnh, tải GPU từ benchmark).

**Quy tắc quyết định (cố định trước):** điểm cuối chính là strict success trên 200 goal cuối; biến thể "tốt hơn B0" chỉ khi hơn ≥ +10 điểm phần trăm và McNemar p < 0,05, hoặc B0 chạm STOP còn biến thể thì không; một seed mỗi biến thể là **thăm dò**, mọi biến thể hứa hẹn phải lặp lại trên ≥ 2 seed mới (cả `experiment_seed` lẫn `train_seed_base`) trước khi khẳng định.

## 6. Ngân sách pilot ngắn đề xuất và điều kiện dừng (STEP 6.6)

- **Pilot-0 (chỉ B0):** 1 seed, ≤ 2 giờ (timeout cứng của `p1_pilot.sh`), gate mỗi 400 iteration, `N` theo benchmark. Cần: benchmark đo trên đúng UUID, hash cổng/giao thức đã ghi, **phê duyệt riêng**.
- **A/B-1 (B0–B3):** mỗi biến thể cùng `N` iteration, ≤ 2 giờ mỗi lần, tuần tự, tổng ≤ 8 giờ GPU; chỉ sau khi Pilot-0 được xem xét và được duyệt riêng.
- **STOP:** G1–G7 đúng như đăng ký; thêm: `actor_grad_norm` không hữu hạn hoặc `bad_grads > 0` ở > 1% iteration; B1 `b1_frac_outside_radius > 0,9` trong 400 iteration liên tiếp (gradient bị clamp nuốt: không có tín hiệu học); B2 thành viên có loss không hữu hạn. STOP = dừng lần chạy đó, giữ sản phẩm, báo cáo; không tự khởi động lại, không đổi ngưỡng.
- **Launcher:** `p1_pilot.sh --gpu-uuid <UUID>` mặc định dry-run, trần 2 giờ, từ chối GPU có tiến trình khác; `--execute` không được dùng nếu chưa có benchmark đo + phê duyệt riêng. Dry-run biến thể: `p1_train_variant --variant b1 … --dry-run`.

## 7. P0 và PickPlace (STEP 5)

**Lệnh đánh giá độc lập P0 trên checkpoint thật** (không retrain; nên chạy trên bản sao; `ROOT` chứa `reaching_long_s{1000,1001,1002}/seed{…}`):
```
sha256sum $ROOT/reaching_long_s*/seed*/checkpoints/*.pt > outputs/p0_final/ckpt_before.sha256     # chứng minh không bị sửa
python -m research.supermimic_openarm.p0_seed_audit --run-dir $ROOT/reaching_long_s1000/seed1000 --run-dir $ROOT/reaching_long_s1001/seed1001 --run-dir $ROOT/reaching_long_s1002/seed1002 --out outputs/p0_final/seed_audit.json
python -m research.supermimic_openarm.p0_final_eval --run-root $ROOT --seeds 1000 1001 1002 --episodes 100 --seed-base 9900000 --ckpts best.pt latest.pt --out outputs/p0_final/eval_<ngày>
sha256sum -c outputs/p0_final/ckpt_before.sha256                                                   # phải toàn OK
```
100 goal mới (9.900.000–9.900.099), cùng bộ cho mọi checkpoint (so cặp best vs latest, McNemar chính xác), tách khỏi huấn luyện, chọn `best.pt` (9.000.000+) và đánh giá trực quan (9.800.000+). Ba seed P0 **độc lập một phần** (độc lập khởi tạo/nhiễu/thứ tự replay; **chung** chuỗi goal huấn luyện và goal chọn checkpoint) — xác nhận bằng `p0_seed_audit` trên config thật.
**PickPlace:** các khiếm khuyết ma sát đã xác nhận được ghi thành **issue riêng** `docs/ai-native/tasks/2026-11-11-pickplace-friction-issue.md` (box–desk/tray: `max`, nửa dải scale vô hiệu; ma sát nắm ngón–hộp luôn 1,0, không được ngẫu nhiên hoá), kèm các lựa chọn và tiêu chí chấp nhận. **Không sửa vật lý PickPlace** trong thay đổi này; không đăng lên GitHub (kết nối GitHub MCP đang lỗi proxy, và đăng bài là việc ra ngoài cần bạn duyệt).

## 8. GO / NO-GO

| Hạng mục | Quyết định | Lý do / điều kiện |
|---|---|---|
| **Benchmark GPU** | **GO** | công cụ bị chặn bởi trần thời gian/VRAM, chỉ đọc đến lệnh cuối, từ chối GPU có job khác, không khởi chạy huấn luyện; đã test các nhánh lỗi. Điều kiện: bạn chạy `--list`, chọn UUID 5090 rảnh (không phải GPU của WAM). Route Colab chỉ là kiểm tra chức năng |
| **Pilot actor B0 (≤ 2 h)** | **NO-GO hôm nay → GO có điều kiện** | chưa có số đo GPU thật (G7) và chưa có phê duyệt riêng. Khi (1) benchmark đo trên đúng UUID cho ≥ 4000 iteration/2 h và VRAM ≤ 16 GiB, (2) hash cổng `a7dc91ca…` và giao thức `126266d5…` được ghi, (3) bạn duyệt riêng, thì GO cho B0 một seed |
| **A/B B0–B3** | **NO-GO** | chờ kết quả Pilot-0; mọi biến thể là giả thuyết từ dữ liệu proxy |
| **Nhiều giờ / nhiều seed** | **NO-GO** | không có bằng chứng pilot |

## 9. Hạn chế và việc cần chủ việc

Hạn chế: không có GPU, không thể xem địa hình/workload thật hay đo 5090; môi trường test (Python 3.13/torch 2.11) khác máy bạn; B1/B2/B3 mới được kiểm bằng test đơn vị trên replay nhỏ, chưa chạy trong huấn luyện thật (cố ý); bằng chứng giảm thiểu đến từ WM proxy với 24 trạng thái; Colab route chưa chạy trên VM thật.
Việc cần bạn: (1) chạy `p1_benchmark --list` và dán kết quả (hoặc chạy lệnh benchmark ở mục 4A); (2) nếu muốn route Colab, tự xác thực và chạy các lệnh ở 4B (mình không xác thực thay bạn, không tải gì lên); (3) chạy lại bộ test trên môi trường của bạn; (4) chạy các lệnh P0 ở mục 7; (5) duyệt commit manifest (27 đường dẫn, nhánh `main-f9g0uh`) và/hoặc ngưỡng/giao thức; (6) quyết định về issue PickPlace.
