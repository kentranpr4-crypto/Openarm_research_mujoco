# Robotics Manipulation Lab + ACT pipeline (OpenArm v1, MuJoCo)
Ngày 2026-10-27. Tự kiểm (không phải kiểm độc lập). **Không commit, không push** theo yêu cầu.
Trạng thái tổng: **PARTIAL** — Phase 1–2 DONE; Phase 3 chạy được end-to-end (overfit PASS, train CPU 85 phút); Phase 4 có báo cáo nhưng ACT chỉ đạt 60% ở tư thế hộp chưa thấy (xem mục 6) và chưa kiểm trên phần cứng thật/Jetson/GPU. Không commit, không push.

## 1. Kiến trúc

```mermaid
flowchart LR
  subgraph SIM[sim/ — môi trường]
    UP[openarm_mujoco v1 (upstream, KHÔNG sửa)] --> LAB[lab_scene.py<br/>MjSpec: sàn/tường/bàn/3 vật/tray/đèn]
    ASSETS[assets/*.png<br/>make_textures.py] --> LAB
    LAB --> CAM[cameras.py<br/>overview, front, wrist_left, wrist_right]
    LAB --> EXP[export_scene.py → outputs/lab_scene/lab_scene.xml]
    IK[ik.py<br/>DLS IK 6D + gravity comp] --> LAB
  end
  subgraph CTRL[control/]
    AA[action_adapter.py<br/>action 16-d → ctrl, clip, slew] --> SL[safety_limits.py]
  end
  subgraph DATA[data/]
    CD[collect_demos.py] --> RAW[(outputs/datasets/*.npz + meta.json)]
    RAW --> VD[validate_dataset.py]
    RAW --> CL[convert_lerobot.py] --> LR[(LeRobot dataset)]
  end
  subgraph POL[policies/]
    BL[baseline.py<br/>scripted IK expert] --> CD
    TR[act_train.py] --> CK[(outputs/checkpoints)]
    INF[act_infer.py]
  end
  EV[eval/evaluate.py + metrics.py]
  LAB --> CD
  LR --> TR
  CK --> INF --> AA --> LAB
  BL --> EV
  INF --> EV
```

Nguyên tắc tách lớp: **appearance** (group 2, `contype=conaffinity=0`, `density=0`, materials/textures) tách khỏi **contact physics** (group 3, `contype=conaffinity=1`, mass/friction/inertia) — đúng quy ước của upstream.

## 2. Khảo sát trước khi sửa (evidence)
- Upstream `openarm_mujoco` @ a3c8a57 sạch (`git status` trống); visual = group 2/contype 0, collision = group 3/contype 1; timestep 1 ms, `implicitfast`.
- Hand frame (đo): local +z = hướng ngón (= world +x), local +x = world lên, local y = world −y. Hand housing cao tới local x=0.029; ngón x∈±0.031, z∈[0.011,0.096], y∈±[0.043,0.077].
- Actuator: position servo, `ctrl` = vị trí đích; ctrl index ≠ qpos index (audit `docs/openarm-mujoco-model-audit.md`). Không bù trọng lực → expert dùng `ctrl = q + qfrc_bias/kp`.
- LeRobot 0.6.1 (PyPI): `requires-python>=3.12`, `numpy<2.3`, `torch<2.12` → **không** cài chung venv hiện tại (numpy 2.5.4). Phase 3 dùng venv riêng. Mirror CPU của PyTorch (`download.pytorch.org`) bị proxy chặn; cloud không GPU.

## 3. PHASE 1 — kết quả
Lệnh (từ `sim/`; cloud headless thêm `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`):
```
python assets/make_textures.py                       # sinh 4 texture PNG (seed cố định)
python lab_scene.py --check --render --export-xml    # kiểm AC + ảnh + xuất MJCF (outputs/lab_scene/)
python export_scene.py --verify                      # roundtrip MJCF
python lab_scene.py --viewer                         # cửa sổ tương tác (cần màn hình)
bash viewer_smoke.sh                                 # mở viewer thật trên Xvfb + chụp màn hình
python -m mujoco.viewer --mjcf=../outputs/lab_scene/lab_scene.xml
```
Files mới: `sim/lab_scene.py`, `sim/cameras.py`, `sim/export_scene.py`, `sim/ik.py`, `sim/assets/make_textures.py` (+4 PNG), `sim/viewer_smoke.sh`. Không sửa: upstream, `desk_scene.py`, `grasp_blue_box.py`, `.gitignore`.

| AC | Kết quả | Evidence (`outputs/lab_scene/lab_check.json`) |
|---|---|---|
| AC-01 scene nạp được | PASS | nq=39, nv=36, nu=16 (18+3×7 / 18+3×6), 32 body, 129 geom, 3 đèn; camera `overview, front, wrist_left, wrist_right` (+2 camera sẵn có của upstream) |
| AC-02 3 vật ổn định 5 s | PASS | 5000 bước, không NaN; lệch ≤0.56 mm; tốc độ cuối ≤2.5e-5 m/s; không va chạm robot↔vật/bàn/tray; quán tính tự tính khớp công thức giải tích (sai số tương đối <1e-15) |
| AC-03 render/viewer thấy đủ | PASS | segmentation overview: robot 13286 px, bàn 27860, tray 1388, hộp xanh 338, trụ đỏ 347, khối xanh lá 356; ảnh `overview/front/wrist_*.png`, `montage.png` |
| AC-04 hộp xanh trong tầm | PASS | IK 6-DoF (giữ hướng tay) tay trái: grasp (0.43,0.12,0.386) sai số 0.26 mm; pregrasp 0.49 mm; trên tray (0.44,0.30,0.414) 0.40 mm. Va chạm trên đường đi được kiểm gián tiếp ở Phase 2 (expert 60/60 thành công trong sim) |
| (mới) Tách visual/collision + upstream nguyên vẹn | PASS | 0 vi phạm group/contype; `git -C sim/openarm_mujoco status` sạch |
| (mới) Không cản workspace | PASS | không geom `deco_/wall_/floor`, đèn, camera thế giới nào nằm trong `WORKSPACE_BOX` x[0.20,0.75] y[−0.45,0.45] z[0.36,0.85] |
| (mới) Export MJCF | PASS | `export_scene.py --verify`: số body/geom/cam/light khớp, không đường dẫn tuyệt đối, sai lệch vị trí ≤4e-6 m (do XML làm tròn; dung sai 1e-5) |
| Viewer | **PASS có điều kiện** | cửa sổ `python -m mujoco.viewer` và `lab_scene.py --viewer` đều mở được **trên Xvfb + GL phần mềm**, ảnh chụp `outputs/lab_scene/viewer_xvfb.png` thấy robot/bàn/3 vật/tray. **Chưa kiểm** trên màn hình thật + GPU + tương tác chuột/bàn phím. |

Thiết kế tray: ban đầu (0.50,0.28), lòng 10×10 cm. Phase 2 cho thấy (a) x=0.50 ở rìa vùng với tới khi giữ hướng tay ngang (IK sai 18–41 mm; bản đồ IK: <1 mm với x≤0.44, >10 mm từ x=0.48) và (b) lòng 10 cm hẹp hơn bề rộng hai ngón khi đang giữ hộp (~11 cm). Đã đổi: tâm (0.44,0.30), lòng 14×14 cm, gờ 12 mm, và nắm hộp ở phần cao (tcp = tâm hộp + 2.5 cm) để ngón không quét qua gờ. Toàn bộ kiểm Phase 1 + ảnh + viewer smoke đã chạy lại sau thay đổi này.

Vấn đề đã gặp & xử lý: (1) camera cổ tay ban đầu không thấy ngón → thử 3 pose, chọn pos (0.04,0,−0.02) pitch 35° FOV 90°, thấy cả hai đầu ngón; (2) camera viewer mặc định thấy cả phòng như cái hộp → đặt `statistic`/azimuth quanh bàn; (3) hàm AABB coi mặt phẳng dày vô hạn → sửa; (4) `pkill -f` tự giết shell (chỉ ảnh hưởng thao tác test).

Giới hạn Phase 1: "soft shadows" chỉ xấp xỉ (shadow map 8192, 3 đèn, ambient) — renderer cổ điển của MuJoCo không có bóng mềm vật lý; còn vài chấm nhiễu nhỏ trên tường ở góc nhìn xiên; kích thước/khối lượng/ma sát là giá trị tạm; thông số camera D435 (FOV 42.5°) từ trí nhớ, chưa đối chiếu datasheet; camera cổ tay chưa đối chiếu với vị trí gắn thật.

## 4. PHASE 2 — expert + demonstrations
Files mới: `control/{action_adapter,safety_limits,test_action_adapter}.py`, `sim/lab_env.py`, `policies/baseline.py`, `eval/metrics.py`, `data/{collect_demos,validate_dataset,clip_actions}.py`. `sim/grasp_blue_box.py` giữ nguyên (demo gốc, vẫn chạy).

**Action contract (đã kiểm bằng `control/test_action_adapter.py`, 9/9 PASS):** action (16,) = vị trí đích tuyệt đối của khớp, đúng thứ tự actuator, ghi thẳng vào `data.ctrl` (rad cho khớp tay, **mét** cho ngón v1). State (16,) = qpos của khớp mà actuator đó điều khiển (ánh xạ `actuator_trnid → jnt_qposadr`; bỏ `finger_joint2` thừa; chỉ số ctrl ≠ qpos từ tay phải: `qadr=[0..7,9..16]`). Test xác nhận ctrl[i] làm đúng khớp i dịch chuyển (tay phải j1→state[8], j4→state[11]), lệnh NaN/Inf/sai shape bị từ chối, clip + giới hạn slew hoạt động. Ngữ nghĩa này **không đổi** so với audit; lệnh expert đã bao gồm offset bù trọng lực `qfrc_bias/kp`.

**Expert** (`policies/baseline.py`): settle → pre-approach → pregrasp → approach → close → lift → transport → lower → release → retreat; IK 6D giữ hướng tay (xoay theo yaw hộp), nắm ở tcp = tâm hộp + 2.5 cm, tốc độ tcp 0.18 m/s, 20 Hz. Vùng random hộp (train): x∈[0.38,0.45], y∈[0.07,0.13], yaw∈[−0.2,0.2] rad; vùng "unseen": x∈[0.36,0.47], y∈[0.05,0.15], |yaw|≤0.3 loại trừ vùng train.

| AC | Kết quả | Evidence |
|---|---|---|
| AC-05 expert grasp khi random | PASS | 60/60 demo thành công (pick, place, task), sai số IK ≤0.50 mm / 0.0049 rad, tốc độ lệnh khớp tối đa 1.70 rad/s (<2.0), hộp nằm cách tâm tray 0.8 mm (tb), 1.5 mm (max). Quét riêng 80 cấu hình rộng hơn (|yaw|≤0.3): 80/80 task nhưng 6% kế hoạch IK lệch >2 mm và có đỉnh 4–9 rad/s khi |yaw|>0.24 — vì vậy vùng train chặt hơn. Eval expert: 30/30 nominal và 30/30 unseen trên 3 seed |
| AC-06 schema dataset | PASS | `outputs/datasets/demo_train_clipped` (60 ep, 133–142 bước, 20 Hz): `observation.state/velocity (T,16)`, `observation.images.{front,wrist_left,wrist_right} (T,120,160,3) u8`, `action (T,16)`, `timestamp`, `gripper_state (T,2)`, `phase`, và `eval.box_pos/box_quat/tcp_pos/box_speed` (ground truth, chỉ để đánh giá). `episodes.jsonl` + `meta.json` ghi seed, config, box pose, ánh sáng, nhãn pick/place/task, version mujoco/numpy, commit repo + upstream |
| AC-07 đọc và visualize độc lập | PASS | `data/validate_dataset.py` chỉ dùng numpy/Pillow/imageio: 0 vấn đề (shape, dtype, finite, timestamp, action trong ctrlrange, ảnh không đen, nhãn nhất quán); sinh `viz/episode3_phases.png`, `episode3.mp4`, `episode3_actions.png` |

Vấn đề tìm ra nhờ validator: ở 23/60 episode `left_joint2` bị bù trọng lực đẩy ctrl vượt `ctrlrange` tới 0.0104 rad nên action ghi ≠ lệnh thực thi. Đã sửa expert (kẹp ctrl) và tạo `demo_train_clipped` từ dữ liệu cũ bằng `data/clip_actions.py` (bản gốc `demo_train` giữ nguyên; lineage trong `meta.json`). Đã kiểm: mô phỏng tất định (chạy 2 lần sai lệch 0), cánh tay khớp ở mức 6e-9 giữa kẹp float32/float64; chênh 8.7e-4 chỉ ở quaternion hộp (làm tròn float32 khuếch đại qua va chạm).
Throughput: render phần mềm 510 ms/ảnh với bóng + phản xạ + MSAA nên thêm `render_profile` (`fast` = không bóng/phản xạ/MSAA, ~100 ms/ảnh) dùng cho dataset/train/infer/eval (phải cùng profile). 60 episode mất 908 s với 3 worker.

## 5. PHASE 3 — ACT
Môi trường: venv riêng `/home/user/.venvs/act` (LeRobot **0.6.1** cần Python ≥3.12, `numpy<2.3`; không cài chung venv sim). Phiên bản đóng băng ở `policies/requirements-act.txt`. Mirror CPU của PyTorch bị proxy chặn → torch 2.11 CUDA wheel chạy CPU; ImageNet weights không tải được → ResNet18 huấn luyện từ đầu.

Files: `data/convert_lerobot.py`, `policies/{act_train,act_infer,open_loop_check}.py`, `eval/audit_policy_inputs.py`.
- Adapter: raw npz → `LeRobotDataset.create` (ảnh PNG, không video), chỉ đưa state/velocity/ảnh/action; **không** đưa `eval.*`. Split theo episode có seed (`openarm_splits.json`): 54 train / 6 val; **đính chính (audit 2026-10-28):** thống kê chuẩn hóa trong checkpoint được tính trên cả 60 episode (8172 khung, gồm 6 episode val) vì `ds_tr.meta.stats` là thống kê toàn dataset; ảnh hưởng lên các chiều có phương sai nhỏ hơn 1% std (xem `2026-10-28-act-v1-audit-v2-plan.md`), không ảnh hưởng các episode đánh giá.
- ACT: input = `observation.state` + ảnh `front`, `wrist_left`; output chunk 20×16 (1 s); `use_vae=False` (expert tất định), dim 256, 4 head, 2 encoder layer, 14.0 M tham số, L1 loss, AdamW 1e-4, batch 16.

| AC | Kết quả | Evidence |
|---|---|---|
| AC-08 train chạy + overfit | PASS | overfit 3 episode, 2500 bước (40.6 phút): L1 chuẩn hóa 0.904 → 0.045 (train), 0.034–0.043 (val trên cùng tập); open-loop (`open_loop_check.json`) sai số cánh tay trái 0.014 rad, gripper 0.56 mm; vòng kín trên chính 3 cấu hình train: pick 3/3, task 2/3 (1 ca `dropped_or_misplaced`). Train đầy đủ: 5383 bước (85 phút, giới hạn thời gian), val L1 0.0234 trên 6 episode giữ lại (train EMA 0.037) |
| AC-09 inference không dùng toạ độ vật | PASS (audit tĩnh + cấu hình) | `eval/audit_policy_inputs.py`: input_features của checkpoint chỉ gồm `observation.state` + 2 ảnh; dataset LeRobot không có feature ground truth; `act_infer.py` không import mujoco/lab_scene và không gọi `env.info`; evaluate gọi `runner.act(o["state"], o["images"])` duy nhất; converter không chuyển `eval.*`. Không có test phản thực nghiệm (đổi tọa độ vật ẩn) — suy ra từ giao diện |

## 6. PHASE 4 — đánh giá (`outputs/eval/full_v1*/report.md`)
Điều kiện: `nominal` (vùng train), `unseen_pose`, `light_dim` (×0.55), `light_bright` (×1.6), `light_shift` (dời đèn), `cam_shift` (camera front +2/+1/+1 cm, nghiêng 3°). n=10 mỗi điều kiện; seed 7 cho tất cả, thêm seed 8 và 9 cho nominal + unseen_pose. Cùng cấu hình cho expert và ACT (so cặp), horizon 200 bước (10 s), ACT chạy chunk 20 bước mở vòng, CPU 1 luồng.

| Điều kiện | Expert task | ACT task (Wilson 95%) | ACT pick / place |
|---|---|---|---|
| nominal, 3 seed gộp | 30/30 | **30/30** = 100% [89–100] | 100% / 100% |
| unseen_pose, 3 seed gộp | 30/30 | **18/30** = 60% [42–75] (theo seed: 3, 7, 8 /10) | pick 19/30 = 63%; place 18/30 |
| light_dim / light_bright / light_shift (seed 7) | 10/10 mỗi điều kiện | 9/10, 10/10, 9/10 | pick 90–100% |
| cam_shift (seed 7) | 10/10 | 8/10 | pick 100%, place 80% |

- Joint tracking (|ctrl − state| trung bình cánh tay): expert 0.0143 rad, ACT 0.0137–0.0138 rad.
- Độ mượt (|jerk| trung bình của ctrl, rad/s³): expert 6.0–6.4; ACT 22.7–29.9 (≈4× nhiễu hơn; nghi do biên chunk mở vòng, chưa thử temporal ensembling).
- Latency suy luận (CPU 1 luồng, tính cả tiền/hậu xử lý): mỗi lần truy vấn chunk p50 50.4 / p95 57.1 / p99 67.1 ms (nominal; các điều kiện khác 50.5–51.4 / 58.4–62.8 / 61.5–76.6 ms); trung bình mỗi bước điều khiển p50 1.4 / p95 7–11 / p99 52 ms (19/20 bước chỉ lấy từ hàng đợi). **Đây không phải số đo trên Jetson Orin Nano/TensorRT.**
- Số lần "clipped" ~1300–1550/điều kiện chủ yếu là lệnh mở gripper = đúng giới hạn trên 0.044 (sai số làm tròn của mạng), không phải vi phạm an toàn; slew-limit kích hoạt 3–18 lần.
- Failed rollouts có video + metadata trong `outputs/eval/full_v1/act/<điều kiện>/videos`; ảnh tổng hợp `outputs/eval/full_v1/failure_frames.png`. Phân loại lỗi `unseen_pose` seed 7: 6 `no_pick` + 1 `dropped_or_misplaced`. Các lỗi chỉ lệch biên vùng train 0.5–2 cm nhưng 3 ca thành công cũng lệch tới 1.7 cm, nên **chưa xác định được nguyên nhân** (giả thuyết, chưa kiểm: chỉ 54 demo, không có augmentation, backbone không pretrained, ảnh 120×160).

## 7. Lệnh chạy lại
```
# sim venv (sim/.venv):   cd sim && bash setup.sh && python assets/make_textures.py
python lab_scene.py --check --render --export-xml            # Phase 1 (cloud: MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa)
python control/test_action_adapter.py                        # (từ thư mục gốc) contract
python data/collect_demos.py --n 60 --seed 10 --workers 3 --light-jitter 0.15 --out outputs/datasets/demo_train
python data/clip_actions.py outputs/datasets/demo_train outputs/datasets/demo_train_clipped
python data/validate_dataset.py outputs/datasets/demo_train_clipped
# ACT venv: uv venv /home/user/.venvs/act --python 3.13 && uv pip install -p /home/user/.venvs/act/bin/python -r policies/requirements-act.txt "lerobot[dataset]==0.6.1"
python data/convert_lerobot.py outputs/datasets/demo_train_clipped outputs/lerobot/pickplace_v1
python policies/act_train.py --dataset outputs/lerobot/pickplace_v1 --out outputs/checkpoints/act_full_v1 --steps 6000 --max-minutes 85
python eval/evaluate.py --agent act --ckpt outputs/checkpoints/act_full_v1/last --conditions all --n 10 --seed 7 --workers 4 --save-rollouts --out outputs/eval/full_v1
python eval/evaluate.py --agent expert --conditions all --n 10 --seed 7 --out outputs/eval/full_v1
python eval/make_report.py outputs/eval/full_v1 ; python eval/audit_policy_inputs.py CKPT outputs/lerobot/pickplace_v1
```

## 8. Quyết định, chưa kiểm, việc cần chủ việc
- Đã tự quyết (đổi được): tray (0.44,0.30) lòng 14 cm; nắm hộp ở phần cao; vùng random; `use_vae=False`; 2 camera cho ACT (`front`, `wrist_left`); 20 Hz; profile render `fast`; chunk 20.
- **Chưa kiểm:** robot thật; Jetson/TensorRT/GPU (latency chỉ trên CPU 1 luồng); viewer trên màn hình thật với chuột/bàn phím (mới chạy trên Xvfb); độ lặp lại của train (1 lần train, 1 seed); "place success" của expert/ACT chỉ đo trong sim; thông số D435 và vị trí camera cổ tay; ma sát/khối lượng/lực kẹp; sim-to-real; temporal ensembling; train dài hơn / nhiều dữ liệu hơn.
- Cần quyết định: có commit/push không (đang để nguyên, chưa commit); có thêm `outputs/` (1.6 GB) vào `.gitignore` hay dùng git-lfs; hướng cải thiện unseen_pose (thêm demo trải rộng hơn, augmentation, pretrained backbone, temporal ensembling).
- Không chạm: upstream `sim/openarm_mujoco`, `desk_scene.py`, `grasp_blue_box.py`, `data/sources.csv` và dữ liệu cào cũ, checkpoints/dataset/log đã có (không xóa; chỉ chuyển một log lỗi sang `outputs/eval/_old/`).
