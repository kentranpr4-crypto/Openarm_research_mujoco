# Mô phỏng OpenArm trong MuJoCo — báo cáo bàn giao
Trạng thái: **PARTIAL** (AC-04 chưa kiểm). Tự kiểm, không phải kiểm độc lập.

Nguồn: github.com/enactic/openarm_mujoco @ a3c8a57 (Apache-2.0, package 2.3.0). MuJoCo 3.17.0, Python 3.12 (venv `sim/.venv`). Ngày 2026-10-27.

| AC | Kết quả | Evidence |
|---|---|---|
| AC-01 model nạp được | PASS | v1/scene.xml và v2/openarm_bimanual.xml: nq=18, nv=18, nu=16 (7 khớp + 1 gripper mỗi tay) |
| AC-02 5 s sim không NaN | PASS | 5000/5000 bước, max abs qvel 11.80 (v1) / 13.42 (v2) rad/s; điều khiển sin biên độ 15% ctrlrange |
| AC-03 ảnh render | PASS | `sim/out/v1_scene.png`, `.mp4` (và v2) |
| AC-04 viewer tương tác | NOT_CHECKED | chưa chạy, cần màn hình |

Chưa kiểm: so khớp với robot thật, mô hình đúng bản phần cứng nào (v1 = OpenArm 1.0?, v2 = 2.0? — chưa xác nhận trong docs), động học/va chạm khi cầm vật, tính hợp lý vật lý của qvel ~12 rad/s với tín hiệu thử.
Lệnh chạy: `cd sim && .venv/bin/python run_sim.py [scene.xml]`; viewer: `.venv/bin/python -m mujoco.viewer --mjcf=openarm_mujoco/v1/scene.xml`.
