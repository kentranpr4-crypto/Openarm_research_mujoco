# Scene bàn + 3 vật cho OpenArm v1 (MuJoCo) — báo cáo bàn giao
Trạng thái: **DONE** cho 4 AC đề xuất (tự kiểm, không phải kiểm độc lập). Ngày 2026-10-27, MuJoCo 3.x, render bằng `MUJOCO_GL=osmesa` (cloud không có EGL; đã `apt install libosmesa6 libegl1`).

Đầu ra: `sim/desk_scene.py` (MjSpec bao quanh `openarm_mujoco/v1/scene.xml`, không sửa file gốc), `sim/out/desk_overview.png`, `sim/out/desk_rs_d435.png`, `sim/out/desk_scene_check.json`.
Chạy: `cd sim && bash setup.sh && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa .venv/bin/python desk_scene.py`

| AC | Kết quả | Evidence |
|---|---|---|
| AC-01 scene nạp được | PASS | nq=39, nv=36, nu=16 (18+3×7 qpos, 18+3×6 qvel; nq≠nv như dự kiến) |
| AC-02 5 s ổn định | PASS | 5000 bước, không NaN; vật lệch ≤0.6 mm, tốc độ cuối ≤2e-5 m/s; không va chạm vật↔robot, bàn↔robot |
| AC-03 ảnh render | PASS | `desk_overview.png` thấy bàn đen + hộp xanh + trụ đỏ + khối lập phương xanh lá + hai tay; `desk_rs_d435.png` (camera đầu D435 mô phỏng) thấy cả 3 vật |
| AC-04 trong tầm với | PASS (có điều kiện) | IK vị trí (damped LS, có giới hạn khớp) đưa tcp tới 3 cm trên mỗi vật, sai số ≤1 mm; khoảng cách ngang tới tcp gần nhất ở tư thế home: 3.5 / 3.3 / 19 cm |

Thiết kế: mặt bàn z=0.36 m, 0.44×0.80 m, tâm x=0.48; hộp xanh 5×4×4 cm (0.05 kg) tại (0.42, +0.12); trụ đỏ r=2 cm h=6 cm (0.08 kg) tại (0.40, −0.12); khối xanh lá 4 cm (0.04 kg) tại (0.52, 0). Ma sát 0.8.
Quyết định: bản đầu để mặt bàn z=0.40 và trụ cao 8 cm → trụ chạm ngón tay phải ở home và bị đẩy 5 cm; đã hạ bàn và trụ (AC-02 đạt sau sửa).

Giới hạn / chưa kiểm:
- AC-04 chỉ IK vị trí: **không** kiểm hướng bàn tay, va chạm trên đường đi, hay khả năng kẹp thực (gripper v1 mở 2×4.4 cm; chưa thử gắp).
- Kích thước bàn/vật, khối lượng, ma sát là giá trị tạm, chưa đo từ bàn và hộp thật của bạn (ảnh robot thật chỉ dùng làm tham khảo bằng mắt).
- Camera D435 vẫn là thông số từ trí nhớ; ảnh nhìn thấy nhiều nền hơn bàn (pitch 35°) — có thể chỉnh.
- Viewer tương tác vẫn NOT_CHECKED (cần màn hình). v1/v2 ↔ OpenArm 1.0/2.0 vẫn chưa xác nhận. Chưa làm bản v2.
