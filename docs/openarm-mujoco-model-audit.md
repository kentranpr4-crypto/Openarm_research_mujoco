# OpenArm MJCF audit — đọc trước khi viết action adapter
Nguồn: enactic/openarm_mujoco @ a3c8a57, MuJoCo 3.17.0, 2026-10-27. Tạo bằng `sim/inspect_model.py` (đo trực tiếp từ model, không suy từ docs). Chưa đối chiếu robot thật.

| | v1/scene.xml | v2/openarm_bimanual.xml |
|---|---|---|
| nq / nv / nu / na | 18 / 18 / 16 / 0 | 18 / 18 / 16 / 0 |
| free/ball joint | không (nq = nv) | không (nq = nv) |
| Khớp tay (7/tay) | hinge, rad | hinge, rad (cùng range) |
| Khớp ngón (2/tay) | **slide, mét**, 0..0.044 | **hinge, rad**, trái 0..0.785, **phải −0.785..0** |
| Ngón ctrlrange | [−0.005, 0.044] | trái [−0.4, 0.785], phải [−0.785, 0.4] |
| Ngón kp / kv / lực | 5664 / 113.3 / ±150.8 | 10 / 0.2 / ±7 |

## Điều đã xác nhận (đo được)
- Mọi actuator là **position servo**: `ctrl[i]` = vị trí đích của khớp (rad với khớp tay), kp/kv qua gainprm/biasprm, không có dynamics (dyn=NONE). Khớp tay: kp 230/190/30, giới hạn lực ±40/±27/±7 N·m theo cặp khớp.
- **nu (16) ≠ nq (18)**: mỗi tay 7 khớp + 1 actuator ngón, nhưng có 2 khớp ngón. `finger_joint2` bị ràng buộc `equality joint` (polycoef 0,1: q2 = q1) vào `finger_joint1`, nên chỉ có 1 bậc tự do gripper/tay.
- **Chỉ số ctrl ≠ chỉ số qpos** từ tay phải: ctrl u[8]↔q[9], u[15]↔q[16]. Phải ánh xạ qua `actuator_trnid` → `jnt_qposadr`, không đánh số bằng nhau.
- Khớp 1–2 trái/phải có range đối gương (trái j1 [−3.49,1.40], phải [−1.40,3.49]); ngón v2 phải mang dấu âm. Adapter dùng chung hai tay cần xử lý dấu.
- ctrlrange của ngón rộng hơn range khớp (v1 −0.005; v2 −0.4/+0.4): cố ý cho phép "siết" quá đóng. Hệ quả: clip theo ctrlrange chưa đủ để bảo đảm trong range khớp.

## Hệ quả cho adapter / dữ liệu
1. Cùng tên `left_finger1_ctrl` nhưng **đơn vị khác nhau giữa v1 (mét) và v2 (rad)**: không dùng chung checkpoint/normalization giữa hai bản.
2. State vector 18 chiều ≠ action vector 16 chiều. Chọn state cho policy là 16 (bỏ finger2 thừa) hay 18 là quyết định cần ghi rõ.
3. Action là *target position*, không phải torque hay delta: nếu dữ liệu thật ghi joint position đo được, cần xác nhận lệnh gửi thật có cùng ngữ nghĩa (NOT_CHECKED).

## Chưa biết
- v1/v2 tương ứng OpenArm 1.0/2.0 phần cứng nào, và gripper thật có đơn vị nào: chưa xác nhận.
- kp/kv trong sim có khớp driver motor thật không: chưa kiểm.
