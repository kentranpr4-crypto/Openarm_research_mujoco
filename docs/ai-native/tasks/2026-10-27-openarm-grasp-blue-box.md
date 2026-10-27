# Gắp hộp xanh trên bàn (OpenArm v1, MuJoCo) — báo cáo bàn giao
Trạng thái: **DONE** cho 4 AC đề xuất (tự kiểm, chạy 1 lần, mô phỏng vòng hở). Ngày 2026-10-27.
Đầu ra: `sim/grasp_blue_box.py`, `sim/out/grasp_blue_box.mp4`, `..._last.png`, `..._mid.png`, `grasp_blue_box_check.json`.
Chạy: `cd sim && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa .venv/bin/python grasp_blue_box.py` (đặt `GRASP_RENDER=0` để chạy nhanh, không quay video).

| AC | Kết quả | Evidence |
|---|---|---|
| AC-A hộp nâng ≥5 cm | PASS | z hộp tăng 7.45 cm (0.380 → 0.454 m) |
| AC-B giữ được sau khi nâng | PASS | giữ 1 s sau nâng, khoảng cách hộp–tcp 1.75 cm, không NaN |
| AC-C chạm cả hai ngón | PASS | contact left_finger và right_finger với blue_box sau khi đóng và sau khi giữ; q ngón sau khi đóng 0.0204 / 0.0205 m, khớp ước tính cho hộp rộng 4 cm (0.044 − 0.0235 ≈ 0.0205) |
| AC-D video render | PASS | mp4 140 khung (camera gần + toàn cảnh); trụ đỏ và khối xanh lá không xê dịch (0.0 m) |

Cách làm: tay trái (hộp ở y=+0.12), IK 6D giữ hướng home (ngón chỉ +x), tiếp cận ngang ở z=0.385, đóng ngón dần, nâng lên z=0.47. Servo vị trí không có bù trọng lực: lần chạy đầu tcp chỉ tới (0.347,0.115,0.446) thay vì (0.43,0.12,0.385) nên gắp hụt (hộp rise 0). Đã sửa bằng ctrl = q + qfrc_bias/kp, sau đó gắp thành công.

Chưa kiểm / giới hạn:
- Vòng hở, một lần chạy: chưa kiểm độ lặp lại, nhiễu vị trí hộp, hay các vật khác (trụ, khối lập phương).
- Ma sát 0.8, khối lượng 0.05 kg, lực kẹp ~50 N (ctrl 0.012) là giá trị tạm; lực kẹp thực của gripper OpenArm và kp/kv thật chưa xác nhận.
- Bù trọng lực dùng mô hình MuJoCo; robot thật cần xác nhận driver có bù hay không.
- Viewer tương tác vẫn NOT_CHECKED; v2 chưa làm; v1/v2 ↔ OpenArm 1.0/2.0 chưa xác nhận.
