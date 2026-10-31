# WM-0 — data audit (OpenArm Action-Conditioned World Model) và kế hoạch WM-1
Ngày 2026-10-31 · script `eval/wm0_audit.py` · kết quả máy đọc được `outputs/wm0/wm0_audit.json`, `wm0_catalog.json` (chưa commit) · phạm vi: chỉ đọc dữ liệu có sẵn; **không train WM, không thu thập mới trong simulator, không sửa ACT/upstream**, không dùng test seeds 31–33 (seeds trong dữ liệu: 7, 8, 9, 21, 22).
Trạng thái: **DONE (tự kiểm)**. Kết luận: dữ liệu **đủ để bắt đầu WM-1**, nhưng audit tìm ra 5 vấn đề bắt buộc sửa trong data contract (mục 3) và 1 giới hạn lớn của dữ liệu hiện có (mục 2.4/2.5).

## 1. Tóm tắt phát hiện
| # | Phát hiện | Hệ quả |
|---|---|---|
| F1 | **Chia theo episode sẽ rò rỉ ~100 %**: 2 170 rollout chỉ có **210 initial state** duy nhất (~10 episode/initial state vì các checkpoint được đánh giá trên cùng pose). Chia ngẫu nhiên 80/10/10 theo episode: 100 % episode val và 98 % episode test có initial state trùng với train | Bắt buộc chia theo **nhóm initial state (x,y,yaw)**, sau đó theo vùng x; không bao giờ chia theo episode/cửa sổ |
| F2 | **Palm contact (và cluster thất bại) chỉ xuất hiện ở vùng x < 0.38**: 60 episode palm contact, 73 cluster trong 282 episode LOW được replay; **0** ở vùng TRAIN (630 replay) và HIGH (168) | WM huấn luyện chỉ trên x ∈ [0.38, 0.45] **không có một ví dụ dương nào** cho palm contact; mục tiêu AUROC palm ≥ 0.80 ở OOD‑low sẽ là zero‑shot về cơ chế ⇒ cần dữ liệu can thiệp (độ sâu grasp) hoặc đưa một phần LOW vào train |
| F3 | Trạng thái robot phản hồi action với **trễ 1–2 bước** (servo vị trí): khớp 1–4 và gripper trễ 1, khớp 5–7 trễ 2 | Lịch sử input **phải gồm action quá khứ** (`A_hist`), bản proposal ban đầu thiếu điều này; quy tắc cửa sổ đổi thành t ≥ H |
| F4 | Action thực thi ≠ output policy: khác nhau ở **53.8 %** số bước (clip/slew; lớn nhất 0.58 rad, trung bình ≲ 1e‑3 rad); WM phải điều kiện theo **executed action** (đã có sẵn trong `action`), không theo `raw_action` | Quy ước đã ghi vào contract |
| F5 | Nhãn contact/box‑yaw/tilt/a‑offset **chỉ có ở replay**: 1 080/2 170 rollout (49.8 %) có replay; 3.1 % replay phân kỳ > 5 mm; 190 rollout schema cũ thiếu `box_speed`/`raw_action`; 20 bản trùng chính xác (audit_v1/seed7 ↔ full_v1) | Replay phần còn lại (~10 phút CPU) trước WM‑1; loại trùng/phân kỳ; 29D cần định nghĩa lại chính xác |

## 2. Kết quả theo 7 mục được yêu cầu
**2.1 Nội dung và đơn vị của "29D state" — FINDING (định nghĩa cũ mơ hồ, đã làm rõ).** Dữ liệu thật (đơn vị đã kiểm trên phạm vi giá trị):
- `state` (16): khớp tay trái 1–7 [rad] + gripper trái `finger_joint1` [m, 0.0007–0.0467], tay phải 7 khớp [rad] + gripper phải [m]. **Tay phải gần như hằng** (khớp 4 ≈ 1.55–1.58, các khớp khác |q| < 0.02, gripper 0.044): không mang thông tin động học (và là kênh shortcut đã biết).
- `box` từ GT sim: `box_pos` (3) [m], `box_speed` [m/s] (tương quan 0.96 với sai phân `box_pos`·20 Hz, cùng chỉ số thời gian), từ replay thêm yaw [rad], tilt [độ], `box_in_tcp` (a,c,v) [m].
- `contact` (replay): `finger_force` [N] (p50 47.9, p99 52.9, max 59.5), `finger_contacts` [số], `hand_force` (palm) [N] (p50 0.53, max 13.5 khi chạm), `table_force`, `tray_force` [N], vị trí contact a/c/v [m, khung TCP], **NaN 71.6 % số bước** (không tiếp xúc) ⇒ cần mask. Tiếp xúc finger 28.4 % số bước, palm 0.77 %, table/tray 81 % (hộp nằm yên).
- "29 = 16 + 7 + 6" của proposal không xác định đủ (số kênh contact thực tế 6 + vị trí contact): contract v1 ở mục 3 thay thế.
**2.2 Thứ tự actuator 16D và ngữ nghĩa executed action — PASS.** Tên `state[k]` và `action[k]` ghép đúng từng cặp (left_joint1…7, left_finger1, right_joint1…7, right_finger1); action là **mục tiêu khớp tuyệt đối (rad; gripper m) đã qua clip/slew** và nằm trong ctrlrange (100 % mẫu). Offset gravity‑compensation tay phải idle: action − state(t+1) ≈ 0.019 / 0.023 / 0.021 rad (khớp 1, 4, 7). Bằng chứng "executed": replay với đúng chuỗi action này tái tạo quỹ đạo hộp (median lệch 0.06 mm; 3.1 % lệch > 5 mm).
**2.3 Đồng bộ timestamps / state / action — PASS với lưu ý.** Rollout không lưu timestamp (ngầm định 20 Hz; demo có timestamp, lệch cực đại 2.9e‑7 s so với 0.05 s). Quy ước: hàng `t` của state = quan sát **trước** action `t`; `state`, `action` có T hàng, `box_pos`, `tcp_pos`, `box_speed` có T+1 hàng (gồm bước cuối). Mọi file đầy đủ schema (1 980) có T = 200, hình dạng nhất quán, hữu hạn. Trễ servo 1–2 bước (F3).
**2.4 Số episode thực, bản trùng, độ phủ nhãn contact — FINDING.** 2 177 file rollout (7 debug/smoke loại ⇒ **2 170**: 2 110 ACT + 60 expert); 2 150 hash action khác nhau ⇒ **20 bản trùng chính xác** (audit_v1/seed7 ↔ full_v1); **210 initial state** duy nhất (nhóm 2: 40; 8: 80; 9: 70; 37: 10; 45: 10); 600 rollout có state‑perturbation (state log là trạng thái thật, policy nhìn bản bị nhiễu), 40 có nhiễu thị giác. Replay: 1 230 file (h23 630, x‑sweep 420, coverage 180) đều duy nhất, không có file replay bị dùng cho hai rollout; 150 file là replay của scripted expert. **1 080/2 170 rollout có nhãn contact**; còn lại chưa replay: e6b 320 (perturbed) + 180, e0 perturbed 160, cov 180, audit_v1 130, full_v1 60.
**2.5 Rò rỉ train/val/test — FINDING (xem F1, F2).** Region split theo x không cắt nhóm initial state nào (0 nhóm trải nhiều vùng) ⇒ an toàn; ACT demo (120 pose) không trùng đánh giá (khoảng cách Chebyshev nhỏ nhất 2.6 mm, không trùng chính xác); không có seed 31–33. Phân bố vùng: LOW 388, TRAIN 1 584, HIGH 198 episode (ACT: success 202/376, 1 058/1 541, 165/193).
**2.6 Mục tiêu có dùng privileged MuJoCo state? — FINDING (đã lập bảng).** Privileged: toàn bộ `box_*`, `box_in_tcp`, tilt, **mọi contact**, mọi nhãn pick/place/cluster. Không privileged: `state`, `action`. `tcp_pos` suy ra được từ khớp qua FK. ⇒ mọi target của V0‑A (box, contact, outcome) là privileged; **không coi kết quả V0‑A là bằng chứng Sim2Real**. Ảnh không lưu trong rollout (chỉ có trong 120 demo).
**2.7 Quy tắc cửa sổ H = 4, K = 20 — PASS (đã có unit test).** Cửa sổ tại t: lịch sử state/box/contact hàng t−H+1…t, **action quá khứ t−H…t−1**, action tương lai t…t+K−1, target hàng t+1…t+K; hợp lệ với **t ∈ [H, T−1−K]** (T hàng action; target state cần hàng t+K ≤ T−1); không bao giờ vượt biên episode; unit test (T = 60, 138, 200, ranh giới H+K) PASS. Stride 5 ⇒ 36 cửa sổ/episode (T = 200), 23 (T = 138). Tổng **71 280 cửa sổ** (1 980 episode đầy đủ schema): 44.5 % "active" (hộp dịch > 5 mm hoặc có đóng gripper trong horizon), 16 738 cửa sổ có finger contact trong horizon (chỉ tính phần đã replay: 38 880 cửa sổ). Theo vùng x: TRAIN 51 840 / LOW 12 672 / HIGH 6 768.

## 3. Data contract v1 (thay thế "29D" của proposal)
Lưu: `float32`, khóa theo `(source, tag, episode_index)`, kèm hash action/box. Bỏ các chiều tay phải khỏi input/target (hằng; lưu để tái tạo).
| Khối | Shape mỗi bước | Nội dung (đơn vị) | Privileged |
|---|---|---|---|
| `s` | 8 | khớp trái 1–7 [rad] + gripper trái [m] | không |
| `b` | 7 | box x, y, z [m], yaw [rad], speed [m/s], tilt [rad], a‑offset `box_in_tcp.a` [m] | có |
| `c` | 7 | finger_force [N], n_finger_contacts, a_contact [m] (mask), a_valid (0/1), palm_force [N], table_force, tray_force | có |
| `a` | 8 | action thực thi tay trái (rad; gripper m) | không |
| `m` (mỗi episode) | – | nguồn, checkpoint, nhóm initial state `g=(x,y,yaw)`, vùng x, nhãn pick/place/palm/cluster, cờ replay_ok | có |
Chuẩn hóa train‑only (như `norm_contract`), `a_contact` NaN → 0 + mask `a_valid`; lực chia cho 50 N.
**Tensor cửa sổ (batch B):** `S_hist (B,4,8)`, `B_hist (B,4,7)`, `C_hist (B,4,7)`, `A_hist (B,4,8)` (action quá khứ), `A_fut (B,20,8)`; target `Y_s (B,20,8)`, `Y_b (B,20,7)`, `Y_c (B,20,7)` (delta so với bước trước, chuẩn hóa; contact kèm mask) và nhãn trong horizon `Y_e (B,4)` = {đóng gripper, palm contact, hộp nâng > 3 cm, hộp trong 3 cm quanh khay ở cuối horizon}. Một đầu ra riêng cho **kết quả cấp episode** nhận initial state + toàn bộ chuỗi action `(T=200,8)`.
**Kiểm tra bắt buộc ở dataset builder:** chia theo nhóm, hash, alignment/shape, loại 20 bản trùng và replay phân kỳ, replay phần còn lại, unit test cửa sổ, kiểm tra rò rỉ (zero nhóm chung giữa các split).

## 4. Intervention protocol (CHỈ thiết kế; chưa chạy; cần bạn duyệt ngân sách và dataset)
**Mục tiêu:** tạo biến thiên action độc lập với state (feed‑forward) để (i) dạy WM tác động nhân quả của action, (ii) có ví dụ palm contact/cluster **trong vùng TRAIN** (F2), (iii) có tập kiểm tra action‑sensitivity cặp.
**Nguồn:** scripted expert (kế hoạch mở‑vòng) qua wrapper/subclass mới (không sửa `policies/baseline.py`); thực thi qua đúng đường `LabEnv.step` + `ActionAdapter` (clip/slew như hiện tại); ghi cùng schema với replay (contact, box in TCP).
**Biến can thiệp (mỗi biến một họ, kèm control không can thiệp):**
| Họ | Mức | Cơ chế |
|---|---|---|
| Độ sâu grasp Δa so với expert (+1 cm mặc định) | −3, −2, −1, 0(+), +1, +2 cm (6) | dịch điểm grasp dọc ngón (trực tiếp tái tạo cơ chế palm/cluster) |
| Thời điểm đóng gripper | ±2, ±5, ±10 bước (6) | dịch lệnh đóng |
| Nhiễu quỹ đạo khớp trái | σ = 0.005, 0.01, 0.02 rad, lọc thông thấp, 2 seed (6) | cộng vào mục tiêu khớp |
| Tốc độ nâng/vận chuyển | ×0.8, ×0.9, ×1.1, ×1.25 (4) | co giãn thời lượng đoạn |
| Độ cao thả | −1, +1, +2 cm (3) | **họ giữ riêng làm tập action OOD (chỉ test)** |
**Initial state ghép cặp:** 100 pose mới (seed 60, không phải 31–33; không trùng pose ACT/đánh giá hiện có): train 48 + val 12 + test‑ID 12 trong x ∈ [0.38, 0.45]; OOD‑low 16 (x 0.34–0.37); OOD‑high 12 (x 0.46–0.48); y, yaw trong dải train. **Mỗi pose chạy mọi mức** (thiết kế ghép) + 1 control ⇒ 25 biến thể × 100 pose = **2 500 episode**. Held‑out: pose (nhóm), chuỗi action (họ thả), vùng (OOD).
**Giới hạn an toàn:** mục tiêu khớp luôn qua clip ctrlrange + giới hạn tốc độ slew hiện hành (≤ 2.0 rad/s cho khớp tay), σ nhiễu tối đa 0.02 rad, gripper trong [0, 0.044] m; loại episode có NaN, hộp ra khỏi bàn/khay bất thường, xuyên thấu, hoặc IK thất bại (ghi lại, không loại im lặng); **chỉ trong mô phỏng**, không liên quan robot thật; không thay đổi vật lý, controller, checkpoint.
**Ngân sách ước tính:** không render ⇒ ≈ 2 s/episode/luồng (như replay) cộng lập kế hoạch IK ≈ 1–2 s ⇒ 2 500 episode ≈ 1–2 giờ trên 4 luồng; dung lượng ≈ 30–50 MB. (Nếu muốn thêm ACT chạy với nhiễu action có render: ~18 s/episode ⇒ 300 episode ≈ 1.5 giờ — tùy chọn, chưa đề xuất.)
**Danh sách quyết định cho bạn:** duyệt/điều chỉnh các họ và mức; số pose (100); có đưa vùng LOW vào train hay giữ OOD nghiêm ngặt; ngân sách giờ.

## 5. Kế hoạch WM-1 (đề xuất; chưa bắt đầu)
**Phạm vi:** V0‑A state‑only trên dữ liệu có sẵn (+ dữ liệu can thiệp nếu bạn duyệt ở WM‑2). Không ảnh, không V0‑B, không WAM.
**Budget WM‑1:** dataset builder + replay phần còn lại ≈ 1.5 giờ; baseline ≈ 2 giờ; V0‑A: mỗi cấu hình ≤ 1 giờ CPU (≈ 2 M tham số, 71 k cửa sổ), 3 cấu hình × 5 seed ensemble ≈ 15 giờ; đánh giá/ablation ≈ 4 giờ ⇒ **≈ 25 giờ CPU**, ≈ 2 tuần lịch (trong 6 tuần). Cổng ngân sách: dừng và xin duyệt nếu dự kiến > 35 giờ.
**Mô hình V0‑A:** encoder MLP cho `[S_hist, B_hist, C_hist, A_hist]` (4×30 = 120 → 256) + encoder chuỗi action tương lai (Transformer 2 lớp hoặc Conv1D) → decoder GRU K = 20 bước tự hồi quy điều kiện theo từng `a_t` → head delta (s, b), head contact (BCE cho tiếp xúc + MSE lực), head sự kiện `Y_e`; ensemble 5.
**Baselines (bắt buộc):** B0 persistence; B1 ridge trên (history, A_hist, A_fut) → delta; B2 MLP một bước tự hồi quy; B3 **outcome‑only classifier** (logistic/GBM từ initial state + trạng thái hiện tại, **không** có chuỗi action) để chứng minh WM dùng action; B4 oracle MuJoCo (cận trên, kiểm tra pipeline).
**Đánh giá riêng:** held‑out initial states (nhóm), held‑out chuỗi action (họ thả; chỉ khi có dữ liệu can thiệp), OOD‑low/OOD‑high, leave‑one‑checkpoint‑out.
**Cổng chấp nhận (đề xuất; đóng băng khi bạn duyệt WM‑1; không sửa sau khi xem kết quả).** Các tiêu chí đã đăng ký trong proposal giữ nguyên (V0 thắng B0/B1/B2 ở K = 20 với khoảng tin cậy bootstrap theo nhóm; AUROC ≥ 0.80 trên OOD‑low; thu hẹp phạm vi nếu không thắng B2/B3) và bổ sung:
| Gate | Nội dung | Ngưỡng đề xuất |
|---|---|---|
| G0 | Leakage audit lặp lại ở đầu và cuối WM‑1 (nhóm/hash/seed 31–33) | 0 nhóm chung giữa split, 0 seed cấm |
| G1 | Sai số theo horizon K = 1, 5, 10, 20 (MAE khớp [rad], sai số TCP/hộp [mm], a‑offset [cm]) | V0 < mọi baseline ở K = 20 trên test‑ID, cận dưới CI của mức cải thiện > 0, và ≥ 20 % tương đối so với baseline tốt nhất về vị trí hộp |
| G2 | OOD‑low / OOD‑high | không tệ hơn baseline tốt nhất (non‑inferior, biên 0) và có cải thiện dương ở ít nhất một OOD |
| G3 | Outcome: AUROC (nâng hộp, palm, trong khay) trên initial state giữ ra | ≥ 0.80 và hơn B3 ≥ 0.05; **palm chỉ được đánh giá nếu train có ví dụ dương (cần dữ liệu can thiệp hoặc LOW trong train); nếu không: "NOT EVALUABLE" (đăng ký trước)** |
| G4 | **Action sensitivity / intervention consistency** (cặp cùng history, khác chuỗi action) | đúng chiều tác động ≥ 80 %; Spearman(dự đoán, Δ thật trong sim) ≥ 0.70; hoán đổi chuỗi action giữa hai episode làm dự đoán dịch về kết quả của chuỗi mới |
| G5 | Calibration | ECE ≤ 0.05, Brier tốt hơn baseline; tương quan độ tản mát ensemble–sai số ≥ 0.3 |
| G6 | Latency / kích thước | rollout K = 20 ≤ 10 ms/CPU 1 luồng (batch 1), ≤ 5 M tham số |
**Rủi ro chính:** (1) F2: thiếu ví dụ palm trong train ⇒ G3 palm không đánh giá được nếu không có dữ liệu can thiệp; (2) dữ liệu closed‑loop gây confound (khắc phục bằng WM‑2); (3) chỉ 3.1 % replay phân kỳ nhưng 600 rollout bị state‑perturbation có action phụ thuộc state nhiễu; (4) tốc độ CPU biến động (đã gặp +25 %): dự phòng 30 %.
**Cần bạn quyết định trước khi bắt đầu WM‑1:** (a) duyệt dataset builder + replay phần còn lại (chỉ đọc/ghi `outputs/`); (b) duyệt/chỉnh intervention protocol và số giờ (mục 4); (c) LOW có vào train hay không; (d) đóng băng các ngưỡng G0–G6.
