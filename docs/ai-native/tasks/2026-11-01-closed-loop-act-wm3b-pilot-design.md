# Closed-loop ACT + WM3-B Pilot — DESIGN (không thực thi) — 2026-11-01

**Trạng thái: PARTIAL (design-only).** Không training, không chạy closed-loop pilot, không đổi frozen protocol/checkpoint. Tự kiểm bởi cùng operator.
Mọi kết quả của pilot sẽ là **DEVELOPMENT** (sim, privileged state, 1 seed/model), không phải independent, không Sim2Real.

## 0. Audit candidate ACT — kết quả thực tế của lần chạy
| Mục | Trạng thái |
|---|---|
| Script `eval/act_candidate_audit.py` (inference-only, cap tự đặt 0.05 CPU-h = 180 s) | Chạy lúc 06:37 UTC, **dừng do chạm cap (`CPU cap reached at act 0.37 20`) trước khi ghi `act_candidate_audit.json`** |
| Số liệu diversity ensemble / MC-dropout, clip-slew, spread J20 | **NOT_CHECKED** (không có JSON). Không suy diễn số từ log |
| CPU đã tiêu | ≈ 0.05 CPU-h (cap), tính vào ledger |
| Nguyên nhân | Chưa chẩn đoán chi tiết; nghi load 6 checkpoint + dựng expert episodes + render quá cap. **Chưa chạy lại** vì cần quyết định ngân sách (xem §9 Q1) |

Fact đã kiểm bằng đọc code/config (không cần chạy): ACT `use_vae=False` ⇒ chunk 20×16 **xác định** cho cùng observation; `chunk_size=n_action_steps=20`, không temporal ensembling; ra 16-D khớp tuyệt đối; qua `ClipAndSlew` (arm 2.0 rad/s, gripper 0.2 m/s, 20 Hz).

## 1. Kiến trúc closed-loop (đề xuất)
```
MuJoCo LabEnv ──obs(state16 + front + wrist_left)──► ACT bank (N nguồn) ──► N raw chunks (20×16)
      │                                                         │
      │ privileged observer (box pose, contacts, 4 hàng lịch sử)│ slew_chain: clip_and_slew tuần tự từ
      ▼                                                         ▼ lệnh đã thực thi cuối → executed seq (20×8 tay trái)
WM window builder (H=4: S/B/C rows + A_hist executed) ──► WM3-B (frozen) ──► dự đoán 20 bước
                                                                  │
                                         J = d_xy + |z−0.384| + 0.05·palm + 0.05·(max rise<0.05)  (frozen)
                                                                  ▼
                                   chọn chunk (A2: argmin J, tie-break SHA-1) ──► thực thi 20 bước ──► lặp
```
Quyết định thiết kế và lý do:
1. **Ranking dùng executed actions, không dùng raw ACT output.** WM3-B được huấn luyện trên action đã qua clip/slew (servo lag 1–2 bước); đưa raw target sai ngữ nghĩa. Vì vậy mỗi candidate được chuyển bằng `slew_chain` từ lệnh thực thi cuối cùng.
2. **Receding horizon = open-loop trọn chunk 20 bước rồi re-plan** (khớp `n_action_steps=20`, không đổi semantics ACT). Không cắt chunk sớm trong pilot tối thiểu (cắt sớm làm ACT ngoài phân phối huấn luyện).
3. **Chunk đầu tiên không ranking được** (cần 4 hàng lịch sử): dùng candidate 0 (ACT mặc định) ở cả 3 arm ⇒ arm giống nhau trước điểm quyết định đầu.
4. **Hạn chế "myopic":** J frozen được thiết kế cho rollout dài (decision-utility: autoregressive 20-step blocks từ t0=4); áp dụng tại cuối **một** block 20 bước ở giữa episode là *ngoại suy* khỏi nghiên cứu đã làm. Giai đoạn approach, J gần như bằng nhau giữa candidate (d_xy lớn chưa đổi) ⇒ nhiều tie. Không đổi J (frozen); chỉ báo cáo tỉ lệ tie.
5. **Privileged observer bắt buộc** cho hàng B, C (pose hộp, contact) — lấy trực tiếp từ MuJoCo; ACT vẫn dùng ảnh + state.

## 2. Sinh candidate không phá ngữ nghĩa action
| Phương pháp | Giữ semantics? | Diversity | Ghi chú |
|---|---|---|---|
| **Ensemble 6 checkpoint xsweep đã freeze** (c0 ×3, s1 ×3 seed) | Có (mỗi cái là policy hợp lệ) | Kỳ vọng trung bình, thật | **Khuyến nghị chính.** Không sửa checkpoint. Lưu ý c0 vs s1 khác recipe ⇒ confound recipe/seed |
| MC-dropout (dropout 0.1, `model.train()`) | Chỉ khi phương sai nhỏ | Chưa biết | Chưa được kiểm; ngoài phân phối huấn luyện; chỉ thử ở audit |
| Nhiễu đầu vào / nhiễu action | **Không** (phá consistency chunk, vượt slew) | Tùy ý | Loại bỏ |
| Chunk cũ (stale) | Có nhưng lệch pha | Thấp | Loại |
Ràng buộc: chỉ nhận candidate nếu sau `slew_chain` số vi phạm clip/slew ≤ ngưỡng đăng ký trước (đề xuất: ≤ 5 % bước bị clip mạnh), nếu không candidate bị loại và ghi lại.

## 3. Ba arm so sánh (paired)
| Arm | Mô tả | Mục đích |
|---|---|---|
| **A0** ACT baseline | Một checkpoint cố định (ví dụ c0_s0), chạy như cũ | Mốc; có thể **tái dùng kết quả xsweep có sẵn** nếu cùng pose/seed/điều kiện — phải xác nhận bằng so sánh trace trước khi dùng, nếu không thì chạy lại |
| **A1** ACT + random pick | Cùng N candidate, chọn ngẫu nhiên (RNG cố định theo episode) | Tách tác dụng của *ensemble switching* khỏi *ranking*. **So sánh chính là A2 vs A1** |
| **A2** ACT + WM3-B ranking | Cùng N candidate, chọn argmin J | Hiệu ứng của world model |
Lưu ý: A1 cũng thay đổi hành vi so với A0 (chuyển giữa các policy mỗi 20 bước), nên A2−A0 gộp cả hai hiệu ứng.

## 4. Protocol
- **Paired initial states:** cùng pose (x,y,yaw) cho 3 arm; pose lấy từ lưới xsweep hiện có (low-x là vùng ACT thất bại: palm contact / grasp quá sâu) — **không dùng seed 31–33**; pose pilot khai báo và băm (hash) trước khi chạy.
- **Đơn vị lấy mẫu = pose** (không phải episode); lặp lại ngẫu nhiên hóa trong pose chỉ để ước lượng noise, không coi là độc lập.
- **Endpoint chính (đăng ký trước):** tỉ lệ success theo pose; hiệu A2 − A1 (và A2 − A0 phụ), paired bootstrap theo pose, 10 000 resample, seed 0, CI 95 %.
- **Phụ:** thời điểm chạm, palm-contact rate, max box rise, J thực đo cuối episode, tỉ lệ tie, tỉ lệ A2 chọn ≠ candidate 0.
- **Phân loại thất bại:** `success / no_pick / dropped_or_misplaced` (từ `eval/evaluate.py::classify`) + cờ `palm_contact`, `too_deep_grasp`, `observer_error`, `ik/plan_fail`, `safety_violation`.
- **Kết luận cho phép:** SUPPORTED_BETTER nếu CI dưới > 0; INCONCLUSIVE nếu cắt 0; SUPPORTED_WORSE nếu CI trên < 0. Không đổi ngưỡng sau khi thấy dữ liệu.

## 5. Latency & compute (ước lượng, chưa đo trong audit này)
| Hạng mục | Số liệu |
|---|---|
| ACT query (1 thread, batch 1) | ~50–110 ms/checkpoint (đo trước đây) |
| N=6 ensemble | ~0.3–0.7 s/quyết định ⇒ **không real-time 20 Hz**; chạy sim đồng bộ (sim dừng khi tính) — chấp nhận được cho nghiên cứu, không cho robot thật |
| WM3-B | median 3.7–4.5 ms/forward (G6) ⇒ không đáng kể |
| Episode A0 | ~18 s wall (xsweep) |
| Episode A1/A2 (≈10 quyết định × ~0.5 s + sim) | ước ~25 s wall (suy luận từ số liệu trên, chưa đo) |
| Pilot tối thiểu 20 pose × 3 arm | ≈ 60 episode ⇒ ước 0.4–0.6 CPU-h (≈ 1 CPU/episode) |
Ledger: đã dùng ≈ 5.9–10.3 CPU-h (+≈0.05 audit này), còn ≈ 11.6–16 / 22. Pilot tối thiểu < 5 % ngân sách còn lại. Khả thi trên Cloud (CPU-only, 1 luồng).

## 6. Giới hạn privileged-state của WM3-B (phải nêu trong mọi báo cáo)
- WM3-B nhận pose hộp và contact **chính xác từ simulator**, không từ ảnh. Trên robot thật cần perception (vision/ước lượng) ⇒ lợi ích đo được là **cận trên (oracle observer)**, không phải kết quả Sim2Real.
- Sai số dự đoán ở closed-loop sẽ khác độ chính xác offline vì phân phối trạng thái do ACT tạo ra (đặc biệt vùng low-x) có thể ngoài phân phối của dữ liệu scripted-expert.
- Yaw hộp: WM3-B dự đoán yaw kém hơn persistence (independent eval) — J không dùng yaw, nhưng dùng d_xy/z/palm.
- 1 seed/model, G3 frozen Gate v2 FAIL trong independent eval (lỗi thiết kế gate), Gate v3 chưa áp dụng cho model này.

## 7. Rủi ro
| ID | Rủi ro | Mức | Giảm thiểu |
|---|---|---|---|
| R1 | ACT xác định ⇒ ensemble ít đa dạng ⇒ A2≈A1 | Cao | Audit diversity trước (Stage 0); dừng nếu spread J nhỏ |
| R2 | J myopic + tie ở giai đoạn approach | Cao | Báo cáo tỉ lệ tie; không đổi J |
| R3 | Confound recipe (c0 vs s1) và seed trong ensemble | TB | Báo cáo theo nguồn; đây là pilot |
| R4 | Ceiling/headroom: nếu A0 đã ~90 % success thì không thấy cải thiện | TB | Chọn pose low-x nơi A0 thất bại; ghi headroom |
| R5 | Right-arm idle brittleness (C0/V1) | TB | Ghi rõ; chỉ ranking trên tay trái |
| R6 | Observer sai / lệch cửa sổ lịch sử | TB | Unit test cửa sổ trên expert episode trước khi chạy |
| R7 | Chồng CPU vượt trần | Thấp | Hard stop §8 |
| R8 | Diễn giải thành bằng chứng độc lập | TB | Gắn nhãn DEVELOPMENT |

## 8. Acceptance criteria & stop conditions (trước khi chạy)
**AC thiết kế (cho phép bắt đầu implementation):**
- AC-01 Audit Stage 0 hoàn tất, có JSON; ≥ 2 candidate cho cùng observation có chênh lệch executed (L∞ trên tay trái) vượt ngưỡng đăng ký (đề xuất 0.02 rad) ở ≥ 50 % điểm quyết định.
- AC-02 Spread J20 giữa candidate ≥ 0.01 (đơn vị J) ở ≥ 30 % điểm quyết định (có headroom để ranking có tác dụng).
- AC-03 Unit test cửa sổ WM (A_hist/A_fut khớp executed) PASS trên expert episode.
- AC-04 Candidate sau slew_chain vi phạm safety = 0.
- AC-05 Pose pilot và RNG được hash/freeze trước khi chạy; không dùng seed 31–33.
**Stop conditions (dừng và báo):** chạm cap CPU; AC-01 hoặc AC-02 FAIL (khi đó pilot không có ý nghĩa — báo cáo "no diversity/headroom", không cố cứu); bất kỳ vi phạm safety; observer/NaN; thay đổi frozen hash; phát hiện A0 ≈ ceiling toàn pose.

## 9. Cấu hình pilot tối thiểu đề xuất & quyết định cần chủ việc
- **Stage 0 (inference-only audit, chạy lại, sửa lỗi cap):** 1 pose, 3 thời điểm, ensemble 6 + (tùy chọn) MC-dropout 8; cap đề xuất **0.15 CPU-h** (cap 0.05 quá thấp, đã hết trước khi ghi kết quả).
- **Stage 1 (pilot):** 20 pose low-x/mid-x (đóng băng trước), 3 arm, N=6 ensemble, chunk 20 mở trọn, chỉ chạy nếu AC-01..AC-05 PASS; cap **0.8 CPU-h**.
- Q1: Duyệt chạy lại Stage 0 với cap 0.15 CPU-h (sửa script để ghi JSON từng phần trước khi dừng)? 
- Q2: Chấp nhận dùng ensemble 6 checkpoint (confound recipe) làm nguồn candidate, hay giới hạn trong một recipe (3 seed)?
- Q3: Tái dùng kết quả A0 xsweep hay chạy lại A0 để paired chặt chẽ?
- Q4: Chấp nhận J myopic (ngoại suy) hay muốn pre-register biến thể J khác (cần phê duyệt vì J đã freeze)?

## 10. AC của tài liệu này
| AC | Kết quả |
|---|---|
| Đủ 9 mục yêu cầu | Có (mục 0–9), audit candidate **NOT_CHECKED** (số liệu) |
| Không training / không chạy pilot / không đổi frozen | PASS (tự kiểm) |
| Không xóa outputs/, không commit | PASS (tự kiểm); file mới untracked |
| Số liệu compute | Ước lượng, chưa đo ⇒ ghi rõ |

## 11. Stage 0 audit — KẾT QUẢ (2026-11-01, DEVELOPMENT, inference-only) — **PASS (có caveat)**
Raw: `outputs/closed_loop_design/act_candidate_audit_stage0.json`; script `eval/act_candidate_audit.py` (v2; bản lần 1 giữ ở `outputs/closed_loop_design/act_candidate_audit_v0_attempt1.py`, log cũ giữ nguyên). Tự kiểm, không phải kiểm độc lập.
- CPU thực dùng **70.6 s = 0.0196 CPU-h** (cap 0.15), 1 lần chạy, không retry. COMPLETE 12/12 decision points (3 pose x 4 phase group), 6 checkpoint, không MC-dropout.
- Diversity (L∞ executed, arm joints, median qua các điểm): approach 0.066 / grasp 0.075 / lift 0.049 / place 0.041 rad; gripper L∞ ≤ 0.005 m. ≥2 candidate khác biệt (ngưỡng 0.02 rad) ở 12/12 điểm; mixed: 5–6 candidate khác biệt.
- Seed vs recipe: L∞ cùng recipe khác seed 0.065 rad, khác recipe 0.075 rad (+15 %); |ΔJ| 0.019 vs 0.017 ⇒ biến thiên chủ yếu do **seed**, không phải recipe.
- J spread (t+20): ≥0.01 ở 75 % điểm (C0, mixed), 42 % (S1); nhưng spread ≈0.05 ở place bằng đúng trọng số palm/lift (0.05) ⇒ phần lớn là bước nhị phân, không phải chênh lệch liên tục. Approach: ties (|ΔJ|≤5e-3) 67–100 %.
- Rank stability (Spearman J@t+20 vs @t+15, chỉ chẩn đoán): approach 0.3–0.7, lift −0.2–0.7, grasp 0.4–0.7, place ≈1.0; argmin trùng 0–100 %.
- Safety: 72/72 chunk hợp lệ (rate và ctrlrange) sau slew_chain; nhưng 955 giá trị bị clip ctrlrange (chưa xác định chiều nào; raw vs executed tay trái lệch tối đa 0.187) và 55 giá trị bị slew.
- Latency (median): load ckpt 1.0 s/cái; ACT 78 ms (p95 114); slew_chain 2.4 ms/6; WM3-B 8.7 ms/6 batch; một lần load WM+ridge 11 s; sim+render ~5.4 s/pose.
- Caveat: 12 điểm không độc lập, observation lấy từ trạng thái expert (không phải trạng thái do ACT tạo ra), y=0.10 và yaw=0 duy nhất, không có outcome thật nên "headroom" chỉ là J dự đoán bởi WM.
