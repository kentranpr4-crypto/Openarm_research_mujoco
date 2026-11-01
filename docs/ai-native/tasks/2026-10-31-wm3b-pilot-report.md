# WM-3B pilot report (DEVELOPMENT evaluation, 1 seed) — 2026-10-31

**Status: PARTIAL.** Không phải independent validation. Mọi so sánh là exploratory (1 seed, development splits của WM-2).

## 1. Phạm vi
Privileged-state world model (box/contact lấy từ MuJoCo, không có vision, không phải Sim2Real).
WM3-B = ridge dự đoán quỹ đạo khớp tương lai (closed form) → GRU nhận (ridge forecast + state history + candidate actions) → object/contact tương lai. Không dùng GT future joints khi inference; không có joint loss trong GRU.

## 2. Freeze (trước khi train)
- Gate v2 config: sha256 `f6c323914a99b4bc884ffc88c4c538615ad3a1ba10649fd95e38282acbaa0627` (`outputs/wm3/gates_v2_config.json`)
- Model config: sha256 bắt đầu `3b74deaa35c729f5`; chi tiết trong `outputs/wm3/freeze.json`
- Unit tests: 19 gate tests + 12 WM3-B tests, tất cả PASS (`outputs/wm3/*_unit_tests.json`)

## 3. Huấn luyện
1 seed, 20 epoch, AdamW 1e-3 cosine, batch 128, 3 threads; 2137 s, best epoch 19, 879,634 tham số.
**Deviation D-1:** trainer crash sau khi train xong (sai input dim 120 thay vì 88 ở MLP reference loader). `wm3b.pt` đã lưu; reload bằng `WM3_REUSE=1`, không train lại. Log từng epoch bị mất → chỉ còn best epoch và thời gian từ stdout.

## 4. Kết quả chính (test_id, K=20)
| Mô hình | Box err (mm) | Joint err (rad) |
|---|---|---|
| WM3-B | 2.53 | 0.0014 (= ridge) |
| WM2-C0 (frozen) | 2.77 | 0.0065 |
| Hybrid post-hoc ridge+GRU | 2.77 | — |
| MLP | 5.11 | — |
| Ridge | 10.79 | 0.0014 |

OOD box (mm, 3 split OOD): WM3-B 4.61 / 6.60 / 4.59; C0 4.55 / 7.20 / 4.95.
Gate v2 G3: close AUROC 0.9993, palm pooled 0.9924. G4 Spearman theo family: depth 0.913, noise 0.949, release 0.941, speed 0.978; timing không eligible. G5 ECE 0.0063 / 0.0084. G6 latency median 4.52 ms.

## 5. Gate v2 verdicts
- WM3-B: G0–G6 PASS
- WM2-C0: G1 FAIL (joint), G2–G5 PASS
- Hybrid post-hoc: G1–G3, G5 PASS
- Negative controls OK: persistence G1 FAIL; action-shuffle box 2.53→52.4 mm; label-shuffle AUROC 0.4995.

## 6. Diễn giải & caveat
1. Lợi ích của WM3-B so với hybrid post-hoc là **nhỏ** (box 2.53 vs 2.77 mm) và 1 seed → chưa kết luận được kiến trúc "ridge-forecast-in-the-loop" tốt hơn hybrid.
2. Phần thắng rõ nhất đến từ ridge giải joint (0.0014 rad), không phải từ GRU.
3. Gate v2 được thiết kế sau khi thấy kết quả WM-2 → khoan dung (C0 chỉ fail mỗi joint). PASS chỉ nên đọc cùng các con số.
4. Dữ liệu là development splits đã dùng ở WM-2, **không độc lập**. Independent held-out evaluation: **NOT RUN**.
5. Ridge forecast trên train windows là in-sample → GRU thấy forecast sạch hơn lúc test (nguy cơ train/test mismatch; ablation trong `evaluation_v2.json`).
6. Mất log per-epoch (D-1).

## 7. AC
| AC | Kết quả | Evidence |
|---|---|---|
| Freeze trước train | PASS (tự kiểm) | freeze.json hashes |
| 8 nhóm unit test | PASS (tự kiểm) | unit_tests.json |
| Không GT future joints / không joint loss | PASS (tự kiểm) | test_wm3b.py |
| So sánh C0, hybrid, ridge/MLP | PASS | evaluation_v2.json |
| Independent held-out | NOT_CHECKED | chưa sinh dữ liệu |
| Training curve | FAIL | log mất (D-1) |

## 8. Chưa commit
Các file `eval/wm3*.py`, `eval/test_wm3*.py` và báo cáo này chưa commit; `outputs/` ngoài git.
