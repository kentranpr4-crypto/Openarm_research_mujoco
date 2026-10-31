# Proposal — OpenArm Action-Conditioned World Model V0
Ngày 2026-10-31 · **DRAFT, chưa được duyệt; không train, không commit/push.** Chỉ là đề xuất cho bước tiếp theo sau khi dừng các thí nghiệm ACT coverage/normalization. Không triển khai World Model/WAM/DreamZero trong đợt này; không dùng test seeds 31–33; không sửa upstream OpenArm.

## 1. Mục tiêu và phạm vi
**Câu hỏi nghiên cứu:** một mô hình động học học từ dữ liệu có điều kiện theo action chunk có thể (a) dự đoán kết quả của một action chunk (nắm được hộp không, đặt có đạt dung sai 3 cm không, hộp có chạm palm không) trước khi thực thi, và (b) làm điều đó tốt hơn các baseline đơn giản **ngoài vùng huấn luyện** (box x < 0.38 m, x > 0.45 m — chính là vùng ACT thất bại ở x‑sweep) hay không?
**V0 (state-space, không ảnh):** dự đoán tương lai của trạng thái robot, trạng thái hộp (ground truth từ mô phỏng) và tín hiệu tiếp xúc, cho một chuỗi action cho trước. Các use‑case sau chưa nằm trong V0: planning/MPC trên ACT chunks, policy evaluation trong "imagination" (cần dự đoán ảnh — V1), WAM/DreamZero.
**Vì sao hợp lý với dữ liệu hiện có:** x‑sweep và H2/H3 đã cho nhãn cơ chế (grasp depth, palm contact, cluster) và các trajectory thành công lẫn thất bại ở 15 giá trị x; V0 kiểm tra xem các cơ chế đó có học được từ (state, action) hay không.

## 2. Data contract
**Nguồn có sẵn (đã kiểm đếm file):** ~2 176 rollout closed-loop của ACT (1 456 trong `outputs/eval/*`, 360 x‑sweep, 360 coverage; 12 checkpoint C0/S1/EXT/VOL + V1 audit), mỗi file có `state (T,16)`, `action (T,16)`, `box_pos (T+1,3)`, `tcp_pos (T+1,3)`, `box_speed (T+1)`, (một phần có `raw_action`); **1 230 bản replay** có thêm box/TCP trong khung TCP, yaw, tilt, lực finger/palm/table/tray, vị trí tiếp xúc (`eval/h23_replay.py`, ≈ 2 s/episode, có thể replay toàn bộ ≈ 2 176 episode trong ~20 phút); 120 expert demo có ảnh (3 camera 120×160), `phase`, `box_quat`. T = 200 bước, 20 Hz (demo ≈ 138).
**Thiếu/giới hạn:** (i) rollout ACT không lưu ảnh (có thể render lại bằng replay tất định; 3 % episode phân kỳ); (ii) dữ liệu là closed‑loop quan sát được: action phụ thuộc state ⇒ nhiễu nhân quả, thiếu đa dạng can thiệp (xem W4); (iii) ground‑truth hộp chỉ có trong sim ⇒ V0 dùng cho đánh giá/chẩn đoán, chưa triển khai được trên robot thật; (iv) vùng x < 0.34 hoặc > 0.48 không có dữ liệu.
**Schema mỗi bước (tensor, float32, đã chuẩn hóa train‑only với floor kiểu `norm_contract`; các chiều tay phải idle bị loại hoặc floor):**
| Ký hiệu | Shape | Nội dung |
|---|---|---|
| `s_t` | (16) | joint tay trái 7 + gripper (m), tay phải 7 + gripper (idle) |
| `b_t` | (7) | box pos (3) + yaw (1) + vận tốc (1) + tilt (1) + `box_in_tcp` a‑offset (cm) (1), từ GT sim |
| `c_t` | (6) | tổng lực finger, số contact finger, vị trí contact a/c (TCP frame), lực palm, lực table/tray |
| `a_t` | (16) | action thực thi (sau clip/slew); `a_raw_t` (16) là output policy trước clip |
| `m` | (3) | metadata mỗi episode: nguồn (expert/C0/S1/EXT/VOL), box pose khởi tạo (x,y,yaw), nhãn pick/place/cluster/palm |
**Mẫu huấn luyện (cửa sổ):** lịch sử H = 4 bước (0.2 s), tầm dự đoán K = 20 bước (= 1 chunk ACT, 1 s). Input `X_hist (B, H, 29)` = [s, b, c]; `A_fut (B, K, 16)`; target `Y (B, K, 29)` (dự đoán delta so với bước trước, chuẩn hóa) + nhãn kết quả cấp episode `(B, 4)` = pick, place, palm, cluster. Với stride 5 ≈ 35 cửa sổ/episode ⇒ ≈ 76 k cửa sổ.
**Chia dữ liệu (theo episode, không theo cửa sổ):** train = x ∈ [0.38, 0.45] (vùng support gốc), val = 10 % episode cùng vùng, **test‑OOD‑low = x ∈ [0.34, 0.38)**, **test‑OOD‑high = x ∈ [0.46, 0.48]**, thêm test theo checkpoint bị giữ ra (leave‑one‑checkpoint‑out). Không dùng test seeds 31–33. Dữ liệu EXT (x thấp) chỉ dùng như một biến thể train có kiểm soát ở bước ablation, không lẫn vào test OOD.
**Kiểm tra bắt buộc trước khi train:** unit test alignment (state/action/box khớp raw), hash dữ liệu, chia episode rời nhau, thống kê train‑only tính lại độc lập, kiểm tra replay‑vs‑live (loại 3 % phân kỳ khỏi các nhãn từ replay).

## 3. Kiến trúc
- **V0 chính (~1–3 M tham số, CPU):** mô hình động học dư (residual) tự hồi quy: encoder MLP cho `X_hist` → latent 256 → GRU 2 lớp (hoặc Transformer 2 lớp, 4 head) điều kiện theo `a_t` → head delta `Δ[s,b,c]` + head xác suất contact (BCE) + head kết quả episode (pick/place/palm/cluster). Dự đoán K = 20 bước tự hồi quy, huấn luyện multi‑step (scheduled sampling), loss = MSE trên delta chuẩn hóa + BCE + loss rollout. Ensemble 5 seed để ước lượng bất định (độ tản mát giữa các thành viên).
- **Baseline:** B0 persistence (giữ nguyên trạng thái, biết lực = 0); B1 hồi quy tuyến tính/ridge trên (s, a) ⇒ delta; B2 MLP một bước tự hồi quy; B3 **oracle** = chính MuJoCo (cận trên, kiểm tra pipeline); B4 bộ phân loại kết quả từ đặc trưng thô (x, y, yaw, a‑offset kế hoạch) để so với head kết quả.
- **Mở rộng sau (V0b, nếu còn thời gian và được duyệt):** thêm latent thị giác từ CNN nhỏ huấn luyện cùng (backbone pretrained bị chặn bởi proxy). Không thuộc phạm vi V0.

## 4. Metric và tiêu chí chấp nhận (đăng ký trước khi chạy)
1. **Dự đoán chuỗi (K = 1, 5, 10, 20):** MAE joint tay trái (rad), sai số vị trí TCP (mm), sai số vị trí hộp (mm), sai số `box_in_tcp` a‑offset (cm), AUROC/Brier của contact (palm, finger). Tiêu chí: V0 tốt hơn B0, B1, B2 ở K = 20 trên cả test‑ID và test‑OOD‑low, mức cải thiện đo bằng khoảng tin cậy bootstrap theo episode.
2. **Dự đoán kết quả:** AUROC/Brier/calibration của pick, place (dung sai 3 cm), palm contact, cluster; đặc biệt **AUROC dự đoán thất bại tại x < 0.38** (điều ACT cần). Tiêu chí: AUROC ≥ 0.80 trên test‑OOD‑low và vượt B4 một biên đã định trước.
3. **Tổng quát hóa vùng:** so ID vs OOD‑low vs OOD‑high; độ suy giảm phải được báo cáo, không chỉ trung bình.
4. **Hiệu quả:** độ trễ rollout K = 20 trên CPU 1 luồng (mục tiêu ≪ thời gian thực), bộ nhớ, thời gian train.
5. **Bất định:** tương quan giữa độ tản mát ensemble và sai số; độ phủ khoảng dự đoán.
6. **Báo cáo trung thực:** mỗi chỉ số có bất định giữa **seed huấn luyện WM** và giữa **checkpoint nguồn/pose block**; tách rõ kết quả đăng ký trước vs khám phá; nêu confound closed‑loop.
**Quy tắc quyết định (đăng ký):** nếu V0 không thắng B2 ở K = 20 (OOD‑low) sau W3, thu hẹp thành bộ dự đoán kết quả (mục 2) thay vì động học đầy đủ; nếu không thắng B4, kết luận "chưa học được cơ chế" và dừng.

## 5. Kế hoạch 6 tuần (CPU 4 core, không GPU)
| Tuần | Việc | Sản phẩm / gate |
|---|---|---|
| W1 | Dựng dataset builder: replay toàn bộ rollout, shard npz, schema + unit test, chia episode, baseline B0/B1/B4 | `data contract v1` + bảng baseline; gate: test alignment/hash đạt |
| W2 | V0 + vòng huấn luyện CPU, loss multi‑step, B2; metric 1 | Gate go/no‑go: V0 vs B0/B2 ở K = 20 trên val |
| W3 | Head kết quả/contact; đánh giá OOD‑low/high; ensemble | Metric 2–3; quyết định thu hẹp theo quy tắc ở trên |
| W4 | Dữ liệu can thiệp: expert + nhiễu action nhỏ trong sim (≈ 300–500 episode không ảnh, ước ~ vài giờ CPU) để giảm confound closed‑loop; huấn luyện lại | So sánh có/không dữ liệu can thiệp |
| W5 | Ablation (head contact, lịch sử, dữ liệu EXT), calibration, phân tích lỗi theo cơ chế H2/H3; khả thi V0b (chỉ đánh giá, không bắt buộc) | Bảng ablation + phân tích lỗi |
| W6 | Tái lập, tài liệu, báo cáo kỹ thuật/trình bày | Repo có script chạy lại + báo cáo |
**Ngân sách tính toán ước tính:** dataset ~ 1 h replay + vài giờ dữ liệu can thiệp; mỗi cấu hình train < 1 h (mô hình nhỏ), 5 seed ensemble × vài cấu hình ≈ 20–40 giờ CPU tổng; trong ngân sách 6 tuần.
**Rủi ro:** (1) dữ liệu closed‑loop ⇒ mô hình có thể học "policy" chứ không học động học (giảm bằng W4 và đánh giá OOD); (2) GT hộp không có trên robot thật (V1 cần ước lượng thị giác); (3) 3 % replay phân kỳ (loại và báo cáo); (4) CPU chậm hơn và biến động (đã gặp 25 % chậm hơn): dự phòng 30 % thời gian; (5) vùng contact giàu động học khó dự đoán ⇒ dùng contact làm nhãn phân loại thay vì hồi quy lực.

## 6. Quyết định cần bạn đưa ra trước khi bắt đầu
1. Use‑case chính của V0: **dự đoán thất bại/kết quả** (đề xuất), đánh giá policy ngoại tuyến, hay planning.
2. Có cho phép **tạo dữ liệu mô phỏng mới** (can thiệp action) ở W4 không, và trong ngân sách giờ CPU nào.
3. State‑only trong 6 tuần (đề xuất) hay đưa latent thị giác vào phạm vi.
4. Tiêu chí chấp nhận (AUROC, biên so với baseline) có cần chỉnh không.
Không có bước nào ở trên được khởi động; chỉ bắt đầu sau khi bạn phê duyệt riêng.
