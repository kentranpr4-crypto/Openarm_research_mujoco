# SuperMimic-OpenArm P1 (Block Pushing): sửa môi trường, kiểm định khoa học, GO/NO-GO cập nhật

Ngày: 2026-11-11 · Nhánh `main-f9g0uh` (HEAD `fcabee9`) · **Trạng thái: DONE cho phạm vi được giao (sửa env + expert + kiểm định WM); pilot GPU chưa chạy.** Tự kiểm (một người chạy), chưa có kiểm độc lập.
Giữ nguyên: không train GPU, không commit/push, không đụng OpenArm-WAM, G2a/G2b, sealed poses, `sim/openarm_mujoco`, checkpoint/dataset hiện có. Đánh giá P0 độc lập là việc riêng (xem mục 9).

> Nhãn: **[đã đo]** = chạy trong phiên này, có file kết quả trong `outputs/p1_validation/`; **[suy luận]** = diễn giải, chưa kiểm bằng thực nghiệm riêng; **[chưa kiểm]** = cần máy/GPU của chủ việc.
> Môi trường đo: cloud sandbox 4 CPU, không GPU. Mọi con số VRAM/GPU là **[chưa kiểm]**.

## 1. Kết luận ngắn

| Hạng mục | Trước (báo cáo sáng nay) | Sau (đo lại) | Kết luận |
|---|---|---|---|
| Ma sát | scale ≤ 1,0 không đổi gì (quãng trượt cùng 0,02342 m) | μ hiệu dụng đo trên `d.contact` = 0,56 / 0,68 / 0,80 / 0,92 / 1,04; quãng trượt 22,8 / 18,9 / 16,2 / 14,1 / 12,3 mm, khớp `v²/(2μg)` trong ≤ 2% | **Đã sửa** |
| Khối lượng | cố định 0,05 kg | 0,031–0,080 kg (300 mẫu, 300 giá trị khác nhau); quán tính = công thức khối hộp, sai số 2e-16 | **Đã sửa** |
| Goal không hợp lệ | ≈ 2% nằm trong khay, goal bị clip | 0 goal không hợp lệ; 35,7% mẫu đầu bị loại và lấy lại | **Đã sửa** |
| Expert, tiêu chí chặt | 20/48 = 41,7% (nhấc hộp > 1 cm: 19 ca) | **260/300 = 86,7%** (Wilson 82,4–90,1%) trên 3 tập mới; nhấc hộp: 1 ca | **Đạt ≥ 80%** |
| Kiểm định môi trường | 8 PASS / 2 FAIL | **11 PASS / 0 FAIL**; test 54/54 | **Đạt** |
| World Model | chưa đo chế độ chạm kéo dài | thắng baseline ở mọi chế độ chạm (vị trí); thua ở vùng tự do; xem mục 5 | **Có điều kiện** |
| **GO / NO-GO pilot P1** | NO-GO | **GO CÓ ĐIỀU KIỆN** cho pilot 1 seed ≤ 2 h (mục 7). **NO-GO** cho multi-seed / nhiều giờ | |

## 2. Sửa môi trường (Phase B) — nguyên nhân gốc và bằng chứng

**Ma sát [đã đo].** MuJoCo lấy `max` ma sát của hai geom khi `geom_priority` bằng nhau. Bàn: 0,8; hộp: 0,8×scale. Scale ≤ 1,0 nên không có tác dụng. Ngón tay có `priority=1` nên ma sát ngón–hộp không phụ thuộc hộp (đã kiểm). Sửa: đặt `priority=1` cho geom hộp ở chế độ `pair_exact` (mặc định cho pushing). Khi đó cặp hộp–bàn dùng đúng ma sát của hộp; cặp ngón–hộp vẫn là `max(1,0; μ_hộp)` ≈ không đổi. Chế độ `legacy` vẫn tái hiện lỗi cũ (có test).
Bằng chứng: μ hiệu dụng = `0,8×scale` với sai số 0 (300 reset); quãng trượt giảm đơn điệu ≥ 5% mỗi bậc; độ lệch so với Coulomb ≤ 2,0% (lớn nhất 2,00% tại scale 1,15).

**Khối lượng và quán tính [đã đo].** Hệ số khối lượng U(0,6; 1,6) × 0,05 kg. Hộp 5×4×4 cm, mật độ đồng nhất, nên quán tính chính tỉ lệ khối lượng (kiểm bằng công thức `m/12·(b²+c²)`). Mật độ ngụ ý 383–999 kg/m³ (xốp đến gỗ nhẹ). Gọi `mj_setConst` trước khi đặt trạng thái. Gia tốc dưới lực ngang 0,6 N: 14,5 / 4,6 / 0,06 m/s² cho khối lượng 0,030 / 0,050 / 0,080 kg. Quãng trượt tự do **không** phụ thuộc khối lượng (đúng vật lý, lực ma sát ∝ m). Mọi tham số mẫu (khối lượng, quán tính, μ, yaw, goal, số lần lấy lại) được ghi trong `episode_info` và trong `sampled_parameters` của `env_validation_v3.json`.

**Goal hợp lệ [đã đo].** Lấy mẫu loại bỏ, tất định theo seed (lần thử đầu dùng cùng thứ tự rút mẫu như bộ lấy mẫu cũ). Loại khi: goal ngoài vùng làm việc (không còn clip); hộp (bán kính ngoại tiếp + 5 mm) chồng lên khay/khối lập phương/trụ ở điểm đầu hoặc đích; đường đẩy thẳng bị chặn. Kết quả 2000 seed: lần đầu hợp lệ 65,0%; ngoài vùng làm việc 15,3%; chồng vật cản 19,3%; đường bị chặn 0,5%; lần thử tối đa 7. Khoảng cách goal còn đúng 0,07–0,13 m. Hình học vật cản được so với mô hình biên dịch bằng test. **Tiêu chí thành công chặt không đổi** (3 cm, 0,03 m/s, nhấc ≤ 1 cm, giữ 5 bước, 150 bước; có test).

**Ghi chú phạm vi.** `PickPlace` dùng chung env và vẫn có lỗi ma sát tương tự; không đổi vì ngoài phạm vi P1 và để P0/pick-place giữ nguyên bit.

## 3. Expert đẩy hộp (Phase B) — chẩn đoán và kết quả

**Chẩn đoán [đã đo].** Trong 19/28 ca thất bại cũ hộp bị nhấc > 1 cm dù cuối bước không có tiếp xúc robot. Theo dõi từng bước vật lý cho thấy ngón tay chạm hộp rồi hộp bị hất lên và bắn với 0,3–0,6 m/s. Nguyên nhân là tiếp cận/đẩy không giới hạn tốc độ TCP. Sau khi giới hạn tốc độ xuất hiện lỗi thứ hai: chu kỳ giới hạn giữa pha đẩy và nâng (kẹt ở z≈0,42). Sau đó là hộp vượt đích do mục tiêu đẩy ở xa tâm hộp.
**Sửa:** máy trạng thái rõ ràng (traverse → descend → push → retreat), giới hạn tốc độ TCP theo pha, đích đẩy bám tâm hộp trên đường hộp→đích, giảm tốc khi gần đích, dò lại nếu vượt đích. Tham số chọn **chỉ trên tập tuning 24,000,000–24,000,047**: `v_push=0,10`, `push_lead=0,01`, `xy_tol=0,006`; sau đó đóng băng.

| Tập seed | Mục đích | Strict success | Wilson 95% | Nguyên nhân thất bại |
|---|---|---|---|---|
| 24,000,000+ (48) | tuning (cấu hình chọn) | 42/48 = 87,5% | 75,3–94,1% | vượt đích 5, lệch ngang 1 |
| 24,100,000+ (100) | **phát triển mới, lần đánh giá đầu** | **83/100 = 83,0%** | 74,5–89,1% | vượt đích > 3 cm 16, nhấc hộp 1 |
| 24,200,000+ (100) | xác nhận | 89/100 = 89,0% | 81,4–93,7% | vượt đích 11 |
| 24,300,000+ (100) | `p1_validate_env` V7 | 88/100 = 88,0% | 80,2–93,0% | dừng thiếu/vượt 12 |
| **3 tập phát triển gộp** | | **260/300 = 86,7%** | **82,4–90,1%** | |

Lỗi còn lại chủ yếu là vượt đích (không phải nhấc hộp). Xuyên thấu robot–hộp nhỏ nhất: −0,56 mm (ngưỡng đăng ký trước −2 mm; trước đó −1,1 mm). Lưu ý trung thực: tập đầu tiên (83%) có Wilson dưới 80%; mức ≥ 80% đạt rõ khi gộp (cận dưới 82,4%).

## 4. Bộ dữ liệu chẩn đoán và hồi quy

**Bộ dữ liệu [đã đo]** (`wm_eval_dataset_v1.npz`, không đưa vào git): train 360 episode / 54.000 chuyển tiếp; validation 120 episode / 18.000 chuyển tiếp, seed tách rời. Nguồn: expert (45%), expert nhiễu + burst (25%), poke nhanh 0,4 m/s (20%), ngẫu nhiên (10%); một nửa episode expert có dừng giữa chừng để tạo mất chạm/chạm lại. Nhãn theo cờ tiếp xúc ngón/tay, làm mượt bằng closing ≤ 4 bước (cờ thô nhấp nháy khi đẩy).

| Chế độ | Chuyển tiếp train | Chuyển tiếp val |
|---|---|---|
| free (không chạm ≥ 3 trạng thái) | 45.949 | 15.108 |
| first_contact | 563 | 199 |
| sustained (≥ 3 trạng thái liên tiếp) | 4.836 | 1.755 |
| contact_loss | 559 | 197 |
| other (loại khỏi bảng) | 2.093 | 741 |

Hạn chế: first_contact và contact_loss chỉ ~200 mẫu/val, nên CI rộng hơn ở các ô nhỏ (n nêu rõ trong JSON).
**Hồi quy [đã đo]:** 54/54 test (44 cũ + 10 mới): ma sát hiệu dụng; chế độ legacy tái hiện lỗi; mass/inertia; mass đổi động lực; hình học vật cản khớp mô hình; goal hợp lệ + tiêu chí chặt không đổi; reset tất định + replay snapshot bit-exact; căn chỉnh hành động thực thi; an toàn hành động (clip + slew, tay phải giữ); reward khớp công thức numpy độc lập; **P0: reaching và pick_place bit-identical với `git archive HEAD`** (băm SHA-256 của quỹ đạo bằng nhau).

## 5. World Model (Phase C) — đánh giá lại

WM dùng chính mã SuperMimic, fit **trên dữ liệu proxy** (6000 cập nhật, cửa sổ 8, loss 0,748 → 0,119, 103 s CPU). **Không phải WM của pilot.** Căn chỉnh hành động thực thi: sai lệch tối đa 0,0 (PASS).

**Dự đoán vị trí hộp, mm (WM / đứng yên / vận tốc không đổi), horizon 1·4·8·16 [đã đo]:**

| Chế độ | h=1 | h=4 | h=8 | h=16 | WM − baseline tốt nhất, h=16 (CI bootstrap) |
|---|---|---|---|---|---|
| free | 0,50 / 0,14 / 0,10 | 1,41 / 0,29 / 0,38 | 2,35 / 0,34 / 0,55 | 4,09 / 0,42 / 0,98 | +3,11 [2,80; 3,39] (**WM tệ hơn**) |
| first_contact | 2,05 / 2,85 / 2,11 | 6,42 / 12,4 / 11,9 | 8,84 / 22,4 / 22,1 | 13,0 / 39,0 / 40,8 | −26,0 [−27,2; −24,8] (WM tốt hơn) |
| sustained | 1,97 / 3,55 / 1,97 | 5,75 / 13,0 / 10,6 | 8,88 / 23,8 / 19,5 | 15,0 / 38,9 / 33,9 | −18,9 [−22,4; −15,4] (WM tốt hơn) |
| contact_loss | 3,40 / 6,45 / 3,31 | 7,31 / 15,7 / 15,3 | 10,4 / 25,1 / 29,5 | 14,6 / 35,7 / 55,7 | −21,1 [−23,1; −19,0] (WM tốt hơn) |

- Một bước (h=1): WM ngang vận tốc không đổi khi chạm (chênh ≈ 0, CI chứa 0), tốt hơn đứng yên. Từ h ≥ 4 WM vượt cả hai baseline khi chạm.
- **Vùng tự do: WM tệ hơn baseline đứng yên** (hộp trôi ma 0,5–4 mm, orientation sai 2,5° ở h=16 so với 0,36°). Số liệu vận tốc cũng tệ (0,006 so với 0,001–0,002 m/s).
- **Orientation khi chạm: WM không tốt hơn đứng yên** (13,3° so với 12,4° ở h=16, first_contact). Hộp bị xoay khi đẩy nhưng WM chưa mô hình hoá được hướng. Điều này khớp với việc lỗi expert là lệch ngang/vượt đích.

**Return tưởng tượng so với thực, khớp quỹ đạo [đã đo]** (cùng trạng thái đầu, cùng hành động thực thi, H=16, 8160 cửa sổ): tương quan 0,991; khoảng lạc quan trung bình **+0,081** (CI +0,045 đến +0,118; |gap| trung bình 0,70; 64% cửa sổ lạc quan). Theo chế độ: free +0,046; first_contact +0,197 (CI 0,067–0,319); sustained +0,382 (CI −0,002–0,755); contact_loss +0,254 (CI −0,058–0,560). Phân rã theo thành phần reward (tái dựng độc lập, khớp reward simulator tới 3,6e-7): chênh chủ yếu từ **số hạng thưởng gần đích** `2·exp(−(d/0,04)²)`: +0,159 (first_contact), +0,365 (sustained). Nghĩa là WM đoán hộp gần đích hơn thực tế. Theo nguồn dữ liệu: poke +0,275 (CI 0,145–0,390); expert +0,041 (CI −0,013–0,094); random −0,150.

**Lạc quan khi tối ưu hoá [đã đo, kiểm tra căng thẳng]**: tối ưu 16 hành động bằng 40 bước Adam trên return tưởng tượng (khởi tạo từ hành động ghi lại), rồi chạy thật từ cùng snapshot (24 trạng thái): khoảng lạc quan **tăng từ −0,08 lên +9,04** (CI 6,2–11,9). Return tưởng tượng tăng +8,57, return thật đổi −0,55; 46% ca thật tệ hơn sau tối ưu. Đây là **khai thác mô hình rõ ràng**: tối ưu tìm ra hành động mà WM tin tốt hơn nhưng thực tế không tốt hơn. Đây là cận trên mạnh (tối ưu mở loop có đối kháng); actor thật được huấn luyện mượt hơn (clip gradient, lr nhỏ), nên mức lạc quan trong huấn luyện thực chưa biết. Số liệu bị giới hạn bởi WM proxy.

**Độ trung thực của hồi đáp theo hành động [đã đo, mẫu nhỏ]**: nhiễu loạn 4 bước trong simulator và trong WM. 10/24 trường hợp có đáp ứng > 0,5 mm trong simulator; tương quan 0,46; **dấu đồng ý 50% (ngang may rủi)**; độ lớn WM 6,2 mm so với 9,6 mm thực. Gradient actor qua động lực hữu hạn và khác 0 (norm 1,9–24,6). Kết luận: gradient *tồn tại* nhưng độ chính xác hướng qua vùng chạm **chưa được chứng minh**; n nhỏ nên chỉ là tín hiệu cảnh báo.

**Diễn giải [suy luận]**: WM đủ tốt để học trong vùng tiếp xúc (dự đoán tốt hơn baseline), nhưng chưa đủ để tin tưởng gradient hoặc tối ưu mạnh. Dữ liệu proxy nhỏ nên pilot (dữ liệu on-policy) là thí nghiệm thật để kiểm tra điều này.

## 6. Trước/sau định lượng (tóm tắt)

| Chỉ số | Trước | Sau |
|---|---|---|
| V1b goal trong khay / không hợp lệ | 2,0% (6/300) | 0/300; 35,7% mẫu đầu bị loại |
| V2 khối lượng | 1 giá trị (0,05 kg) | 300 giá trị, 0,031–0,080 kg |
| V3b slide 0,7 → 1,0 → 1,3 (mm) | 23,4 → 23,4 → 19,8 (bản cũ, v0 = 0,6 m/s, 100 bước) | 22,8 → 16,2 → 12,3 (v0 = 0,5 m/s; điều kiện đo khác nhau nên chỉ so xu hướng) |
| Xuyên thấu robot–hộp | −1,12 mm | −0,56 mm |
| Expert strict | 20/48 (41,7%) | 260/300 (86,7%) |
| Nhấc hộp > 1 cm | 19/48 | 1/300 |
| Mất chạm / chạm kéo dài trong dữ liệu WM | sustained n = 0 | sustained 4.836 train / 1.755 val |
| Khoảng lạc quan khi tối ưu (imag − real) | +7,4 (12 trạng thái, WM nhỏ) | +9,0 (24 trạng thái, WM lớn hơn); khớp quỹ đạo ghi lại: +0,08 |

## 7. GO / NO-GO cập nhật

| Thành phần | Quyết định | Căn cứ |
|---|---|---|
| Môi trường P1 | **GO** | 11/11 kiểm định, 54/54 test, P0 bit-identical |
| Expert / tập phát triển | **GO** | 260/300 = 86,7% (cận dưới 82,4%), tiêu chí chặt giữ nguyên |
| World Model | **CÓ ĐIỀU KIỆN** | tốt khi chạm, nhưng lạc quan lớn khi tối ưu, hồi đáp theo hành động chưa tin cậy, orientation kém |
| Pilot 1 seed ≤ 2 h | **GO CÓ ĐIỀU KIỆN** | điều kiện bên dưới |
| Multi-seed / nhiều giờ | **NO-GO** | chưa có bằng chứng pilot |

Điều kiện trước khi chạy pilot (đều cần chủ việc):
1. **Duyệt riêng** việc chạy GPU; chạy `p1_benchmark.sh` trên máy bạn (VRAM đỉnh, throughput, tải GPU) và chốt số iteration vừa trần 2 giờ **[chưa kiểm]**. Cấu hình hiện tại đã cập nhật đặc tả môi trường mới (mass/friction/goal) qua mặc định, không cần đổi YAML.
2. **Duyệt ngưỡng giám sát** (đề xuất, **chưa kiểm**, phải cố định trước khi chạy): sau mỗi 400 iteration chạy `p1_wm_eval --checkpoint <ckpt>` trên bản sao checkpoint. Cờ đỏ nếu (a) tỉ lệ sai số hộp WM/đứng yên khi chạm ở h=8 > 1,0; (b) khoảng lạc quan return khớp quỹ đạo của actor hiện tại > 3,0 ở 3 lần đo liên tiếp; (c) success thật trên 32 goal vẫn 0 ở nửa ngân sách. Cờ đỏ ⇒ dừng và báo cáo, không tự sửa ngưỡng.
3. Trainer hiện chưa ghi sẵn các chỉ số khoảng lạc quan/orientation; kế hoạch dùng script ngoài (chạy offline trên checkpoint). Nếu muốn ghi trong trainer thì phải sửa mã trainer (chưa làm, vì chưa được duyệt).
4. Cần biết: pilot có thể thất bại vì khai thác mô hình hoặc orientation, đây là kết quả khoa học hợp lệ, không phải lỗi môi trường.

## 8. Diff chính xác và tệp thay đổi (chưa commit, chưa push)

Tệp **đã theo dõi, bị sửa** (2): `research/supermimic_openarm/envs/tasks.py` (+36/−3), `research/supermimic_openarm/envs/openarm_env.py` (+8/−3). `git diff --stat`: 2 files changed, 40 insertions(+), 7 deletions(−). Không có tệp upstream/`sim/`, G2a/G2b, WAM hay checkpoint nào bị sửa.
Tệp **mới, chưa theo dõi** (không gồm `research/openarm_wam/`): `baselines/push_expert.py`, `configs/pushing_p1_pilot.yaml`, `p0_final_eval.py`, `p1_expert_eval.py`, `p1_validate_env.py`, `p1_wm_diagnostics.py` (bản cũ, giữ để so sánh), `p1_wm_eval.py`, `scripts/p1_pilot.sh`, `scripts/p1_benchmark.sh`, `tests/test_p1_env.py`, cùng 3 tài liệu trong `docs/ai-native/tasks/` (kế hoạch P0, aggregator, báo cáo trước, báo cáo này).
**Patch đầy đủ**: `outputs/p1_validation/p1_repair.patch` (920 dòng, 12 tệp: 2 diff đã theo dõi + 10 tệp mới; không gồm tài liệu). `outputs/` nằm ngoài git.

Tóm tắt diff đã theo dõi:
- `tasks.py`: `Pushing` thêm `mass_scale=(0.6,1.6)`, `friction_mode="pair_exact"`, hình học vật cản, `goal_valid()`, lấy mẫu loại bỏ (tối đa 200 lần) không clip, ghi `mass_scale/friction_mode/rejected_samples`.
- `openarm_env.py`: lưu priority/khối lượng/quán tính gốc; khi `cfg` yêu cầu: đặt `geom_priority`, khối lượng + quán tính (từ giá trị gốc, không cộng dồn) và `mj_setConst` **trước** `reset_home`; ghi tham số mẫu trong `episode_info`. Reaching/pick_place không có các khoá này nên hành vi không đổi.

## 9. Bảng AC

| AC | Nội dung | Trạng thái | Evidence |
|---|---|---|---|
| 1 | Ma sát hiệu dụng của cặp hộp–bàn, khác biệt vật lý đo được | PASS | V1a, V3b, test; μ 0,56–1,04, trượt 22,8→12,3 mm |
| 2 | Khối lượng + quán tính nhất quán, ghi tham số | PASS | V2a/V2b, test, `sampled_parameters` |
| 3 | Loại goal không hợp lệ; tiêu chí chặt không đổi | PASS | V1b (0 không hợp lệ), test |
| 4 | Expert ≥ 80% strict trên tập mới, có tử/mẫu, CI, loại thất bại | PASS | 260/300 (82,4–90,1%); mục 3 |
| 5 | Bộ dữ liệu 4 chế độ | PASS | mục 4 |
| 6 | WM 1 bước + nhiều bước vs baseline; return tưởng tượng vs thực; khoảng lạc quan | PASS (đo xong) / kết quả: có điều kiện | mục 5 |
| 7 | Hồi quy: snapshot, căn chỉnh, an toàn, reward, P0 | PASS | 54/54 test; V4/V5/V6 |
| 8 | Không nhầm lẫn về đánh giá checkpoint thật | PASS | 30 episode trực quan trên best.pt thật được ghi nhận; đánh giá độc lập 100 goal/seed vẫn tách riêng, NOT_CHECKED |
| 9 | Không train GPU; không đụng WAM/G2/test bảo vệ | PASS | không chạy trainer; `git status` |
| 10 | Diff chính xác, trước/sau, báo cáo GO/NO-GO, không commit/push | PASS | mục 6–8 |
| — | Đo VRAM/throughput/GPU trên GPU thật | NOT_CHECKED | cần máy bạn |
| — | `view_live` với best.pt thật; P0 100 goal/seed độc lập | NOT_CHECKED | cần máy bạn |

Việc cần chủ việc: (a) duyệt ngưỡng giám sát và việc chạy benchmark/pilot; (b) chạy `p0_final_eval` trên checkpoint thật (tách riêng); (c) quyết định có sửa lỗi ma sát tương tự cho `PickPlace` không; (d) có commit/push không, và nhóm tệp nào.

## 10. Hạn chế

- WM đánh giá là WM proxy (dữ liệu nhỏ, không on-policy). Kết luận về lạc quan/khai thác là cận trên có đối kháng, chưa phải hành vi huấn luyện thực.
- Expert chọn tham số trên tập tuning; ba tập phát triển được xem như độc lập nhưng tập 24,100,000+ là lần đánh giá đầu và có cận dưới Wilson 74,5%.
- Tập first_contact/contact_loss nhỏ (~200 mẫu val); CI rộng hơn ở các ô đó.
- Kết quả hồi đáp theo hành động chỉ có 10 trường hợp đo được (mẫu nhỏ).
- Tự kiểm, không phải kiểm độc lập; mọi thứ chạy CPU trong sandbox.

## 11. Lệnh tái tạo (từ thư mục gốc repo)

```
python -m pytest research/supermimic_openarm/tests -q
python -m research.supermimic_openarm.p1_validate_env --out outputs/p1_validation/env_validation_v3.json
python -m research.supermimic_openarm.p1_expert_eval --seed-base 24100000 --episodes 100 --out outputs/p1_validation/expert_devset_24100000_n100.json
python -m research.supermimic_openarm.p1_wm_eval --out outputs/p1_validation/wm_eval_v1.json --dataset-out outputs/p1_validation/wm_eval_dataset_v1.npz
bash research/supermimic_openarm/scripts/p1_pilot.sh --device cuda:N        # dry-run; --execute chỉ sau khi duyệt riêng
```
