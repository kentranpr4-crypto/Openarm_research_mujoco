# Quy trình làm việc của dự án: AI Work Method 2.0

Dự án này làm việc theo **AI Work Method** (bộ AI Native, biên soạn từ hệ phương pháp của Lâm Nguyễn).
Mọi yêu cầu không tầm thường đều đi theo chu trình:

**Hiểu việc → Chốt đầu ra → Làm → Kiểm → Sửa → Bàn giao.**

## Luật bắt buộc

1. **Việc nhỏ** (câu hỏi, tra cứu, sửa một dòng): làm thẳng, không ép quy trình.
2. **Việc nhiều phần**: trước khi làm, ghi **TIP ngắn** ngay trong chat theo mẫu ở
   `docs/ai-native/templates.md`: mục tiêu, input, đầu ra, AC-01..AC-n (tiêu chí quan sát được + cách kiểm),
   giới hạn, phần được tự làm, quyết định cần chủ việc, giả định.
   Việc lớn hoặc kéo dài nhiều phiên: lưu TIP và báo cáo vào `docs/ai-native/tasks/<YYYY-MM-DD>-<ten-viec>.md`.
3. Đổi “hay / đẹp / chuẩn” thành tiêu chí kiểm được. Chỉ hỏi khi thiếu thông tin làm đổi kết quả;
   trong lúc chờ thì làm tiếp phần độc lập.
4. **Kỷ luật dữ liệu**: tách evidence / claim / fact đã kiểm / suy luận / đề xuất. Không dùng
   “AI tự tin X%” làm bằng chứng. Ô chưa biết để trống có nhãn, không đoán cho đủ. Mâu thuẫn giữ cả hai kèm nguồn và ngày.
5. **Bàn giao** luôn có báo cáo: `DONE | PARTIAL | BLOCKED`, bảng AC → PASS / FAIL / NOT_CHECKED kèm evidence,
   kiểm đã chạy, phần chưa kiểm, quyết định đã đưa ra, việc cần chủ việc. Chưa chạy thì không ghi PASS.
   AC bắt buộc FAIL hoặc NOT_CHECKED thì không được báo DONE. Tự kiểm thì ghi rõ là tự kiểm, không gọi là kiểm độc lập.
6. Không tự mở agent `lead`/`peer`. Khi người dùng cần nhiều agent, họ bật SLP cho dự án bằng `/slp-on` (tắt bằng `/slp-off`).
7. Không tự gửi thông tin, phát hành ra ngoài hay làm việc khó đảo ngược khi chưa được uỷ quyền.

## Chọn module (skill trong `.claude/skills/`) chỉ khi cần

| Tình huống | Skill |
|---|---|
| Điều phối, giao việc, nghiệm thu mọi lĩnh vực | `ai-work-method` (lõi, luôn áp dụng) |
| Xây hoặc sửa phần mềm | `vibecode-kit` (SCAN → RRI → VISION → BLUEPRINT → TASK GRAPH → BUILD → VERIFY → REFINE, co giãn theo cỡ việc) |
| Dữ liệu nội bộ có owner, dùng lặp lại, cần phát hiện thay đổi | `sot-kernel-in-a-box` |
| Gom dữ liệu từ nhiều nguồn ngoài thành registry truy nguồn được | `refinery` |
| Slide HTML trình chiếu có 3D, tương tác | `html3d-slides` |

Các module bổ sung nhau, không bắt buộc chạy đủ. Script Python của SOT/Refinery cần Python 3.10+ và
`pip install -r docs/ai-native/requirements.txt` (cần internet lần đầu).

Tài liệu đầy đủ: `docs/ai-native/HANDBOOK.md`, ví dụ theo lĩnh vực: `docs/ai-native/domain-examples.md`.
