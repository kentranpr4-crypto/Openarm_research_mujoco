# Vibecode Kit v6.3-transfer · Profile phần mềm

Giữ mô hình Homeowner/Contractor/Builder, TIP và Completion Report, thêm nghiệm thu có bằng chứng. Kế thừa bản skill đang cài v6.2; bản chuyển giao làm rõ uỷ quyền, bỏ quota phỏng vấn và đồng bộ đủ references.

- [SKILL.md](SKILL.md): chỉ dẫn thực thi cho AI.
- [Lý thuyết](LY_THUYET_CHUYEN_GIAO.md): giải thích cho người học.
- [Index](REFERENCES_INDEX.md): 10 references đọc theo nhu cầu.
- [Jev](JEV_DECISION_SEAM.md): thiết kế tuỳ chọn, OFF; chưa có provider runtime.

Task nhỏ: scan → TIP → build → verify. Task nhiều phần: thêm requirements, blueprint và dependency. Chủ việc có thể đã uỷ quyền qua yêu cầu; không cần luôn gõ “APPROVED”. Hỏi khi quyết định còn thiếu làm đổi scope, kiến trúc hoặc quyền; làm xong bản review được trước quyết định phát hành.

Ba vai là trách nhiệm, không bắt buộc ba người/model. Một AI có thể tách bước implement và review, nhưng không gọi đó là kiểm độc lập. Với việc ngoài code, dùng [lõi đa lĩnh vực](../00_AI_Work_MethodBox/SKILL.md).

Gói .skill cũ thực tế thiếu references; các file giờ nằm trực tiếp trong hộp và được đóng gói từ cùng nguồn bằng build_package.py. PDF cũ được giữ trong backup trước nâng cấp để tránh phát hành hai bản nội dung khác nhau.
