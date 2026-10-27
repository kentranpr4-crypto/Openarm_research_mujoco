# Vibecode Kit · Lý thuyết chuyển giao phần mềm

**Bản 6.3-transfer · 30/09/2026.** Profile chuyên cho phần mềm của hệ phương pháp làm việc với AI.

## Vấn đề và trách nhiệm

Yêu cầu mơ hồ dẫn đến xây sai sản phẩm; ép từng thao tác dễ bỏ qua cách làm phù hợp codebase; tin báo cáo mà không kiểm sản phẩm khiến lỗi lọt qua. Giải bằng đầu ra/AC rõ, quyền quyết rõ và kiểm hành vi thực.

Chủ việc chốt mục tiêu, ưu tiên, scope và quyền phát hành. Contractor làm rõ yêu cầu, thiết kế, chia TIP và verify. Builder implement, kiểm và báo cáo. Trong chế độ phân vai, Contractor không implement thay Builder. Đây không phải quy định cấm người dùng code hoặc yêu cầu thuê ba người. Một AI có thể làm các bước riêng, nhưng tự review có giới hạn.

## Tám bước co giãn

SCAN → RRI → VISION → BLUEPRINT → TASK GRAPH → BUILD → VERIFY → REFINE.

Task nhỏ dùng scan/TIP/build/verify; task lớn thêm requirements/kiến trúc/phụ thuộc. Không có quota RRI, số giờ hoặc số TIP chung cho mọi dự án. Blueprint ghi những gì đã chốt; uỷ quyền sẵn trong yêu cầu hợp lệ. Khi cần đổi quyết định vượt uỷ quyền, làm rõ với người có quyền trước phần phụ thuộc.

TIP là **Task Instruction Pack**, có mục tiêu, input, AC, constraints và phạm vi. Completion Report có trạng thái, thay đổi, bằng chứng chạy test, vấn đề, deviations và đề xuất. VERIFY đối chiếu từng REQ/AC, đọc artifact và chạy kiểm phù hợp; không chỉ đọc report. Tỷ lệ coverage cao không bù cho lỗi nghiêm trọng. NOT_CHECKED không được viết thành PASS.

## Escalation

Chi tiết trong TIP, không đổi contract và đủ context → Builder tự xử lý, ghi quyết định đáng kể. Mơ hồ làm đổi output hoặc ảnh hưởng ngoài TIP → Contractor đánh giá. Scope/kiến trúc/quy tắc nghiệp vụ/chi phí/quyền chưa uỷ quyền → chủ việc quyết. Không dùng mặc định fail rate 20% hoặc budget +30% làm ranh giới duy nhất; một lỗi critical cần xử lý dù tổng fail rate thấp.

## References và giới hạn

[REFERENCES_INDEX](REFERENCES_INDEX.md) liệt kê các protocol thực sự kèm. X-Ray/QA/debug là công cụ theo nhu cầu, không ép chạy đủ cho sửa nhỏ. Jev chỉ là thiết kế decision seam mặc định OFF, không tích hợp sẵn và không quyết READY/SHIP.

Bản cũ mô tả >500 task production nhưng không kèm log/case dataset kiểm lại được; bản chuyển giao không dùng con số này như bằng chứng. Kết quả kiểm package chỉ chứng minh khả năng nạp/cấu trúc và các công cụ mẫu; chất lượng phần mềm của người nhận cần verify trên dự án thật.
