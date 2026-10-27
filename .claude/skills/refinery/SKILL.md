---
name: refinery
description: >-
  Tập hợp dữ liệu nguồn ngoài thành registry có claims, snapshots, provenance,
  disputed và honest-null bằng engine tất định. Dùng khi cần danh mục hoặc dataset
  truy nguồn được; không dùng cho dữ liệu nội bộ có owner hoặc coi gate máy là xác
  minh ngữ nghĩa. Bản chuyển giao 1.4 có demo đào tạo giả lập và audit cell độc lập.
---

# Refinery v1.4-transfer

Đổi domain bằng config, claims và snapshots. Engine không gọi LLM hay cào web. Đọc [README](README.md) trước khi chạy; đọc [lý thuyết](LY_THUYET_CHUYEN_GIAO.md) khi thiết kế một registry mới.

1. Khai schema, phạm vi, kỳ dữ liệu và mẫu số có basis. Dùng `new_domain.py` để tạo domain trống; không tự lấy con số 100 làm mẫu số.
2. Thu thập nguồn được phép dùng, lưu snapshot. Trích claims với entity, field, value, evidence_span, extraction, tier và capture. Mỗi value cần review ngữ cảnh; đoạn trích hiện trong nguồn chưa chứng minh value đúng.
3. Khai `capture.origin_id` chung cho các trang chép cùng nguồn gốc. Tier A/B/C là nhãn chất lượng nguồn theo domain, không phải xác suất đúng. Nếu không có origin_id, engine dùng source làm định danh, nên maintainer phải kiểm độc lập nguồn.
4. Chạy `python refinery.py <domain>` rồi `python bites.py <domain>`. PASS chỉ chứng minh các cổng đã triển khai. N/A phải ghi rõ. Auditor đối chiếu cells/provenance và aggregate từ claims; không xác minh thực tế ngoài đời.
5. Giữ disputed và null; không chọn nguồn thắng tự động. Review thay đổi và bàn giao lineage cùng output. `review.py` hỗ trợ staging, không thay human review value/span.

`sourced`: một giá trị có nguồn; `corroborated`: cùng giá trị từ ít nhất hai origin A/B đã khai; `disputed`: nhiều giá trị, không tự phân xử; `null`: không có claim. Corroborated chưa đồng nghĩa fact đã xác minh. Claims output giữ capture và evidence_span.

Engine hỗ trợ snapshot text/HTML; PDF cần bước trích text có thể truy về trang/file gốc. Matcher giải HTML entity và Unicode NFC khi so khớp, giữ snapshot nguyên trạng. Không sửa evidence để che mất nguồn gốc.

Insight dùng quy trình review riêng: nhãn quan sát/suy luận, evidence, điều kiện bác bỏ và người có quyền duyệt. Engine fact chỉ nhận A/B/C; không đưa P/S/H/X vào `claims.jsonl` và không quảng bá các cổng insight chưa có trong runtime.

[Jev/System One](JEV_DECISION_SEAM.md) là thiết kế lựa chọn, OFF mặc định. Chỉ đọc khi người dùng yêu cầu; không có provider runtime, không thay gate hoặc quyền con người.
