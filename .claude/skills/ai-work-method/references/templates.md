# Hai mẫu tối thiểu và profile domain

Có thể ghi ngay trong chat. Một file ghép các phần cũng hợp lệ. Xoá các trường không có tác dụng; giữ mục tiêu, tiêu chí, bằng chứng và trạng thái.

## TIP · Gói giao việc

```text
ID / Chủ việc:
Mục tiêu / Người sử dụng:
Đầu ra / Nơi lưu:
Input / Nguồn / Scope và ngày dữ liệu:
Tiêu chí:
  AC-01 [kết quả quan sát được + cách kiểm]
  AC-02 [...]
Giới hạn / Ngoài phạm vi:
Được tự làm / Quyết định cần chủ việc:
Giả định / Thiếu dữ liệu / Phụ thuộc:
```

## Completion + Verify · Báo cáo bàn giao

```text
Trạng thái: DONE | PARTIAL | BLOCKED
Artifact: [đường dẫn hoặc nội dung]
AC-ID | PASS / FAIL / NOT_CHECKED | evidence | mức ảnh hưởng
Kiểm đã chạy: [công cụ/người, input, kết quả, thời điểm]
Chưa kiểm / Nguồn thiếu / Mâu thuẫn:
Quyết định và thay đổi: [ai quyết, vì sao, tác động]
Cần chủ việc: [chỉ quyết định chưa có uỷ quyền]
```

PASS cần evidence. Chưa chạy không được ghi PASS. FAIL/NOT_CHECKED ở AC bắt buộc → PARTIAL hoặc BLOCKED theo khả năng tiếp tục. Với nhiều REQ-ID, thêm bảng REQ → AC → artifact → evidence; tính coverage bằng số REQ đạt/total và giữ danh sách còn thiếu.

## Profile domain

```text
Tên lĩnh vực / Loại đầu ra:
Nguồn được dùng / Nguồn không đủ để kết luận:
Tiêu chí chuyên môn / Đơn vị / Kỳ và phạm vi:
Người có quyền quyết / Mức truy cập:
Kiểm tự động / Kiểm con người:
Điều kiện phải kiểm lại nguồn:
```

Profile không thay tiêu chí cụ thể của task. Khi mục tiêu đổi, sửa TIP và ghi quyết định thay vì âm thầm đổi bài.
