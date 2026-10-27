# Năm ví dụ áp dụng · Dữ liệu và kết quả đều giả lập

Các ví dụ mô tả cách giao việc và kiểm; chúng chưa phải thí nghiệm năng suất hoặc kết quả thực tế.

| Lĩnh vực | Việc/đầu ra | Input | Tiêu chí nghiệm thu | Bằng chứng cần kiểm | Module |
|---|---|---|---|---|---|
| Truyền thông | Bài giới thiệu dịch vụ 600–800 từ | Brief, đối tượng đọc, nguồn số liệu | Có thông điệp/CTA đã chốt; mọi con số có nguồn; đúng độ dài | Bản bài, số từ, đối chiếu con số/brief | Lõi; skill giọng viết khi được yêu cầu |
| Nhân sự/đào tạo | Kế hoạch onboarding hai tuần | Vai trò, lịch, người hướng dẫn | Mỗi buổi có mục tiêu/người phụ trách; không trùng lịch; có bài đánh giá | Bảng lịch, mapping mục tiêu→bài tập, review người phụ trách | Lõi; sheet nếu cần |
| Mua hàng/vận hành | So sánh ba nhà cung cấp giả lập | Báo giá cùng scope/ngày, đơn vị | Cùng cơ sở so sánh; giữ điểm chưa biết; không chọn giá thấp bằng giả định thiếu | Giá/thuế/phạm vi giao đối chiếu từng báo giá; người mua chọn | Lõi; SOT nếu giữ báo giá và tái dùng |
| Nghiên cứu thị trường | Danh mục chương trình đào tạo | Nguồn ngoài được phép thu thập | Mỗi học phí có snapshot/span; nguồn mâu thuẫn hiện disputed; phạm vi rõ | Claims, snapshots, pipeline/bites, đọc ngữ cảnh nguồn | Lõi + Refinery |
| Phần mềm | Sửa lọc danh sách theo trạng thái | Code, bug repro, hành vi mong muốn | Repro hết lỗi; trạng thái khác đúng; test liên quan qua | Diff, kết quả chạy, kiểm hành vi thực | Lõi + Vibecode |

## Ví dụ TIP cụ thể · Mua hàng

**ID:** BUY-DEMO-01. **Input giả lập:** A: 100 đơn vị/chiếc, giao tại kho; B: 95, chưa có phí giao; C: 110, gồm phí giao. Chưa rõ thuế của cả ba.

**Đầu ra:** bảng so sánh cho người mua, chưa chọn nhà cung cấp.

- AC-01: đủ ba đơn vị, giữ giá và scope đúng input. Kiểm từng dòng đối chiếu input.
- AC-02: thuế ghi “chưa rõ”; không điền 0. Kiểm ô trống có nhãn.
- AC-03: B ghi phí giao chưa biết; không kết luận B rẻ nhất theo tổng chi phí. Đọc câu kết luận.

**Uỷ quyền:** làm bảng và đề xuất câu hỏi cần bổ sung. Người mua quyết việc đặt hàng. Báo cáo DONE chỉ khi ba AC có evidence; nếu thiếu báo giá thật mà task yêu cầu kết luận mua, phải báo PARTIAL.

## Ví dụ kiểm lại suy luận · Đào tạo

Giả thuyết: “thiếu bài thực hành khiến người mới cần hỗ trợ nhiều”. Evidence: log câu hỏi trong tuần; ghi cả khả năng tài liệu không rõ hoặc phân công mentor thiếu. Điều kiện bác bỏ: nhóm có bài thực hành vẫn cần hỗ trợ tương đương trong cùng điều kiện. Đây là giả thuyết, không trở thành fact chỉ vì có ba quan sát.
