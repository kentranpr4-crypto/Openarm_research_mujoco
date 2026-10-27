---
name: html3d-slides
description: "Dựng hoặc nâng cấp slide HTML trình chiếu có 3D và tương tác. Dùng khi cần nội dung rõ, điều khiển theo người nói, render sắc nét, kiểm trạng thái và gói HTML dùng lại."
---

# Slide HTML 3D

Tạo trải nghiệm trình chiếu mà người xem đọc được, hiểu được và thao tác được. Đồ hoạ giải thích nội dung. Kết quả kiểm kỹ thuật không thay thế đánh giá thẩm mỹ hoặc thử người xem.

## Xác định phạm vi

Đọc nội dung, mã nguồn và yêu cầu hiện có trước khi sửa. Giữ giới hạn số trang, cách bấm, phần được giữ và phần đã được duyệt. Nếu thiếu thông tin quan trọng, hỏi ngắn trong khi làm phần độc lập. Với bộ mới, dùng [mẫu brief](assets/brief.template.json) để ghi người xem, mục tiêu, nguồn, phong cách, cách trình chiếu và đầu ra.

Không tự áp số trang, màu, giới tính người xem, ký tên hoặc QR của bản mẫu vào chủ đề khác. Không thêm nghiên cứu nguồn ngoài, AI backend, tài khoản hoặc phát hành công khai nếu nhiệm vụ không cần. Đây là phương pháp làm slide HTML, không thay thế hồ sơ chuyên môn hoặc tài liệu gốc.

## Quy trình

1. **Chốt thông điệp và bằng chứng.** Mỗi trang có một điều cần hiểu và một vật mang nội dung: bảng, lịch, nguồn, bản sửa hoặc mô hình. Giữ phần chưa biết. Gắn nhãn ví dụ giả lập, tách kết quả tính được với mô phỏng. Đọc [workflow và thiết kế](references/workflow-and-design.md).
2. **Dựng storyboard.** Ghi message, artifact, scene, action, states và ghi chú người nói. Chọn một ví dụ xuyên suốt nếu phù hợp. Phân nhịp giữa giải thích, kiểm, lựa chọn, thực hành và kết thúc. Chọn ngôn ngữ hình theo nội dung, tránh lặp một đạo cụ cho mọi trang.
3. **Chốt thiết kế trên vài trang đại diện.** Có một trang mở đầu, một trang dày dữ liệu và một tương tác khó. Sửa tất cả trạng thái của các trang này trước khi nhân rộng. Giá trị trong [cấu hình Studio](assets/studio-profile.json) chỉ là điểm khởi đầu.
4. **Dựng nội dung và runtime.** Một nguồn dữ liệu sinh slide, ghi chú và file ví dụ. DOM giữ chữ chính, nút, bảng và QR. 3D giữ mô hình. DOM và Raycaster gọi cùng action, cùng state. Đọc [kỹ thuật](references/engineering.md) khi viết renderer/presenter.
5. **Render và sửa.** Kiểm đầu, giữa, cuối từng build và mọi nhánh lựa chọn. Soi chữ/hình, dòng mồ côi, khoảng footer, cạnh mô hình, phông trong texture, bóng vật ẩn, tốc độ và QR. Có thể nạp [measure-ui.js](scripts/measure-ui.js) vào trang để lấy số đo. Script cần selector đúng với DOM của bộ đang làm và không chấm được thẩm mỹ.
6. **Kiểm và đóng gói.** Áp [QA và phát hành](references/qa-and-release.md). Render lại hình dự phòng sau thay đổi scene, dựng lại HTML, kiểm đúng hash bản cuối. Gói viewer và nguồn riêng. Chỉ sinh PDF nếu cần, kiểm đủ trang bằng render. Báo rõ việc cần máy chiếu hoặc người xem thực tế.

## Những điều phải giữ khi trình chiếu thủ công

- Chỉ action người nói mới đổi ý, ví dụ hoặc kết luận. Thời gian không quyết định phần thông tin nào được mở. Đoạn vào cảnh và nội suy phản hồi có thể dùng thời gian, nhưng không tự mở kết luận.
- Ambient có thể loop, nhưng không lật giấy sang dữ liệu khác hoặc tự đóng dấu đạt. Pause dừng RAF và CSS. Tôn trọng reduced motion.
- Tương tác trên vật thể có nút tương đương. Space/Enter kích hoạt nút đang focus. Phím qua trang vẫn làm việc sau select. Người xem không cần đoán một hotspot vô hình.
- Chữ chính và QR giữ ngoài hiệu ứng 3D/hậu kỳ. QR dùng đúng nguồn được giao, có nền và quiet zone, không xoay hoặc glow.
- Chỉ renderer đang dùng chạy. WebGPU là tăng cường khi phù hợp. Có WebGL hoặc ảnh dự phòng đúng slide/build/case, cùng bố cục và nội dung.
- Kiểm đầu ra từ dữ liệu hoặc hành vi thực. Không dựng trạng thái “đã kiểm” bằng một animation định sẵn. Thiếu phí hoặc người phụ trách không được biến thành 0 hoặc một tên bịa.

## Khi dùng lại source Studio 1.1

Source tham chiếu được đóng cùng bộ phương pháp tại `starter/Nguon-slide-HTML-3D.zip` trong gói đầy đủ. Nếu chỉ nhận file skill, source là tuỳ chọn, không phải dependency để hiểu phương pháp.

Đây là bản đã chạy cho **10 trang cụ thể**, chưa phải generator bất kỳ số trang. Xem [kỹ thuật](references/engineering.md#dùng-lại-source-studio) trước khi đổi số trang hoặc schema. Thay minh hoạ, chữ ký, QR, bộ tài liệu và dữ liệu cũ. Không dùng test cũ như bằng chứng cho chủ đề mới.
