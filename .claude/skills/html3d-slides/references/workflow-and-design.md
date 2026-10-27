# Workflow và thiết kế

## Từ yêu cầu tới một trang dùng được

Brief cần đủ: người xem và kiến thức nền, mục tiêu sau buổi, số trang/thời lượng, nội dung và nguồn, giọng nói, nơi trình chiếu, asset phải giữ, tương tác và đầu ra cần nhận. Không buộc người dùng điền mọi trường cho một sửa nhỏ.

Một storyboard có các cột:

`id | mục đích | thông điệp | artifact | nguồn | layout | scene | action | trạng thái | phần chưa rõ | lời dẫn`

Ví dụ của bản Studio: một lịch đào tạo đi qua nguồn, giao việc, bản nháp, kiểm trùng giờ, bản sửa và bàn giao. Người xem thấy cùng một sản phẩm thay đổi, thay vì nghe tên phương pháp rời rạc. Khi mở ví dụ đa ngành, giữ cách giao/kiểm, thay artifact và tiêu chí. Không dùng lịch đào tạo làm khuôn cho mọi ngành.

Mỗi tương tác trả lời một câu hỏi. “Mở nguồn này”, “thử bộ lọc rỗng”, “so giá với bảng nguồn” có kết quả quan sát được. Nhấc một khối hoặc đổi màu nút chỉ là phản hồi, chưa phải nội dung của tương tác.

## Ba lớp thiết kế

1. **Đọc:** tiêu đề, lời dẫn, bảng, kết luận, nút. Giữ trong DOM để nét, focus được, copy được và in được.
2. **Hiểu:** mô hình hoặc vật mang nội dung, ưu tiên hình học và ánh sáng làm rõ quan hệ. Khi cần ghi chữ trên mô hình, giữ nhãn ngắn và nạp phông trước khi raster.
3. **Không khí:** nền ánh sáng, lưới, quỹ đạo và điểm sáng chậm. Giảm tương phản sau lớp đọc. Không chạy qua QR. Pause phải dừng cả lớp này.

Dùng vài họ bố cục theo chức năng. Sân khấu lớn phù hợp mở đầu, tuyến ngang phù hợp quy trình, hai cột phù hợp văn bản và mô hình, vùng bảng lớn phù hợp kiểm dữ liệu. Chọn vài cao trào, giảm chuyển động khi cần người xem làm bài. Không bắt mọi trang cùng mạnh như trang mở đầu.

## Chữ và khoảng cách

Viết tiêu đề như một điều muốn người xem hiểu. Câu ngắn nhưng không chặt thành khẩu hiệu mơ hồ. Đọc thành tiếng, bỏ câu nói quá khả năng hoặc tự khen AI. Đầu ra và tiêu chí phải cụ thể.

Với profile của buổi này: xưng mình/bạn, không em dash hoặc dấu chấm phẩy trong thuyết minh. Đây là lựa chọn biên tập của chủ buổi. Chỉ dùng câu ngắn không đủ chứng minh nội dung hết “AI slop”. Vẫn cần chủ ngữ, ví dụ, nguồn và giới hạn rõ.

Sửa dòng mồ côi theo thứ tự: viết lại câu, điều chỉnh chiều rộng, text-wrap: pretty, rồi giữ cụm ngắn không tách dòng. Dùng `white-space: nowrap` cho khoảng giờ, giá hoặc đơn vị cần đi liền. Không nowrap cả đoạn dài. Không sửa bằng cách giảm toàn bộ chữ đến khó đọc.

Đo khoảng chữ/hình từ vùng nội dung nhìn thấy, không chỉ bounding box của host có nhiều khoảng trống. Host gap là cảnh báo kỹ thuật, vẫn cần xem hình. Chú thích và chữ ký có hai vùng riêng. Kiểm nút, trạng thái lỗi và lời giải thích dài nhất để tránh lấn footer.

## 3D tinh tế trong bản Studio

- Giấy có độ dày mỏng và roughness cao. Đồ sơn có clearcoat hạn chế. Điểm kim loại chỉ chiếm phần nhỏ.
- Nhân vật monoline nằm trên lớp giấy cong. Đây là hình 2.5D trên sân khấu 3D, không phải nhân vật có rig.
- Hoa dùng cánh có nếp gấp, tuyến dùng ribbon/tube. Đạo cụ liên quan tới nội dung: lịch, tờ giao việc, kính lúp, sách, màn hình.
- Pen tip, cap và thân thuộc một group, dùng chung trục cục bộ. Chuyển động cả group giữ các mối nối.
- Nền hi tech có lưới phối cảnh, quỹ đạo và điểm sáng. Độ mạnh thấp để không cạnh tranh với chữ. Cobalt dùng làm nhịp riêng, không nhuộm cả bộ.
- Không dùng hậu kỳ phủ toàn trang để che thiếu chi tiết. Nếu thêm bloom/DOF, giới hạn vào vùng hình và kiểm frame budget. Bản thực tế chưa dùng các hiệu ứng đó.

## Chốt trang mẫu trước khi làm toàn bộ

Chọn các trang có rủi ro khác nhau. Render cả trạng thái lỗi/thiếu, bảng dài, nhãn dài, nhánh lựa chọn và full build. Một ảnh đẹp ở trạng thái mặc định không chứng minh toàn bộ thiết kế ổn. Duyệt thẩm mỹ bằng hình render thực, không chỉ đọc code hoặc điểm số do AI tự chấm.
