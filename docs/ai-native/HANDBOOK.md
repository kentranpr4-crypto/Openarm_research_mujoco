# Làm việc với AI · Hướng dẫn chuyển giao đa lĩnh vực

**Bản 2.0 · 30/09/2026.** Bạn không cần học hết các hộp. Bắt đầu bằng một đầu ra cần tạo và cách xác định nó đạt yêu cầu.

## 1. Lõi chung

Ba trách nhiệm: **chủ việc** xác định mục tiêu và có quyền quyết; **điều phối** làm rõ yêu cầu, chia việc và nghiệm thu; **thực hiện** tạo sản phẩm, kiểm và báo cáo. Một người và một AI có thể đảm nhiệm các bước này. Việc lớn hoặc hệ quả cao có thể cần người kiểm riêng. Chỉ gọi kiểm tra độc lập khi có nguồn bằng chứng hoặc người/công cụ kiểm tách khỏi quá trình tạo.

Chu trình thường dùng:

**Hiểu việc → Chốt đầu ra → Làm → Kiểm → Sửa → Bàn giao.**

Giao việc bằng TIP (Task Instruction Pack): kết quả cần đạt, input, tiêu chí nghiệm thu, giới hạn và quyền được làm. Tiêu chí phải kiểm được trên sản phẩm. “Hay”, “đẹp”, “chuẩn” cần đổi thành yêu cầu quan sát được, ví dụ đối tượng đọc, độ dài, cấu trúc, nguồn số liệu hoặc mẫu thiết kế đã thống nhất.

Khi xong, báo sản phẩm ở đâu, mỗi tiêu chí PASS/FAIL/NOT_CHECKED, bằng chứng kiểm, vấn đề còn lại và những quyết định quan trọng. Task nhỏ có thể ghi TIP và báo cáo ngay trong chat; không bắt buộc tạo hai file hoặc mở nhiều agent.

## 2. Dữ liệu có những lớp khác nhau

| Lớp | Nghĩa | Cách trình bày |
|---|---|---|
| Nguồn/evidence | File, đoạn trích, bản đo hoặc snapshot | Giữ định danh, vị trí, ngày, owner nếu là nội bộ |
| Claim | Nguồn nói một điều | Dẫn nguồn, không mặc định đúng |
| Fact đã kiểm | Đã kiểm theo phạm vi và quy trình cụ thể | Nêu kiểm gì, khi nào, giới hạn nào |
| Suy luận/giả thuyết | Diễn giải từ evidence | Gắn nhãn, nêu giả định và điều kiện bác bỏ |
| Đề xuất/quyết định | Việc nên làm hoặc đã được chọn | Nêu trade-off và người có quyền chọn |

Không dùng “AI confidence 95%” làm bằng chứng. Nguồn lặp lại cùng một thông cáo không phải nhiều nguồn độc lập. Một giá trị đúng năm trước có thể cần kiểm lại trước khi dùng hiện nay. Ô chưa biết giữ trống có nhãn, không đoán cho đủ bảng.

## 3. Chọn module theo nhu cầu

| Tình huống | Chọn | Đầu ra điển hình |
|---|---|---|
| Viết bài, soạn slide, lập kế hoạch, sửa một file | Lõi + skill định dạng/phong cách nếu cần | Artifact + kiểm theo TIP |
| Phân tích dữ liệu do tổ chức sở hữu | Lõi; thêm SOT nếu dùng lặp lại | Bảng có owner, đơn vị, kỳ dữ liệu |
| Dựng danh mục từ nhiều nguồn ngoài | Lõi + Refinery | Claims, snapshots, registry có disputed/null |
| Số liệu đổi thường xuyên và phải dùng trong nhiều tài liệu | Lõi + SOT | Canonical source, registry, nhật ký kiểm |
| Xây hoặc sửa phần mềm | Lõi + Vibecode | TIP, implementation, test evidence, Verify report |
| Hồ sơ chuyên ngành hoặc xuất bản | Lõi + profile chuyên môn/định dạng | Sản phẩm theo chuẩn áp dụng và kiểm chuyên môn |

Các module bổ sung nhau nhưng không có luật phải chạy đủ cả ba. Refinery → SOT cũng không phải thao tác copy trực tiếp: người phụ trách cần quyết định claim nào được chấp nhận, giữ lineage, dùng adapter phù hợp và kiểm lại; không ánh xạ `sourced` thành `verified_primary` tự động.

## 4. Quy mô vừa đủ

**Việc nhỏ, dễ sửa:** một TIP ngắn → làm → kiểm artifact → báo cáo. Không hỏi lại nội dung người dùng đã uỷ quyền. Không tạo test chỉ để lặp lại câu chữ tài liệu.

**Việc nhiều phần hoặc có dữ liệu quan trọng:** thêm danh sách yêu cầu, nguồn, phụ thuộc và bảng nghiệm thu. Chỉ hỏi những thiếu hụt làm đổi kết quả; tiếp tục phần độc lập trong lúc chờ.

**Việc có hệ quả cao hoặc khó đảo ngược:** làm xong bản có thể review, thêm kiểm chuyên môn thích hợp; người có thẩm quyền quyết trước hành động vượt uỷ quyền. Không thêm một checklist pháp lý cho mọi việc thông thường.

Không dùng tỷ lệ test fail hoặc phần trăm vượt ngân sách mặc định làm cửa duy nhất. Một lỗi nghiêm trọng có thể chặn dù 99% tiêu chí khác đạt. Ngưỡng chi phí, deadline và quyền phát hành do chủ việc chốt theo thực tế.

## 5. Một prompt để bắt đầu

```text
Áp dụng AI Work Method cho việc sau: [mục tiêu].
Đầu ra: [file/định dạng/đối tượng dùng]. Input: [nguồn có sẵn].
Đạt khi: [2–5 tiêu chí quan sát được]. Giới hạn: [scope, thời gian, chi phí].
Được tự làm: [phạm vi đã uỷ quyền]. Cần tôi quyết: [quyết định còn thiếu].
Lập TIP ngắn, thực hiện, kiểm sản phẩm và trả báo cáo với bằng chứng.
Đánh dấu giả định, dữ liệu thiếu và phần chưa kiểm; chọn module chỉ khi cần.
```

## 6. Chuyển giao cho nhiều người

Chọn một bài tập gần công việc mỗi nhóm; dùng mẫu trong [templates](00_AI_Work_MethodBox/references/templates.md). Người nhận tự chạy task, giải thích được bằng chứng và biết khi nào phải hỏi chủ việc. Nghiệm thu bằng đầu ra cụ thể, không bằng việc học thuộc thuật ngữ.

Pilot với vài việc lặp lại, ghi thời gian, số vòng sửa, lỗi bị bỏ sót và mức phù hợp với nhu cầu. Chỉ đưa vào quy trình sau khi nhìn thấy lợi ích; các ví dụ trong bộ này là bài tập giả lập, không phải số liệu chứng minh tăng năng suất.

Mỗi tổ chức duy trì profile domain ngắn: loại đầu ra, nguồn được dùng, tiêu chí chuyên môn, người có quyền quyết, yêu cầu truy cập và nhịp cập nhật. Không sao chép dữ liệu thật, thông tin cá nhân hoặc bí mật của tổ chức gốc vào gói huấn luyện.
