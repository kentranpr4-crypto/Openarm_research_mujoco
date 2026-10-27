# Refinery · Lý thuyết chuyển giao

**Bản 1.4-transfer · 30/09/2026.** Registry truy nguồn được là registry cho phép người nhận lần từ ô dữ liệu về claim, đoạn trích và snapshot. Điều đó giúp kiểm lại; không bảo đảm nguồn đã nói đúng hoặc người trích hiểu đúng.

## Bảy giai đoạn của phương pháp

**0. Định khung:** thực thể nào, field nào, vùng/kỳ nào, mẫu số từ đâu. **1. Thu thập:** nguồn hợp lệ, chụp nội dung. **2. Trích:** value, span, extraction và provenance. **3. Đối chiếu:** chất lượng nguồn, nguồn gốc chung, mâu thuẫn. **4. Chuẩn hoá:** alias và đơn vị có giải thích. **5. Hợp nhất:** cells và aggregate từ claims. **6. Kiểm và trình bày:** audit, bites, review nghĩa và báo phần chưa biết.

Các bước là mô hình suy nghĩ, không phải bảy màn hình bắt buộc. Engine thực thi phần build/check từ input đóng băng. Crawler, LLM extraction và merge giữa các kho không được cung cấp sẵn.

## Bốn trạng thái

- sourced: có một giá trị được nguồn nêu, chưa có đủ hai origin mạnh.
- corroborated: cùng giá trị, ít nhất hai origin A/B đã khai. Người review chịu trách nhiệm nguồn có thật sự độc lập.
- disputed: có nhiều giá trị; giữ cả claims, value null. Nguồn A không tự thắng nguồn C.
- honest-null: không có claim. Null là thông tin về độ bao phủ, không phải chỗ cần AI điền đoán.

A/B/C phân hạng chất lượng nguồn theo quy tắc domain. Không quy mọi cơ quan thành A hoặc mọi nhà cung cấp thành C: chất lượng phụ thuộc claim, chuyên môn, phương pháp, thời điểm và scope. Engine kiểm nhãn hợp lệ, không quyết nhãn đúng.

## Những lớp kiểm khác nhau

**Máy:** snapshot tồn tại, span tìm thấy sau chuẩn hoá biểu diễn, field/tier hợp lệ, alias mơ hồ, các gate được bật, deterministic build, audit cells/provenance và aggregate.

**Con người/người kiểm:** value có được span hỗ trợ, đơn vị, scope, kế hoạch so với hiện trạng, độ tươi, tính độc lập và quyền sử dụng nguồn. Có thể cần kiểm chuyên môn khi kết luận có hệ quả cao.

Auditor tự dựng lại cell/state/provenance và đếm aggregate từ claims, không gọi build/aggregates của builder. Tuy vậy, nó dùng cùng input/config; một sai sót trong nguồn hoặc định nghĩa schema vẫn có thể qua cả hai lớp. Bites chỉ chứng minh các tình huống inject đã chạy, N/A không phải PASS.

## Insight

Quan sát và giả thuyết cần evidence, bối cảnh, cách bác bỏ và người review. Quan sát trực tiếp không chứng minh động cơ của người khác. Không dùng quy tắc “ba nguồn → sự thật” hoặc ba data point → kết luận chắc. Có thể dùng hygiene theo schema riêng; Refinery fact runtime không nhận P/S/H/X.

## Chuyển giao

Bắt đầu với một danh mục nhỏ, cùng scope, vài nguồn phù hợp. Giữ claims/snapshots và output cùng phiên bản. Khi đưa sang SOT, owner quyết claim được chấp nhận, lưu lineage và adapter; không chuyển sourced/corroborated thành verified_primary tự động.

Demo đào tạo trong gói là giả lập. Các case study UAV/sầu riêng và số lượng hàng trăm record trong bản cũ không có đầy đủ artefact đối chứng tại thư mục này nên không dùng làm bằng chứng bản phát hành. Lợi ích thực tế cần đo ở pilot của từng tổ chức.
