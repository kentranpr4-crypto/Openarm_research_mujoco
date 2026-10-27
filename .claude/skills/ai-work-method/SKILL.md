---
name: ai-work-method
description: >-
  Điều phối công việc với AI theo mục tiêu, bằng chứng và nghiệm thu, dùng cho viết,
  nghiên cứu, vận hành, phân tích và phần mềm. Dùng khi cần làm rõ việc nhiều phần,
  giao việc, kiểm kết quả hoặc chuyển giao phương pháp đa lĩnh vực; việc đơn giản xử lý
  trực tiếp, không ép quy trình hay tự mở nhiều agent.
---

# AI Work Method v1.0 · Lõi đa lĩnh vực

Kế thừa giao việc/kiểm của Vibecode, provenance của SOT và giữ mâu thuẫn của Refinery. Lõi dùng được với một người và một AI; các module là lựa chọn theo bài toán.

## Phân trách nhiệm

- Chủ việc: mục tiêu, ưu tiên, quyền quyết và nhận kết quả.
- Điều phối: hiểu input, làm rõ khoảng trống, chia việc và kiểm đầu ra.
- Thực hiện: tạo artifact, tự kiểm, báo bằng chứng và phần còn thiếu.

Đây là trách nhiệm, không phải số lượng người/model. Một AI có thể làm các bước ở các lượt riêng. Không gọi tự review là kiểm độc lập. Không tự spawn agent, gửi thông tin hay phát hành ra ngoài khi task chưa uỷ quyền.

## Quy trình

1. Đọc input, xác định đầu ra và tiêu chí quan sát được; tận dụng thông tin và uỷ quyền đã có. Chỉ hỏi khi thiếu thông tin làm thay đổi kết quả.
2. Ghi TIP ngắn: mục tiêu, input, đầu ra, AC-ID, giới hạn, uỷ quyền, giả định. Task nhỏ ghi trong chat; task nhiều phần thêm yêu cầu và phụ thuộc. Dùng [mẫu](references/templates.md) khi cần.
3. Chọn module: SOT cho dữ liệu nội bộ cần bảo trì; Refinery cho registry từ nguồn ngoài; Vibecode cho phần mềm; skill định dạng/phong cách cho artifact tương ứng. Không yêu cầu module chưa cài. Nếu thiếu, nêu giới hạn và xử lý phần vẫn làm được.
4. Thực hiện trong scope. Tự quyết chi tiết có thể đảo ngược khi đủ context; thay đổi kết quả/scope/ngân sách/quyền đã chốt thì chuyển cho người có quyền. Hoàn thiện bản có thể review trước khi xin quyết định phát hành.
5. Kiểm sản phẩm theo từng AC, dùng file/nguồn/kết quả chạy làm bằng chứng. Dùng kiểm chuyên môn theo hệ quả; tránh test chỉ sao chép implementation. Sửa lỗi có ảnh hưởng đến mục tiêu.
6. Bàn giao artifact và báo cáo: DONE/PARTIAL/BLOCKED; AC PASS/FAIL/NOT_CHECKED; evidence, vấn đề, quyết định và bước cần người dùng. Không báo DONE khi tiêu chí bắt buộc chưa đạt hoặc chưa kiểm.

## Kỷ luật dữ liệu

Tách evidence, claim, fact đã kiểm, suy luận và đề xuất. Nhãn nguồn mạnh hoặc độ tự tin AI không xác nhận sự thật. Mâu thuẫn giữ cả hai với scope/ngày; thiếu nguồn ghi chưa biết. Tỷ lệ phải có mẫu số và phạm vi. Dữ liệu biến động kiểm lại theo nhu cầu dùng, không áp một nhịp cố định cho mọi domain.

## Nghiệm thu

Tỷ lệ coverage chỉ bổ trợ: mọi AC quan trọng phải đạt; nêu rõ tiêu chí còn thiếu dù tổng phần trăm cao. Nếu AI tự kiểm thì ghi rõ; khi cần người/chuyên gia/công cụ riêng, nêu phần đã có bằng chứng và phần chờ kiểm. Không tự nhận quyết định thay chủ việc.

Đọc [ví dụ domain](references/domain-examples.md) khi người dùng cần chuyển phương pháp sang một lĩnh vực; thay input, tiêu chí và bằng chứng, không ép tất cả đầu ra thành code hoặc registry YAML.
