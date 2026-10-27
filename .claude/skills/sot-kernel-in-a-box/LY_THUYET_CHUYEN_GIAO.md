# SOT · Lý thuyết và phạm vi kiểm

**Bản chuyển giao 1.3 · 30/09/2026.**

## Bài toán

Cùng một giá, quy định, spec hoặc chỉ số được chép vào nhiều tài liệu; file gốc đổi nhưng dẫn xuất chưa đổi; ước tính bị mất nhãn sau vài lần sao chép. SOT giữ một nguồn có owner cho từng nhóm dữ liệu và tạo dẫn xuất có provenance.

```text
Nguồn có owner → adapter → registry → baseline + guard → gate → artifact được review
```

Nguồn chuẩn là điểm cập nhật dữ liệu. Registry là dẫn xuất máy đọc. File xuất là một dẫn xuất tiếp theo. Người quản trị quyết nguồn nào có thẩm quyền, kỳ/ngày dữ liệu, đơn vị và quy tắc cập nhật; engine không tự quyết các điều này.

## Ba mức dữ liệu

- `verified_primary`: đã được owner phân loại là có bằng chứng sơ cấp theo chuẩn của project.
- `claim`: chưa đạt chuẩn đó; giữ claim và `verified_value: null`, không phơi `value` như fact chắc chắn.
- `flagged`: có mâu thuẫn hoặc vấn đề cần quyết.

Đây là taxonomy của SOT, không đồng nhất với A/B/C hoặc sourced/corroborated của Refinery. Không tự nâng nguồn ngoài lên verified_primary chỉ vì hai trang viết giống nhau.

## Những gì các cổng kiểm

1. **Adapter:** chuyển input theo schema; cần kiểm tính tất định và review diff. Adapter mẫu không phải bộ hiểu Excel/Word tổng quát.
2. **Baseline:** hash nội dung nguồn để phát hiện thêm, mất hoặc đổi. Mtime khác khi copy mà byte không đổi chỉ là thông tin.
3. **Guard:** kiểm các quy tắc provenance đã khai trong cấu hình. Mapping status sai có thể tạo PASS sai về nghiệp vụ.
4. **Publish gate:** token ràng buộc byte registry, nguồn, baseline, cấu hình và adapter. Chúng đổi sau kiểm thì token hết hiệu lực.
5. **Bites:** inject lỗi trên bản tạm để chứng minh các cửa cụ thể chặn được tình huống đã mô tả. Không chứng minh mọi lỗi có thể xảy ra.

## Nghi thức thay đổi

Owner uỷ quyền sửa nguồn → cập nhật nguồn → chạy adapter và hygiene → review diff → baseline accept → check → bites nếu logic/config đổi → gate mới → review artifact → phát hành trong phạm vi đã uỷ quyền. Không dùng accept chỉ để xoá báo động.

Với insight, phân biệt quan sát trực tiếp, nghe lại và giả thuyết; ghi bằng chứng, giả định và điều kiện bác bỏ. Toolkit hygiene có access/ethical/falsify nhưng chỉ chạy khi gọi lệnh và có config phù hợp; `sot_check` mặc định chỉ baseline và guard. Không mặc định coi các cổng insight đã được bật.

## Chuyển giao

Bắt đầu bằng vài dữ liệu có giá trị cao và một owner rõ; dùng project_template và demo giả lập. Mỗi tổ chức tự giữ dữ liệu thật ngoài skill. Chỉ mở rộng schema/hạ tầng khi nhu cầu đồng thời, phân quyền hoặc dữ liệu lớn đòi hỏi; không yêu cầu DB cho mọi nhóm.

Lần phát hành này chứng minh các phép kiểm trong VALIDATION.json trên dữ liệu mẫu. Những số liệu case study 167 facts/13 file trong bản cũ không có log đối chứng đi kèm nên không dùng làm bằng chứng phát hành. Token không xác thực nội dung tài liệu xuất, tính đúng chuyên môn, quyền phát hành hoặc sự độc lập của người kiểm.
