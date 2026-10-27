# Refinery v1.4-transfer · Registry truy nguồn được

Toolkit tất định cho dữ liệu nguồn ngoài; không có crawler hoặc LLM runtime. Ví dụ có sẵn là catalogue đào tạo **hoàn toàn giả lập**, dùng để học mà không gửi dữ liệu thật.

## Chạy từ thư mục gốc

```bash
python 02_Refinery_MethodBox/refinery.py 02_Refinery_MethodBox/domains/training-demo
python 02_Refinery_MethodBox/bites.py 02_Refinery_MethodBox/domains/training-demo
python 02_Refinery_MethodBox/new_domain.py my-domain "Danh mục của tôi"
```

Scaffold tạo claims rỗng, mẫu số null và không bật rollup. Điền schema/claims/snapshots thật trước khi dùng. Có `--output-root` để đặt domain ngoài hộp. `--help` không tạo thư mục. Không tự ghi đè domain đã có.

Demo: 9 claims → 3 chương trình; một học phí disputed, nhiều honest-null, một nguồn mirror khai chung origin_id. Sáu bites cơ bản được áp dụng; các gate chuyên ngành chưa khai thì báo N/A. Kết quả lần chạy thật ở VALIDATION.json trong thư mục gốc.

## Hợp đồng

```text
domains/<name>/domain.yaml     schema, alias, scope, universe basis, gates
               claims.jsonl   một claim mỗi dòng
               snapshots/     bằng chứng văn bản/HTML
```

`capture` giữ url, fetched_at, snapshot, source; dùng thêm origin_id để nhóm nguồn gốc chung. `evidence_span` nằm trong snapshot sau chuẩn hoá HTML entity/Unicode NFC. Giá trị số có thể dùng extraction=normalized, nhưng người trích phải kiểm đơn vị và nghĩa.

`review.py <domain> list|approve|reject` hỗ trợ staging; approve/reject cần người duyệt qua `--by`. Đây là thao tác thay đổi dữ liệu, không phải kiểm xác thực nội dung bằng máy.

## Giới hạn phải biết

- Span có trong snapshot không chứng minh nó hỗ trợ value; cần đọc ngữ cảnh.
- Source/origin_id do maintainer khai, không tự phát hiện nguồn sao chép. Corroborated phản ánh metadata đã khai.
- Auditor độc lập với hàm build/aggregates nhưng vẫn dùng chung config và quy tắc; không phải thẩm định chuyên môn độc lập.
- refresh_days là metadata; chưa có gate độ tươi. Trước dùng dữ liệu biến động, người phụ trách kiểm thời điểm phù hợp.
- Merge vào SOT, thu thập PDF và kiểm insight P/S/H/X cần adapter/quy trình riêng. Không mặc định đã có runtime.
- Các gate chuyên ngành là heuristic/config cụ thể, không chứng minh đúng ngữ nghĩa. Không giới thiệu engine như chống mọi hallucination.
- Export portal LAE, monitor, seed URL chuyên ngành đã được loại khỏi lõi; bản gốc nằm trong backup trước nâng cấp.

Đọc [lý thuyết](LY_THUYET_CHUYEN_GIAO.md); chỉ bật [Jev](JEV_DECISION_SEAM.md) khi có yêu cầu và implementation riêng.
