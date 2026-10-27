# Jev decision seam · Refinery v1.3

## Mục tiêu

Giảm thời gian đọc và định tuyến candidate mà không đưa AI xác suất vào engine build/audit.

```text
snapshot + candidate tối thiểu hóa
        → typed decision provider
        → schema validator + risk policy
        → review priority / queue proposal
        → Refinery deterministic gates
```

## Mode

- `OFF` mặc định: Refinery vận hành như trước.
- `SHADOW`: ghi prediction để đánh giá, không đổi thứ tự hoặc output.
- `SELECTIVE`: chỉ đổi thứ tự review; không được loại record hay tự ghi claim.
- Không có mode tự động resolve hoặc publish.

## Node được phép

1. `claim_source_relation`: `supports`, `contradicts`, `unrelated`, `missing_context`,
   `same_origin`, `unknown`.
2. `review_priority`: score/ranking để con người đọc trước.
3. `ethical_risk_hint`: cảnh báo bổ sung; ethical gate hiện có vẫn quyết định.
4. `tier_hint`: chỉ là hint đầu vào review, không phải tier cuối.

## Ranh giới cấm

Provider không được tạo/sửa `value`, `evidence_span`, `capture`, `tier`, alias, scope,
corroborated/disputed/honest-null hoặc aggregate. Provider không được chọn nguồn thắng,
ẩn contradiction, bỏ record, hay thay exit code của gate.

## Hợp đồng audit tối thiểu

Mỗi prediction ghi: `node_id`, `mode`, `case_id`, input digest, model/version, output,
probabilities/confidence, policy result, final human disposition, override reason và thời
gian. Không ghi raw secret hoặc dữ liệu chưa được phép gửi ra ngoài.

## Cổng bật

Chỉ bật `SELECTIVE` khi holdout nội bộ chứng minh không bỏ sót contradiction nghiêm trọng,
không kém baseline tốt nhất và giảm đáng kể thời gian/chi phí review. Timeout, malformed,
unknown hoặc confidence thấp đều quay về review theo quy trình cũ.

