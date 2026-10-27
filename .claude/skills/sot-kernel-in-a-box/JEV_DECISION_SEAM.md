# Jev decision seam · SOT Kernel v1.2

## Mục tiêu

Giảm thời gian maintainer đọc candidate mà không biến model thành nguồn sự thật.

```text
candidate + canonical excerpt chỉ đọc
        → typed decision provider
        → schema validator + risk policy
        → append-only suggestion log / review queue
        → maintainer disposition
        → adapter/check/bites/publish như cũ
```

## Mode

- `OFF` mặc định: SOT vận hành nguyên bản.
- `SHADOW`: prediction chỉ dùng để đánh giá.
- `SELECTIVE`: được sắp thứ tự queue; không tự ghi hoặc resolve.
- Không tồn tại mode auto-verify/auto-publish.

## Node được phép

1. `candidate_relation`: `new`, `duplicate`, `conflict`, `legacy`, `unsure`.
2. `review_priority`: xếp case có khả năng conflict/impact cao lên trước.
3. `owner_hint`: gợi ý queue owner khi mapping deterministic không đủ.
4. `sensitive_content_hint`: cảnh báo bổ sung, không thay access/DLP policy.

## Ranh giới cấm

Provider không được sửa canonical file, registry, source hash, baseline, verification log,
provenance status hoặc publish token. Nó không được nâng `claim` thành `verified_primary`,
resolve `flagged`, chấp nhận drift, loại candidate, hoặc đưa một con số vào export.

## Chống label leakage

Khi benchmark, không đưa `kernel_check`, `kernel_ref`, reviewer note hoặc final disposition
vào state. Chỉ đưa candidate và canonical excerpts mà maintainer có thể thấy tại thời điểm
ra quyết định. Ground truth lưu local, tách khỏi request.

## Hợp đồng audit

Mỗi suggestion ghi `node_id`, `mode`, case ID, input digest, model/version, output,
probabilities/confidence, policy result, maintainer disposition và override reason. Raw
secret hoặc dữ liệu chưa được phê duyệt xử lý ngoài tổ chức không được gửi provider.

## Cổng bật

Chỉ bật `SELECTIVE` khi holdout nội bộ không có false negative nghiêm trọng ở `conflict`,
không kém baseline tốt nhất, fallback pass 100% và giảm đáng kể review minutes. Provider
lỗi hoặc không chắc chắn thì giữ quy trình cũ và đẩy maintainer review.

