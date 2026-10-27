# Jev decision seam · Vibecode Kit v6.2

## Vị trí trong workflow

```text
EVIDENCE → typed decision provider → schema/policy → Contractor
                                                → INVESTIGATE/VERIFY như cũ
```

Provider tăng tốc đọc bằng cách đưa ra một snap judgment có kiểu. Contractor vẫn sở hữu
process, root cause, TIP, Blueprint, VERIFY và escalation cuối.

## Mode

- `OFF` mặc định: Vibecode vận hành nguyên bản.
- `SHADOW`: prediction chỉ vào evaluation log.
- `SELECTIVE`: route case tới checklist điều tra; không tự sửa hoặc approve.
- Production cho từng node phải có owner phê duyệt và đường rollback.

## Node được phép

1. `failure_category`: `RUNTIME`, `LOGIC`, `UI_RENDER`, `API_NETWORK`, `STATE_DATA`,
   `BUILD_CONFIG`, `TIP_CONFLICT`, `UNKNOWN`.
2. `semantic_report_gap`: flag báo cáo có field nhưng thiếu evidence có nghĩa.
3. `escalation_hint`: khuyến nghị L1/L2/L3; policy không được hạ mức mà luật Vibecode yêu cầu.
4. `hypothesis_priority`: xếp thứ tự giả thuyết, không tuyên bố root cause.

## Ranh giới cấm

Provider không được sửa code, đổi architecture/scope, thay TIP/Blueprint, bỏ test, xác nhận
AC, đặt severity cuối, hoặc phát verdict READY/SHIP. Output `UNKNOWN`, confidence thấp hay
nhiều nguyên nhân phải được escalate, không ép chọn cho đủ dữ liệu.

## Completion Report addendum khi seam hoạt động

```text
DECISION_ASSIST:
  node_id / mode / model_version / prediction / confidence
  policy_result / contractor_disposition / override_reason
```

Không có section này khi mode `OFF`; audit trail TIP và Completion Report cũ vẫn đủ.

## Cổng bật

Chỉ bật `SELECTIVE` nếu holdout lịch sử không có false negative ở `TIP_CONFLICT/UNKNOWN`,
không kém baseline tốt nhất, fallback pass 100% và giảm có ý nghĩa time-to-root-cause hoặc
review minutes. Provider lỗi thì route về Contractor theo Debug Protocol.

