# Project Template · Khung dựng SOT cho một tổ chức/dự án mới

Copy nguyên thư mục này, đổi tên, làm theo checklist. Từ con số không đến kho sự thật
có kiểm định + cửa xuất trong một buổi.

## Cấu trúc (tên mặc định; sửa config/adapter nếu thay đổi)

```
<ten_du_an>/
├── PROJECT.yaml          ← meta: tên, owner, mục tiêu (điền đầu tiên)
├── sot.yaml              ← config trung tâm (sửa mỗi dòng `org`)
├── sot.hygiene.yaml      ← từ vựng status/evidence CỦA BẠN (file duy nhất phải "dịch")
├── source_canonical/     ← file gốc: Excel, Word, CSV. Nơi DUY NHẤT sửa sự thật.
├── registry/             ← fact YAML máy đọc (adapter sinh ra — không sửa tay tuỳ tiện)
├── adapters/             ← máy xay: file gốc → registry (tự viết theo adapter_csv.py mẫu)
├── index/                ← verify_log.jsonl + .publish_token (tool tự ghi)
├── _queue/               ← conflicts_pending.md: mâu thuẫn chờ người quyết
└── exports/              ← file đã qua cửa publish (deck, báo giá, đề án)
```

## Checklist khởi động (7 bước · ~nửa ngày)

1. **Meta:** điền `PROJECT.yaml` + sửa `org:` trong `sot.yaml`.
2. **Từ vựng:** mở `sot.hygiene.yaml`, map status nội bộ của bạn (VERIFIED/INT/nháp/đã duyệt...)
   vào 3 mức: `verified_primary` (giữ value) / `claim` (giấu value) / `flagged` (tranh chấp).
3. **Đổ file gốc** vào `source_canonical/`. Mỗi file phải có owner (ai được sửa nó).
4. **Viết adapter** cho từng loại file (bắt đầu từ `adapters/adapter_csv.py`). Chạy adapter
   → registry có fact. Rồi ép A3:
   ```bash
   python <tools>/hygiene.py fill  registry --config sot.hygiene.yaml --apply
   python <tools>/hygiene.py split registry --config sot.hygiene.yaml --apply
   ```
5. **Vân tay:** `python <tools>/sot_baseline.py . init`
6. **Kiểm + chứng minh răng cắn:**
   ```bash
   python <tools>/sot_check.py .        # phải PASS
   python <tools>/sot_bites.py .        # phải "TẤT CẢ RĂNG CẮN"
   ```
7. **Vận hành:** đặt cron chạy `sot_check.py` hàng tuần. Mọi file gửi ra ngoài đi qua:
   ```bash
   python <tools>/sot_publish.py . gate
   python <tools>/sot_publish.py . publish <file> exports/
   ```

## Nghi thức bảo trì (thuộc lòng 3 câu)

- **Sửa sự thật = sửa file gốc** → chạy adapter → `sot_baseline.py . accept` → `sot_check.py .`
- **Phát hiện mâu thuẫn** → ghi `_queue/conflicts_pending.md`, KHÔNG tự quyết — người có thẩm quyền quyết.
- **Không bao giờ** gửi file ra ngoài mà không qua `publish` — token chứng minh dữ liệu/config hiện tại khớp lần gate; cần review số trong artifact riêng.
