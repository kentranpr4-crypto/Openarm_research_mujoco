# SOT Kernel-in-a-Box v1.3-transfer

Kho dữ liệu có owner, nguồn gốc và kiểm nội dung trước khi xuất. Kernel là module dữ liệu, không phải điều kiện cho mọi việc dùng AI.

## Chạy demo

Cài dependency theo README gốc, rồi chạy từ thư mục gốc:

```bash
python 01_SOT_KernelInABox/tools/sot_check.py 01_SOT_KernelInABox/examples/demo_org
python 01_SOT_KernelInABox/tools/sot_bites.py 01_SOT_KernelInABox/examples/demo_org
python 01_SOT_KernelInABox/tools/sot_publish.py 01_SOT_KernelInABox/examples/demo_org gate
python 01_SOT_KernelInABox/tools/sot_publish.py 01_SOT_KernelInABox/examples/demo_org status
```

Demo là dữ liệu giả lập. `sot_check` ghi log; `gate` ghi token trong demo; bites tiêm lỗi trên bản tạm. Gói phát hành không mang log, backup hoặc token đã tạo từ máy người soạn.

## Dựng project thật

Copy [project_template](project_template/README.md) sang vị trí dự án. Điền owner trong PROJECT.yaml, sources và từ vựng status. Chạy adapter → review registry → hygiene fill/split → baseline init → check → bites → gate. Chấp nhận baseline mới chỉ sau khi owner đã uỷ quyền thay đổi nguồn và registry đã cập nhật.

Token bản 1.3 ràng buộc registry, nội dung nguồn, baseline, cấu hình và adapter Python trong project. Token cũ không có input_digest phải tạo lại. Chỉ thay mtime do copy/unzip, nội dung giữ nguyên, không chặn gate. Sáu bites kiểm sửa nguồn, phơi value chưa verify, registry đổi sau gate, thiếu token, nguồn đổi sau gate và policy đổi sau gate.

## Giới hạn

- Hash phát hiện thay đổi, không chứng minh nội dung đúng hoặc adapter hiểu đúng nguồn.
- `verified_primary` dựa trên mapping/status do owner khai; guard không xác minh chứng từ ngoài đời.
- `publish` là copy qua cổng kiểm. Nó **không kiểm các con số trong file xuất** có đúng với registry; cần review artifact riêng. Token không phải quyền phát hành.
- File-based, một người quản trị; không có phân quyền hay log chống sửa. Người có quyền ghi file vẫn có thể bỏ qua công cụ. Không dùng như cơ chế bảo mật.
- Jev/System One chỉ có [thiết kế tuỳ chọn](JEV_DECISION_SEAM.md), OFF mặc định, không có provider runtime trong gói.

Đọc [lý thuyết](LY_THUYET_CHUYEN_GIAO.md) khi thiết kế ownership, adapter hoặc quy trình vận hành.
