# AI Work Method · Bộ phương pháp làm việc với AI đa lĩnh vực

**Lõi chuyển giao 2.0 · Profile slide HTML 3D 1.0 bổ sung ngày 01/10/2026 · Biên soạn từ hệ phương pháp của Lâm Nguyễn.**

Một cách làm chung: xác định đầu ra, giao việc rõ, giữ bằng chứng và nghiệm thu kết quả. Có thể dùng cho viết, nghiên cứu, vận hành, phân tích dữ liệu và phần mềm. Người mới chỉ cần lõi; bật từng module khi bài toán cần.

## Bắt đầu

1. Đọc [Handbook](HANDBOOK.md), chọn một việc thật nhỏ và có đầu ra rõ.
2. Copy mẫu trong [AI Work Method](00_AI_Work_MethodBox/references/templates.md), điền mục tiêu và tiêu chí nghiệm thu.
3. Chọn profile trong [ví dụ đa lĩnh vực](00_AI_Work_MethodBox/references/domain-examples.md). Chạy một vòng giao việc → thực hiện → kiểm → sửa.
4. Chỉ bật SOT khi cần duy trì dữ liệu có owner; chỉ bật Refinery khi cần chuẩn hoá dữ liệu nguồn ngoài; bật Vibecode khi đầu ra là phần mềm.

## Các hộp

| Hộp | Công dụng | Người dùng |
|---|---|---|
| [00 · AI Work Method](00_AI_Work_MethodBox/SKILL.md) | Lõi giao việc và nghiệm thu đa domain | Mọi người, không cần Python |
| [01 · SOT](01_SOT_KernelInABox/README.md) | Giữ dữ liệu nội bộ có owner, phát hiện thay đổi, kiểm trước xuất | Người quản trị dữ liệu |
| [02 · Refinery](02_Refinery_MethodBox/README.md) | Tập hợp claims nguồn ngoài, giữ mâu thuẫn và dấu vết | Nghiên cứu, phân tích, danh mục |
| [03 · Vibecode](03_Vibecode_MethodBox/README.md) | Adapter quy trình cho phát triển phần mềm | Đội kỹ thuật |
| [04 · Slide HTML 3D](04_HTML3D_Slide_MethodBox/START_HERE.md) | Làm slide HTML tương tác, dựng 3D, sửa bố cục và kiểm bản trình chiếu | Người làm thuyết trình, đào tạo, giới thiệu sản phẩm |

Chuẩn tài liệu Nhà nước, NAFOSTED, phong cách RtR/Lâm Nguyễn và chuẩn xuất bản là **profile riêng**: dùng khi công việc yêu cầu. Chúng không phải điều kiện để áp dụng lõi. Thư mục này không chứa hộp Văn bản Nhà nước; README cũ đã mô tả nhầm phạm vi.

## Chia sẻ và dùng làm skill

- Gói lõi 2.0 gồm bốn skill ban đầu. Profile slide HTML được bổ sung riêng để người nhận chỉ tải phần cần dùng.
- Tải [gói phương pháp slide HTML 3D](dist/HTML3D-Slide-Method-1.0.zip) hoặc [skill html3d-slides](dist/html3d-slides.skill).
- Các file `.skill` trong `dist/` là ZIP có một thư mục gốc mang tên skill, chứa `SKILL.md` và tài nguyên đi cùng. Đây là định dạng đóng gói, không bảo đảm mọi ứng dụng tự cài được.
- Cách trung lập: giải nén, đưa `SKILL.md` cho AI đọc; chỉ đưa references cần cho task. Với hệ hỗ trợ skill theo thư mục, đặt thư mục đã giải nén vào vị trí skill mà hệ đó quy định. Không trộn các bản cùng tên.
- Tất cả tài liệu nguồn đều mở trực tiếp được; không cần giải nén archive để tìm references.

## Chạy toolkit (tuỳ chọn)

Cần Python 3.10+; phiên bản đã chạy kiểm được ghi trong VALIDATION.json. Tại thư mục gốc:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/verify_package.py
python tools/build_package.py
```

Bộ kiểm dùng dữ liệu mẫu và bản tạm, không gọi LLM hoặc internet. Hướng dẫn cài dependency cần internet lần đầu. [Báo cáo đối chiếu](AUDIT_VA_NANG_CAP.md) ghi thay đổi, phần đã loại và giới hạn còn lại; [kết quả kiểm](VALIDATION.json) ghi lần chạy thực tế. [Manifest](dist/MANIFEST.json) chứa SHA-256 để đối chiếu bản phát hành.

## Giấy phép và nguồn

README gốc ghi CC BY-SA 4.0 cho tài liệu và MIT cho code; bản này giữ thông tin đó và attribution Lâm Nguyễn. Các nội dung cũ không có hồ sơ nguồn/quyền tái phân phối riêng được ghi trong báo cáo; người phát hành cần xác nhận quyền với phần bên thứ ba trước khi phân phối thương mại. Không có dữ liệu nghiệp vụ thật trong các ví dụ mới: chúng được gắn nhãn giả lập.

## Profile làm slide HTML 3D

[04 · Phương pháp slide HTML 3D](04_HTML3D_Slide_MethodBox/PHUONG_PHAP.md) tổng kết cách dựng, sửa UI và kiểm bản Studio 1.1. Đây là profile đầu ra riêng, dùng cùng lõi AI Work Method khi cần làm bài trình chiếu.

Skill `html3d-slides` chứa quy trình từ nội dung tới bản trình chiếu, mẫu brief, profile thiết kế và script đo UI. Các tài liệu đi kèm hướng dẫn giữ text ở lớp HTML, dựng hình Three.js/WebGPU theo nội dung, điều khiển bước bằng phím, loop animation, giữ QR tĩnh, kiểm khoảng thoáng, dòng mồ côi và độ nét canvas. Áp dụng tone, số trang và cách xưng hô theo brief mới. Ví dụ Studio 10 trang không phải thiết kế bắt buộc cho mọi lĩnh vực.

Trên Drive, bắt đầu tại [thư mục phương pháp HTML 3D](https://drive.google.com/drive/folders/1GLaqaIJ39hkFFkbQ7XCSlmeurvwYjaGu). Tải [skill html3d-slides](https://drive.google.com/file/d/1dAgojWsQ377817vg943_hBjgFqErRZPA/view?usp=drivesdk) hoặc [gói phương pháp, skill và mẫu giao việc](https://drive.google.com/file/d/1c_yqv-1CLC3ZE__imNjqDWwSK_0mvez3/view?usp=drivesdk).

Bản Drive hiện chia sẻ skill và tài liệu phương pháp. Source và tài nguyên của bộ slide Studio được giữ riêng, chưa đưa vào thư mục công khai. SOURCE_MAP.json ghi nguồn và hash của bản tham chiếu trong workspace gốc. Gói skill không cần source Studio để hoạt động.

Skill, templates và công cụ đo UI đã được kiểm. Chưa có đánh giá độc lập trên một deck đề tài mới. Xem [hướng dẫn dùng](04_HTML3D_Slide_MethodBox/START_HERE.md) và [kết quả kiểm](04_HTML3D_Slide_MethodBox/VALIDATION.json).
