# QA và phát hành

## Kiểm đúng nội dung

Đối chiếu message/artifact/source của từng trang. Ví dụ hư cấu được ghi rõ. Nguồn chưa có và kết quả chưa kiểm để thành việc cần bổ sung. Ghi phạm vi pass: hết trùng giờ không có nghĩa cả lịch đủ người hướng dẫn.

Thử lỗi có chủ đích vào verifier. Lịch cùng người/cùng ngày bị giao nhau phải fail, tiếp giáp phải pass, khác người không báo nhầm, người chưa xác nhận giữ unknown. Báo giá thiếu phí giao không được coi là 0. Các kiểm này chỉ là ví dụ cách viết test có ý nghĩa, không bắt mọi deck có lịch và bảng giá.

## Kiểm từng trạng thái

| Nhóm | Phép kiểm | Điều cần quan sát |
|---|---|---|
| Manual | Giữ build đầu ở 0/30/120 giây | Không tự đổi thông tin hoặc kết luận |
| Nhánh | Initial/middle/final, mọi case/filter, lỗi/rỗng | Chữ và model cùng state, thiếu vẫn rõ |
| Bàn phím | Focus button + Space/Enter, mũi tên, PageDown sau select | Activate đúng nút, không nhảy trang ngoài ý muốn |
| Pause | P, reduced motion, tab ẩn | RAF, ánh sáng, lưới, trace đều dừng |
| In | before/after print | Đủ trang, panel đúng trạng thái in, khôi phục state |
| Offline | file://, network disabled, downloads | Bundle/fonts/data local hoạt động, link ngoài phân biệt rõ |
| Backend | WebGPU, WebGL, effects off | Không lỗi, fallback đúng bước/domain/filter |
| QR | Byte nguồn, bounds/pixels khi giữ trang, quét thật | Không animation, không che quiet zone, link đúng |
| Render | Màn hình ngang/dọc, resize, Retina, 720p | Không cắt chữ, orphan, sát hình hoặc footer |
| Resource | Vài vòng đổi toàn bộ deck | Không tăng vô hạn scene/geometry/texture |

`measure-ui.js` đọc DOM, không gọi mạng, không sửa trang. Nó đo orphan theo vị trí Range, footer gap, column gap và tỷ lệ canvas. Dùng selectors phù hợp, kiểm khung giờ bằng nowrap. Một đèn xanh từ script không chứng minh không có chạm chữ vào hình nằm bên trong host hoặc không có UI xấu.

Ví dụ chạy trong Playwright:

```js
await page.addScriptTag({path: 'html3d-slides/scripts/measure-ui.js'});
const measured = await page.evaluate(() => measureSlideUI());
```

Lặp theo từng slide/build/case của runtime đang làm. Bản Studio dùng API `slideDemo.seek` để đặt trạng thái và capture. Bộ khác cần adapter theo state của nó. Muốn chạy từ DevTools có thể dán nội dung script và gọi `measureSlideUI()`.

## Bốn cửa phát hành

1. **Nội dung:** nguồn, ví dụ, giọng nói, quyền chốt và tiêu chí đúng phạm vi.
2. **Hình và UX:** render trực tiếp các trạng thái, đọc từ khoảng cách mục tiêu, keyboard/focus, fallback, QR.
3. **Kỹ thuật:** không lỗi chưa xử lý, manual/pause/in/offline đúng, số đo hiệu năng có điều kiện, dữ liệu kiểm chạy thực.
4. **Môi trường buổi thật:** máy chiếu, quét QR từ chỗ ngồi, thử hiểu với một nhóm nhỏ nếu buổi quan trọng.

Cửa 4 không được ghi pass nhờ kiểm headless. Trong Studio, thử 5 người với ít nhất 4 người kể lại được đầu vào/đầu ra/cách kiểm là mục tiêu pilot, chưa có kết quả thực tế và không phải nghiên cứu thống kê.

## Bản phát hành có thể dựng lại

Chuỗi cuối khi đã sửa code/scene: bundle -> render stills -> build HTML -> kiểm hash HTML cuối -> export/render PDF nếu cần -> package -> giải nén source, build lại và so hash. Không tiếp tục sửa HTML rồi giữ báo cáo pass của hash cũ. Khi asset/fonts/content/skill source đổi, dựng lại các output phụ thuộc.

Gói viewer chứa HTML, ghi chú, tài liệu tải và PDF nếu được yêu cầu. Gói nguồn chứa dữ liệu, CSS, runtime, scenes, bundle/lockfile, assets/licenses và hướng dẫn build. Không kéo archive, node_modules, screenshot scratch hoặc token/tệp riêng tư vào viewer.

Manifest ghi SHA-256 theo đường dẫn thật trong ZIP. Kiểm ZIP integrity, từng payload và build lại từ thư mục tạm. Viết báo cáo gồm hash bản được kiểm, môi trường, kết quả, phần chưa kiểm và dependency còn cần trên máy khác. Không dùng điểm thẩm mỹ do mình tự chấm để thay cho hình và phản hồi người xem.
