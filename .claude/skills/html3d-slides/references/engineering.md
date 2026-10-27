# Kỹ thuật

## Dữ liệu và state

Một file dữ liệu chính sinh nội dung DOM, notes, ví dụ tải và nhãn scene. Source của bản Studio là `deck.studio.json`. Không đồng bộ bằng cách copy cùng câu vào nhiều script. Không lấy markup từ archive vào bản đang chạy.

State mẫu:

```js
{ slideId, buildId, caseId, methodId, focusId, motionState, time }
```

- `slideId/buildId/caseId` quyết định nội dung. `methodId` giữ lựa chọn. `focusId` phục vụ điều khiển nếu cần.
- `time` chỉ cho ambient, vào cảnh và nội suy hình đã được action chọn. Không làm `if(time > 12) showResult()`.
- DOM và raycast gọi `action(name, value)` chung. Tính kết quả từ dữ liệu, render chữ và model cùng trạng thái.
- Hủy/giữ RAF đúng lúc pause, tab ẩn, reduced motion. Không bắt scene ẩn chạy độc lập.
- Trước in, lưu state và đưa từng trang tới trạng thái cần in. Sau in khôi phục state. Chỉ cập nhật panel động của trang đang xử lý để panel trước không bị reset bởi vòng render trang sau.

## Renderer và scene

Dùng một renderer/canvas cho bộ vừa phải như Studio, chuyển canvas vào host trang hiện hành. Tái dùng geometry, material và texture có ích. Đây là lựa chọn đã dùng, không phải luật cho mọi trải nghiệm có nhiều viewport.

WebGPURenderer với WebGL fallback được thử trong bản Three.js đã khoá. Khi cập nhật thư viện, kiểm API trong source/docs chính thức của phiên bản đó. Không suy rằng mọi shadow filter hoặc hậu kỳ của WebGL hoạt động tương tự trong WebGPU.

Mô hình tương tác có hit target đủ lớn và DOM button tương đương. Khi raycast, kiểm cả visibility của ancestor. Khi model ẩn, gate cả castShadow theo visibility gốc và ancestor để không để lại bóng “ma”. Các mối nối hình học thuộc cùng group/cùng hệ trục.

## Độ nét

Stage có thể là 1600 x 900, nhưng đó là hệ bố cục, không phải độ phân giải canvas cuối.

```js
const scale = host.getBoundingClientRect().width / host.clientWidth;
const pixelRatio = Math.min(cap, Math.max(1, devicePixelRatio * scale));
renderer.setPixelRatio(pixelRatio);
renderer.setSize(host.clientWidth, host.clientHeight, false);
```

Trong Studio `cap=3`, `quality=low` dùng 1x. Đây là giới hạn thử thực tế, cần điều chỉnh theo máy và kích thước host. Kiểm resize, browser zoom, fullscreen và thay DPR khi sang màn hình khác. Trước cap, canvas physical size phải khớp `display size × DPR`; sau cap, ghi rõ sự giảm mật độ có chủ đích.

Texture chữ: nạp đúng phông và weight bằng `document.fonts.load()` trước `fillText`. Chờ `document.fonts.ready` không bảo đảm weight chưa từng dùng đã nạp. Giữ canvas aspect bằng aspect mặt phẳng, tránh đặt min-height quá lớn rồi kéo bẹt glyph. Bản sửa dùng texture width 2048, không cần ép mọi label tới con số này.

Phông nhúng phải có glyph tiếng Việt và giấy phép phù hợp. Shader antialias, mesh đủ phân đoạn, map bóng và ánh sáng được kiểm ở kích thước thật. Hình nét gốc có thể vẫn là bitmap, không tuyên bố vô hạn độ nét khi phóng lớn.

## Dự phòng, offline, QR

Baked images đánh khóa theo slide/build/case/filter. Dự phòng không hiển thị trạng thái cuối khi người nói đang ở bước đầu. Bản Studio bake 2x, dùng một pool binary khử trùng SHA-256 cho fallback và print. SHA khử trùng chỉ nói hai file giống nhau, không chứng minh nội dung đúng.

HTML tự chứa bundle, phông, hình, dataset và downloads nếu yêu cầu offline. Kiểm cả `file://` và network disabled. File tự chứa vẫn có link ngoài; QR/link ngoài cần mạng. Giữ QR nguyên nguồn được giao, so byte và pixel theo thời gian, rồi thử quét tại khoảng cách thực.

## Hiệu năng

Đo sau warmup và ở cảnh nặng. Ghi viewport, DPR, quality, browser, backend, p95 frame interval, CPU render call, drawCalls, triangles, memory counters và errors. Đo frame interval bằng RAF là nhịp trình duyệt, không phải GPU frame time. Không biến một phép đo 1x thành bảo đảm Retina/Windows/máy chiếu.

Theo dõi resource counts qua vài vòng toàn bộ deck. Counts ổn không tự chứng minh không có mọi memory leak. Giữ chất lượng fallback phù hợp trước khi tăng pixel ratio, shadows hoặc hậu kỳ.

## Dùng lại source Studio

Giải nén `starter/Nguon-slide-HTML-3D.zip`, giữ cấu trúc `AI-technique/animation-mock`. Python 3.10+ dựng HTML từ bundle đã có:

```sh
cd AI-technique
python3 animation-mock/.build/studio/build.py
```

Nội dung: `deck.studio.json`. CSS: `.build/studio/studio.css`. State/actions: `runtime.js`. Scene: `scenes.js`. Nếu sửa 3D, `npm ci` trong `.build/ten/vendor`, bundle bằng esbuild ở đó, render lại stills rồi chạy builder. Giữ lockfile và các giấy phép OFL/MIT.

Starter dùng 10 trang cố định, ids 1..10, kind/layout và dataset cụ thể. Khi đổi số trang: sửa clamp/boundary/next/previous/getState trong runtime, navigation/counters trong builder, mapping scene/stills và assertions trong tests. Không chỉ thêm slide vào JSON. Khi đổi domain: thay verifier, nguồn, trạng thái thiếu và file download. Không dùng kết quả kiểm lịch để xác nhận một hồ sơ ngành khác.

Các test hiện có chứa đường dẫn Node/Playwright/Chrome của máy tác giả. Thay bằng runtime máy mới hoặc env config trước khi chạy. Trong quá trình tổng kết, chưa tạo generator tự thích nghi mọi schema/số trang, chưa kiểm tự động các domain mới.
