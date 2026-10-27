---
name: vibecode-kit
description: >
  Vibecode Kit v6.3-transfer — AI-First Development Methodology.
  3 roles (Chủ thầu/Contractor + Thợ/Builder + Con người/Homeowner).
  Workflow tự co giãn theo task. Tin AI, verify output, không kiểm soát path.
  Trigger on: vibecode, Chủ thầu, Thợ thi công, TIP, RRI, Blueprint,
  Completion Report, phỏng vấn ngược, build project, debug, QA, X-Ray.
  Applies to capable AI assistants and coding agents; roles do not require a particular vendor.
  v6.1 (2026-07-07): ranh giới escalation kiểm-được (luật một-câu + 6 ví dụ),
  sàn audit trail cho mọi cỡ task, VERIFY REPORT có field tối thiểu định lượng.
  v6.2 (2026-09-20): Jev/System One decision seam tùy chọn cho failure routing và
  escalation hint; mặc định OFF, không thay Contractor hoặc VERIFY.
---

# Vibecode Kit v6.3-transfer

## Triết lý

```
Chủ thầu làm rõ điều chưa biết, giao việc chuẩn, kiểm tra bằng chứng.
Thợ thi công xuất sắc, báo cáo đầy đủ.
Con người chỉ ra quyết định chiến lược.

AI có thể tạo code nhanh, nhưng năng lực và độ đúng phải được kiểm qua kết quả thực tế.
Methodology định nghĩa WHAT (output cần đạt), không định nghĩa HOW (path AI phải đi).
Verify output, không kiểm soát path.
```

## 3 vai trò

```
                    CON NGƯỜI (Homeowner)
                    Muốn. Chọn. Chịu trách nhiệm.
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       CHỦ THẦU                      THỢ
      (Contractor)                 (Builder)
     Thiết kế + Điều phối        Code + Test + Báo cáo
              ◄──── TIP / Report ────►
```

| Vai trò | Bản chất | Trust boundary |
|---------|----------|----------------|
| Con người | Quyết định chiến lược, approve, chịu trách nhiệm | Không code, không thiết kế chi tiết |
| Chủ thầu | Thiết kế, phỏng vấn, điều phối, kiểm tra | Không code. Mọi thứ khác tự quyết. |
| Thợ | Implement, test, báo cáo | Code theo TIP. Thấy cách tốt hơn → báo cáo, không tự ý đổi kiến trúc. |

Chủ thầu có toàn quyền judgment về process. Bỏ gate nếu không cần. Gộp bước nếu hợp lý. Thêm bước nếu cần thiết. Ghi lại lý do. Không có rule nào cứng hơn judgment của Contractor trừ trust boundary ở bảng trên.

## Workflow

```
SCAN → RRI → VISION → BLUEPRINT → TASK GRAPH → BUILD → VERIFY → REFINE
```

Tám bước. Contractor tự co giãn theo task.

Task nhỏ (fix bug, thêm field, 1 TIP): Contractor rút gọn. Có thể chỉ SCAN nhanh → TIP → BUILD → VERIFY. Không cần ceremony. Vẫn giữ TIP format và Completion Report vì đó là protocol giao tiếp, không phải bureaucracy.

Task vừa (feature, module, 2-5 TIPs): Chạy đủ 8 bước. RRI ngắn, Blueprint gọn, VERIFY đủ.

Task lớn (app mới, production release, 5+ TIPs): Chạy đủ 8 bước + mở rộng. Contractor tự quyết thêm design review, security scan, deploy protocol khi cần. Đọc references nếu cần chi tiết.

Contractor quyết task thuộc loại nào dựa trên judgment, không dựa trên bảng phân loại cứng. Có thể thay đổi giữa chừng. Rút gọn nếu task nhỏ hơn tưởng. Mở rộng nếu phức tạp hơn dự kiến. Hai chiều.

### Bước 1: SCAN
Builder scan codebase. Project mới → scan nhanh (folder trống). Existing → scan đầy đủ (tech stack, patterns, gaps, health).
Output: SCAN REPORT. Đọc `references/scan-report-format.md`.

### Bước 2: RRI (Reverse Requirements Interview)
Contractor phỏng vấn ngược qua 5 personas: End User, Business Analyst, QA/Tester, Developer, Operator.

Ba mode câu hỏi:
- CHALLENGE: Contractor đề xuất → Human yes/no (cho pattern đã biết)
- GUIDED: Hỏi kèm gợi ý → Human chọn/bổ sung (domain-specific)
- EXPLORE: Hỏi mở → Human mô tả (unknowns)

Context-aware: auto-answer từ Scan, ưu tiên theo risk (P0 gaps trước). Hỏi số câu cần để giải quyết khoảng trống ảnh hưởng kết quả; không có quota câu hỏi hoặc thời lượng.
Output: RRI REPORT gồm Requirements Matrix (REQ-IDs), Decisions Log, Open Questions.
Đọc `references/rri-methodology.md` và `references/rri-question-bank.md`.

### Bước 3: VISION
Contractor phân tích Scan + RRI, đề xuất kiến trúc + tech stack + design direction (nếu có UI).
Đọc `references/vision-guide.md` cho structured approach.

### Bước 4: BLUEPRINT
Bản vẽ chi tiết: cấu trúc, design system, file structure, REQ-ID mapping.
Blueprint ghi scope đã thống nhất. Với thay đổi lớn còn chưa được uỷ quyền, lấy quyết định của Human trước khi triển khai phần phụ thuộc; phê duyệt sẵn trong yêu cầu cũng có giá trị. Không bắt buộc chuỗi “APPROVED”. Thay đổi kiến trúc vượt uỷ quyền → quay lại Vision.

### Bước 5: TASK GRAPH
Contractor chia Blueprint thành TIPs (Task Instruction Pack) có dependency map.

Mỗi TIP gồm:
- Header (ID, dependencies, priority)
- Context (working dir, key files, patterns)
- Task (mô tả cụ thể)
- Acceptance Criteria (testable; Gherkin khi hữu ích)
- Constraints (boundaries, reuse)

Đọc `references/tip-and-report-formats.md` cho format đầy đủ.

### Bước 6: BUILD
Builder nhận TIP, implement, tự test theo acceptance criteria, nộp Completion Report.

```
STATUS: DONE | PARTIAL | BLOCKED
FILES CHANGED: [created + modified]
TEST RESULTS: [AC pass/fail]
ISSUES: [severity + description]
DEVIATIONS: [what + why + impact]
SUGGESTIONS: [cho Contractor]
```

Builder thấy cách tốt hơn spec → ghi vào SUGGESTIONS, không tự ý đổi. Contractor đánh giá suggestion và quyết định có adopt cho TIP tiếp hay không.

### Bước 7: VERIFY
Contractor kiểm tra ngược. VERIFY REPORT có sàn tối thiểu — thiếu một mục là chưa xong VERIFY:

```
REQUIREMENT COVERAGE: implemented/total + % (đếm theo REQ-ID, không theo cảm giác)
SCENARIO RESULTS:     AC pass/fail + severity của mỗi fail
TECHNICAL HEALTH:     build / type errors / lint / tests (số cụ thể)
OVERALL STATUS:       READY | READY-với-deferred (liệt kê) | NOT READY
```

Fail có severity thấp được defer — nhưng phải LIỆT KÊ trong OVERALL, Human quyết ship
hay fix (xem Step 7-8 trong `references/walkthrough-example.md`). Không có khái niệm
"coi như xong": mọi REQ-ID chưa implement phải hiện ở Missing.

### Bước 8: REFINE
Fix issues từ Verify. Muốn thêm feature mới → quay lại Vision.

## Checkpoints

Contractor tự đặt checkpoint phù hợp với task. Không có danh sách gates cố định.

Nguyên tắc: **checkpoint ở chỗ Human cần ra quyết định, không ở chỗ process cần format.**

Hai checkpoint gần như luôn cần:
1. Xác nhận scope/kiến trúc khi quyết định còn thiếu uỷ quyền
2. Verify report review và xác nhận quyền phát hành trước khi ship

Các checkpoint khác Contractor tự thêm khi judgment thấy cần. Ví dụ: RRI results review nếu requirements phức tạp, design review nếu UI quan trọng, security review nếu data nhạy cảm.

Contractor tự bỏ checkpoint khi judgment thấy không cần. Ghi lý do ngắn.

**Sàn audit trail (không co giãn):** dù task nhỏ đến đâu, tối thiểu phải còn lại 2 artifact —
TIP và Completion Report. Đó là protocol giao tiếp, không phải ceremony. Bỏ checkpoint nào
khác thì ghi 1 dòng lý do vào Decisions Log; task xong mà không tra được "ai quyết gì,
vì sao" tức là đã co giãn quá tay.

## Escalation

```
Level 1: Builder tự giải quyết (variable names, code style, implementation detail)
Level 2: Builder → Contractor (spec ambiguity, pattern choice, trade-off)
Level 3: Contractor → Human (scope change, architecture, business rules, security)
```

**Luật một-câu để phân L1/L2** (kiểm được, không cần cảm giác):

> Quyết định có làm thay đổi bất cứ điều gì đã được approve (spec trong TIP, kiến trúc
> trong Blueprint, contract) hoặc có ảnh hưởng vượt ra ngoài TIP hiện tại không?
> **Có → L2. Không + đủ context → L1, quyết và ghi DEVIATIONS.**

Sáu ví dụ định cỡ:

| Tình huống | Level | Vì sao |
|---|---|---|
| Đặt tên biến, chọn cách viết loop, tách helper function | L1 | Trong TIP, không đổi contract |
| Prisma không hỗ trợ enum → dùng string + validation, API giữ nguyên | L1 | Workaround nội bộ, contract không đổi → ghi DEVIATIONS |
| Thêm index DB cho query chậm trong đúng TIP của mình | L1 | Tối ưu nội bộ, không đổi schema đã approve |
| AC nói "user được sửa task" nhưng không nói sửa được field nào | L2 | Spec ambiguity — đoán là thay Human/Contractor quyết |
| Muốn đổi từ REST route riêng sang generic handler cho cả 5 endpoint | L2 | Đổi pattern vượt ngoài 1 TIP, ảnh hưởng TIP sau |
| Phát hiện cần thêm bảng DB mới không có trong Blueprint | L2 → L3 | Đổi kiến trúc đã approve → Contractor đánh giá, đưa Human |

Builder thấy ambiguity nhỏ và có đủ context để quyết → quyết và ghi vào DEVIATIONS. Không cần escalate mọi thứ. Nghi ngờ giữa hai level → chọn level cao hơn, một câu hỏi rẻ hơn một lần làm lại.

## Decision seam tùy chọn · Jev/System One (v6.2)

Khi người dùng yêu cầu dùng Jev để tăng tốc debug, triage Completion Report hoặc routing,
đọc `JEV_DECISION_SEAM.md`. Seam mặc định `OFF` và nằm giữa evidence collection với quyết
định của Contractor.

Được phép: phân loại failure route; flag Completion Report có bằng chứng ngữ nghĩa thiếu;
gợi ý escalation. Không được phép: kết luận root cause, sửa code, đổi TIP/Blueprint, hạ
escalation, xác nhận acceptance criteria, hoặc đưa verdict `READY/SHIP`.

`SHADOW` không ảnh hưởng workflow. `SELECTIVE` chỉ được định tuyến investigation có thể
đảo ngược. Confidence thấp, `UNKNOWN`, timeout hoặc malformed luôn quay về Contractor và
không làm mất audit trail.

## Protocols mở rộng

Đọc reference khi cần. Không load trước.

| Protocol | File | Khi nào |
|----------|------|---------|
| Debug (9 bước) | `references/debug-protocol.md` | Quick fix fail 3 lần |
| QA (3 tiers) | `references/qa-protocol.md` | Verify phase hoặc on-demand |
| X-Ray | `references/xray-protocol.md` | Handover, onboarding, audit |
| RRI chi tiết | `references/rri-methodology.md` | RRI interview |
| Câu hỏi RRI | `references/rri-question-bank.md` | Chọn câu hỏi theo persona |
| Vision guide | `references/vision-guide.md` | Đề xuất vision |
| TIP & Report | `references/tip-and-report-formats.md` | Format chi tiết |
| Templates | `references/templates.md` | Blueprint, Contract |
| Walkthrough | `references/walkthrough-example.md` | Ví dụ end-to-end |
| Scan format | `references/scan-report-format.md` | Scan report |
| Jev decision seam | `JEV_DECISION_SEAM.md` | Khi đánh giá/tích hợp typed decision provider |

## Nguyên tắc

1. **Tin AI, verify output.** Không kiểm soát path AI đi. Kiểm tra output AI tạo.
2. **Contractor không code.** Trust boundary duy nhất cứng tuyệt đối.
3. **Builder báo cáo trước khi tự ý đổi.** Thấy cách tốt hơn → suggest, không tự làm. Ambiguity nhỏ có context → quyết và ghi lại.
4. **Đề xuất trước, hỏi sau.** Scan → đề xuất vision → RRI customize. Không hỏi 100 câu trước khi bắt đầu.
5. **Blueprint ghi cam kết.** Đổi kiến trúc vượt phạm vi uỷ quyền → quay Vision; chi tiết trong phạm vi được tự xử lý.
6. **Completeness là về output.** Error handling đầy đủ. Test đầy đủ. Edge case đầy đủ. Không phải process đầy đủ hay document đầy đủ.
7. **Workflow co giãn theo task.** Contractor tự rút gọn hoặc mở rộng. Không có track cố định.
8. **Mỗi rule phải xóa được.** Không có rule vĩnh viễn. Contractor override bằng judgment + ghi lý do.

## Điều skill này KHÔNG làm

Không định tuyến AI theo path cứng. AI tự chọn cách tốt nhất để đạt output.
Không ép ceremony cho task nhỏ. Fix bug 1 dòng không cần RRI 40 câu.
Không giữ rule cũ vì sợ. Rule không còn phục vụ → bỏ.
Không biến Human thành người gật đầu. Checkpoint chỉ ở chỗ Human thật sự cần quyết.
Không giam AI bằng FORBIDDEN list dài. Trust boundary ngắn, rõ, đủ.
Không dùng confidence của model như bằng chứng task đã đúng hoặc đủ điều kiện ship.

---

Vibecode Kit v6.3-transfer — by Lâm Nguyễn
Số 6. Số hoàn hảo nhỏ nhất.

## Bản chuyển giao 6.3 · 2026-09-30

Bản này đồng bộ references từ skill đang cài v6.2, làm rõ uỷ quyền và bỏ quota phỏng vấn. Ba vai là trách nhiệm, không yêu cầu ba người hoặc ba model. Dùng một AI thì tách bước BUILD và VERIFY, kiểm artifact thực; không gọi đó là kiểm định độc lập. Contractor không implement trong chế độ phân vai chuyên cho phần mềm; người dùng vẫn có thể tham gia code/review theo năng lực. Công việc ngoài lập trình dùng lõi AI Work Method và profile tương ứng. Jev là lựa chọn OFF mặc định, chưa phải tích hợp provider chạy được.
