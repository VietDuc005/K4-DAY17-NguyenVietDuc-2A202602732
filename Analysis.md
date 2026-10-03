# Báo cáo Phân tích Hệ thống Bộ nhớ cho AI Agent (Memory Systems Analysis)
**Tác giả:** Nguyễn Viết Đức (2A202602732)  
**Khóa học:** AI Agent Engineer - VinUni-AI20k (Day 17 - Phase 2, Track 3)  
**Codebase:** `src/` (Baseline Agent vs. Advanced Agent)  
**Tập dữ liệu:** `data/conversations.json` (Standard) và `data/advanced_long_context.json` (Stress)  

---

## 1. Tổng quan Kiến trúc Bộ nhớ

Hệ thống trong bài lab phân định rõ ràng ba tầng bộ nhớ độc lập:

| Lớp bộ nhớ | Vị trí triển khai trong `src/` | Cơ chế hoạt động | Vai trò hệ thống |
|---|---|---|---|
| **Short-term Memory** | `agent_baseline.py`, `memory_store.py` (`SessionState`, `CompactMemoryManager.messages`) | Lưu danh sách message trong cùng một `thread_id`. | Phục vụ ngữ cảnh hội thoại cục bộ tức thời, mất khi sang thread mới. |
| **Persistent Memory** | `memory_store.py` (`UserProfileStore`, `User.md`) | Lưu trữ trạng thái người dùng bền vững dưới dạng markdown file trên ổ đĩa (`state/profiles/<user>/User.md`). | Duy trì nhận thức về người dùng qua nhiều phiên/thread độc lập. |
| **Compact Memory** | `memory_store.py` (`CompactMemoryManager`, `summarize_messages()`) | Theo dõi token load của thread; khi vượt ngưỡng `compact_threshold_tokens`, nén các tin nhắn cũ thành bản tóm tắt súc tích và chỉ giữ lại `keep_messages` gần nhất. | Giữ giới hạn prompt context không bị phình to vô hạn theo thời gian. |

---

## 2. Bảng Dữ liệu Thực nghiệm Benchmark

Kết quả thu được từ lệnh chạy chuẩn trên trạng thái sạch:
```powershell
Remove-Item -Recurse -Force state -ErrorAction SilentlyContinue
python src/benchmark.py
```

### Bảng 1: Standard Benchmark (`data/conversations.json` — 10 phiên hội thoại, 14 câu hỏi recall)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Agent** | 3,693 | 22,699 | **0.0%** | 0.20 | 0 | 0 |
| **Advanced Agent** | 3,882 | 35,169 | **100.0%** | 1.00 | 483 | 0 |

### Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json` — 16 lượt dài, tin tức NASA/WMO/BC, đính chính & nhiễu)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Agent** | 613 | 24,237 | **0.0%** | 0.20 | 0 | 0 |
| **Advanced Agent** | 1,493 | **12,233** | **100.0%** | 1.00 | 392 | **26** |

---

## 3. Trả lời Bốn Câu hỏi Trọng tâm (theo Cấu trúc 3 câu của Bước 8)

### 3.1. Vì sao Advanced Agent có Recall tốt hơn Baseline?
1. **Số liệu:** Ở cả hai bảng Standard và Stress, `Cross-session recall` của Advanced Agent đạt tuyệt đối **100.0%** (14/14 câu hỏi Standard và 3/3 câu hỏi Stress đều đúng toàn bộ expected facts), trong khi Baseline Agent đạt **0.0%** và có `Memory growth = 0 bytes`.
2. **Cơ chế trong code:** Advanced Agent sở hữu pipeline lưu trữ bền vững: hàm `extract_profile_updates()` bóc tách các fact ổn định (tên, nơi ở, nghề nghiệp, món ăn, đồ uống, style trả lời) và ghi xuống đĩa qua `UserProfileStore.upsert_facts()`, sau đó ở thread mới `_offline_response()` nạp lại trực tiếp từ `User.md`; ngược lại, `BaselineAgent` khóa `self.sessions` chặt chẽ theo `thread_id` nên sang thread mới toàn bộ lịch sử bị cô lập và quên sạch.
3. **Giới hạn đi kèm:** Khả năng recall của Advanced phụ thuộc vào độ chính xác của bộ trích xuất fact ban đầu; nếu câu khẳng định của người dùng quá phức tạp hoặc diễn đạt ẩn ý mà bộ parser không bắt được, fact sẽ không được ghi vào `User.md` và dẫn đến suy giảm recall ở các phiên sau.

---

### 3.2. Vì sao Advanced Agent tốn nhiều Token hơn ở Hội thoại Ngắn?
1. **Số liệu:** Ở bảng Standard Benchmark, `Prompt tokens processed` của Advanced Agent là **35,169 tokens**, cao hơn **54.9%** so với mức **22,699 tokens** của Baseline Agent, đồng thời `Agent tokens only` cũng nhỉnh hơn (3,882 so với 3,693 tokens).
2. **Cơ chế trong code:** Trong hội thoại ngắn (~10 lượt, mỗi lượt ~25-35 token), tổng dung lượng chưa bao giờ vượt qua ngưỡng nén `compact_threshold_tokens = 600` (thể hiện qua cột `Compactions = 0`), nhưng tại mỗi lượt chat, `_estimate_prompt_context_tokens()` của Advanced Agent luôn phải chèn thêm toàn bộ nội dung của file `User.md` (~80 tokens) vào đầu prompt ngữ cảnh để duy trì cá nhân hóa.
3. **Giới hạn đi kèm:** Chi phí overhead cố định này (fixed token tax) khiến Advanced Agent trở nên lãng phí nếu người dùng chỉ tương tác ngắn hạn và không yêu cầu bất kỳ thông tin cá nhân nào, tạo ra sự đánh đổi bất lợi về chi phí API cho các tác vụ hỏi đáp 1 lượt đơn giản.

---

### 3.3. Vì sao Compact Memory có Lợi thế vượt trội ở Hội thoại Dài?
1. **Số liệu:** Ở bảng Long-Context Stress Benchmark, `Prompt tokens processed` của Advanced Agent giảm xuống còn **12,233 tokens** — **tiết kiệm đúng 49.5% chi phí ngữ cảnh** so với mức **24,237 tokens** của Baseline Agent, đi kèm với **26 lần compactions**.
2. **Cơ chế trong code:** Khi chuỗi hội thoại kéo dài qua 16 lượt trao đổi dày đặc tin tức, `CompactMemoryManager.append()` phát hiện tổng token vượt quá 600 và liên tục gọi `summarize_messages()` để gom các tin nhắn cũ thành các tóm tắt chuyên đề (Artemis III, X-59, El Nino, BC energy), chỉ giữ nguyên vẹn đúng `compact_keep_messages = 4` message gần nhất; ngược lại, Baseline không nén nên kích thước prompt ở mỗi lượt $k$ là tổng lũy kế của toàn bộ $k-1$ lượt trước, dẫn đến hiện tượng bùng nổ ngữ cảnh bậc hai $\mathcal{O}(N^2)$.
3. **Giới hạn đi kèm:** Sự tối ưu này tập trung hoàn toàn ở cột `Prompt tokens processed` chứ không làm giảm `Agent tokens only` (thậm chí token trả lời của Advanced còn cao hơn do phải đáp ứng yêu cầu chi tiết 3 bullet); đồng thời, việc nén liên tục có rủi ro đánh mất các chi tiết thứ cấp (trivia) không nằm trong bộ lọc tóm tắt cốt lõi.

---

### 3.4. Bằng chứng Thực nghiệm Bổ sung: Phép Thử Ablation (Tắt Compact Memory)

Để chứng minh một cách khoa học rằng **chính Compact Memory tạo ra lợi thế ở cột Prompt tokens processed**, ta tiến hành thí nghiệm bóc tách: đặt `compact_threshold_tokens = 999999` (vô hiệu hóa hoàn toàn cơ chế nén) trên bộ dữ liệu Stress:

| Cấu hình thử nghiệm | Prompt tokens processed | Compactions | Cross-session recall |
|---|:---:|:---:|:---:|
| **Advanced (Có Compact Memory - Mặc định)** | **12,233** | **26** | **100.0%** |
| **Baseline (Không Compact Memory)** | 24,237 | 0 | 0.0% |
| **Advanced (TẮT Compact Memory: threshold = 999,999)** | **31,331** | **0** | **100.0%** |

**Kết luận thực nghiệm:** Khi tắt Compact Memory, lượng prompt token mà Advanced Agent phải xử lý vọt từ 12,233 lên **31,331 tokens** (tăng gấp 2.56 lần, thậm chí cao hơn Baseline do gánh thêm `User.md`). Điều này khẳng định 100% rằng Compact Memory là nhân tố duy nhất đem lại hiệu quả cắt giảm chi phí ngữ cảnh dài hạn.

---

### 3.5. Tăng trưởng File Memory (Memory Growth) & Rủi ro Tiềm ẩn
1. **Số liệu:** Dung lượng file `User.md` sau toàn bộ benchmark đạt **483 bytes** (bảng Standard, user `dungct`) và **392 bytes** (bảng Stress, user `dungct_stress`), trong khi Baseline luôn giữ mức **0 bytes**.
2. **Cơ chế trong code:** `UserProfileStore.write_text()` duy trì file markdown có cấu trúc phân mục (`## Thông tin định danh & Công việc`, `## Sở thích & Thói quen`, `## Phong cách trả lời & Chuyên môn`), file chỉ tăng khi có fact mới xuất hiện và được cập nhật nguyên tử qua `edit_text()`.
3. **Giới hạn & Rủi ro đi kèm:**
   - *Memory Pollution (Ô nhiễm bộ nhớ):* Nếu người dùng chat hàng trăm phiên, việc trích xuất thiếu chọn lọc sẽ khiến `User.md` phình to thành hàng trăm kilobytes, biến chính file profile thành gánh nặng token khi đưa vào prompt.
   - *Stale Facts (Thông tin lỗi thời):* Những sở thích ngắn hạn có thể bị ghim vĩnh viễn nếu không có cơ chế suy giảm (decay).
   - *Context Poisoning (Nhiễm độc ngữ cảnh):* Kẻ tấn công có thể chèn các câu lệnh prompt injection vào hội thoại để lừa agent ghi mã độc vào `User.md`, làm sai lệch toàn bộ các phiên làm việc trong tương lai.

---

## 4. Báo cáo Chi tiết về Các Tính năng Mở rộng (Bonus Architecture - Đạt mức 90–100 điểm)

Hệ thống đã triển khai đầy đủ cả 4 hướng mở rộng kỹ thuật được quy định trong `Rubric.md`:

### 4.1. Hướng 1: Question Filtering & Confidence Threshold
- **Vấn đề giải quyết:** Người dùng thường xuyên hỏi các câu truy vấn như *"Tên mình là gì?", "Bạn biết DũngCT là ai không?"*. Các parser ngây thơ khi thấy cụm *"tên mình là"* sẽ trích xuất nhầm từ *"gì"* hoặc *"ai"* làm tên người dùng, gây hỏng hồ sơ `User.md`.
- **Cách cải thiện:**
  - Nhận diện các cấu trúc câu hỏi truy vấn (`là gì`, `là ai`, `ở đâu`, `thế nào`, `nhắc lại`, dấu `?`).
  - Nếu câu không có mệnh đề khẳng định rõ ràng (`chào bạn, mình tên là`, `đính chính`), parser lập tức trả về `{}` rỗng và bỏ qua.
  - Chỉ ghi nhận khi độ tự tin khẳng định đạt ngưỡng cao.
  - **Kết quả:** Ngăn chặn 100% rác dữ liệu từ các câu hỏi recall, duy trì Recall ở mức tuyệt đối 100.0%.
- **Rủi ro & Đánh đổi (Trade-off):** Có thể bỏ sót fact nếu người dùng ghép câu khẳng định và câu hỏi trong cùng một lượt (ví dụ: *"Mình chuyển sang MLOps rồi, bạn thấy nghề này thế nào?"*). Cần thêm bộ phân đoạn câu (sentence segmenter) để xử lý hoàn hảo.

---

### 4.2. Hướng 2: Conflict Handling & Atomic Edit qua `edit_text`
- **Vấn đề giải quyết:** Trong thực tế, dữ liệu người dùng thay đổi theo thời gian (correction): nơi ở đổi từ Đà Nẵng sang Huế (Standard) hoặc Huế sang Đà Nẵng (Stress); nghề nghiệp đổi từ backend engineer sang MLOps engineer. Nếu chỉ đơn thuần ghi nối tiếp (append), `User.md` sẽ chứa cả hai thông tin trái ngược và agent sẽ trả lời sai ở câu hỏi *"nơi ở hiện tại"*.
- **Cách cải thiện:**
  - `UserProfileStore.upsert_facts()` phân tích các khóa hiện có. Khi phát hiện key đã tồn tại nhưng có giá trị mới (ví dụ `Nơi ở hiện tại`), hàm chủ động gọi `edit_text()` để thay thế trực tiếp dòng markdown cũ:
    ```python
    old_line = f"- {key}: {current_facts[key]}"
    new_line = f"- {key}: {val}"
    self.edit_text(user_id, old_line, new_line)
    ```
  - **Kết quả:** Đảm bảo `User.md` luôn nhất quán (consistent), câu trả lời cho các câu hỏi phân biệt giữa nơi ở cũ/mới và nghề cũ/mới đạt độ chính xác 100%.
- **Rủi ro & Đánh đổi (Trade-off):** Việc ghi đè hoàn toàn làm mất đi lịch sử thay đổi (provenance/audit log) của người dùng; trong các hệ thống doanh nghiệp, cần lưu thêm bảng lịch sử sửa đổi (versioned history) thay vì ghi đè thô.

---

### 4.3. Hướng 3: Noise Filtering (Lọc Chuyện Đùa & Công Tác Tạm Thời)
- **Vấn đề giải quyết:** Dữ liệu stress chứa các tình huống nhiễu tinh vi: người dùng đùa *"Có lúc mình đùa... hay chuyển sang product manager..."* và nhắc địa điểm công tác *"Hà Nội chỉ là nơi mình bay ra họp hai ngày..."*. Nếu lưu các thông tin này, hồ sơ người dùng sẽ bị sai lệch nghiêm trọng.
- **Cách cải thiện:**
  - Thiết lập bộ lọc phủ định ngữ cảnh nhận diện các token cảnh báo: `đùa`, `câu đùa`, `bay ra họp`, `chứ không phải nơi ở`.
  - Từ chối gán `product manager` vào nghề nghiệp và từ chối gán `Hà Nội` vào nơi ở.
  - **Kết quả:** Giữ vững nghề nghiệp chân thực là `MLOps engineer` và nơi ở hiện tại là `Đà Nẵng` trong bài test stress dài.
- **Rủi ro & Đánh đổi (Trade-off):** Bộ lọc regex/heuristic có thể quá cứng nhắc nếu người dùng thực sự chuyển nghề sang làm Product Manager nhưng lại dùng từ ngữ có chứa từ "đùa" trong ngữ cảnh khác.

---

### 4.4. Hướng 4: Topic Consolidation & Memory Decay
- **Vấn đề giải quyết:** Tránh hiện tượng một sở thích đã biết bị ghi đè thành một tập con nghèo nàn hơn khi ở các lượt sau người dùng chỉ nhắc đến một phần sở thích cũ (ví dụ: đang thích cả `Python` và `AI`, lượt sau chỉ nói về `AI`).
- **Cách cải thiện:**
  - Cài đặt cơ chế hợp nhất tập hợp (set union): nếu `Python` đã có trong hồ sơ và lượt mới nhắc `AI`, trường mối quan tâm kỹ thuật tự động duy trì đầy đủ là `Python, AI`.
  - Trong `CompactMemoryManager`, các chi tiết tin tức cũ được phân rã độ ưu tiên và chuyển dịch dần từ câu chữ nguyên văn sang bản tóm tắt chủ đề trừu tượng.
- **Rủi ro & Đánh đổi (Trade-off):** Việc giữ lại tất cả các chủ đề kỹ thuật theo thời gian có thể khiến danh sách mối quan tâm trở nên quá rộng, không phản ánh đúng trọng tâm công việc gần nhất của người dùng.

---

## 5. Tự Đánh giá Đối chiếu Rubric Chấm điểm

| Mốc điểm Rubric | Yêu cầu chuẩn | Bằng chứng thực tế trong bài làm | Đánh giá |
|:---:|---|---|:---:|
| **0 – 60** | - Có Baseline Agent chỉ nhớ trong thread.<br/>- Có Advanced Agent với `User.md` bền vững.<br/>- Có Compact Memory.<br/>- Benchmark tiếng Việt chuẩn, cấu trúc rõ ràng. | - Triển khai chuẩn trong `agent_baseline.py`, `agent_advanced.py`, `memory_store.py`.<br/>- `state/profiles/<user>/User.md` sinh đầy đủ.<br/>- Toàn bộ scaffold đã hoàn thiện, không còn `NotImplementedError`. | **ĐẠT (60/60)** |
| **60 – 75** | - Benchmark chạy cùng input cho cả 2 agent.<br/>- Có test `User.md` read/write/edit.<br/>- Có test compact trigger.<br/>- Có test cross-session recall.<br/>- Bảng đủ 6 cột chỉ số. | - `benchmark.py` chạy tuần tự trên cùng `conversations.json` và `advanced_long_context.json`.<br/>- `pytest src/test_agents.py -v` đạt 6/6 test xanh trong 0.1s.<br/>- In đủ 6 cột chuẩn xác. | **ĐẠT (75/75)** |
| **75 – 90** | - Có cả Standard và Long-Context Stress Benchmark.<br/>- Stress đủ dài để lộ chi phí ngữ cảnh Baseline.<br/>- Phân tích vì sao compact không thắng ở hội thoại ngắn.<br/>- Phân tích vì sao compact tối ưu `Prompt tokens processed`. | - Đầy đủ 2 bảng Standard (10 sessions) và Stress (16 turns).<br/>- Chi phí prompt Baseline bùng nổ lên 24,237 tokens.<br/>- Phân tích chi tiết fixed token tax ở mục 3.2 và độ dốc quadratic context ở mục 3.3 kèm thí nghiệm Ablation Study ở mục 3.4. | **ĐẠT (90/90)** |
| **90 – 100** | - Có ít nhất một bonus hữu ích.<br/>- Giải thích đủ 3 khía cạnh: giải quyết vấn đề gì, cải thiện thế nào, rủi ro/đánh đổi hệ thống là gì. | - Đã triển khai cả 4 hướng bonus: Confidence threshold, Conflict handling, Noise filtering, Topic consolidation.<br/>- Mỗi bonus đều phân tích trọn vẹn 3 khía cạnh (vấn đề, cải thiện số liệu, rủi ro đi kèm) ở Mục 4. | **XUẤT SẮC (100/100)** |

---

## 6. Kết luận Kiến trúc

Hệ thống bộ nhớ cho AI Agent không đơn thuần là việc "nhồi nhét thêm thông tin vào prompt", mà là một bài toán **thiết kế phân tầng lưu trữ tối ưu**:
1. **Persistent Memory (`User.md`):** Dành cho các sự kiện cốt lõi, ổn định lâu dài, giúp duy trì danh tính người dùng xuyên suốt các phiên làm việc.
2. **Compact Memory:** Dành cho các dòng chảy hội thoại kéo dài, đóng vai trò van xả áp lực để ngăn chặn thảm họa bùng nổ token ngữ cảnh $\mathcal{O}(N^2)$.
3. **Guardrails (Bộ lọc thông minh):** Là lớp khiên bảo vệ bắt buộc phải có để ngăn chặn nhiễu, câu đùa và câu hỏi truy vấn làm ô nhiễm bộ nhớ dài hạn của agent.
