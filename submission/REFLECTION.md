# Reflection — Lab 19

**Tên:** Cao Minh Quang
**Cohort:** _<điền mã lớp của bạn, vd. A20-K1>_
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Đo trên 50 golden queries (NB2 + `make benchmark`): `exact` — BM25 và
hybrid ngang nhau (96.7%), vì từ khoá kỹ thuật xuất hiện verbatim nên BM25
đã đủ tín hiệu. `paraphrase` — bất ngờ nhất: BM25 (33.3%) > hybrid (32.0%)
> semantic (24.0%), vì embedding model `bge-small-en-v1.5` train tiếng Anh,
yếu với câu diễn đạt lại tiếng Việt thuần — trái với kỳ vọng "vector thắng
paraphrase". `mixed` — hybrid thắng rõ nhất (100% vs 97–98.5%) vì RRF cộng
tín hiệu từ cả hai phía, không phụ thuộc một retriever duy nhất.

Không dùng hybrid khi: (1) query là tra cứu chính xác (mã lỗi, ID, log) —
BM25 đơn đã đủ và nhanh hơn nhiều (P50 0.8ms vs 33ms); (2) corpus nhỏ,
đồng nhất và embedding model đã khớp tốt với ngôn ngữ dữ liệu — vector đơn
đủ, thêm BM25 chỉ tốn latency; (3) ngân sách latency rất chặt và loại
query đã biết trước là exact-match.

---

## Điều ngạc nhiên nhất khi làm lab này

BM25 thắng cả `paraphrase` — lý thuyết nói vector phải thắng ở đó, nhưng
với embedding tiếng Anh trên câu hỏi tiếng Việt thì không đúng: chọn model
sai ngôn ngữ làm hỏng đúng chỗ vector lẽ ra mạnh nhất.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
