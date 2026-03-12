# WI+LOCNESS Dataset — Tài liệu Thuyết trình

> **Sử dụng trong dự án:** Dữ liệu huấn luyện cho task **Grammatical Error Correction (GEC)** của mô hình Qwen trong hệ thống LexiLingo.

---

## 1. Tổng quan

**WI+LOCNESS** là bộ dữ liệu chuẩn quốc tế dành cho bài toán sửa lỗi ngữ pháp tiếng Anh (Grammatical Error Correction), được phát hành trong khuôn khổ **BEA-2019 Shared Task** (Building Educational Applications).

| Thông tin | Chi tiết |
|-----------|---------|
| **Tổ chức phát hành** | Cambridge English / University of Cambridge |
| **Năm phát hành** | 2019 (v2.1 – 02/07/2020) |
| **Mục đích** | Huấn luyện & đánh giá mô hình GEC |
| **Giấy phép** | Phi thương mại – nghiên cứu và giáo dục |
| **Định dạng** | JSON + M2 |

---

## 2. Nguồn dữ liệu

### 2.1 Write & Improve (WI)

- **Nguồn gốc:** Nền tảng học viết tiếng Anh trực tuyến [writeandimprove.com](https://writeandimprove.com/), hoạt động từ năm 2014.
- **Nội dung:** Các bài luận (thư, truyện, bài viết) do học sinh không phải người bản ngữ (ESL/EFL) nộp để nhận phản hồi tự động.
- **Gán nhãn:** Các annotator của Cambridge English đã thủ công chú thích lỗi và gán cấp độ CEFR.
- **Cấp độ:** A (Beginner), B (Intermediate), C (Advanced).

### 2.2 LOCNESS

- **Nguồn gốc:** Corpus LOCNESS từ Trung tâm Ngôn ngữ học Corpus, Đại học Louvain (Bỉ).
- **Nội dung:** Bài luận của sinh viên người bản ngữ tiếng Anh (native).
- **Mục đích:** Cho phép đánh giá mô hình trên toàn dải trình độ, kể cả người bản ngữ vẫn có thể mắc lỗi.

---

## 3. Thống kê Dataset

### 3.1 Tổng quan theo Split

| Split | Texts | Sentences | Tokens |
|-------|------:|----------:|-------:|
| **Train (A+B+C)** | 3,000 | 34,308 | 628,720 |
| **Dev (A+B+C+N)** | 350 | 4,384 | 86,973 |
| **Test (A+B+C+N)** | 350 | 4,477 | 85,668 |
| **Tổng** | **3,700** | **43,169** | **801,361** |

### 3.2 Chi tiết theo Cấp độ CEFR

| Cấp độ | Mô tả | Train Texts | Train Sentences | Dev Sentences |
|--------|-------|------------:|----------------:|--------------:|
| **A** | Beginner | 1,300 | 10,493 | 1,037 |
| **B** | Intermediate | 1,000 | 13,032 | 1,290 |
| **C** | Advanced | 700 | 10,783 | 1,069 |
| **N** | Native (LOCNESS) | 0 | — | 988 |

### 3.3 Số lượng Annotation trong dự án

| File | Sentences | Annotations (lỗi) |
|------|----------:|------------------:|
| A.train | 10,493 | 29,096 |
| A.dev | 1,037 | 3,030 |
| B.train | 13,032 | 24,416 |
| B.dev | 1,290 | 2,552 |
| C.train | 10,783 | 10,171 |
| C.dev | 1,069 | 1,100 |
| N.dev | 988 | 950 |
| **Tổng** | **38,662** | **71,315** |

---

## 4. Định dạng Dữ liệu

### 4.1 M2 Format (Dùng trong dự án)

M2 là định dạng chuẩn cho GEC, được sử dụng từ CoNLL Shared Task 2013. Mỗi câu gồm:

```
S <câu gốc đã tokenize>
A <start> <end>|||<loại lỗi>|||<sửa thành>|||REQUIRED|||-NONE-|||<annotator_id>
```

**Ví dụ thực tế:**

```
S It 's difficult answer at the question " what are you going to do " ?
A 3 3|||M:VERB:FORM|||to|||REQUIRED|||-NONE-|||0
A 4 5|||U:PREP||||||REQUIRED|||-NONE-|||0
```

Giải thích:
- `S` — câu gốc (lỗi), đã tách token
- `A 3 3` — vị trí token 3 đến 3, **M**issing từ `to` (thiếu VERB:FORM)
- `A 4 5` — vị trí token 4–5, **U**nnecessary `at` (giới từ thừa)

**Các ký hiệu loại lỗi:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| `M:XXX` | **Missing** — thiếu từ/cụm |
| `U:XXX` | **Unnecessary** — thừa từ/cụm |
| `R:XXX` | **Replacement** — sai, cần thay thế |
| `noop` | Câu đúng, không cần sửa |
| `UNK` | Lỗi không xác định loại |

### 4.2 JSON Format (Dữ liệu thô)

```json
{
  "id": "...",
  "userid": "...",
  "text": "Câu gốc chưa tokenize...",
  "cefr": "A2.ii",
  "edits": [[annotator_id, [[char_start, char_end, correction], ...]]]
}
```

---

## 5. Phân tích Lỗi

### 5.1 Top 15 Loại Lỗi Phổ Biến Nhất

| Hạng | Loại Lỗi | Số lượng | Mô tả |
|------|----------|--------:|-------|
| 1 | `M:PUNCT` | 9,038 | Thiếu dấu câu |
| 2 | `R:OTHER` | 7,839 | Thay thế khác (không phân loại rõ) |
| 3 | `R:PREP` | 4,529 | Sai giới từ |
| 4 | `M:DET` | 3,431 | Thiếu mạo từ (a/an/the) |
| 5 | `R:ORTH` | 3,391 | Lỗi chính tả (orthography) |
| 6 | `R:VERB:TENSE` | 3,325 | Sai thì động từ |
| 7 | `R:VERB` | 3,311 | Sai động từ |
| 8 | `U:DET` | 2,974 | Thừa mạo từ |
| 9 | `R:NOUN:NUM` | 2,831 | Sai số (singular/plural) |
| 10 | `R:SPELL` | 2,771 | Sai chính tả (spelling) |
| 11 | `R:PUNCT` | 2,481 | Sai dấu câu |
| 12 | `R:NOUN` | 2,437 | Sai danh từ |
| 13 | `R:VERB:FORM` | 2,122 | Sai dạng động từ |
| 14 | `UNK` | 1,820 | Không xác định |
| 15 | `R:VERB:SVA` | 1,568 | Sai chia động từ (Subject-Verb Agreement) |

### 5.2 Nhận xét

- **Dấu câu và mạo từ** là hai nhóm lỗi phổ biến nhất — ánh xạ đúng đặc điểm của người học ESL ở mọi trình độ.
- **Sai thì & dạng động từ** (`R:VERB:TENSE`, `R:VERB:FORM`, `R:VERB:SVA`) chiếm tỷ lệ lớn ở học sinh trình độ A, B.
- **Giới từ** (`R:PREP`) là loại lỗi khó nhất do phụ thuộc nhiều vào ngữ cảnh.
- Cấp độ **C và N** có ít annotation hơn — người học nâng cao mắc ít lỗi cú pháp hơn.

---

## 6. Files trong Dự án

```
datasets/wi+locness/
├── m2/
│   ├── A.train.gold.bea19.m2   # 10,493 câu — cấp độ Beginner train
│   ├── A.dev.gold.bea19.m2     #  1,037 câu — cấp độ Beginner dev
│   ├── B.train.gold.bea19.m2   # 13,032 câu — Intermediate train
│   ├── B.dev.gold.bea19.m2     #  1,290 câu — Intermediate dev
│   ├── C.train.gold.bea19.m2   # 10,783 câu — Advanced train
│   ├── C.dev.gold.bea19.m2     #  1,069 câu — Advanced dev
│   ├── N.dev.gold.bea19.m2     #    988 câu — Native (LOCNESS) dev
│   ├── ABC.train.gold.bea19.m2 # Gộp A+B+C train (34,308 câu)
│   └── ABCN.dev.gold.bea19.m2  # Gộp A+B+C+N dev (4,384 câu)
├── json/
│   ├── A.dev.json, B.dev.json, ...   # Dữ liệu thô JSON
│   └── (các file JSON tương ứng)
├── test/                        # Tokenized test data
├── json_to_m2.py               # Script chuyển đổi JSON → M2 (cần ERRANT)
├── readme.txt                   # Tài liệu gốc của dataset
├── licence.wi.txt               # Giấy phép Write & Improve
└── license.locness.txt          # Giấy phép LOCNESS
```

---

## 7. Pipeline Sử dụng trong LexiLingo

```
M2 Files (raw)
      │
      ▼
json_to_m2.py (ERRANT toolkit)
      │  Chuyển đổi character-level edits → token-level annotations
      ▼
grammar_data.json
      │
      ▼
download_and_inspect_datasets.py
      │  Chuẩn hóa → Instruction format
      ▼
unified_training_data.json / .csv
      │
      ▼
merge_explanation_data.py
      │  Thêm giải thích tiếng Việt
      ▼
train_with_explanation.jsonl  ←── Fine-tune Qwen3
val_with_explanation.jsonl    ←── Validation
```

**Instruction format sau khi xử lý:**
```json
{
  "instruction": "Correct the grammatical errors in the following sentence.",
  "input": "It 's difficult answer at the question.",
  "output": "It 's difficult to answer the question.",
  "explanation": "Thiếu 'to' trước động từ 'answer' (VERB:FORM). Giới từ 'at' không cần thiết sau 'answer' (U:PREP)."
}
```

---

## 8. Anonymization (Ẩn danh hóa)

Dataset đã được Cambridge xử lý ẩn danh để loại bỏ thông tin cá nhân:

| Loại thông tin | Cách xử lý |
|---------------|-----------|
| Tên người | Thay bằng tên cùng quốc tịch |
| Ngày sinh | Ngẫu nhiên hóa |
| Địa chỉ | Đổi thành địa chỉ hư cấu |
| Email/Username | Ngẫu nhiên hóa phần local-part |
| Số điện thoại | Giữ mã vùng, đổi các số còn lại |

---

## 9. Trích dẫn

Nếu sử dụng dataset này trong nghiên cứu, cần trích dẫn:

```bibtex
@inproceedings{bryant-etal-2019-bea,
    title = "The {BEA}-2019 Shared Task on Grammatical Error Correction",
    author = "Bryant, Christopher and Felice, Mariano and Andersen, {\O}istein E. and Briscoe, Ted",
    booktitle = "Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    pages = "52--75"
}

@article{yannakoudakis-2018-wi,
    title = "Developing an automated writing placement system for {ESL} learners",
    author = "Yannakoudakis, Helen and Andersen, {\O}istein E. and Geranpayeh, Ardeshir and Briscoe, Ted and Nicholls, Diane",
    journal = "Applied Measurement in Education",
    volume = "31",
    number = "3",
    pages = "251--267",
    year = "2018"
}
```

---

## 10. Điểm Mạnh & Hạn chế

### Điểm mạnh
- **Đa trình độ:** Bao phủ từ Beginner đến Native — mô hình có thể học sửa lỗi theo từng cấp.
- **Quy mô lớn:** ~43,000 câu với 71,000+ annotation chất lượng cao (thủ công).
- **Chuẩn ngành:** Được dùng rộng rãi trong cộng đồng NLP/GEC quốc tế.
- **Song ngữ thân thiện:** Dữ liệu từ học sinh nhiều quốc tịch → phù hợp cho ứng dụng đa văn hóa.

### Hạn chế
- **Chỉ tiếng Anh:** Không có dữ liệu tiếng Việt → cần tự xây dựng explanation tiếng Việt (`vietnamese_explanations.jsonl`).
- **Không có Native train set:** Cấp độ N chỉ có dev, không có tập train.
- **Giấy phép hạn chế:** Không dùng cho mục đích thương mại.
- **Phụ thuộc annotation:** Chất lượng phụ thuộc vào annotator — có thể có inconsistency giữa các annotator.
