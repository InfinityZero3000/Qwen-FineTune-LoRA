# Explanation Task Documentation

## 📝 Overview

**Task Name:** `explanation`  
**Task Type:** Grammar Error Explanation (Vietnamese)  
**Role:** Model đóng vai trò như **giáo viên dạy ngữ pháp**, giải thích lỗi bằng tiếng Việt một cách thân thiện, dễ hiểu

## 🎯 Purpose

Thay vì chỉ sửa lỗi ngữ pháp (task `grammar`), task này giúp model:
1. **Giải thích TẠI SAO** câu sai
2. **Hướng dẫn** cách sửa đúng
3. **Dạy** quy tắc ngữ pháp bằng ngôn ngữ dễ hiểu
4. Xây dựng **kết nối thầy-trò** với người học

## 📊 Dataset Statistics

- **Source:** `vietnamese_explanations.json`
- **Total samples:** 4,869 entries
- **Quality filtered (≥50):** 4,132 samples
- **Training samples:** 3,926 (95%)
- **Validation samples:** 206 (5%)

### Error Type Distribution
Common error types include:
- `modal_verb` - Lỗi động từ khuyết thiếu
- `verb_form_wrong` - Lỗi chia động từ
- `tense_error` - Lỗi thì
- `preposition_wrong` - Lỗi giới từ
- `article_missing` - Thiếu mạo từ
- And more...

## 📐 Data Format

### Input Format
```json
{
  "task": "explanation",
  "messages": [
    {
      "role": "user",
      "content": "Error: 'He go to school yesterday.' → Correct: 'He went to school yesterday.'"
    },
    {
      "role": "assistant",
      "content": "Khi nói về hành động trong quá khứ (yesterday), động từ phải chia ở thì quá khứ đơn. 'Go' là hiện tại, phải đổi thành 'went' nhé em."
    }
  ],
  "metadata": {
    "source": "vietnamese_explanations",
    "index": 0,
    "error_type": "verb_form_wrong",
    "quality_score": 85
  }
}
```

### Quality Score System
- **100:** Perfect explanation - Clear, accurate, pedagogical
- **85:** Excellent - Very good teaching style
- **70:** Good - Correct explanation, could be clearer
- **50:** Acceptable - Basic explanation, minimal quality threshold
- **<50:** Filtered out (not included in training)

## 🎓 Teaching Style

The explanations follow a **friendly tutor approach:**

### Key Characteristics:
1. **Personal pronouns:** "em", "con", "chúng ta" (informal, friendly)
2. **Clear structure:**
   - Identify the error
   - Explain why it's wrong
   - Show the correct form
   - Provide the rule/pattern
3. **Natural Vietnamese:** Conversational, not overly formal
4. **Encouraging tone:** Build confidence while correcting

### Example Explanation Styles:

**Style 1: Rule-Based**
```
"Các động từ khuyết thiếu như 'should' thì sau chúng động từ phải ở dạng 
nguyên mẫu không 'to'. Vậy nên, 'should study' là đúng, không phải 'should 
to study' nhé."
```

**Style 2: Context-Based**
```
"Khi nói về hành động trong quá khứ (yesterday), động từ phải chia ở thì 
quá khứ đơn. 'Go' là hiện tại, phải đổi thành 'went' nhé em."
```

**Style 3: Comparison-Based**
```
"'Can' dùng cho hiện tại, còn 'could' là dạng quá khứ của 'can'. Vì câu 
nói về quá khứ (when I was young), nên phải dùng 'could ride' nhé!"
```

## 🔄 Integration with Pipeline

### Task Flow:
```
User Input → Model → Explanation Task
    ↓
"Error: X → Correct: Y"
    ↓
Model analyzes error type, grammatical context
    ↓
Generates Vietnamese explanation in tutor style
    ↓
Output: Friendly, pedagogical explanation
```

### Relationship with Other Tasks:

| Task | Input | Output | Role |
|------|-------|--------|------|
| `grammar` | Incorrect sentence | Corrected sentence | **Corrector** |
| `explanation` | Error → Correct pair | Vietnamese teaching | **Tutor** |
| `fluency` | Any sentence | Fluency score | **Evaluator** |
| `vocabulary` | Word/sentence | CEFR level | **Classifier** |
| `dialogue` | Conversation | Response | **Conversationalist** |

## 📈 Training Integration

### Step 1: Merge Data
```bash
python scripts/merge_explanation_data.py
```

### Step 2: Load in Notebook
```python
# Use merged data
train_file = "datasets/datasets/train_with_explanation.jsonl"
val_file = "datasets/datasets/val_with_explanation.jsonl"
```

### Step 3: Task Distribution (After Merge)
```
Total: 30,806 training samples
- dialogue:    6,649 (21.6%)
- explanation: 3,926 (12.7%)  ← NEW TASK
- fluency:     7,255 (23.6%)
- grammar:     5,881 (19.1%)
- vocabulary:  7,095 (23.0%)
```

## 🎯 Expected Model Behavior

### When user asks for explanation:
```
User: "Error: 'She have a car.' → Correct: 'She has a car.'"

Model: "Với chủ ngữ số ít 'She' (ngôi thứ 3 số ít), động từ 'have' phải 
chia thành 'has' trong thì hiện tại đơn nhé em. Quy tắc: He/She/It + has, 
còn I/You/We/They + have."
```

### Comparison with Grammar Task:
```
Grammar Task:
User: "She have a car."
Model: "She has a car."  ← Just correction

Explanation Task:
User: "Error: 'She have a car.' → Correct: 'She has a car.'"
Model: "Với chủ ngữ số ít 'She'..." ← Full explanation
```

## 🔧 Configuration Parameters

```python
# merge_explanation_data.py parameters:
val_split_ratio = 0.05      # 5% for validation
quality_threshold = 50      # Minimum quality score
```

## 📝 Merge Report

After running merge script, check:
```json
{
  "merge_date": "2026-01-27",
  "statistics": {
    "before_merge": {"train": 26880, "val": 1412},
    "explanation_data": {
      "total": 4869,
      "filtered": 4132,
      "train": 3926,
      "val": 206
    },
    "after_merge": {"train": 30806, "val": 1618}
  }
}
```

## 🎓 Pedagogical Benefits

1. **Deeper Understanding:** Not just "what's wrong" but "why it's wrong"
2. **Pattern Recognition:** Helps learners identify similar errors
3. **Confidence Building:** Friendly tone reduces learning anxiety
4. **Cultural Relevance:** Vietnamese explanations for Vietnamese learners
5. **Active Learning:** Engages student thinking through clear reasoning

## 🚀 Production Usage

### API Request Example:
```python
POST /api/v1/explanation
{
  "error_sentence": "He go to school yesterday.",
  "correct_sentence": "He went to school yesterday."
}

Response:
{
  "task": "explanation",
  "explanation": "Khi nói về hành động trong quá khứ (yesterday), 
  động từ phải chia ở thì quá khứ đơn. 'Go' là hiện tại, phải đổi 
  thành 'went' nhé em.",
  "error_type": "verb_form_wrong",
  "confidence": 0.92
}
```

## 📚 References

- Original dataset: `datasets/datasets/vietnamese_explanations.json`
- Merged dataset: `datasets/datasets/train_with_explanation.jsonl`
- Merge script: `scripts/merge_explanation_data.py`
- Merge report: `datasets/datasets/merge_explanation_report.json`

## ⚠️ Important Notes

1. **Quality Threshold:** Set to 50 to balance quantity vs quality
2. **Train/Val Split:** 95/5 ratio to maximize training data
3. **Data Shuffling:** Both explanation and merged data are shuffled
4. **Backup Recommendation:** Keep original train.jsonl before using merged version
5. **Task Balance:** Explanation task now represents ~12.7% of training data

## 🔮 Future Enhancements

1. Add more error types (punctuation, word order, etc.)
2. Increase quality scores through human review
3. Add difficulty levels (A1-C2)
4. Include example sentences for each explanation
5. Multi-language support (English explanations for international learners)

---

**Last Updated:** 2026-01-27  
**Author:** LexiLingo Team  
**Task Version:** 1.0
