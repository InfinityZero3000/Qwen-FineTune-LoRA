# Quick Reference: Explanation Task

## 🚀 One-Command Setup

```bash
# Merge explanation data vào training set
python scripts/merge_explanation_data.py
```

## 📊 Quick Stats

```
Before: 26,880 samples (4 tasks)
After:  30,806 samples (5 tasks)
Added:  +3,926 explanation samples (12.7%)
```

## 💡 Example Usage

### Input Format:
```
Error: '[Wrong sentence]' → Correct: '[Correct sentence]'
```

### Output Format:
```
Vietnamese explanation in friendly tutor style
```

## 🎯 5 Tasks Overview

| Task | Input | Output | Role |
|------|-------|--------|------|
| `fluency` | Any sentence | Score (0.0-1.0) | Evaluator |
| `vocabulary` | Word/sentence | CEFR level (A1-C2) | Classifier |
| `grammar` | Wrong sentence | Corrected sentence | Corrector |
| `dialogue` | User message | Assistant response | Conversationalist |
| `explanation` | Error → Correct | Vietnamese teaching | **Tutor** (NEW) |

## 📁 Important Files

```
Data Source:
└── datasets/datasets/vietnamese_explanations.json (4,869 samples)

Merged Output:
├── train_with_explanation.jsonl (30,806 samples)
├── val_with_explanation.jsonl (1,618 samples)
└── merge_explanation_report.json (statistics)

Scripts:
└── scripts/merge_explanation_data.py (merge tool)

Docs:
├── docs/EXPLANATION_TASK.md (full documentation)
├── README.md (project overview)
└── PIPELINE_UPDATE_SUMMARY.md (update summary)
```

## ⚙️ Configuration

```python
# Edit in scripts/merge_explanation_data.py

quality_threshold = 50    # Min quality score (default: 50)
val_split_ratio = 0.05    # Validation % (default: 5%)
```

## 🎓 Teaching Style Examples

**Rule-Based:**
```
"Các động từ khuyết thiếu như 'should' thì sau chúng động từ phải ở dạng 
nguyên mẫu không 'to' nhé."
```

**Context-Based:**
```
"Khi nói về hành động trong quá khứ (yesterday), động từ phải chia ở thì 
quá khứ đơn."
```

**Comparison-Based:**
```
"'Can' dùng cho hiện tại, còn 'could' là dạng quá khứ của 'can'."
```

## 🔧 Common Commands

```bash
# Run merge
python scripts/merge_explanation_data.py

# Check merge report
cat datasets/datasets/merge_explanation_report.json

# View sample explanation
head -n 100 datasets/datasets/train_with_explanation.jsonl | grep "explanation"

# Count tasks
python check_datasets.py
```

## 📈 Task Distribution (After Merge)

```
dialogue:    6,649 samples (21.6%)
explanation: 3,926 samples (12.7%) ← NEW
fluency:     7,255 samples (23.6%)
grammar:     5,881 samples (19.1%)
vocabulary:  7,095 samples (23.0%)
```

## 🎯 When to Use Which Task?

### Use `grammar` when:
- User wants just the corrected sentence
- Quick fixes needed
- No explanation required

### Use `explanation` when:
- User wants to **learn WHY** it's wrong
- Teaching/tutoring mode
- Building understanding of grammar rules

## 🔍 Quality Scores

| Score | Quality | Description |
|-------|---------|-------------|
| 100 | Perfect | Crystal clear, excellent teaching |
| 85 | Excellent | Very good explanation |
| 70 | Good | Correct but could be clearer |
| **50** | **Acceptable** | **Minimum threshold** |
| <50 | Poor | Filtered out |

## ✅ Checklist

- [x] Run merge script
- [x] Check merge report
- [x] Verify sample count (30,806)
- [x] Check task distribution (~12.7% explanation)
- [ ] Update notebook training path
- [ ] Train model with new data
- [ ] Test explanation quality
- [ ] Deploy to production

## 🐛 Quick Troubleshooting

**Too few samples?**
→ Lower `quality_threshold` from 50 to 40

**Validation set too small?**
→ Increase `val_split_ratio` from 0.05 to 0.10

**Need to re-merge?**
→ Just run `python scripts/merge_explanation_data.py` again

## 📞 Need Help?

- Full docs: [EXPLANATION_TASK.md](docs/EXPLANATION_TASK.md)
- Update summary: [PIPELINE_UPDATE_SUMMARY.md](PIPELINE_UPDATE_SUMMARY.md)
- Project overview: [README.md](README.md)

---

**Version:** 2.0  
**Last Updated:** 2026-01-27
