# XPANCEO DB Evaluation Report

**Generated**: 2024-01-21 15:30:00

---

## Summary

- **Total Questions**: 50
- **Successful Judgments**: 48
- **Failed Judgments**: 2

---

## Overall Metrics

| Metric | Score |
|--------|-------|
| Faithfulness | 0.823 |
| Relevancy | 0.791 |
| Context Precision | 0.756 |
| Context Recall | 0.682 |

---

## Metrics by Slice

### Overall (n=20)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.834 |
| Relevancy | 0.801 |
| Context Precision | 0.778 |
| Context Recall | 0.712 |


### Table (n=12)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.845 |
| Relevancy | 0.812 |
| Context Precision | 0.789 |
| Context Recall | 0.734 |


### Image (n=10)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.756 |
| Relevancy | 0.723 |
| Context Precision | 0.698 |
| Context Recall | 0.612 |


### No-Answer (n=8)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.912 |
| Relevancy | 0.889 |
| Context Precision | 0.445 |
| Context Recall | 0.312 |
| No-Answer Accuracy | 0.875 |
| False Positive Rate | 0.125 |


---

## Notable Cases

### Low Faithfulness Examples

**Q**: What specific numerical values are shown in Figure 3?...

**A**: The figure shows values ranging from 0.1 to 0.9 with a peak at 0.75...

**Score**: 0.35

**Notes**: Answer includes values not visible in the retrieved image context.

---

**Q**: What is the exact training time mentioned in Table 2?...

**A**: The training time was approximately 48 hours on 8 GPUs...

**Score**: 0.42

**Notes**: Table chunk shows "training_time" column but specific value not clearly retrieved.

---

### No-Answer False Positives

**Q**: What is the current stock price of Apple?...

**A**: Based on the documents, Apple's stock price is discussed in terms of market cap...

*(System should have refused to answer)*

---

## Interpretation

1. **Text-based questions** perform best with high faithfulness (>0.8)
2. **Table questions** show good precision, suggesting effective Markdown conversion
3. **Image questions** have lower recall - Vision fallback helps but OCR limitations remain
4. **No-answer detection** is strong (87.5% accuracy) but 1 false positive observed

## Recommendations

1. Increase `top_k` for image-heavy documents
2. Fine-tune OCR confidence threshold (currently 60)
3. Add more explicit no-answer training examples
