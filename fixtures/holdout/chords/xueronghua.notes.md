# Ground Truth Notes: 雪绒花 (xueronghua)

## Summary
- **Pages**: 2
- **Rows**: 18
- **Measures**: 72
- **Chords**: 80
- **Key**: D
- **Time Signature**: 3/4
- **Chord Notation**: Page 1: circled chords below melody with stacked fractions (`circled_below`). Page 2: chord-only grid (`chord_only`).

## Model Transcription & Disagreement Resolutions
Dual model transcriptions were performed using:
1. **Gemini 2.5 Pro** (`role='reader'`, `model='gemini-2.5-pro'`)
2. **Claude Opus 5.5** (`role='arbiter'`)

All disagreements between models were reviewed against 3x resolution crops of each row:
- Page 1 Row 4 M2-M4: Opus shifted stacked fractions across measures. Gemini correctly assigned M2 [(6)7], M3 [(#1)/(6)7 (stacked: true), (2)m7/9], M4 [(#4)/(2)7/9 (stacked: true), (5)7/11, (5)].
- Page 1 Row 10 M2-M3: Boundary placement of altered chord (2)7/13 and (b3)7/13. Verified: M2 has [(5)], M3 has [(2)7/13, (b3)7/13].
- Page 2 Rows 4–8: Re-audited all 8 rows on page 2 using individual row crops (system height ~107px). Resolved sequence: Row 4 `[1, 5, 4, 1]`, Row 5 `[5, 5, 1, 6m]`, Row 6 `[4, 2, 5, 5]`, Row 7 `[1, 5, 1, 4]`, Row 8 `[1, 5, 4, 1]`. Corrected earlier crop offset that skipped Row 5 and duplicated Row 8.

## Unreadable Cells
- Total '?' cells: **0** (all chord symbols clearly resolved).
