# Ground Truth Notes: 七里香 (qilixiang)

## Summary
- **Pages**: 1
- **Rows**: 10
- **Measures**: 41
- **Chords**: 90
- **Key**: Cm
- **Time Signature**: 4/4
- **Chord Notation**: Taiwanese boxed chords with altered degrees (`number`)

## Model Transcription & Disagreement Resolutions
Dual model transcriptions were performed using:
1. **Gemini 2.5 Pro** (`role='reader'`, `model='gemini-2.5-pro'`)
2. **Claude Opus 5.5** (`role='arbiter'`)

All disagreements between models were reviewed against 3x resolution crops of each row:
- Row 1 M1-M5: M1 is pickup (2356 ||). Gemini shifted chords left into M1; Opus correctly put [] in M1 and aligned M2-M5.
- Row 3 M2-M3: Boundary placement of [37/5#]. Verified: sits above note 17 at end of M2.
- Row 6 M2-M4: Measure distribution of chords [57, 1, 5/7], [6m, 4], [57, 1, 37/5#]. Opus matched visual ground truth.
- Row 7 M1-M4: Gemini misassigned chords across barlines; Opus aligned M1 [6m, 4], M2 [57, 37/5#, 6m], M3 [2m7, 57], M4 [1, 37/5#].
- Row 8 M2-M4: Same pattern as Row 6; Opus aligned accurately.
- Row 9 M4: Verified against 3x crop: contains exactly one boxed `[1]` above beat 1; trailing numbers are instrumental fill annotations (`EG+Vio`, `i5 35`, `(D.r fill)`), not chords.

## Unreadable Cells
- Total '?' cells: **0** (all chord symbols clearly resolved).
