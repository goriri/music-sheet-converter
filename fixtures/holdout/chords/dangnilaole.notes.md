# Ground Truth Notes: 当你老了 (dangnilaole)

## Summary
- **Pages**: 2
- **Rows**: 18
- **Measures**: 76
- **Chords**: 76
- **Key**: C
- **Time Signature**: 6/8
- **Chord Notation**: Page 1: 专业版 with circled chords below melody (`circled_below`). Page 2: 唯美版 simplified circled chords.

## Model Transcription & Disagreement Resolutions
Dual model transcriptions were performed using:
1. **Gemini 2.5 Pro** (`role='reader'`, `model='gemini-2.5-pro'`)
2. **Claude Opus 5.5** (`role='arbiter'`)

All disagreements between models were reviewed against 3x resolution crops of each row:
- Page 1 Row 2 M1-M4: Opus repeated [1add9] and [3m7]. Gemini accurately read [1], [3m7], [4sus2], [4sus2].
- Page 1 Rows 3 & 5: Pickup bar handling. M0 is pickup (no chord); chords start on M1.
- Page 1 Row 8 M3: Beat index in 6/8 meter (beat 3.0 vs 4.0; beat 4.0 represents dotted-quarter second pulse).
- Page 2 Rows 1 & 3: Pickup bar handling. M0 has no chord; chords start on M1.

## Unreadable Cells
- Total '?' cells: **0** (all chord symbols clearly resolved).
