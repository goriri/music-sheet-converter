# Ground Truth Notes: 天使 (tianshi)

## Summary
- **Pages**: 1
- **Rows**: 10
- **Measures**: 41
- **Chords**: 72
- **Key**: D
- **Time Signature**: 4/4
- **Chord Notation**: Taiwanese boxed chords (`number`)

## Model Transcription & Disagreement Resolutions
Dual model transcriptions were performed using:
1. **Gemini 2.5 Pro** (`role='reader'`, `model='gemini-2.5-pro'`)
2. **Claude Opus 5.5** (`role='arbiter'`)

All disagreements between models were reviewed against 3x resolution crops of each row:
- Row 1 M2: Beat placement disagreement for [5] (beat 2.0 vs beat 3.0 in 4/4 measure).
- Row 4 M3-M4: Opus shifted [5] from M3 into M4 (yielding 1 chord in M3 and 3 in M4). Verified against 3x crop: M3 has [4, 5] and M4 has [5sus, 5] (2 chords in each measure).

## Unreadable Cells
- Total '?' cells: **0** (all chord symbols clearly resolved).
