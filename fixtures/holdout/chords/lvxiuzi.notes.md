# Ground Truth Notes: 绿袖子 (lvxiuzi)

## Summary
- **Pages**: 1
- **Rows**: 11
- **Measures**: 46
- **Chords**: 44
- **Key**: G
- **Time Signature**: 3/4
- **Chord Notation**: Circled chords below melody with modal alterations (`circled_below`)

## Model Transcription & Disagreement Resolutions
Dual model transcriptions were performed using:
1. **Gemini 2.5 Pro** (`role='reader'`, `model='gemini-2.5-pro'`)
2. **Claude Opus 5.5** (`role='arbiter'`)

All disagreements between models were reviewed against 3x resolution crops of each row:
- Row 2 M2: Minor beat placement disagreement (beat 1.0 vs 2.0 for (3)add9).
- Row 3 M1-M2: M1 is pickup '6 |' without chord. Gemini put [(6)m] in M1; Opus correctly put [] in M1 and [(6)m] in M2.
- Row 5 M1-M2: Same pickup pattern; Opus placed chord in M2 after pickup bar.

## Unreadable Cells
- Total '?' cells: **0** (all chord symbols clearly resolved).
