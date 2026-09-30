# Ground Truth Notes: 听妈妈的话 (tingmamadehua)

## Summary
- **Pages**: 1
- **Rows**: 10
- **Measures**: 40
- **Chords**: 77
- **Key**: Bb
- **Time Signature**: 4/4
- **Chord Notation**: Taiwanese boxed degree numbers with slash bass (`number`)

## Model Transcription & Disagreement Resolutions
Dual model transcriptions were performed using:
1. **Gemini 2.5 Pro** (`role='reader'`, `model='gemini-2.5-pro'`)
2. **Claude Opus 5.5** (`role='arbiter'`)

All disagreements between models were reviewed against 3x resolution crops of each row:
- Row 1 M1-M4: Gemini transcribed melody intro notes (15, 35, 25, 45) as chords. Opus correctly identified only the boxed `[1]` in M1, with M2-M4 empty.
- Row 3 M4: Gemini omitted the third chord `[5]` at the bar end (`1`, `2m/1`, `5`). Opus captured all 3 chords accurately.
- Row 7 M2: Gemini read `67/5`; Opus read `[/5]`. Verified against 3x crop: printed symbol is boxed `[/5]` above note i.
- Row 7 M4: Gemini omitted `2m/1` and `5`; Opus captured `[1]`, `[2m/1]`, `[5]`. Verified against 3x crop.
- Row 8 M4: Minor beat alignment disagreement (Gemini beat 2.0 vs Opus beat 3.0 for `57`). Beat 3.0 matches harmonic half-note rhythm.

## Unreadable Cells
- Total '?' cells: **0** (all chord symbols clearly resolved).
