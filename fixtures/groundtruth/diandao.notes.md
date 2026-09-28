# Ground Truth Notes: 倒带 (diandao)

## Summary
- **Pages**: 2
- **Rows**: 16 rows total
  - Page 1: 9 rows × 4 measures = 36 measures
  - Page 2: 7 rows (rows 1–6 have 4 measures; row 7 has 1 measure = 25 measures)
- **Measures**: 61 measures total
- **Key**: C (`1=C`)
- **Time Signature**: 4/4
- **Chord Notation**: Taiwanese degree numbers (`number`)
- **Key Changes**: None (the pipeline baseline had an erroneous modulation copied from diaole, verified no key change on original sheet)

## Disagreements and Resolutions
1. **Row 1 (前奏)**:
   - Chords: `6m | 4maj7 | 1 | 57/11 |`
   - Gemini transcribed circled chord notation (`⑥m`, `④maj7`, `①`, `⑤7/11`). Opus normalized degrees to standard digits (`6m`, `4maj7`, `1`, `57/11`). Both agreed on chord identities and measure structure.
2. **Page 2, Row 5, Measure 4**:
   - Chords: `4m` (beat 1), `b6` (beat 3), `5` (beat 4).
   - Opus accurately detected all 3 passing chords in this cadence bar before the repeat.
3. **Key Change Verification**:
   - The original sheet is in C throughout. The baseline pipeline `verified.json` falsely asserted a modulation at m78 (borrowed from diaole fallback). Ground truth strictly has `key_changes: []`.
