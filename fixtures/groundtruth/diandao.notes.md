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

## Visual Audit for Small Flats (Accidentals)
- Audited all rows against original sheet crops at 3x zoom for small preceding flat signs (`♭`):
  - `p1 r2`: m2 has `♭⑦add9` (`b7add9`).
  - `p1 r3`: m2 has `♭⑦add9` (`b7add9`); m4 has `♭⑥/④m` (`b6/4m`) and `♭⑦` (`b7`).
  - `p1 r4`: m4 has `♭⑦add9` (`b7add9`).
  - `p1 r5`: m3 has `♭⑦add9` (`b7add9`).
  - `p1 r6`: m4 has `♭⑥` (`b6`).
  - `p1 r8`: m2 has `♭⑦add9` (`b7add9`).
  - `p1 r9`: m2 has `♭⑦add9` (`b7add9`).
  - `p2 r1` & `p2 r2`: m2 has `♭⑦` (`b7`).
  - `p2 r3`: m4 has `♭⑦` (`b7`).
  - `p2 r4`: m2 is natural `⑥` (major 6th secondary dominant, no flat engraved), m3 has `♭⑦` (`b7`).
  - `p2 r5`: m4 has `♭⑥` (`b6`).
- All flat symbols are accurately captured and verified 100% consistent with the engraved sheet.

