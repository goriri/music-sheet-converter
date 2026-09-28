# Ground Truth Notes: 听海 (tinghai)

## Summary
- **Pages**: 2 (Page 1: portrait; Page 2: landscape 弹唱版)
- **Rows**: 20 rows total
  - Page 1: 12 rows (measure distribution: [4, 4, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4] = 50 measures)
  - Page 2: 8 rows (measure distribution: [5, 4, 4, 4, 4, 4, 4, 4] = 33 measures)
- **Measures**: 83 measures total
- **Key**: Bb
- **Time Signature**: 4/4
- **Chord Notation**: Taiwanese degree numbers (`number`)
- **Key Changes**: None

## Pickup Measures (弱起)
- Page 1, Row 3 has 5 measures: Measure 1 is printed `弱起 | 012 |` with `chords: []`. Measures 2–5 contain the 4 main phrases.
- Page 1, Row 5 has 5 measures: Measure 1 is a 1-beat pickup `chords: []`.
- Page 2, Row 1 has 5 measures: Measure 1 is a pickup bar into 弹唱版.

## Disagreements and Resolutions
1. **Row Barlines vs Beat Groups**:
   - Gemini repeatedly mistook beat groupings and chord shifts for measure boundaries (e.g. reporting 11 measures for Page 1 Row 3, and 9 measures for Row 4).
   - Claude Opus precisely identified the printed vertical barlines (`|` and `||`), matching human visual inspection.
2. **Complex Altered Chords**:
   - Page 1, Row 2, Measure 4: printed `⑤ 7/11^-9` over beat 1 and `⑤` over beat 4. Opus transcribed `57/11-9`, accurately preserving the suspended eleventh flat ninth notation.
   - Page 1, Row 1, Measure 1: `1add9` on beat 1, `1/2m7-5` (slash half-diminished) on beat 3.
   - Circled degree symbols (`①`, `②`, `⑤`, etc.) are standardized into canonical integer degrees (`1`, `2`, `5`) in accordance with the OMR normalization spec.
