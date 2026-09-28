# Ground Truth Notes: 千里之外 (qianlizhiwai)

## Summary
- **Pages**: 1
- **Rows**: 10 rows
- **Measures**: 33 measures total (Distribution: 3, 3, 3, 3, 3, 3, 4, 4, 4, 3)
- **Key**: D (Original key)
- **Time Signature**: 4/4
- **Chord Notation**: Taiwanese degree numbers (`number`)
- **Key Changes**: None

## Disagreements and Resolutions
1. **Row 1 (Intro)**:
   - **Disagreement**: Opus transcribed 3 measures (`1 1sus | 1 1sus | 1 - - - |`). Gemini reported 5 measures, treating each chord change as a separate measure.
   - **Resolution**: Eye inspection confirms printed barlines define exactly 3 measures in the intro line. Opus is correct.
2. **Row 4 (Verse 1 Line 2)**:
   - **Disagreement**: Opus identified 3 measures (`2m7 5sus7 57 | 1(2) 5/7 | 6m7 /5 |`). Gemini reported 5 measures and had `5sus?` on beat 3 of measure 1.
   - **Resolution**: The bar contains `2m7` (b1), `5sus7` (b3), and `57` (b4) with clear printed superscripts. Row has 3 measures. Opus accepted.
3. **Row 5 (Verse 1 Line 3)**:
   - **Disagreement**: Opus identified 3 measures ending with altered chord `7b(6.9)` in measure 3. Gemini split chords across 4 measures.
   - **Resolution**: Measure 3 contains the rare altered chord `7b(6.9)` leading into the next phrase. 3 measures verified by barlines.
4. **Row 6 (Pre-chorus Line 1)**:
   - **Disagreement**: Measure count Opus=3 vs Gemini=5.
   - **Resolution**: 3 measures verified by visual barlines.
5. **Row 7 (Chorus Line 1)**:
   - **Disagreement**: Opus=4 measures, Gemini=5 measures. Measure 4 has chords `2m7`, `/5`, `5/7`.
   - **Resolution**: 4 measures verified. All 3 chords in m4 are distinct beats within one bar.
6. **Row 8 (Chorus Line 2)**:
   - **Disagreement**: Opus=4 measures (`1 5/7 | 6m7 5m7 1/3 | 4 1/3 | 2m7 4/5 |`), Gemini=8 measures.
   - **Resolution**: Gemini split every single beat/chord box into a measure. Visual inspection confirms 4 measures with dense chord progressions.
7. **Row 10 (Outro)**:
   - **Disagreement**: Opus=3 measures (`4/6 1/5 | 4 1/3 | 1`), Gemini=5 measures.
   - **Resolution**: 3 measures verified.

## Chord Notation Nuances
- Dense multi-chord measures with bass slash notations (`/5`, `1/3`, `4/5`, `4/6`, `1/5`).
- Complex altered chords: `7b(6.9)` and suspensions: `5sus7`.
