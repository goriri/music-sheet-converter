# Ground Truth Notes: 流沙 (liusha)

## Summary
- **Pages**: 2 (Page 1: 伴奏版 with melody & chords in key A; Page 2: 弹唱版 in key Ab)
- **Rows**: 20 rows total
  - Page 1: 13 rows (measure distribution: [5, 4, 4, 4, 4, 2, 4, 4, 4, 4, 2, 4, 1] = 46 measures)
  - Page 2: 7 rows (measure distribution: [4, 4, 4, 2, 4, 4, 1] = 23 measures)
- **Measures**: 69 measures total
- **Key**: A (`1=A` on Page 1; `1=bA` on Page 2 lead sheet)
- **Time Signature**: 4/4
- **Chord Notation**: Circled degree numbers (`number`)
- **Key Changes**: None within performance flow

## Notation Details & Disagreements
1. **Circled Degree Notation**:
   - The sheet prints circled degrees (`①`, `⑦/⑤`, `⑥m`, `⑤/①`, `④maj7`, `③/①add9`, `②m7`, `⑤7/11`).
   - All are standardized to numerical degree representations in accordance with the OMR specification.
2. **Page 1 Row 5 & 10 (Bridge / Variation)**:
   - Contains a pickup measure (`01 |`) into `4maj7` and `37`.
   - The phrase `1. 76 65 523 3 22 3 03 |` contains slurred sixteenths (`523 3`), which Gemini repeatedly split into 2-3 spurious bars. Visual inspection confirms 1 full 4/4 measure.
3. **Page 2 Slash Chords Across Lines**:
   - In Chorus on Page 2, chords like `⑦/⑤`, `⑤/①`, `③/①` have root degree on top and bass degree below the slash.
   - When cropped tightly, single-model OCR cut between the numerator and denominator; cross-model visual review resolved each slash chord accurately.
