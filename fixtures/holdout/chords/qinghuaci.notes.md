# Ground Truth Notes: 青花瓷 (qinghuaci)

## Summary
- **Pages**: 1
- **Rows**: 10
- **Measures**: 31
- **Chords**: 52
- **Key**: A-Bb
- **Time Signature**: 4/4
- **Chord Notation**: Taiwanese boxed chords with slash bass and altered degrees (`number`)

## Model Transcription & Disagreement Resolutions
Dual model transcriptions were performed using:
1. **Gemini 2.5 Pro** (`role='reader'`, `model='gemini-2.5-pro'`)
2. **Claude Opus 5.5** (`role='arbiter'`)

All disagreements between models were reviewed against 3x resolution crops of each row:
- Row 1 M1-M3: M1 is pickup (0.56 5323 ||). Gemini shifted chords left into M1; Opus correctly placed M2 [4, 1/3], M3 [2m7, 1, 3m/7], M4 [6m7, 2].
- Row 2 M1: Gemini missed [4/5] before [5]. Verified against 3x crop: row 2 begins with boxed [4/5] over sixteenth-note figure.
- Row 4 M3: Gemini transcribed cue text notes '43' as chord; Opus correctly extracted [1], [5m7], [1/3].
- Row 5 M2-M3: Beat index disagreement (Gemini beats 4.0/6.0 vs Opus beat 3.0). Harmonic rhythm is 2 beats per chord (beats 1.0, 3.0).
- Row 6 M1: Gemini missed [17] above note 235. Opus captured both [1] and [17].
- Row 7 M2: Verified against 4x high-res crop that the note '5' with `<>` harmonic marking is an instrumental fill note for `Vio+(筝)`, not a chord. The sole chord in M2 is `[1(2)]` (annotated EG above). Row 7 has 5 chords: `[2m7]`, `[5sus7]`, `[1(2)]`, `[4]`, `[4m6]`.
- Row 8 M3: Minor beat alignment disagreement on 57 (beat 2.0 vs 3.0).
- Row 10 M2: Minor beat alignment disagreement on 1 (beat 2.0 vs 1.0).

## Unreadable Cells
- Total '?' cells: **0** (all chord symbols clearly resolved).
