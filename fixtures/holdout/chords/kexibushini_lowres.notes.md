# Ground Truth Notes: 可惜不是你 (kexibushini_lowres)

## Summary
- **Pages**: 1
- **Rows**: 10
- **Measures**: 42
- **Chords**: 81
- **Key**: F#
- **Time Signature**: 4/4
- **Chord Notation**: Taiwanese boxed chords (`number`), low-res scan (800x1132)

## Model Transcription & Disagreement Resolutions
Dual model transcriptions were performed using:
1. **Gemini 2.5 Pro** (`role='reader'`, `model='gemini-2.5-pro'`)
2. **Claude Opus 5.5** (`role='arbiter'`)

All disagreements between models were reviewed against 3x resolution crops of each row:
- Row 2 M3-M4: Gemini placed both [3m7] and [6m9] into M3. Opus correctly separated [3m7] in M3 and [6m9] in M4 across barline.
- Row 4 M3-M4: Gemini placed chord [1] in M3; Opus placed [1], [1sus], [1] in M4 above melody 1 - - 0 0.2.
- Row 7 M2-M3: Gemini misaligned [6m] into M2; Opus placed [2m7, /5, 1] in M2 and [6m, 3m7] in M3.
- Row 8 M2: Gemini transcribed slash bass as root [5]; Opus transcribed [/5]. Verified: slash bass without root digit.
- Row 9 M3-M4: Gemini placed [1sus] in M3; Opus placed [2m7, 57] in M3 and [1sus, 1] in M4.
- Row 10 M3-M4: Gemini hallucinated bass figure '4' as chord in M3; Opus correctly identified [2m7, 57] in M3 and [1sus, 1] in M4.

## Unreadable Cells
- Total '?' cells: **0** (all chord symbols clearly resolved).
