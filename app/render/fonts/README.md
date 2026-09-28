# Bundled Fonts

This directory contains Open Font License (OFL) fonts used by the renderer:

1. **NotoSansSC.ttf** (~17MB)
   - Source: https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf
   - Upstream: https://github.com/google/fonts/tree/main/ofl/notosanssc
   - License: SIL Open Font License, Version 1.1 (`OFL.txt`)
   - Supports: Latin, Simplified Chinese (CJK), digits, punctuation, musical symbols (♯, ♭).

2. **NotoSans.ttf** (~2MB)
   - Source: https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf
   - Upstream: https://github.com/google/fonts/tree/main/ofl/notosans
   - License: SIL Open Font License, Version 1.1
   - Supports: Latin, digits, punctuation.

If neither font is present at runtime, the renderer falls back gracefully to `ImageFont.load_default()`.
