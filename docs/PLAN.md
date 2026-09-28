# 台湾谱伴奏生成器 — Design & Module Contracts

Root: `/Users/jush/projects/music-sheet-converter`  · Python 3.12 venv at `.venv` (`.venv/bin/python`, deps in `requirements.txt`, already installed).
GCP project `cellular-cider-495602-r9` (gcloud already authenticated, ADC available). Deploy target: Cloud Run, region `asia-east1`.

## Product
User uploads a multi-page Taiwanese band chart (jianpu melody + boxed number chords like `1(2)`, `5/7`, `2m7/5`,
texture labels like `Only PN+AG(RHY)+Bs in`, `Bs: 11 11 11 112`, `D.r fill`, key changes like `轉成2調(Ab)`).
User picks a start key (or clicks printed 原调/男调/女调), a difficulty (初级/中级/高级) and instrument (piano now;
guitar/ukulele later). Output: a PDF where each original page is kept, every chord box gets the concrete chord name
(e.g. `G/B`) next to it, and **below each printed row a new accompaniment strip** is inserted, aligned bar-by-bar with
the row above, containing a right-hand line and left-hand line in movable-do jianpu (relative to the key in force),
with finger numbers.
Sample input: `fixtures/pages/page1.jpg`, `fixtures/pages/page2.jpg` (song 掉了, Slow Soul 4/4, ♩=81, printed keys
original F#→Ab, male Bb→C, female F→G; modulation up a whole step mid page 2).

## Data contract
`app/models.py` (conductor-owned; do not rename/remove fields; if you need an extra field, report it as a concern).

## Pipeline & module ownership
| Module | Public API | Notes |
|---|---|---|
| `app/omr/gemini_omr.py` | `parse_pages(images: list[bytes]) -> ParsedSheet` | Gemini on Vertex AI (`google-genai`, `vertexai=True`, project above, location `global`), structured JSON output, one call per page + stitch global measure indices. Model name from env `OMR_MODEL` (default: best available Gemini pro model, verified by a live call). |
| `app/omr/barlines.py` | `refine_measure_boxes(image: bytes, sheet: ParsedSheet, page: int) -> None` | OpenCV: detect bar lines inside each System bbox and snap Measure x-ranges; must not crash if detection fails (keep model boxes). |
| `app/theory/chords.py` | `parse_chord(raw) -> ChordSpec`, `resolve_chord(raw, tonic_pc, beat=1.0) -> ResolvedChord`, `key_name_to_pc(name) -> int`, `spell(pc, tonic_pc) -> str` | Taiwanese number notation → pitch classes & letter names; correct enharmonic spelling per key (Bb not A# in F). |
| `app/arrange/piano.py` | `arrange(sheet: ParsedSheet, start_key: str, difficulty: Difficulty) -> Arrangement` | Style patterns × difficulty, section-texture aware, voice-leading DP, fingering. |
| `app/arrange/styles.py` | pattern tables used by piano.py | |
| `app/render/jianpu.py` | drawing primitives for jianpu events | |
| `app/render/overlay.py` | `render_pages(pages: list[bytes], sheet, arrangement) -> list[PIL.Image.Image]`, `render_pdf(...) -> bytes` | |
| `app/main.py`, `app/storage.py`, `app/static/*` , `Dockerfile`, `deploy.sh` | web app | |

## Web API (FastAPI)
- `POST /api/sheets` multipart `files[]` (jpg/png/pdf; PDFs split into page images with PyMuPDF at 200 dpi) → `{sheet_id}`; OMR runs in a background thread; state in GCS `sheets/{id}/state.json` (`status: queued|parsing|ready|error`, `progress`, `error`).
- `GET /api/sheets/{id}` → state + `ParsedSheet` when ready.
- `PUT /api/sheets/{id}/parsed` → user-corrected `ParsedSheet` (UI lets the user fix chord text, beat, and key changes).
- `POST /api/sheets/{id}/render` `{start_key, difficulty, instrument}` → `{pdf_url, preview_urls[], arrangement}` (sync; arrangement+render is seconds).
- `GET /api/pages/{id}/{n}` page image; static UI at `/`.
Storage: bucket from env `BUCKET` (default `${PROJECT}-sheet-converter`); signed URLs are hard on Cloud Run default creds → serve files through the app (`GET /api/files/{path}`) instead.
Cloud Run: `--no-cpu-throttling` (background thread), `--timeout 900`, 2 CPU / 2Gi, service account with `roles/aiplatform.user` + `roles/storage.objectAdmin`.

## Arrangement rules (piano)
- Key handling: start key tonic; each `KeyChange` shifts tonic by `semitones` from its measure on. Number chords always relative to the tonic in force.
- Difficulty
  - beginner: RH close-position triads (7ths reduced to triad unless the 7th is the bass or the chord is sus/7b5 where it defines colour: keep ≤3 notes), LH single root (or slash bass) as whole/half notes. Rhythm: chord changes only, max 2 attacks per bar.
  - intermediate: RH 3–4 note voicings, 7ths kept; LH root–5–8 eighth-note broken pattern for ballad/slow soul; RH chord on 1 and 3 (+ anticipations when chord changes on "and").
  - advanced: LH wider arpeggio (1-5-1'-2'/3'), RH voicings with extensions (add9 for `(2)`), syncopated comping, fill patterns on `fill` bars, passing bass on slash-chord bass lines.
- Texture from labels: `Only PN(RHY)` → sparser (drop one density level for RH), sections gaining `+Bs in` / `OG in` / tempo drums → full density; `is_stop` → hit on beat 1 and hold/rest as the melody; `fill` → fill figure on last 1–2 beats; follow `bass_hint` rhythm for LH when present (e.g. `11 11 11 112` = eighths with a 16th pickup).
- Voice leading: choose RH voicings by dynamic programming over candidates per chord, cost = semitone movement between consecutive voicings (+ reward common tones) + register drift from centre (RH top ≈ C5–E5, never above A5, never below E4 lowest note; LH bass C2–C3 range) + hand span (>octave forbidden for beginner, >9th forbidden for all) + black-key-thumb penalty. LH follows slash-bass lines stepwise where possible.
- Fingering: standard piano fingering (RH root pos 1-3-5, 1st inv 1-2-5, 2nd inv 1-3-5, 4-note 1-2-3-5 or 1-2-4-5 by interval; LH 5-2-1 for 1-5-8, 5-3-1 for 1-5-10, 5-4-2-1 etc.), avoid thumb on black keys in arpeggios, keep finger continuity across repeated patterns.
- Melody collision: RH top note should avoid the melody's pitch-class on strong beats where feasible (it's an accompaniment, the singer carries the tune) — low priority.

## Render rules
- Keep original page pixels. Split each page at System boundaries and insert a strip under each System (height ≈ 0.75× system height, min 140 px at 1000-px-wide page scale); non-system regions (title, margins) are copied unchanged.
- Chord names: small red text immediately above-right of each chord `bbox` (does not cover original ink).
- Strip content per Measure, x-aligned to the Measure bbox x-range: RH row (upper), LH row (lower), both movable-do jianpu relative to `tonic_pc` of that measure: digits 1–7 with ♯/♭ prefix, octave dots above/below (reference octave: RH C4–B4 = no dots... per tonic, LH one dot below baseline register), chord stacks drawn vertically (lowest note at bottom), rhythm by jianpu convention (underline = eighth, double = sixteenth, `-` = extra beat, dot = dotted), rests `0`. Finger numbers in small blue digits above RH notes / below LH notes. Thin bar lines at measure x-ranges; row labels "R"/"L" in margin.
- Header strip on page 1: "Key: F (1=F) · 中级 · Piano" and modulation notice at the measure where the key changes ("1=G").
- Font: bundled `app/render/fonts/` (download Noto Sans / Noto Sans SC or DejaVu Sans at build time; fall back to PIL default).

## Quality assurance (production guardrails)
Goal: never return a wrong note; minimise what the user must confirm. Findings are `QualityIssue`s
(`app/models.py`) on `ParsedSheet.issues` / `Arrangement.issues`. Only `needs_review` reaches the user as a
yellow "请核对" item (generation is still allowed; the PDF ends with a list of them). Everything else is
auto-resolved and logged.

### OMR verification — `app/qa/omr_verify.py` → `verify_sheet(pages: list[bytes], sheet) -> ParsedSheet`
Ensemble + music priors, escalate only unresolved cases:
1. Structural: measure count vs OpenCV barlines, beat sum of melody vs time signature, chord grammar
   (`app.theory.chords.parse_chord`), chord bbox inside its measure, header keys vs detected key change
   (e.g. printed `F#-Ab` ⇒ +2).
2. Crop re-read: every chord box cropped (with margin) and read independently by Gemini Flash; agreement ⇒ conf↑.
3. Music priors scored per candidate reading: melody/chord fit (strong-beat melody notes ∈ chord tones),
   Bs-line first note == chord bass, typical progressions; used to pick between disagreeing readings and to
   recover dropped accidentals.
4. Arbiter: remaining disagreements/low scores → Claude Opus on Vertex AI (`app/qa/llm.py`) with the crop,
   row image, candidate readings, melody and neighbouring chords; structured answer + confidence.
5. Result: auto-correct when the combined evidence is decisive (issue `auto_fixed`), else keep best reading,
   `alternatives` filled, issue `needs_review`.

### Arrangement verification — `app/qa/arrange_check.py` → `check_and_repair(sheet, arrangement) -> Arrangement`
Runs on every render request before the PDF is produced:
1. Independent validator (chord pitch classes recomputed from `ResolvedChord.name` with **music21**, not our
   own theory code): every chord voiced at its beat; RH contains defining tones (3rd, 7th, sus4, add2);
   LH notes ⊂ chord tones except weak-subdivision passing tones; fills diatonic to the key in force; tiling;
   ranges; spans; fingering physically valid (no reused finger on different simultaneous/consecutive pitches
   without crossing/shift).
2. Any failing measure is replaced by a verified safe fallback (block chord + bass, simple fingering) and
   re-validated; issue `auto_fixed`. A failing fallback is a hard error (500), never a silent wrong note.
3. Optional LLM musical review (env `QA_LLM_REVIEW=1`): Opus reads a compact text rendering of the song
   (chords, RH/LH notes+fingers per measure) and flags implausible measures; flagged measures that also fail
   a stricter heuristic get the fallback.
4. CI: property-based tests (hypothesis) over all chord qualities × 12 keys × 3 levels × random sheets; golden
   regression outputs in `tests/golden/` updated only via `scripts/update_golden.py` after human approval.

### Render verification
Layout self-check in the renderer: every event x inside its measure, no label/text bbox overlaps; failures
logged as `render` issues.

## v2 OMR architecture (after testing on 14 external charts)
Finding: Gemini page-level bboxes are fine on sparse charts but wrong on dense ones (strips cut through rows,
3/5-bar rows forced into 4, chords shifted by one bar). v2 separates *where* from *what*:
1. `app/omr/layout.py` — classical CV → `PageGeometry` (`app/omr/geometry.py`): deskew, watermark suppression,
   line classification (melody/chord/label/lyrics/Bs), barlines (any count per row), chord boxes (rounded-rect
   contours; unboxed text tokens for letter-chord charts), beat_geo from x position, confidence.
2. `app/omr/reader.py` — LLM reads *content* of upscaled crops with numbered measures/boxes; answers mapped by
   number only. Model: `OMR_READER_MODEL` (Gemini 2.5 Pro vs Claude Opus 5.5, chosen by A/B on the answer key).
3. `parse_pages` = layout → reader; falls back to the v1 whole-page method only for low-confidence pages.
4. Letter-chord charts (`SongHeader.chord_notation='letter'`) are transposed from the printed key.
5. Gate: `ParsedSheet.layout_confidence < 0.6` or >25% of measures needing review ⇒ no PDF until the user
   confirms structural issues (HTTP 409).
6. Evaluation: hand-checked answer key `fixtures/groundtruth/*.json` (7 songs) + `scripts/eval_omr.py`.
   Third-party charts live in `fixtures/external/` (git-ignored, local testing only).
