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
