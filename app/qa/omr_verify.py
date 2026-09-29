"""OMR Verification Layer for Taiwanese Band Charts (台湾谱).

========================================================================================
DECISION RULE SPECIFICATION
========================================================================================
1. Inputs per chord box:
   - raw_full: Full-page global transcription from primary OMR.
   - raw_crop: Isolated chord-box crop re-read from Gemini 2.5 Flash (reader role).
   - priors:
       * melody_fit: fraction of strong-beat & long melody notes that are chord tones.
       * bass_hint_fit: 1.0 if printed 'Bs:' line's first note matches chord bass PC, else 0.0.
       * progression_plausibility: transition score from prev_chord & to next_chord.
   - candidates: OCR confusion variants generated via generate_candidates().

2. Verification Flow & Decision Logic:
   a. Concordance Check:
      If raw_crop is available, matches raw_full (raw_full == raw_crop), and passes
      chord grammar (parse_chord succeeds):
      -> Concordant reading: Confidence set to 0.95. No modification needed.

   b. Discordance / Ambiguity / Grammar Failure:
      Gather all candidate readings: [raw_full, raw_crop] + generated confusion candidates.
      Filter for grammatically valid candidates (parse_chord succeeds).
      Score each candidate:
          Score(cand) = PriorScore(cand) + 0.45 * (cand == raw_crop) + 0.35 * (cand == raw_full)

      - If raw_full is grammatically invalid (parse_chord fails):
        * If valid candidates exist and top candidate has prior score >= 0.60:
          -> Auto-fix to top candidate with severity='auto_fixed', code='chord_corrected'.
        * Else:
          -> Flag with severity='needs_review', code='invalid_chord_grammar'.

      - If raw_full is valid, but raw_crop differs:
        * If raw_crop has strong prior support (score >= 0.75, margin >= 0.25 over raw_full,
          e.g. matching bass hint while raw_full fails):
          -> Auto-Fix to raw_crop with severity='auto_fixed', code='chord_corrected'.
        * Else:
          -> Escalate to Arbiter (Claude Opus 5.5 on Vertex AI).
             Batch all undecided chords on the page in one call: send chord crop,
             candidate options, melody, bass hint, and neighbouring chords.
             * If Arbiter selects with confidence >= 0.70:
               -> Auto-Fix if different from raw_full (severity='auto_fixed'), or confirm raw_full.
             * If Arbiter confidence < 0.60 or still undecided:
               -> Retain top candidate, confidence < 0.60, populate chord.alternatives.
               -> Emit QualityIssue with severity='needs_review', code='chord_ambiguous'.

   c. Structural Checks:
      - Measure count & barline alignment: Mismatches flagged with severity='needs_review'.
      - Melody beat-sum: If melody beat-sum deviates from measure.beats by >= 1.0,
        flag with severity='needs_review'.
      - Key change consistency: If header indicates key modulation but key_changes is empty,
        auto-restore from header with severity='auto_fixed'.
      - Multi-chord beat alignment: If 2nd chord has invalid beat (e.g. beat 1.0 or > beats),
        auto-correct to beat 3.0 in 4/4 with severity='auto_fixed'.

   d. Graceful Degradation:
      If LLM calls fail or QA_OFFLINE=1:
      Verify using grammar + priors only. Emit an informational QualityIssue (severity='info').
      The function NEVER raises exceptions.
========================================================================================
"""

from __future__ import annotations

import concurrent.futures
import io
import logging
import re
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.models import ChordSymbol, KeyChange, Measure, ParsedSheet, QualityIssue, SongHeader, System
from app.qa.llm import LLMUnavailable, ask_json, llm_available
from app.qa.melody import parse_melody
from app.qa.priors import (
    _simplify_chord,
    bass_hint_fit,
    generate_candidates,
    melody_fit,
    progression_plausibility,
    score_candidate,
)
from app.theory.chords import parse_chord
from app.theory.keys import key_name_to_pc

logger = logging.getLogger(__name__)

BASS_ONLY_PATTERN = re.compile(r"^/[#b♯♭]?[1-7]$")
NON_CHORD_VOICING_PATTERN = re.compile(r"^(?:0|\d{3,})$")


def is_valid_chord_grammar(raw: str) -> bool:
    """Check whether raw chord text is grammatically valid.

    Supports standard Taiwanese chord notation via parse_chord,
    as well as bass-only slash changes (e.g. '/5', '/#4', '/b7').
    """
    if not raw or not raw.strip():
        return False
    r = raw.strip()
    if BASS_ONLY_PATTERN.match(r):
        return True
    try:
        parse_chord(r)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pydantic schemas for reader & arbiter batch calls
# ---------------------------------------------------------------------------

class CropItemReading(BaseModel):
    crop_index: int = Field(description="1-based crop index")
    chord: str = Field(description="Chord in Taiwanese number notation, e.g. '1(2)', '5m/7b'")


class PageCropReadings(BaseModel):
    readings: list[CropItemReading] = Field(default_factory=list)


class ArbiterItemDecision(BaseModel):
    chord_id: int = Field(description="Identifier matching requested item")
    chosen_chord: str = Field(description="Selected canonical chord symbol")
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    reason: str = Field("", description="Brief explanation in Chinese or English")


class ArbiterBatchResponse(BaseModel):
    decisions: list[ArbiterItemDecision] = Field(default_factory=list)


class LocatedKeyAnnotation(BaseModel):
    system_index: int = Field(0, description="0-based system row index on this page (top row is 0)")
    measure_index: int = Field(0, description="0-based measure index within this system row (0 is 1st bar in row)")
    approx_x: Optional[float] = Field(None, description="Normalized horizontal position x in [0, 1] where annotation appears")
    verbatim_text: str = Field(description="Exact printed text of the key change annotation")
    semitones: Optional[int] = Field(None, description="Semitone shift if stated or deduced (e.g. 2 for 2 semitones)")


class PageKeyLocateResponse(BaseModel):
    annotations: list[LocatedKeyAnnotation] = Field(default_factory=list)


class DetectedMeasureChordBox(BaseModel):
    chord: str = Field(description="Chord symbol inside the box in Taiwanese number notation, e.g. '5', '2m7'")
    beat: float = Field(3.0, description="Starting beat, e.g. 1.0 or 3.0")
    confidence: float = Field(0.8, ge=0.0, le=1.0)


class MeasureCropChordResponse(BaseModel):
    chords: list[DetectedMeasureChordBox] = Field(default_factory=list)


class MeasureMelodyCropReading(BaseModel):
    melody: str = Field(
        description="Jianpu melody for this measure grouped into space-separated beats (e.g. '2323 35. 2321 12.')."
    )


class MissingChordArbiterConfirmation(BaseModel):
    confirmed: bool = Field(description="True if the chord box is clearly printed on the sheet")
    confirmed_chord: Optional[str] = Field(None, description="The confirmed chord text in Taiwanese number notation")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = Field("", description="Visual evidence and harmonic reason")


class RowTranscribedChord(BaseModel):
    measure_in_row: int = Field(default=1, description="1-based measure index within this row (1 for leftmost bar, 2 for next, etc.)")
    chord: str = Field(description="Exact printed text inside the chord box, e.g. '1(2)', '5/7', '5m6/2', '67/1#', '6m7-5', '5', '5sus'")
    approx_x: Optional[float] = Field(default=0.5, description="Approximate horizontal position across this row image from 0.0 (left edge) to 1.0 (right edge)")
    beat: float = Field(default=1.0, description="Approximate starting beat in the measure (1.0 for first half, 3.0 for second half)")


class SystemRowChords(BaseModel):
    row_index: int = Field(description="0-based index of the row crop image (0 for Image #0, 1 for Image #1, etc.)")
    chords: list[RowTranscribedChord] = Field(default_factory=list)
    measure_count: Optional[int] = Field(default=None, description="Total number of measures separated by barlines in this row image")


class PageRowTranscriptionResponse(BaseModel):
    rows: list[SystemRowChords] = Field(default_factory=list)


class CandidateInsertionDecision(BaseModel):
    item_id: int = Field(description="1-based ID of the candidate insertion item")
    confirmed: bool = Field(description="True if the printed chord box ink is clearly present in the measure crop")
    confirmed_chord: Optional[str] = Field(None, description="Exact confirmed chord text if confirmed")
    confidence: float = Field(default=0.9, description="Confidence in the decision between 0.0 and 1.0 (e.g. 0.95)")
    reason: str = Field(default="", description="Concise reason")


class CandidateInsertionBatchResponse(BaseModel):
    decisions: list[CandidateInsertionDecision] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level helper functions (flat module scope)
# ---------------------------------------------------------------------------

def align_measure_chords(
    sheet_chords: list[ChordSymbol],
    transcribed_chords: list[RowTranscribedChord],
) -> tuple[list[ChordSymbol], list[RowTranscribedChord]]:
    """Align parsed sheet chords with row-transcribed chords for a single measure.

    Returns:
        (unmatched_sheet_chords, candidate_insertions)
    """
    matched_sheet: set[int] = set()
    matched_row: set[int] = set()

    for r_idx, r in enumerate(transcribed_chords):
        r_txt = r.chord.strip()
        r_simp = _simplify_chord(r_txt)
        for s_idx, s in enumerate(sheet_chords):
            if s_idx in matched_sheet:
                continue
            s_txt = s.raw.strip()
            if r_txt == s_txt or r_simp == _simplify_chord(s_txt):
                matched_sheet.add(s_idx)
                matched_row.add(r_idx)
                break

    unmatched_sheet = [sheet_chords[i] for i in range(len(sheet_chords)) if i not in matched_sheet]
    candidate_insertions = [transcribed_chords[i] for i in range(len(transcribed_chords)) if i not in matched_row]
    return unmatched_sheet, candidate_insertions


def map_transcribed_chords_to_measures(
    system: System,
    transcribed: list[RowTranscribedChord],
) -> dict[int, list[RowTranscribedChord]]:
    """Map each row-transcribed chord to a measure in the system by measure_in_row, using approx_x when index overflows or deep in measure."""
    meas_map: dict[int, list[RowTranscribedChord]] = {m.index: [] for m in system.measures}
    if not system.measures:
        return meas_map

    n = len(system.measures)
    w_sys = max(0.001, system.bbox[2] - system.bbox[0])
    splits = [0.0]
    for k in range(1, n):
        mid = ((system.measures[k - 1].bbox[2] + system.measures[k].bbox[0]) / 2.0 - system.bbox[0]) / w_sys
        splits.append(mid)
    splits.append(1.0)

    for tc in transcribed:
        assigned_k = None
        k_by_x = None
        dist_to_boundary = 1.0
        if tc.approx_x is not None and 0.0 <= tc.approx_x <= 1.0:
            for k in range(n):
                if splits[k] <= tc.approx_x < splits[k + 1]:
                    k_by_x = k
                    dist_to_boundary = min(abs(tc.approx_x - splits[k]), abs(tc.approx_x - splits[k + 1]))
                    break

        if 1 <= tc.measure_in_row <= n:
            k_by_meas = tc.measure_in_row - 1
            if k_by_x is not None and k_by_x != k_by_meas and dist_to_boundary > 0.04:
                assigned_k = k_by_x
            else:
                assigned_k = k_by_meas
        elif k_by_x is not None:
            assigned_k = k_by_x

        if assigned_k is None:
            m_idx = tc.measure_in_row - 1
            assigned_k = max(0, min(n - 1, m_idx))
        meas_map[system.measures[assigned_k].index].append(tc)

    return meas_map


def resolve_chord_beat_geo(
    chord: ChordSymbol,
    measure: Measure,
    competing_beats: list[float],
) -> tuple[Optional[float], bool]:
    """Resolve chord starting beat geometrically from chord bbox x0 relative to measure x-range.

    Returns:
        (resolved_beat, is_ambiguous)
    """
    if not chord.bbox or not measure.bbox:
        return None, True

    m_x0, m_x1 = measure.bbox[0], measure.bbox[2]
    w_m = max(0.001, m_x1 - m_x0)
    c_x0 = chord.bbox[0]

    frac = max(0.0, min(1.0, (c_x0 - m_x0) / w_m))
    raw_beat_geo = 1.0 + frac * measure.beats
    beat_geo = round(raw_beat_geo * 2.0) / 2.0
    beat_geo = max(1.0, min(float(measure.beats), beat_geo))

    if not competing_beats:
        return beat_geo, False

    matching_reading = None
    min_dist = 999.0
    for cand_b in competing_beats:
        dist = min(abs(raw_beat_geo - cand_b), abs(beat_geo - cand_b))
        if dist < min_dist:
            min_dist = dist
            if dist <= 0.3:
                matching_reading = cand_b

    if matching_reading is not None:
        return matching_reading, False

    if min_dist <= 0.3:
        return beat_geo, False

    return beat_geo, True


def _crop_chord_box(img_bgr: np.ndarray, chord: ChordSymbol) -> Optional[bytes]:
    """Extract a tight image crop of a chord box based on its normalized bounding box."""
    if not chord.bbox:
        return None
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = chord.bbox
    pad_x = 0.005
    pad_y = 0.005
    y_start = int(max(0.0, y0 - pad_y) * h)
    y_end = int(min(1.0, y1 + pad_y) * h)
    x_start = int(max(0.0, x0 - pad_x) * w)
    x_end = int(min(1.0, x1 + pad_x) * w)

    if y_end <= y_start or x_end <= x_start:
        return None

    c_img = img_bgr[y_start:y_end, x_start:x_end]
    if c_img.size == 0 or c_img.shape[0] < 5 or c_img.shape[1] < 5:
        return None

    _, buf = cv2.imencode(".jpg", c_img)
    return buf.tobytes()


def _crop_measure(img_bgr: np.ndarray, system: System, measure: Measure) -> Optional[bytes]:
    """Extract an image crop of a measure column spanning the whole system row height."""
    h, w = img_bgr.shape[:2]
    x0, x1 = measure.bbox[0], measure.bbox[2]
    y0, y1 = system.bbox[1], system.bbox[3]
    pad_x = 0.005
    pad_y = 0.005
    y_start = int(max(0.0, y0 - pad_y) * h)
    y_end = int(min(1.0, y1 + pad_y) * h)
    x_start = int(max(0.0, x0 - pad_x) * w)
    x_end = int(min(1.0, x1 + pad_x) * w)

    if y_end <= y_start or x_end <= x_start:
        return None

    c_img = img_bgr[y_start:y_end, x_start:x_end]
    if c_img.size == 0 or c_img.shape[0] < 5 or c_img.shape[1] < 5:
        return None

    _, buf = cv2.imencode(".jpg", c_img)
    return buf.tobytes()


def parse_header_key_pair(
    header_raw: str, header: Optional[SongHeader] = None
) -> tuple[int, Optional[str], Optional[str]]:
    """Parse key pairs from header into (semitone_shift, k1, k2).

    Handles formats like:
      - '(X - Y)', '(X-Y)', 'X→Y', 'X->Y', 'X - Y'
      - '男調(X-Y)', '女調(X-Y)', '原調(X-Y)'
      - '[X - Y]', '降B - C', '升F - 降A'

    Returns (diff, k1, k2) where diff is the semitone shift mod 12 (-5..+6).
    Returns (0, None, None) if no valid key modulation pair is detected.
    """
    candidates_text: list[str] = []
    if header_raw:
        candidates_text.append(header_raw)
    if header:
        for val in (header.original_key, header.male_key, header.female_key, header.raw):
            if val and val not in candidates_text:
                candidates_text.append(val)

    key_token = r"(?:[#b♯♭]?[A-Ga-g][#b♯♭]?|(?:降|升)[A-Ga-g])"
    pair_re = re.compile(
        rf"(?:(?:\(|（|\[)?\s*({key_token})\s*(?:[-–—~→]|->)\s*({key_token})\s*(?:\)|）|\])?)"
    )

    for text in candidates_text:
        matches = pair_re.findall(text)
        for k1, k2 in matches:
            try:
                pc1 = key_name_to_pc(k1)
                pc2 = key_name_to_pc(k2)
                diff = (pc2 - pc1) % 12
                if diff > 6:
                    diff -= 12
                if diff != 0:
                    return diff, k1, k2
            except Exception:
                continue

    return 0, None, None


def parse_header_key_changes(header_raw: str, header: Optional[SongHeader] = None) -> int:
    """Parse key pairs from header into expected total modulation."""
    diff, _, _ = parse_header_key_pair(header_raw, header)
    return diff


def _parse_semitones_from_annotation(
    text: str,
    expected_semitones: int,
    start_key: Optional[str] = None,
) -> Optional[int]:
    """Extract or verify semitone amount from annotation text."""
    m_shift = re.search(r"轉[成]?\s*([1-9])\s*調", text)
    if m_shift:
        return int(m_shift.group(1))

    if "半調" in text or "半音" in text:
        return 1

    m_key = re.search(r"(?:1=|Key:\s*|\()([A-Ga-g][b#]?)", text)
    if m_key and start_key:
        try:
            target_pc = key_name_to_pc(m_key.group(1))
            start_pc = key_name_to_pc(start_key)
            diff = (target_pc - start_pc) % 12
            if diff > 6:
                diff -= 12
            return diff
        except Exception:
            pass

    return None


def _transcribe_page_row(
    p_idx: int,
    page_img: Optional[np.ndarray],
    p_systems: list[System],
) -> tuple[int, list[System], dict[int, list[RowTranscribedChord]], dict[int, int]]:
    """Transcribe all row chord bands for a single page in one Gemini Flash call."""
    if page_img is None or not p_systems:
        return p_idx, [], {}, {}
    h_p, w_p = page_img.shape[:2]
    row_crops: list[bytes] = []
    valid_p_systems: list[System] = []
    for s in p_systems:
        x0 = int(max(0.0, s.bbox[0] - 0.005) * w_p)
        x1 = int(min(1.0, s.bbox[2] + 0.005) * w_p)
        m_y0 = min(m.bbox[1] for m in s.measures) if s.measures else s.bbox[3]
        y0 = int(max(0.0, s.bbox[1] - 0.005) * h_p)
        y1 = int(min(1.0, m_y0 + 0.015) * h_p)
        crop = page_img[y0:y1, x0:x1]
        if crop.size > 0 and crop.shape[0] >= 5 and crop.shape[1] >= 5:
            _, buf = cv2.imencode(".jpg", crop)
            row_crops.append(buf.tobytes())
            valid_p_systems.append(s)

    if not row_crops:
        return p_idx, [], {}, {}

    row_prompt = (
        f"You are transcribing chord boxes from a Taiwanese band chart sheet music (page {p_idx + 1}). "
        "Each numbered image (Image #0, Image #1, ...) is a horizontal system row crop covering the chord band above the melody for that row. "
        "Each row image contains measures separated by vertical barlines. "
        "For each row, report: "
        "- measure_count: total number of measures in this row (separated by vertical barlines) "
        "- chords: list of chord boxes in this row from left to right: "
        "  1. measure_in_row: 1-based measure index within this row (count barlines: 1 for leftmost measure, 2 for second, etc.) "
        "  2. beat: 1.0 for the first chord in the measure, 3.0 for the second chord in the measure "
        "  3. chord: exact printed chord text (e.g. '1(2)', '5/7', '5m6/2', '67/1#', '6m7-5', '5', '5sus', '2m7/5') "
        "  4. approx_x: horizontal position from 0.0 (left edge) to 1.0 (right edge). "
        "IMPORTANT: If a measure has TWO chord boxes, both have the SAME measure_in_row! Ignore volta brackets [1. 2.] and section names."
    )
    try:
        row_resp = ask_json(
            prompt=row_prompt,
            schema=PageRowTranscriptionResponse,
            images=row_crops,
            role="reader",
            timeout_s=45.0,
        )
        # Normalize row_index to 0-based if LLM used 1-based indexing
        min_idx = min((r.row_index for r in row_resp.rows), default=0)
        max_idx = max((r.row_index for r in row_resp.rows), default=0)
        is_1_based = min_idx == 1 and max_idx == len(row_resp.rows)

        row_dict: dict[int, list[RowTranscribedChord]] = {}
        meas_counts: dict[int, int] = {}
        for idx, r in enumerate(row_resp.rows):
            actual_row = (r.row_index - 1) if is_1_based else r.row_index
            if not (0 <= actual_row < len(valid_p_systems)):
                actual_row = idx
            row_dict[actual_row] = r.chords
            if r.measure_count is not None and r.measure_count > 0:
                meas_counts[actual_row] = r.measure_count
            else:
                max_tc = max((tc.measure_in_row for tc in r.chords), default=0)
                if max_tc > 0:
                    meas_counts[actual_row] = max_tc
                elif actual_row < len(valid_p_systems):
                    meas_counts[actual_row] = len(valid_p_systems[actual_row].measures)
        return p_idx, valid_p_systems, row_dict, meas_counts
    except Exception as exc:
        logger.warning("Row chord transcription failed on page %d: %s", p_idx + 1, exc)
        return p_idx, valid_p_systems, {}, {}



def _read_page_crops(page_num: int, items: list[dict[str, Any]]) -> dict[tuple[int, int], str]:
    """Read all chord box crops for a single page in one Gemini Flash call."""
    if not items:
        return {}
    page_crops = [it["crop_bytes"] for it in items]
    prompt = (
        f"You are given {len(page_crops)} chord box images from a Taiwanese band chart (page {page_num + 1}), "
        f"numbered 1 to {len(page_crops)} in order. "
        "Transcribe the chord symbol inside each box verbatim using Taiwanese number notation "
        "(e.g. '1(2)', '5/7', '2m7', '5m/7b', '67/1#', '6m7-5', '57sus', '4M7', '17/7b', "
        "'3m/7', '6b', '5sus', '5', '57', '1/3', '4', '2m7/5', '6m7/5', '7b'). "
        "Do NOT convert to Western letter chords (never C, Dm, etc.). Exclude 'PN' or 'EG'."
    )
    try:
        batch_resp = ask_json(
            prompt=prompt,
            schema=PageCropReadings,
            images=page_crops,
            role="reader",
            timeout_s=60.0,
        )
        res: dict[tuple[int, int], str] = {}
        idx_to_reading = {
            r.crop_index: r.chord.strip() for r in batch_resp.readings if r.crop_index
        }
        for i, it in enumerate(items, 1):
            chord_str = idx_to_reading.get(i)
            if chord_str:
                res[it["key"]] = chord_str
            elif not idx_to_reading and i <= len(batch_resp.readings):
                # Fallback to positional index only if model omitted crop_index entirely
                clean_c = batch_resp.readings[i - 1].chord.strip()
                if clean_c:
                    res[it["key"]] = clean_c
        return res
    except Exception as exc:
        logger.warning("Crop re-read failed on page %d: %s", page_num + 1, exc)
        return {}


def _finalize_undecided(item: dict[str, Any], issues: list[QualityIssue]) -> None:
    """Fallback finalizer for an undecided chord when Arbiter is unavailable."""
    c: ChordSymbol = item["chord"]
    m: Measure = item["measure"]
    orig_raw: str = item["raw_full"]
    cand_scores = item["cand_scores"]
    prior_full = item.get("prior_full", 0.5)

    top_cand, top_score = cand_scores[0]
    c.alternatives = [x[0] for x in cand_scores[1:4]]

    is_full_valid = is_valid_chord_grammar(orig_raw)

    if top_cand != orig_raw and (not is_full_valid or (prior_full < 0.40 and top_score >= 0.75)):
        c.raw = top_cand
        c.confidence = 0.85
        issues.append(
            QualityIssue(
                stage="omr",
                measure_index=m.index,
                severity="auto_fixed",
                code="chord_corrected",
                message=f"第{m.index + 1}小节和弦由 {orig_raw} 更正为 {top_cand}（先验规则更优）",
                detail={"original": orig_raw, "corrected": top_cand, "score": top_score},
            )
        )
    else:
        raw_crop = item.get("raw_crop", "")
        if raw_crop and raw_crop != orig_raw:
            c.confidence = 0.55
            issues.append(
                QualityIssue(
                    stage="omr",
                    measure_index=m.index,
                    severity="needs_review",
                    code="chord_ambiguous",
                    message=f"第{m.index + 1}小节和弦存疑（当前：{c.raw}，裁剪复读：{raw_crop}，备选：{c.alternatives}），请核对",
                    detail={"raw": c.raw, "crop": raw_crop, "alternatives": c.alternatives},
                )
            )
        elif not is_full_valid:
            c.confidence = 0.20
            issues.append(
                QualityIssue(
                    stage="omr",
                    measure_index=m.index,
                    severity="needs_review",
                    code="invalid_chord_grammar",
                    message=f"第{m.index + 1}小节和弦 '{orig_raw}' 格式无法识别，请核对",
                    detail={"raw": orig_raw},
                )
            )
        else:
            c.confidence = 0.85


def _arbitrate_page(
    p_idx: int,
    p_items: list[dict[str, Any]],
    crop_images_map: dict[tuple[int, int], bytes],
    issues: list[QualityIssue],
) -> None:
    """Escalate a page's undecided chords to Claude Opus in one batch."""
    arbiter_images: list[bytes] = []
    prompt_lines = [
        "You are an expert Taiwanese band chart (台湾简谱 / 流行乐队总谱) chord arbiter. "
        "Review the undecided chord readings below using visual ink, harmonic context, and bass line evidence.\n"
    ]

    for i, it in enumerate(p_items):
        cid = i + 1
        it["arbiter_id"] = cid
        m = it["measure"]
        c_crop = crop_images_map.get(it["key"])
        if c_crop:
            arbiter_images.append(c_crop)

        cands = [x[0] for x in it["cand_scores"][:4]]
        prompt_lines.append(
            f"Item {cid} (Measure {m.index + 1}):\n"
            f"- Full-page transcription: '{it['raw_full']}'\n"
            f"- Crop re-read: '{it['raw_crop']}'\n"
            f"- Candidates: {cands}\n"
            f"- Melody jianpu: '{m.melody}'\n"
            f"- Bass line: '{m.bass_hint}'\n"
            f"- Previous chord: '{it['prev_c']}', Next chord: '{it['next_c']}'\n"
        )

    prompt_lines.append(
        "For each item, determine the correct chord in Taiwanese number notation, "
        "confidence (0.0 to 1.0), and concise reason."
    )
    arbiter_prompt = "\n".join(prompt_lines)

    try:
        arbiter_resp = ask_json(
            prompt=arbiter_prompt,
            schema=ArbiterBatchResponse,
            images=arbiter_images if arbiter_images else None,
            role="arbiter",
            timeout_s=60.0,
        )
        decision_map = {d.chord_id: d for d in arbiter_resp.decisions}
        for it in p_items:
            cid = it["arbiter_id"]
            c = it["chord"]
            dec = decision_map.get(cid)
            m = it["measure"]
            orig_raw = it["raw_full"]

            if dec and dec.chosen_chord:
                chosen = dec.chosen_chord.strip()
                c.confidence = dec.confidence
                c.alternatives = [x[0] for x in it["cand_scores"] if x[0] != chosen][:3]

                if chosen != orig_raw and dec.confidence >= 0.70:
                    c.raw = chosen
                    issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=m.index,
                            severity="auto_fixed",
                            code="chord_corrected",
                            message=f"第{m.index + 1}小节和弦由 {orig_raw} 更正为 {chosen}（仲裁确认：{dec.reason}）",
                            detail={"original": orig_raw, "chosen": chosen, "reason": dec.reason},
                        )
                    )
                elif dec.confidence < 0.60:
                    issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=m.index,
                            severity="needs_review",
                            code="chord_ambiguous",
                            message=f"第{m.index + 1}小节和弦存疑（当前：{c.raw}，备选：{c.alternatives}），请核对",
                            detail={"raw": c.raw, "alternatives": c.alternatives},
                        )
                    )
            else:
                _finalize_undecided(it, issues)
    except Exception as exc:
        logger.warning("Arbiter batch call failed: %s, falling back to priors", exc)
        for it in p_items:
            _finalize_undecided(it, issues)


# ---------------------------------------------------------------------------
# Main Verification Function
# ---------------------------------------------------------------------------

def verify_sheet(
    pages: list[bytes],
    sheet: ParsedSheet,
    *,
    use_llm: bool = True,
) -> ParsedSheet:
    """Verify and refine a ParsedSheet using structural checks, crop re-read, and music priors.

    Args:
        pages: List of original page images (JPEG or PNG bytes).
        sheet: ParsedSheet produced by OMR.
        use_llm: Whether to use vision LLMs (Gemini Flash reader and Claude Opus arbiter).

    Returns:
        New ParsedSheet with corrected chords, refined confidences, alternatives, and issues.
    """
    try:
        verified = sheet.model_copy(deep=True)
    except Exception:
        verified = ParsedSheet.model_validate_json(sheet.model_dump_json())

    new_issues: list[QualityIssue] = list(verified.issues)
    is_v2_geometry = (
        verified.layout_source == "cv"
        and verified.layout_confidence >= 0.6
    )

    can_use_llm = use_llm and llm_available()
    if not can_use_llm:
        new_issues.append(
            QualityIssue(
                stage="omr",
                severity="info",
                code="llm_unavailable",
                message="LLM验证服务不可用或未启用，降级为先验规则验证",
            )
        )

    # Decode page images with OpenCV for cropping
    cv2_pages: list[Optional[np.ndarray]] = []
    for p_bytes in pages:
        arr = np.frombuffer(p_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        cv2_pages.append(img)

    # 1. Structural Checks: Key Change Detection from Header
    header_raw = verified.header.raw or ""
    expected_shift, k1, k2 = parse_header_key_pair(header_raw, verified.header)
    detected_shift = sum(kc.semitones for kc in verified.key_changes)

    if expected_shift != 0 and expected_shift != detected_shift:
        key_change_restored = False
        key_change_annotations_read: list[LocatedKeyAnnotation] = []
        if can_use_llm and pages:
            locate_prompt = (
                "You are analyzing a page of a Taiwanese band chart sheet music. "
                f"Find any mid-page key-change annotations (such as '轉 N 調', '轉{'成'} N 調', '轉調', 'Key:', '1=X'). "
                "Do NOT include initial song header key signatures at the top of page 1. "
                "For each annotation found: "
                "1. system_index: 0-based index of the system row on this page (0 is top row, 1 is 2nd row, etc.) "
                "2. measure_index: 0-based measure index within that row (0 for 1st bar, 1 for 2nd bar, 2 for 3rd bar, etc.) "
                "3. approx_x: horizontal position from 0.0 (left edge) to 1.0 (right edge) where annotation appears "
                "4. verbatim_text: exact printed text "
                "5. semitones: signed integer semitones modulated (e.g. 2 for 2 semitones)."
            )
            for p_idx, p_bytes in enumerate(pages):
                try:
                    resp = ask_json(
                        prompt=locate_prompt,
                        schema=PageKeyLocateResponse,
                        images=[p_bytes],
                        role="reader",
                        timeout_s=45.0,
                    )
                    if resp.annotations:
                        key_change_annotations_read.extend(resp.annotations)
                    page_systems = [s for s in verified.systems if s.page == p_idx]
                    for ann in resp.annotations:
                        s_idx = ann.system_index
                        target_sys = None
                        if 0 <= s_idx < len(page_systems):
                            target_sys = page_systems[s_idx]
                        elif 1 <= s_idx <= len(page_systems):
                            target_sys = page_systems[s_idx - 1]
                        if target_sys is None:
                            continue

                        target_meas = None
                        if ann.approx_x is not None:
                            target_meas = next(
                                (m for m in target_sys.measures if m.bbox[0] - 0.05 <= ann.approx_x <= m.bbox[2] + 0.05),
                                None,
                            )
                        if target_meas is None:
                            m_idx = ann.measure_index
                            if 0 <= m_idx < len(target_sys.measures):
                                target_meas = target_sys.measures[m_idx]
                            elif 1 <= m_idx <= len(target_sys.measures):
                                target_meas = target_sys.measures[m_idx - 1]
                        if target_meas is None:
                            continue

                        ann_semi = ann.semitones
                        parsed_semi = _parse_semitones_from_annotation(
                            ann.verbatim_text,
                            expected_shift,
                            verified.header.original_key,
                        )
                        match_semi = (
                            ann_semi == expected_shift
                            or parsed_semi == expected_shift
                            or ("轉" in ann.verbatim_text and str(abs(expected_shift)) in ann.verbatim_text)
                        )

                        if match_semi:
                            global_m = target_meas.index
                            verified.key_changes.append(
                                KeyChange(
                                    at_measure=global_m,
                                    raw=ann.verbatim_text,
                                    semitones=expected_shift,
                                )
                            )
                            new_issues.append(
                                QualityIssue(
                                    stage="omr",
                                    measure_index=global_m,
                                    severity="auto_fixed",
                                    code="key_change_restored",
                                    message=f"根据曲谱标题及画面定位恢复第{global_m + 1}小节转调标记：{ann.verbatim_text} ({expected_shift:+d}半音)",
                                    detail={"at_measure": global_m, "raw": ann.verbatim_text, "semitones": expected_shift},
                                )
                            )
                            key_change_restored = True
                            break
                    if key_change_restored:
                        break
                except Exception as exc:
                    logger.warning("Key change locate failed on page %d: %s", p_idx + 1, exc)

        if not key_change_restored:
            pair_str = f"{k1}→{k2}" if (k1 and k2) else f"{expected_shift:+d}半音"
            if len(pages) <= 1 and not key_change_annotations_read:
                warn_msg = f"谱头标示 {pair_str} 转调，但所给页面中未找到转调记号；如转调在后续页请一并上传"
                verified.warnings.append(f"[omr_verify] {warn_msg}")
                new_issues.append(
                    QualityIssue(
                        stage="omr",
                        measure_index=None,
                        severity="warning",
                        code="key_change_unlocated",
                        message=warn_msg,
                        detail={
                            "expected_semitones": expected_shift,
                            "detected_semitones": detected_shift,
                            "header_pair": pair_str,
                        },
                    )
                )
            else:
                new_issues.append(
                    QualityIssue(
                        stage="omr",
                        measure_index=None,
                        severity="needs_review",
                        code="key_change_unlocated",
                        message="谱头显示有转调但未能定位，请在核对表中设置转调小节",
                        detail={"expected_semitones": expected_shift, "detected_semitones": detected_shift},
                    )
                )

    # Collect all measures and pre-parse melodies
    all_measures = verified.measures()
    measure_map: dict[int, Measure] = {m.index: m for m in all_measures}
    parsed_melodies: dict[int, Any] = {
        m.index: parse_melody(m.melody, beats=m.beats) for m in all_measures
    }

    # 2. Parallel Vision Phase: Row Transcription & Crop Re-Reading concurrently
    page_transcription_results: list[tuple[int, list[System], dict[int, list[RowTranscribedChord]], dict[int, int]]] = []
    crop_readings_map: dict[tuple[int, int], str] = {}
    crop_images_map: dict[tuple[int, int], bytes] = {}
    page_row_meas_counts: dict[int, dict[int, int]] = {}

    if can_use_llm and cv2_pages:
        # Prepare chord box crops for crop re-reading
        chords_to_read: list[dict[str, Any]] = []
        for s in verified.systems:
            p_idx = s.page
            img_bgr = cv2_pages[p_idx] if p_idx < len(cv2_pages) else None
            if img_bgr is None:
                continue
            for m in s.measures:
                for c_idx, c in enumerate(m.chords):
                    crop_bytes = _crop_chord_box(img_bgr, c)
                    if crop_bytes:
                        key = (m.index, c_idx)
                        crop_images_map[key] = crop_bytes
                        chords_to_read.append({
                            "key": key,
                            "crop_bytes": crop_bytes,
                            "raw_full": c.raw,
                            "page": p_idx,
                            "system": s,
                            "measure": m,
                            "chord": c,
                        })

        pages_with_crops = sorted(set(it["page"] for it in chords_to_read))
        num_workers = max(1, len(cv2_pages) + len(pages_with_crops))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            row_futs = []
            if not is_v2_geometry:
                row_futs = [
                    executor.submit(
                        _transcribe_page_row,
                        p_idx,
                        cv2_pages[p_idx],
                        [s for s in verified.systems if s.page == p_idx],
                    )
                    for p_idx in range(len(cv2_pages))
                ]
            crop_futs = {
                executor.submit(
                    _read_page_crops, p, [it for it in chords_to_read if it["page"] == p]
                ): p
                for p in pages_with_crops
            }

            for f in row_futs:
                try:
                    res = f.result()
                    page_transcription_results.append(res)
                    p_res_idx, _, _, meas_counts = res
                    page_row_meas_counts[p_res_idx] = meas_counts
                except Exception as exc:
                    logger.warning("Row transcription future failed: %s", exc)

            for f in concurrent.futures.as_completed(crop_futs):
                try:
                    crop_readings_map.update(f.result())
                except Exception as exc:
                    logger.error("Crop reading future failed: %s", exc)

    # 3. Structural Checks: Barline Warnings cross-checked with row transcription / geometry
    if is_v2_geometry:
        reader_meas_counts: dict[tuple[int, int], int] = {}
        chord_only_systems: set[tuple[int, int]] = set()
        for w in verified.warnings:
            m_v2 = re.search(r"\[v2_reader_measures_seen\]\s+page=(\d+)\s+system=(\d+)\s+measures_seen=(\d+)", w)
            if m_v2:
                reader_meas_counts[(int(m_v2.group(1)), int(m_v2.group(2)))] = int(m_v2.group(3))
            m_co = re.search(r"\[v2_chord_only\]\s+page=(\d+)\s+system=(\d+)", w)
            if m_co:
                chord_only_systems.add((int(m_co.group(1)), int(m_co.group(2))))

        total_meas = len(all_measures)
        for p_idx in range(len(cv2_pages)):
            page_sys = [s for s in verified.systems if s.page == p_idx]
            for s_idx, target_sys in enumerate(page_sys):
                geometry_count = len(target_sys.measures)
                first_m = target_sys.measures[0].index if target_sys.measures else None
                reader_count = reader_meas_counts.get((p_idx, s_idx))
                beat_sums = [round(parsed_melodies[m.index].beat_sum, 2) for m in target_sys.measures]
                is_chord_only_sys = (p_idx, s_idx) in chord_only_systems

                # (a) Check melody beat-sum anomalies (exclude pickup/first measure, final measure, and chord_only measures)
                has_beat_anomaly = False
                if not is_chord_only_sys:
                    for m in target_sys.measures:
                        if m.index == 0 or m.index == total_meas - 1:
                            continue
                        if not m.melody or not m.melody.strip():
                            continue
                        b_sum = parsed_melodies[m.index].beat_sum
                        exp_b = m.beats
                        if b_sum > 0:
                            # Missed barline -> merged measure ~2x beats; Extra barline -> under-full measure
                            if b_sum >= 1.75 * exp_b or b_sum <= 0.55 * exp_b:
                                has_beat_anomaly = True
                                break

                # (b) Cross-check reader measures_seen with geometry_count
                has_count_mismatch = (reader_count is not None and reader_count != geometry_count)

                if has_count_mismatch or has_beat_anomaly:
                    new_issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=first_m,
                            severity="needs_review",
                            code="barline_count_mismatch",
                            message=f"第{p_idx + 1}页第{s_idx + 1}行小节线检测与阅读器或节拍总数不符（几何={geometry_count}，阅读器={reader_count}，节拍={beat_sums}），请核对小节划分",
                            detail={
                                "page": p_idx,
                                "system": s_idx,
                                "geometry_count": geometry_count,
                                "reader_count": reader_count,
                                "beat_sums": beat_sums,
                            },
                        )
                    )
                else:
                    new_issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=first_m,
                            severity="info",
                            code="barline_count_consistent",
                            message=f"第{p_idx + 1}页第{s_idx + 1}行小节线与阅读器及节拍一致（共{geometry_count}小节，布局正常）",
                            detail={
                                "page": p_idx,
                                "system": s_idx,
                                "geometry_count": geometry_count,
                                "reader_count": reader_count,
                                "beat_sums": beat_sums,
                            },
                        )
                    )
    else:
        for w in verified.warnings:
            m_match = re.search(r"Page\s+(\d+)\s+System\s+(\d+):\s+detected\s+(\d+)\s+barlines\s+for\s+(\d+)\s+measures", w)
            if m_match:
                p_idx = int(m_match.group(1)) - 1
                s_idx = int(m_match.group(2)) - 1
                page_sys = [s for s in verified.systems if s.page == p_idx]
                target_sys = page_sys[s_idx] if 0 <= s_idx < len(page_sys) else None
                first_m = target_sys.measures[0].index if target_sys and target_sys.measures else None
                parsed_count = len(target_sys.measures) if target_sys else 0

                is_monotonic_non_overlapping = True
                if target_sys and len(target_sys.measures) > 1:
                    for k in range(len(target_sys.measures) - 1):
                        m_curr = target_sys.measures[k]
                        m_next = target_sys.measures[k + 1]
                        if not (m_curr.bbox[0] < m_next.bbox[0] and m_curr.bbox[2] <= m_next.bbox[0] + 0.015):
                            is_monotonic_non_overlapping = False
                            break

                transcribed_count = None
                if p_idx in page_row_meas_counts:
                    transcribed_count = page_row_meas_counts[p_idx].get(s_idx)

                agrees = (
                    transcribed_count is not None
                    and transcribed_count == parsed_count
                    and is_monotonic_non_overlapping
                )

                if agrees:
                    new_issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=first_m,
                            severity="info",
                            code="barline_count_consistent",
                            message=f"第{p_idx + 1}页第{s_idx + 1}行小节线检测与行级转录一致（共{parsed_count}小节，布局正常）",
                            detail={"warning": w, "measure_count": parsed_count},
                        )
                    )
                else:
                    new_issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=first_m,
                            severity="needs_review",
                            code="barline_count_mismatch",
                            message=f"第{p_idx + 1}页第{s_idx + 1}行小节线检测数量不符，请核对小节划分",
                            detail={"warning": w, "parsed_count": parsed_count, "transcribed_count": transcribed_count},
                        )
                    )

    # 4. Visual Row-Level Chord Transcription & Sequence Alignment
    row_aligned_restored_measures: set[int] = set()
    m_tc_map_global: dict[int, list[RowTranscribedChord]] = {}
    candidate_insertions: list[dict[str, Any]] = []
    for p_idx, valid_p_systems, row_dict, _ in page_transcription_results:
        page_img = cv2_pages[p_idx]
        if page_img is None or not valid_p_systems:
            continue
        for r_idx, s in enumerate(valid_p_systems):
            tc_list = row_dict.get(r_idx, [])
            m_tc_map = map_transcribed_chords_to_measures(s, tc_list)
            m_tc_map_global.update(m_tc_map)
            for m in s.measures:
                m_tcs = m_tc_map.get(m.index, [])
                unmatched_sheet, cands = align_measure_chords(m.chords, m_tcs)
                for us in unmatched_sheet:
                    us.confidence = min(us.confidence, 0.70)
                for cand in cands:
                    candidate_insertions.append({
                        "system": s,
                        "measure": m,
                        "candidate_chord": cand.chord.strip(),
                        "candidate_beat": cand.beat,
                        "page_img": page_img,
                        "page_idx": p_idx,
                    })

        if candidate_insertions:
            arb_images: list[bytes] = []
            arb_items: list[dict[str, Any]] = []
            for item in candidate_insertions:
                m = item["measure"]
                s = item["system"]
                p_img = item["page_img"]
                c_crop = _crop_measure(p_img, s, m)
                if c_crop:
                    arb_images.append(c_crop)
                    item["img_idx"] = len(arb_images) - 1
                    item["item_id"] = len(arb_items) + 1
                    arb_items.append(item)

            if arb_items:
                prompt_lines = [
                    "You are an expert chord arbiter for Taiwanese band charts. "
                    "Review each measure crop image below to evaluate a candidate missing chord box.\n"
                ]
                for item in arb_items:
                    m = item["measure"]
                    prompt_lines.append(
                        f"Item {item['item_id']} (Image #{item['img_idx']}, Measure {m.index + 1} in song):\n"
                        f"- Reading 1 (parsed sheet): {[c.raw for c in m.chords] if m.chords else 'No chords'}\n"
                        f"- Reading 2 (row transcription): suggests missing chord box '{item['candidate_chord']}' around beat {item['candidate_beat']:.1f}\n"
                    )
                prompt_lines.append(
                    "For each item, inspect the corresponding measure image: is there an actual printed chord box "
                    "matching Reading 2 printed in this measure? "
                    "Confirm only if the chord box ink is clearly visible. Provide confidence score (0.0 to 1.0)."
                )
                arbiter_prompt = "\n".join(prompt_lines)
                try:
                    arbiter_resp = ask_json(
                        prompt=arbiter_prompt,
                        schema=CandidateInsertionBatchResponse,
                        images=arb_images,
                        role="arbiter",
                        timeout_s=60.0,
                    )
                    dec_map = {d.item_id: d for d in arbiter_resp.decisions}
                    for item in arb_items:
                        m = item["measure"]
                        cand_c = item["candidate_chord"]
                        cand_b = item["candidate_beat"]
                        dec = dec_map.get(item["item_id"])
                        is_confirmed = dec and dec.confirmed and dec.confidence >= 0.8
                        confirmed_c = (dec.confirmed_chord.strip() if dec and dec.confirmed_chord else cand_c) if is_confirmed else None

                        row_aligned_restored_measures.add(m.index)
                        if dec and not dec.confirmed:
                            # Candidate was rejected by arbiter upon inspecting the measure crop.
                            # Discard false candidate without flagging needs_review.
                            continue

                        if is_confirmed and confirmed_c:
                            if is_valid_chord_grammar(confirmed_c):
                                matching_existing = next(
                                    (
                                        c
                                        for c in m.chords
                                        if abs(c.beat - cand_b) < 0.5
                                        and (
                                            c.raw in generate_candidates(confirmed_c)
                                            or confirmed_c in generate_candidates(c.raw)
                                            or _simplify_chord(c.raw) == _simplify_chord(confirmed_c)
                                        )
                                    ),
                                    None,
                                )
                                if matching_existing is not None:
                                    orig_raw = matching_existing.raw
                                    if orig_raw != confirmed_c:
                                        matching_existing.raw = confirmed_c
                                        matching_existing.confidence = dec.confidence
                                        new_issues.append(
                                            QualityIssue(
                                                stage="omr",
                                                measure_index=m.index,
                                                severity="auto_fixed",
                                                code="chord_corrected",
                                                message=f"第{m.index + 1}小节和弦由 {orig_raw} 更正为 {confirmed_c}（行级转录与仲裁确认）",
                                                detail={
                                                    "measure": m.index,
                                                    "original": orig_raw,
                                                    "corrected": confirmed_c,
                                                    "confidence": dec.confidence,
                                                },
                                            )
                                        )
                                else:
                                    assigned_beat = cand_b
                                    existing_beats = [c.beat for c in m.chords]
                                    if any(abs(b - assigned_beat) < 0.5 for b in existing_beats):
                                        if not any(abs(b - 3.0) < 0.5 for b in existing_beats):
                                            assigned_beat = 3.0
                                        elif not any(abs(b - 1.0) < 0.5 for b in existing_beats):
                                            assigned_beat = 1.0
                                    if not any(abs(c.beat - assigned_beat) < 0.5 for c in m.chords):
                                        m.chords.append(
                                            ChordSymbol(
                                                raw=confirmed_c,
                                                beat=assigned_beat,
                                                confidence=dec.confidence,
                                            )
                                        )
                                        m.chords.sort(key=lambda c: c.beat)
                                        new_issues.append(
                                            QualityIssue(
                                                stage="omr",
                                                measure_index=m.index,
                                                severity="auto_fixed",
                                                code="chord_restored",
                                                message=f"第{m.index + 1}小节行级转录识别并经仲裁确认恢复和弦 '{confirmed_c}'（置信度 {dec.confidence:.2f}）",
                                                detail={"measure": m.index, "restored": confirmed_c, "beat": assigned_beat, "confidence": dec.confidence},
                                            )
                                        )
                            else:
                                new_issues.append(
                                    QualityIssue(
                                        stage="omr",
                                        measure_index=m.index,
                                        severity="needs_review",
                                        code="missing_chord_suspected",
                                        message=f"第{m.index + 1}小节行级转录疑似存在和弦 '{cand_c}'，但格式无法识别",
                                        detail={"measure": m.index, "candidate": cand_c, "beat": cand_b},
                                    )
                                )
                        else:
                            conf_val = dec.confidence if dec else 0.0
                            new_issues.append(
                                QualityIssue(
                                    stage="omr",
                                    measure_index=m.index,
                                    severity="needs_review",
                                    code="missing_chord_suspected",
                                    message=f"第{m.index + 1}小节行级转录疑似存在和弦 '{cand_c}'（拍数 {cand_b:.1f}），请核对",
                                    detail={"measure": m.index, "candidate": cand_c, "beat": cand_b, "arbiter_confidence": conf_val},
                                )
                            )
                except Exception as exc:
                    logger.warning("Candidate insertion arbiter call failed: %s", exc)
                    for item in arb_items:
                        m = item["measure"]
                        cand_c = item["candidate_chord"]
                        cand_b = item["candidate_beat"]
                        row_aligned_restored_measures.add(m.index)
                        new_issues.append(
                            QualityIssue(
                                stage="omr",
                                measure_index=m.index,
                                severity="needs_review",
                                code="missing_chord_suspected",
                                message=f"第{m.index + 1}小节行级转录疑似存在和弦 '{cand_c}'（拍数 {cand_b:.1f}），请核对",
                                detail={"measure": m.index, "candidate": cand_c, "beat": cand_b},
                            )
                        )
    # Drop non-chord voicing tokens outside closed boxes on boxed charts
    is_boxed_chart = any(
        c.bbox is not None for s in verified.systems for m_i in s.measures for c in m_i.chords
    )
    if is_boxed_chart:
        for s in verified.systems:
            for m in s.measures:
                retained_chords = []
                for c in m.chords:
                    raw_c = c.raw.strip()
                    if NON_CHORD_VOICING_PATTERN.match(raw_c) and c.bbox is None:
                        new_issues.append(
                            QualityIssue(
                                stage="omr",
                                measure_index=m.index,
                                severity="info",
                                code="non_chord_token_dropped",
                                message=f"第{m.index + 1}小节非和弦记号 '{c.raw}' 未处于和弦框内，已自动丢弃",
                                detail={"raw": c.raw, "beat": c.beat},
                            )
                        )
                    else:
                        retained_chords.append(c)
                m.chords = retained_chords

    # 2a-2. Melody Beat-Sum Check (info unless >=2 chords and beat disagreement)
    for m in all_measures:
        pm = parsed_melodies.get(m.index)
        if pm and pm.notes and abs(pm.beat_sum - m.beats) >= 1.0:
            if can_use_llm and m.bbox:
                m_sys = next(
                    (s for s in verified.systems if any(meas.index == m.index for meas in s.measures)),
                    None,
                )
                p_idx = m_sys.page if m_sys else 0
                img_bgr = cv2_pages[p_idx] if p_idx < len(cv2_pages) else None
                if img_bgr is not None and m_sys is not None:
                    crop_bytes = _crop_measure(img_bgr, m_sys, m)
                    if crop_bytes:
                        try:
                            m_meter = f"{int(m.beats)}/4" if m.beats.is_integer() else f"{m.beats} beats"
                            melody_prompt = (
                                f"You are an expert music assistant reading Jianpu (numbered musical notation) in {m_meter} meter. "
                                f"Transcribe the melody for this single measure. "
                                f"Group notes into exactly {int(m.beats)} space-separated beat groups so the measure has {m.beats:.1f} beats in total. "
                                f"For example: '2323 35. 2321 12.' or '1 2 3 4'."
                            )
                            crop_resp = ask_json(
                                prompt=melody_prompt,
                                schema=MeasureMelodyCropReading,
                                images=[crop_bytes],
                                role="reader",
                                timeout_s=30.0,
                            )
                            if crop_resp and crop_resp.melody:
                                new_pm = parse_melody(crop_resp.melody, beats=m.beats)
                                if abs(new_pm.beat_sum - m.beats) < 1.0:
                                    orig_melody = m.melody
                                    m.melody = crop_resp.melody
                                    parsed_melodies[m.index] = new_pm
                                    pm = new_pm
                                    new_issues.append(
                                        QualityIssue(
                                            stage="omr",
                                            measure_index=m.index,
                                            severity="auto_fixed",
                                            code="melody_corrected",
                                            message=f"第{m.index + 1}小节旋律拍数经单小节重读更正（{new_pm.beat_sum:.1f}拍，原={orig_melody}，更正后={crop_resp.melody}）",
                                            detail={
                                                "original": orig_melody,
                                                "corrected": crop_resp.melody,
                                                "beat_sum": new_pm.beat_sum,
                                            },
                                        )
                                    )
                        except Exception as exc:
                            logger.warning("Measure %d melody crop re-read failed: %s", m.index + 1, exc)

            if abs(pm.beat_sum - m.beats) >= 1.0:
                has_unresolved_beat_disagreement = False
                if len(m.chords) >= 2:
                    beats = [c.beat for c in m.chords]
                    m_tcs = m_tc_map_global.get(m.index, [])
                    tc_beats = [tc.beat for tc in m_tcs] if m_tcs else []

                    has_potential_disagreement = (
                        len(set(beats)) < len(beats)
                        or any(c2.beat <= c1.beat for c1, c2 in zip(m.chords[:-1], m.chords[1:]))
                        or (len(tc_beats) >= len(beats) and any(abs(b1 - b2) >= 0.5 for b1, b2 in zip(beats, tc_beats)))
                    )

                    if has_potential_disagreement:
                        is_all_geometrically_resolved = True
                        resolved_beats = []
                        for idx, c in enumerate(m.chords):
                            competing = [c.beat]
                            if idx < len(tc_beats):
                                competing.append(tc_beats[idx])
                            resolved_b, is_ambiguous = resolve_chord_beat_geo(c, m, competing)
                            if is_ambiguous or resolved_b is None:
                                is_all_geometrically_resolved = False
                                break
                            resolved_beats.append(resolved_b)

                        if is_all_geometrically_resolved:
                            if len(set(resolved_beats)) < len(resolved_beats) or any(
                                b2 <= b1 for b1, b2 in zip(resolved_beats[:-1], resolved_beats[1:])
                            ):
                                is_all_geometrically_resolved = False

                        if not is_all_geometrically_resolved:
                            has_unresolved_beat_disagreement = True
                        else:
                            for c, rb in zip(m.chords, resolved_beats):
                                c.beat = rb
                            new_issues.append(
                                QualityIssue(
                                    stage="omr",
                                    measure_index=m.index,
                                    severity="info",
                                    code="chord_beat_resolved_geo",
                                    message=f"第{m.index + 1}小节和弦起始拍依据打印框几何位置判定正常",
                                    detail={"measure": m.index, "beats": [c.beat for c in m.chords]},
                                )
                            )

                if len(m.chords) >= 2 and has_unresolved_beat_disagreement:
                    new_issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=m.index,
                            severity="needs_review",
                            code="melody_beat_sum_mismatch",
                            message=f"第{m.index + 1}小节旋律拍数（{pm.beat_sum:.1f}拍）与小节拍数（{m.beats:.1f}拍）不符且和弦起始拍存疑，请核对",
                            detail={"melody": m.melody, "beat_sum": pm.beat_sum, "expected_beats": m.beats},
                        )
                    )
                else:
                    new_issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=m.index,
                            severity="info",
                            code="melody_beat_sum_mismatch",
                            message=f"第{m.index + 1}小节旋律拍数（{pm.beat_sum:.1f}拍）与小节拍数（{m.beats:.1f}拍）不符（简谱下划线省略，仅供参考）",
                            detail={"melody": m.melody, "beat_sum": pm.beat_sum, "expected_beats": m.beats},
                        )
                    )

    # 2b. Structural Checks: Missing Chord Inspection (Crop Re-Read + Arbiter Confirmation)
    suspected_missing_measures: list[dict[str, Any]] = []
    for m in all_measures:
        if m.index in row_aligned_restored_measures:
            continue
        pm = parsed_melodies.get(m.index)
        m_notes = pm.notes if pm else []

        prev_c = None
        if (m.index - 1) in measure_map and measure_map[m.index - 1].chords:
            prev_c = measure_map[m.index - 1].chords[-1].raw

        next_c = None
        if (m.index + 1) in measure_map and measure_map[m.index + 1].chords:
            next_c = measure_map[m.index + 1].chords[0].raw

        if len(m.chords) == 0:
            carried_fit = melody_fit(prev_c, m_notes, 1.0, m.beats) if prev_c else 0.0
            if m.fill or carried_fit < 0.35:
                suspected_missing_measures.append({
                    "measure": m,
                    "beat": 1.0,
                    "reason": "fill_empty" if m.fill else "empty_measure_melody_clash",
                })
        elif len(m.chords) == 1 and m.beats >= 4.0:
            c0 = m.chords[0]
            if c0.beat >= 2.5:
                suspected_missing_measures.append({
                    "measure": m,
                    "beat": 1.0,
                    "reason": "starts_late",
                })
            elif c0.beat <= 2.0:
                h2_notes = [n for n in m_notes if n[0] >= 2.8]
                fit_h2 = melody_fit(c0.raw, h2_notes, chord_beat=3.0, chord_duration=2.0)
                is_cadence = False
                if next_c:
                    try:
                        s0 = parse_chord(c0.raw)
                        s1 = parse_chord(next_c)
                        if s0.degree in (2, 4) and s1.degree in (1, 6):
                            is_cadence = True
                    except Exception:
                        pass
                if m.fill:
                    suspected_missing_measures.append({
                        "measure": m,
                        "beat": 3.0,
                        "reason": "fill_measure",
                    })
                elif is_cadence and len(h2_notes) >= 2:
                    if fit_h2 < 0.30:
                        suspected_missing_measures.append({
                            "measure": m,
                            "beat": 3.0,
                            "reason": "cadence_measure",
                        })

    is_boxed_chart = (
        is_v2_geometry
        and any(c.bbox is not None for s in verified.systems for m_i in s.measures for c in m_i.chords)
    )

    for it in suspected_missing_measures:
        m = it["measure"]
        sus_beat = it["beat"]
        m_sys = next((s for s in verified.systems if any(meas.index == m.index for meas in s.measures)), None)
        p_idx = m_sys.page if m_sys else 0
        img_bgr = cv2_pages[p_idx] if p_idx < len(cv2_pages) else None

        # Check for unmapped ink in the chord band
        has_unmapped_ink = False
        if is_boxed_chart and img_bgr is not None and m_sys is not None:
            h_img, w_img = img_bgr.shape[:2]
            sys_boxed_chords = [c for m_i in m_sys.measures for c in m_i.chords if c.bbox]
            if sys_boxed_chords:
                y0_band = min(c.bbox[1] for c in sys_boxed_chords)
                y1_band = max(c.bbox[3] for c in sys_boxed_chords)
            else:
                y0_band = m_sys.bbox[1]
                y1_band = m_sys.bbox[1] + 0.35 * (m_sys.bbox[3] - m_sys.bbox[1])

            if sus_beat <= 2.0:
                x0_b = m.bbox[0]
                x1_b = m.bbox[0] + 0.50 * (m.bbox[2] - m.bbox[0])
            else:
                x0_b = m.bbox[0] + 0.45 * (m.bbox[2] - m.bbox[0])
                x1_b = m.bbox[2]

            y0_px = max(0, int(y0_band * h_img))
            y1_px = min(h_img, int(y1_band * h_img))
            x0_px = max(0, int(x0_b * w_img))
            x1_px = min(w_img, int(x1_b * w_img))

            if y1_px > y0_px and x1_px > x0_px:
                band_crop = img_bgr[y0_px:y1_px, x0_px:x1_px]
                if band_crop.size > 0:
                    g_crop = cv2.cvtColor(band_crop, cv2.COLOR_BGR2GRAY)
                    crop_std = float(np.std(g_crop))
                    _, bin_crop = cv2.threshold(g_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    ink_frac = float(np.sum(bin_crop > 0)) / float(g_crop.size)

                    # Compute median ink fraction of known chord boxes in this system
                    box_ink_fracs: list[float] = []
                    for c in sys_boxed_chords:
                        bx0 = max(0, int(c.bbox[0] * w_img))
                        by0 = max(0, int(c.bbox[1] * h_img))
                        bx1 = min(w_img, int(c.bbox[2] * w_img))
                        by1 = min(h_img, int(c.bbox[3] * h_img))
                        if by1 > by0 and bx1 > bx0:
                            b_crop = img_bgr[by0:by1, bx0:bx1]
                            if b_crop.size > 0:
                                bg = cv2.cvtColor(b_crop, cv2.COLOR_BGR2GRAY)
                                _, b_bin = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                                box_ink_fracs.append(float(np.sum(b_bin > 0)) / float(bg.size))

                    median_box_frac = float(np.median(box_ink_fracs)) if box_ink_fracs else 0.10
                    min_frac = max(0.015, 0.15 * median_box_frac)

                    if ink_frac >= min_frac and crop_std >= 12.0:
                        has_unmapped_ink = True

        # In boxed charts, absence of a chord box and ink token is strong evidence no chord is printed -> info carry-over
        if is_boxed_chart and not has_unmapped_ink:
            new_issues.append(
                QualityIssue(
                    stage="omr",
                    measure_index=m.index,
                    severity="info",
                    code="chord_carry_over",
                    message=f"第{m.index + 1}小节未检测到和弦框及油墨，判定为沿用前序和弦进行（carry-over）",
                    detail={"measure": m.index, "beat": sus_beat, "reason": "boxed_chart_no_box_carry_over"},
                )
            )
            continue

        # If unmapped ink is present or chart is unboxed / low-confidence:
        if not can_use_llm or img_bgr is None or m_sys is None:
            new_issues.append(
                QualityIssue(
                    stage="omr",
                    measure_index=m.index,
                    severity="needs_review",
                    code="missing_chord_suspected",
                    message=f"第{m.index + 1}小节疑似遗漏和弦，请核对",
                    detail={"measure": m.index, "beat": sus_beat},
                )
            )
            continue

        crop_bytes = _crop_measure(img_bgr, m_sys, m)
        if not crop_bytes:
            new_issues.append(
                QualityIssue(
                    stage="omr",
                    measure_index=m.index,
                    severity="needs_review",
                    code="missing_chord_suspected",
                    message=f"第{m.index + 1}小节疑似遗漏和弦，请核对",
                    detail={"measure": m.index, "beat": sus_beat},
                )
            )
            continue

        reader_prompt = (
            f"Examine this measure crop from a Taiwanese band chart sheet music (measure {m.index + 1}). "
            f"Are there any printed chord boxes (e.g. '1(2)', '5', '5/7', '2m7/5', etc.) around beat {sus_beat}? "
            "Return any chord box printed in this measure with its raw text and beat."
        )
        try:
            reader_resp = ask_json(
                prompt=reader_prompt,
                schema=MeasureCropChordResponse,
                images=[crop_bytes],
                role="reader",
                timeout_s=30.0,
            )
            valid_cands: list[DetectedMeasureChordBox] = []
            for b in reader_resp.chords:
                if is_valid_chord_grammar(b.chord.strip()):
                    if not any(abs(c.beat - b.beat) < 0.5 for c in m.chords):
                        valid_cands.append(b)

            if valid_cands:
                top_cand = valid_cands[0]
                cand_chord = top_cand.chord.strip()
                cand_beat = top_cand.beat

                arbiter_prompt = (
                    f"You are an expert chord arbiter for Taiwanese band charts. "
                    f"Review this measure crop image (measure {m.index + 1}). "
                    f"The reader detected a possible chord box '{cand_chord}' around beat {cand_beat}. "
                    f"Inspect the image: is there an actual printed chord box with text '{cand_chord}' in this measure? "
                    "Confirm only if you clearly see the chord box ink printed in the image."
                )
                arbiter_resp = ask_json(
                    prompt=arbiter_prompt,
                    schema=MissingChordArbiterConfirmation,
                    images=[crop_bytes],
                    role="arbiter",
                    timeout_s=45.0,
                )
                if (
                    arbiter_resp.confirmed
                    and arbiter_resp.confidence >= 0.8
                    and arbiter_resp.confirmed_chord
                ):
                    clean_c = arbiter_resp.confirmed_chord.strip()
                    if is_valid_chord_grammar(clean_c):
                        m.chords.append(
                            ChordSymbol(
                                raw=clean_c,
                                beat=cand_beat,
                                confidence=arbiter_resp.confidence,
                            )
                        )
                        m.chords.sort(key=lambda c: c.beat)
                        new_issues.append(
                            QualityIssue(
                                stage="omr",
                                measure_index=m.index,
                                severity="auto_fixed",
                                code="chord_restored",
                                message=f"第{m.index + 1}小节在图像中识别并经仲裁确认恢复和弦 '{clean_c}'（置信度 {arbiter_resp.confidence:.2f}）",
                                detail={"measure": m.index, "restored": clean_c, "beat": cand_beat, "confidence": arbiter_resp.confidence},
                            )
                        )
                    else:
                        new_issues.append(
                            QualityIssue(
                                stage="omr",
                                measure_index=m.index,
                                severity="needs_review",
                                code="missing_chord_suspected",
                                message=f"第{m.index + 1}小节疑似遗漏和弦（候选：'{cand_chord}'），请核对",
                                detail={"measure": m.index, "candidate": cand_chord, "beat": cand_beat},
                            )
                        )
                else:
                    new_issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=m.index,
                            severity="needs_review",
                            code="missing_chord_suspected",
                            message=f"第{m.index + 1}小节疑似遗漏和弦（候选：'{cand_chord}'），请核对",
                            detail={"measure": m.index, "candidate": cand_chord, "beat": cand_beat, "arbiter_confidence": arbiter_resp.confidence},
                        )
                    )
            else:
                new_issues.append(
                    QualityIssue(
                        stage="omr",
                        measure_index=m.index,
                        severity="needs_review",
                        code="missing_chord_suspected",
                        message=f"第{m.index + 1}小节疑似遗漏和弦，请核对",
                        detail={"measure": m.index, "beat": sus_beat},
                    )
                )
        except Exception as exc:
            logger.warning("Missing chord check failed on m%d: %s", m.index, exc)
            new_issues.append(
                QualityIssue(
                    stage="omr",
                    measure_index=m.index,
                    severity="needs_review",
                    code="missing_chord_suspected",
                    message=f"第{m.index + 1}小节疑似遗漏和弦，请核对",
                    detail={"measure": m.index, "beat": sus_beat},
                )
            )
    # 4. Multi-chord beat alignment & Candidate Evaluation
    undecided_chords: list[dict[str, Any]] = []

    for s in verified.systems:
        for m in s.measures:
            # Check 2nd chord beat alignment
            if len(m.chords) == 2:
                c1, c2 = m.chords[0], m.chords[1]
                if c2.beat <= c1.beat or c2.beat > m.beats:
                    orig_beat = c2.beat
                    c2.beat = 3.0
                    new_issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=m.index,
                            severity="auto_fixed",
                            code="chord_beat_corrected",
                            message=f"第{m.index + 1}小节第2个和弦起始拍由 {orig_beat:.1f} 更正为 3.0",
                            detail={"original_beat": orig_beat, "corrected_beat": 3.0},
                        )
                    )

            for c_idx, c in enumerate(m.chords):
                key = (m.index, c_idx)
                raw_full = c.raw.strip()
                raw_crop = crop_readings_map.get(key, "").strip()

                # Neighboring chords for progression analysis
                prev_c = None
                if c_idx > 0:
                    prev_c = m.chords[c_idx - 1].raw
                elif (m.index - 1) in measure_map and measure_map[m.index - 1].chords:
                    prev_c = measure_map[m.index - 1].chords[-1].raw

                next_c = None
                if c_idx + 1 < len(m.chords):
                    next_c = m.chords[c_idx + 1].raw
                elif (m.index + 1) in measure_map and measure_map[m.index + 1].chords:
                    next_c = measure_map[m.index + 1].chords[0].raw

                pm = parsed_melodies.get(m.index)
                m_notes = pm.notes if pm else []
                dur = m.beats - c.beat + 1.0 if c_idx == len(m.chords) - 1 else 2.0

                is_full_valid = is_valid_chord_grammar(raw_full)
                is_crop_valid = bool(raw_crop and is_valid_chord_grammar(raw_crop))

                # 4a. Concordance Check
                if is_crop_valid and raw_crop == raw_full:
                    c.confidence = 0.95
                    continue

                # 4b. Candidate Generation and Prior Scoring
                cand_pool: set[str] = set()
                cand_pool.update(generate_candidates(raw_full))
                if is_crop_valid:
                    cand_pool.update(generate_candidates(raw_crop))
                    cand_pool.add(raw_crop)
                if is_full_valid:
                    cand_pool.add(raw_full)

                valid_cands = [cand for cand in cand_pool if is_valid_chord_grammar(cand)]

                if not valid_cands:
                    c.confidence = 0.2
                    new_issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=m.index,
                            severity="needs_review",
                            code="invalid_chord_grammar",
                            message=f"第{m.index + 1}小节和弦 '{raw_full}' 格式无法识别，请核对",
                            detail={"raw": raw_full},
                        )
                    )
                    continue

                cand_scores: list[tuple[str, float]] = []
                for cand in valid_cands:
                    prior_s = score_candidate(
                        cand,
                        melody_notes=m_notes,
                        bass_hint=m.bass_hint,
                        prev_chord=prev_c,
                        next_chord=next_c,
                        chord_beat=c.beat,
                        chord_duration=dur,
                    )
                    boost = 0.0
                    if is_v2_geometry:
                        if is_full_valid and cand == raw_full:
                            boost += 0.65
                        if is_crop_valid and cand == raw_crop:
                            boost += 0.30
                    else:
                        if is_crop_valid and cand == raw_crop:
                            boost += 0.65
                        if is_full_valid and cand == raw_full:
                            boost += 0.30
                    total = prior_s + boost
                    cand_scores.append((cand, total))

                cand_scores.sort(key=lambda x: x[1], reverse=True)
                top_cand, top_score = cand_scores[0]
                second_score = cand_scores[1][1] if len(cand_scores) > 1 else 0.0

                # 4c. Auto-fix Decision when raw_full is invalid
                if not is_full_valid:
                    cand_of_raw = set(generate_candidates(raw_full))
                    if is_crop_valid and top_cand == raw_crop and (top_cand in cand_of_raw or len(cand_of_raw) > 0):
                        orig_raw = raw_full
                        c.raw = top_cand
                        c.confidence = 0.88
                        c.alternatives = [x[0] for x in cand_scores[1:4]]
                        new_issues.append(
                            QualityIssue(
                                stage="omr",
                                measure_index=m.index,
                                severity="auto_fixed",
                                code="chord_corrected",
                                message=f"第{m.index + 1}小节和弦由 '{orig_raw}' 更正为 '{top_cand}'（修复格式错误）",
                                detail={"original": orig_raw, "corrected": top_cand, "score": top_score},
                            )
                        )
                    else:
                        c.confidence = 0.20
                        c.alternatives = [x[0] for x in cand_scores[:3]]
                        new_issues.append(
                            QualityIssue(
                                stage="omr",
                                measure_index=m.index,
                                severity="needs_review",
                                code="invalid_chord_grammar",
                                message=f"第{m.index + 1}小节和弦 '{raw_full}' 格式无法识别，请核对",
                                detail={"raw": raw_full},
                            )
                        )
                    continue

                # 4d. Compute prior score for raw_full
                prior_full = score_candidate(
                    raw_full,
                    melody_notes=m_notes,
                    bass_hint=m.bass_hint,
                    prev_chord=prev_c,
                    next_chord=next_c,
                    chord_beat=c.beat,
                    chord_duration=dur,
                )

                # If raw_full is valid, supported by priors, and not contradicted by a valid crop:
                min_prior = 0.35 if is_v2_geometry else 0.55
                if is_full_valid and prior_full >= min_prior and (not is_crop_valid or raw_crop == raw_full or (is_v2_geometry and top_cand == raw_full)):
                    c.confidence = 0.95 if is_v2_geometry else 0.92
                    c.alternatives = [x[0] for x in cand_scores if x[0] != raw_full][:3]
                    continue

                # 4e. Check for decisive prior superiority
                if (
                    is_crop_valid
                    and top_cand == raw_crop
                    and raw_crop in generate_candidates(raw_full)
                    and (top_score - second_score >= 0.10)
                ):
                    orig_raw = raw_full
                    c.raw = top_cand
                    c.confidence = 0.92
                    c.alternatives = [x[0] for x in cand_scores[1:4]]
                    reason_parts = ["裁剪复读一致"]
                    if bass_hint_fit(top_cand, m.bass_hint) == 1.0:
                        reason_parts.append("低音行一致")
                    if melody_fit(top_cand, m_notes, c.beat, dur) >= 0.8:
                        reason_parts.append("旋律音吻合")
                    reason_str = "与".join(reason_parts)

                    new_issues.append(
                        QualityIssue(
                            stage="omr",
                            measure_index=m.index,
                            severity="auto_fixed",
                            code="chord_corrected",
                            message=f"第{m.index + 1}小节和弦由 {orig_raw} 更正为 {top_cand}（{reason_str}）",
                            detail={"original": orig_raw, "corrected": top_cand, "score": top_score},
                        )
                    )
                elif top_cand == raw_full and (top_score - second_score >= (0.10 if is_v2_geometry else 0.25)):
                    c.confidence = 0.92 if is_v2_geometry else 0.90
                    c.alternatives = [x[0] for x in cand_scores[1:4]]
                else:
                    undecided_chords.append({
                        "key": key,
                        "chord": c,
                        "measure": m,
                        "system": s,
                        "raw_full": raw_full,
                        "raw_crop": raw_crop if is_crop_valid else "",
                        "cand_scores": cand_scores,
                        "prev_c": prev_c,
                        "next_c": next_c,
                        "m_notes": m_notes,
                        "prior_full": prior_full,
                    })

    # 5. Escalate Undecided Chords to Arbiter (Claude Opus on Vertex AI)
    if can_use_llm and undecided_chords:
        logger.info("Escalating %d undecided chords to Arbiter (Claude Opus)...", len(undecided_chords))
        page_to_undecided: dict[int, list[dict[str, Any]]] = {}
        for it in undecided_chords:
            p = it["system"].page
            page_to_undecided.setdefault(p, []).append(it)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(page_to_undecided))) as executor:
            futs = [
                executor.submit(_arbitrate_page, p_idx, p_items, crop_images_map, new_issues)
                for p_idx, p_items in page_to_undecided.items()
            ]
            concurrent.futures.wait(futs)
    else:
        for it in undecided_chords:
            _finalize_undecided(it, new_issues)

    verified.issues = new_issues
    return verified
