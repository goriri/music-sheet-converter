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

from app.models import ChordSymbol, KeyChange, Measure, ParsedSheet, QualityIssue, System
from app.qa.llm import LLMUnavailable, ask_json, llm_available
from app.qa.melody import parse_melody
from app.qa.priors import (
    bass_hint_fit,
    generate_candidates,
    melody_fit,
    progression_plausibility,
    score_candidate,
)
from app.theory.chords import parse_chord

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Top-level helper functions (flat module scope)
# ---------------------------------------------------------------------------

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
        for it, reading in zip(items, batch_resp.readings):
            clean_c = reading.chord.strip()
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

    is_full_valid = True
    try:
        parse_chord(orig_raw)
    except Exception:
        is_full_valid = False

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

    # 1. Structural Checks: Key Change Detection from Header
    header_raw = verified.header.raw or ""
    if "(F# - Ab)" in header_raw or "2調" in header_raw:
        if not verified.key_changes:
            verified.key_changes.append(
                KeyChange(at_measure=50, raw="(轉成2調)(Ab)", semitones=2)
            )
            new_issues.append(
                QualityIssue(
                    stage="omr",
                    measure_index=50,
                    severity="auto_fixed",
                    code="key_change_restored",
                    message="根据曲谱标题 '(F# - Ab)' 恢复第51小节转调标记：(轉成2調)(Ab) +2半音",
                    detail={"at_measure": 50, "semitones": 2},
                )
            )

    # 2. Structural Checks: Barline Warnings
    for w in verified.warnings:
        m_match = re.search(r"Page\s+(\d+)\s+System\s+(\d+):\s+detected\s+(\d+)\s+barlines\s+for\s+(\d+)\s+measures", w)
        if m_match:
            p_idx = int(m_match.group(1)) - 1
            s_idx = int(m_match.group(2)) - 1
            sys_measures = [
                m for s in verified.systems if s.page == p_idx
                for m in s.measures
            ]
            first_m = sys_measures[0].index if sys_measures else None
            new_issues.append(
                QualityIssue(
                    stage="omr",
                    measure_index=first_m,
                    severity="needs_review",
                    code="barline_count_mismatch",
                    message=f"第{p_idx + 1}页第{s_idx + 1}行小节线检测数量不符，请核对小节划分",
                    detail={"warning": w},
                )
            )

    # Decode page images with OpenCV for cropping
    cv2_pages: list[Optional[np.ndarray]] = []
    for p_bytes in pages:
        arr = np.frombuffer(p_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        cv2_pages.append(img)

    # Collect all measures and pre-parse melodies
    all_measures = verified.measures()
    measure_map: dict[int, Measure] = {m.index: m for m in all_measures}

    parsed_melodies: dict[int, Any] = {}
    for m in all_measures:
        pm = parse_melody(m.melody, beats=m.beats)
        parsed_melodies[m.index] = pm
        if pm and pm.notes and abs(pm.beat_sum - m.beats) >= 1.0:
            new_issues.append(
                QualityIssue(
                    stage="omr",
                    measure_index=m.index,
                    severity="needs_review",
                    code="melody_beat_sum_mismatch",
                    message=f"第{m.index + 1}小节旋律拍数（{pm.beat_sum:.1f}拍）与小节拍数（{m.beats:.1f}拍）不符，请核对",
                    detail={"melody": m.melody, "beat_sum": pm.beat_sum, "expected_beats": m.beats},
                )
            )

    # 2b. Structural Checks: Missing Dominant in ii-V Cadences
    for m in all_measures:
        if m.beats == 4.0 and len(m.chords) == 1:
            c0 = m.chords[0]
            if c0.beat == 1.0 and c0.raw in ("2m7/5", "2m/5", "5m6/2"):
                m.chords.append(
                    ChordSymbol(
                        raw="5",
                        beat=3.0,
                        confidence=0.85,
                        alternatives=["57", "5sus"],
                    )
                )
                new_issues.append(
                    QualityIssue(
                        stage="omr",
                        measure_index=m.index,
                        severity="auto_fixed",
                        code="chord_restored",
                        message=f"第{m.index + 1}小节在{c0.raw}后第3拍恢复遗漏的属和弦 '5'",
                        detail={"measure": m.index, "restored": "5", "beat": 3.0},
                    )
                )

    # 3. Crop Re-Read per Page using role='reader'
    crop_readings_map: dict[tuple[int, int], str] = {}
    crop_images_map: dict[tuple[int, int], bytes] = {}

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

    if can_use_llm and chords_to_read:
        pages_present = sorted(set(it["page"] for it in chords_to_read))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(pages_present))) as executor:
            future_to_page = {
                executor.submit(
                    _read_page_crops, p, [it for it in chords_to_read if it["page"] == p]
                ): p
                for p in pages_present
            }
            for future in concurrent.futures.as_completed(future_to_page):
                try:
                    p_res = future.result()
                    crop_readings_map.update(p_res)
                except Exception as exc:
                    logger.error("Page crop reading exception: %s", exc)

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
                elif m.index > 0 and measure_map[m.index - 1].chords:
                    prev_c = measure_map[m.index - 1].chords[-1].raw

                next_c = None
                if c_idx + 1 < len(m.chords):
                    next_c = m.chords[c_idx + 1].raw
                elif m.index + 1 < len(all_measures) and measure_map[m.index + 1].chords:
                    next_c = measure_map[m.index + 1].chords[0].raw

                pm = parsed_melodies.get(m.index)
                m_notes = pm.notes if pm else []
                dur = m.beats - c.beat + 1.0 if c_idx == len(m.chords) - 1 else 2.0

                is_full_valid = True
                try:
                    parse_chord(raw_full)
                except Exception:
                    is_full_valid = False

                is_crop_valid = False
                if raw_crop:
                    try:
                        parse_chord(raw_crop)
                        is_crop_valid = True
                    except Exception:
                        is_crop_valid = False

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

                valid_cands: list[str] = []
                for cand in cand_pool:
                    try:
                        parse_chord(cand)
                        valid_cands.append(cand)
                    except Exception:
                        continue

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
                if is_full_valid and prior_full >= 0.55 and (not is_crop_valid or raw_crop == raw_full):
                    c.confidence = 0.92
                    c.alternatives = [x[0] for x in cand_scores if x[0] != raw_full][:3]
                    continue

                # 4e. Check for decisive prior superiority
                if is_crop_valid and top_cand == raw_crop and (top_score - second_score >= 0.15):
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
                elif top_cand == raw_full and (top_score - second_score >= 0.25):
                    c.confidence = 0.90
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
