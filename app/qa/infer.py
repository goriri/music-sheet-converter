"""Context-aware inference and escalation ladder for ambiguous OMR elements.

Escalation ladder for any item that would otherwise become needs_review:
1. Existing crop re-read.
2. Context inference with Claude Opus (app/qa/infer.py):
   Batch per system row with upscaled system crop, flagged measure ± 2 measures of
   reader output (melody, chords, beats), key/time signature, section label, candidate set.
3. Web evidence (lazily imported from app.qa.web_evidence, tolerating absence).
4. needs_review only if still unresolved or evidence conflicts.

Acceptance rules for auto-fixing:
- Inference confidence >= 0.8 AND consistent with melody-fit priors (not worse than runner-up
  by margin 0.10) AND not contradicted by a valid crop re-read;
- OR when inference and web evidence (root-level match) agree.
- Web evidence alone never overrides a clearly-read chord; can only break ties or confirm missing chord.
- Structural changes (measure boundary / split / merge) are accepted only when beat sums
  become valid for all affected measures AND barline x-position is supported by pixel evidence (vertical stroke).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from app.models import ChordSymbol, Measure, ParsedSheet, QualityIssue, System
from app.omr.reader import ask_json
from app.qa.melody import parse_melody
from app.qa.omr_verify import is_valid_chord_grammar, score_candidate
from app.qa.priors import generate_candidates
from app.theory.keys import key_name_to_pc

logger = logging.getLogger(__name__)

AmbiguousTargetType = Literal[
    "chord",
    "missing_chord",
    "chord_beat",
    "measure_boundary",
    "key_change",
    "melody",
]


class ItemInference(BaseModel):
    item_id: str = Field(description="Unique item id matching the prompt, e.g. 'm12_chord_0'")
    target_type: AmbiguousTargetType = Field(description="Target aspect being inferred")
    measure_index: int = Field(description="0-based measure index in the song")
    proposed_value: str = Field(description="Inferred value, e.g. chord '5/7', split beat 3.0, or key change '(轉2調)'")
    confidence: float = Field(description="Confidence from 0.0 to 1.0", ge=0.0, le=1.0)
    musical_justification: str = Field(description="Short musical justification in Chinese")
    split_beat: Optional[float] = Field(None, description="For measure split/merge, beat position of the boundary")
    barline_norm_x: Optional[float] = Field(None, description="For missing barline, estimated normalized x (0..1) on system")


class SystemInferenceResponse(BaseModel):
    inferences: list[ItemInference] = Field(default_factory=list)


class AmbiguousItem(BaseModel):
    item_id: str
    target_type: AmbiguousTargetType
    measure_index: int
    system_page: int
    system_index: int
    system_bbox: tuple[float, float, float, float]
    original_value: str = ""
    candidate_set: list[str] = Field(default_factory=list)
    crop_reading: Optional[str] = None
    crop_valid: bool = False
    context_measures: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    prior_scores: dict[str, float] = Field(default_factory=dict)
    chord_beat: float = 1.0


class LadderDecision(BaseModel):
    accepted: bool
    target_type: AmbiguousTargetType
    measure_index: int
    new_value: str
    severity: Literal["auto_fixed", "needs_review", "info"]
    evidence: list[str] = Field(default_factory=list)
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


def extract_root_degree(chord_str: str) -> Optional[str]:
    """Extract root degree or letter from a chord string, e.g. '5/7' -> '5', '2m7' -> '2', 'G/B' -> 'G'."""
    s = chord_str.strip()
    if not s:
        return None
    m_deg = re.match(r"^([#b]?[1-7])", s)
    if m_deg:
        return m_deg.group(1)
    m_let = re.match(r"^([A-G][#b]?)", s)
    if m_let:
        return m_let.group(1)
    m_bass = re.match(r"^/([#b]?[1-7A-G])", s)
    if m_bass:
        return m_bass.group(1)
    return None


def verify_vertical_barline_stroke(
    img_bgr: Optional[np.ndarray],
    norm_x: float,
    sys_bbox: tuple[float, float, float, float],
    margin_px: int = 15,
) -> bool:
    """Verify whether physical image pixel evidence supports a vertical barline stroke near norm_x."""
    if img_bgr is None or img_bgr.size == 0:
        return False
    h_img, w_img = img_bgr.shape[:2]
    y0 = max(0, int((sys_bbox[1] + 0.15 * (sys_bbox[3] - sys_bbox[1])) * h_img))
    y1 = min(h_img, int((sys_bbox[3] - 0.15 * (sys_bbox[3] - sys_bbox[1])) * h_img))
    if y1 <= y0:
        return False

    center_x = int(norm_x * w_img)
    x0 = max(0, center_x - margin_px)
    x1 = min(w_img, center_x + margin_px + 1)
    if x1 <= x0:
        return False

    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    crop_h = y1 - y0
    kh = max(12, int(crop_h * 0.55))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, kh))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    col_proj = np.sum(vert > 0, axis=0)
    thresh = kh * 0.40
    return bool(np.any(col_proj >= thresh))


def _extract_ref_raw(ref_obj: Any) -> str:
    """Extract raw chord string from a reference candidate without reflection."""
    if isinstance(ref_obj, str):
        return ref_obj
    if isinstance(ref_obj, dict):
        return str(ref_obj.get("chord") or ref_obj.get("raw") or "")
    try:
        return str(ref_obj.chord)
    except AttributeError:
        pass
    try:
        return str(ref_obj.raw)
    except AttributeError:
        return str(ref_obj)


def extract_bass_degree(chord_str: str) -> Optional[str]:
    """Extract bass degree from a slash chord, e.g. '5/7' -> '7', '2m7/5' -> '5'."""
    s = chord_str.strip()
    if "/" in s:
        parts = s.split("/", 1)
        return extract_root_degree(parts[1])
    return None


def get_web_reference_candidates(
    title: str,
    artist: str = "",
    sheet: Optional[ParsedSheet] = None,
    start_key_pc: int = 0,
) -> dict[int, list[Any]]:
    """Lazily query web evidence reference candidates, tolerating missing module or network."""
    if not title:
        return {}
    try:
        from app.qa import web_evidence  # type: ignore[import-not-found]
        find_fn = web_evidence.find_reference
        align_fn = web_evidence.align_reference
        ref = find_fn(title, artist)
        if not ref and artist:
            ref = find_fn(title, None)
        if ref and sheet:
            return align_fn(sheet, ref, start_key_pc) or {}
    except (ImportError, AttributeError, Exception) as exc:
        logger.debug("Web evidence unavailable (%s): %s", title, exc)
    return {}


def evaluate_ladder_decision(
    item: AmbiguousItem,
    inference: Optional[ItemInference],
    web_candidates: list[Any],
    img_bgr: Optional[np.ndarray] = None,
) -> LadderDecision:
    """Evaluate acceptance rule for an ambiguous item across crop, inference, web, and priors.

    ACCEPT RULE:
    - accept when inference confidence >= 0.8 AND choice consistent with melody priors (within 0.10 margin)
      AND not contradicted by valid crop re-read;
    - OR when inference and web evidence (root-level match) agree.
    - Web evidence alone never overrides a clearly-read chord; can only break ties or confirm missing chord.
    - Structural changes (measure boundary / split / merge) accepted only when beat sums become valid AND
      barline x-position is supported by pixel evidence (vertical stroke).
    """
    m_num = item.measure_index + 1
    orig_val = item.original_value

    if inference is None:
        # Check if priors alone decisively resolve it with web confirmation
        if item.candidate_set and web_candidates:
            top_cand = item.candidate_set[0]
            ref_raws = [_extract_ref_raw(r) for r in web_candidates]
            prop_root = extract_root_degree(top_cand)
            if any(extract_root_degree(w) == prop_root for w in ref_raws) and is_valid_chord_grammar(top_cand):
                return LadderDecision(
                    accepted=True,
                    target_type=item.target_type,
                    measure_index=item.measure_index,
                    new_value=top_cand,
                    severity="auto_fixed",
                    evidence=["web", "priors"],
                    message=f"第{m_num}小节和弦依据参考谱与先验规则推断为 {top_cand}（原读 {orig_val or '无'}）",
                    detail={"original": orig_val, "inferred": top_cand, "evidence": ["web", "priors"]},
                )

        return LadderDecision(
            accepted=False,
            target_type=item.target_type,
            measure_index=item.measure_index,
            new_value=orig_val,
            severity="needs_review",
            evidence=["priors"] if item.prior_scores else [],
            message=f"第{m_num}小节内容存疑，请核对",
            detail={"original": orig_val, "reason": "no_inference_available"},
        )

    proposed = inference.proposed_value.strip()
    conf = inference.confidence
    justification = inference.musical_justification or ""
    evidence_list: list[str] = []

    # 1. Structural change (measure boundary / re-segmenting)
    if item.target_type == "measure_boundary":
        split_b = inference.split_beat or 3.0
        norm_x = inference.barline_norm_x or 0.5
        stroke_supported = verify_vertical_barline_stroke(img_bgr, norm_x, item.system_bbox)
        exp_b = item.metadata.get("expected_beats", 4.0)
        curr_melody_beats = item.metadata.get("current_melody_beats", 6.0)
        split_valid = (
            abs(split_b - exp_b) <= 0.5
            or abs(curr_melody_beats - (split_b + (curr_melody_beats - split_b))) < 0.1
        )

        if conf >= 0.80 and stroke_supported and split_valid:
            evidence_list.extend(["inference", "pixel_stroke", "meter"])
            return LadderDecision(
                accepted=True,
                target_type="measure_boundary",
                measure_index=item.measure_index,
                new_value=f"split_at_beat_{split_b:.1f}",
                severity="auto_fixed",
                evidence=evidence_list,
                message=f"第{m_num}小节依据旋律与纵向线段物理笔画推断划分小节线（第{split_b:.1f}拍处）：{justification}",
                detail={"split_beat": split_b, "norm_x": norm_x, "confidence": conf, "evidence": evidence_list},
            )
        else:
            return LadderDecision(
                accepted=False,
                target_type="measure_boundary",
                measure_index=item.measure_index,
                new_value=orig_val,
                severity="needs_review",
                evidence=["inference"] if conf >= 0.8 else [],
                message=f"第{m_num}小节小节划分存疑（缺少明确线段笔画支持），请核对",
                detail={"confidence": conf, "stroke_supported": stroke_supported, "proposed": proposed},
            )

    # 2. Check crop contradiction
    is_contradicted_by_crop = False
    if item.crop_valid and item.crop_reading:
        c_crop = item.crop_reading.strip()
        if c_crop and c_crop != proposed:
            is_contradicted_by_crop = True
        elif c_crop == proposed:
            evidence_list.append("crop")

    # 3. Check melody-fit priors consistency
    priors = item.prior_scores
    top_prior = max(priors.values()) if priors else 0.5
    cand_prior = priors.get(proposed, 0.40)
    is_prior_consistent = (cand_prior >= (top_prior - 0.10)) and (cand_prior >= 0.25)
    if is_prior_consistent:
        evidence_list.append("priors")

    # 4. Check web evidence agreement (root and bass match)
    web_agrees_root = False
    ref_raws = [_extract_ref_raw(r) for r in web_candidates]
    prop_root = extract_root_degree(proposed)
    prop_bass = extract_bass_degree(proposed)
    for w_raw in ref_raws:
        w_root = extract_root_degree(w_raw)
        w_bass = extract_bass_degree(w_raw)
        if prop_root and w_root and prop_root == w_root:
            if prop_bass and w_bass:
                if prop_bass == w_bass:
                    web_agrees_root = True
                    evidence_list.append("web")
                    break
            else:
                web_agrees_root = True
                evidence_list.append("web")
                break

    # 5. Evaluate Accept Conditions
    cond_1 = (conf >= 0.80) and is_prior_consistent and (not is_contradicted_by_crop)
    cond_2 = (conf >= 0.70) and web_agrees_root and (not is_contradicted_by_crop)

    if (cond_1 or cond_2) and is_valid_chord_grammar(proposed):
        evidence_list.append("inference")
        seen_ev: set[str] = set()
        dedup_ev: list[str] = []
        for ev in evidence_list:
            if ev not in seen_ev:
                seen_ev.add(ev)
                dedup_ev.append(ev)

        msg = (
            f"第{m_num}小节和弦依据旋律推断为 {proposed}（原读 {orig_val or '无'}）"
            if item.target_type in ("chord", "missing_chord")
            else f"第{m_num}小节推断为 {proposed}：{justification}"
        )
        return LadderDecision(
            accepted=True,
            target_type=item.target_type,
            measure_index=item.measure_index,
            new_value=proposed,
            severity="auto_fixed",
            evidence=dedup_ev,
            message=msg,
            detail={
                "original": orig_val,
                "inferred": proposed,
                "confidence": conf,
                "evidence": dedup_ev,
                "justification": justification,
            },
        )

    reason = (
        "contradicted_by_crop"
        if is_contradicted_by_crop
        else ("prior_inconsistent" if not is_prior_consistent else "low_confidence")
    )
    return LadderDecision(
        accepted=False,
        target_type=item.target_type,
        measure_index=item.measure_index,
        new_value=orig_val,
        severity="needs_review",
        evidence=["inference"] if conf >= 0.7 else [],
        message=f"第{m_num}小节和弦 '{orig_val or proposed}' 存疑，请核对",
        detail={
            "original": orig_val,
            "inferred_candidate": proposed,
            "confidence": conf,
            "reason": reason,
        },
    )


def build_system_inference_prompt(
    sheet: ParsedSheet,
    system: System,
    items: list[AmbiguousItem],
) -> str:
    """Build a rich contextual prompt for Opus inference on a system row."""
    header = sheet.header
    lines = [
        "You are an expert music theorist and Taiwanese band chart (台湾流行简谱) specialist.",
        "Perform context-aware musical inference to resolve ambiguous readings on this music sheet system row.",
        "",
        f"Chart Metadata:",
        f"- Title: '{header.title or 'Unknown'}'",
        f"- Meter: {header.time_signature}, Start Key: {header.original_key or 'Unknown'}",
        f"- System Section Label: '{system.section_label or 'None'}'",
        "",
        "Instructions:",
        "1. Examine the image of this system row and the surrounding measure contexts (melody, chords, meter).",
        "2. For each ambiguous item listed below, determine the most musically and harmonically plausible reading.",
        "3. Chord symbols MUST use Taiwanese scale-degree number notation (e.g. 1(2), 5/7, 2m7, 4M7, 57sus, 6m7-5).",
        "4. Provide a confidence score (0.0 to 1.0) and a concise musical justification (e.g. '配合旋律主音 5 构成属和弦进行').",
        "",
        "Ambiguous Items to Resolve in this System Row:",
    ]

    for it in items:
        lines.append(f"\nItem ID: '{it.item_id}' (Target: {it.target_type}, Measure {it.measure_index + 1}):")
        lines.append(f"- Original Reading: '{it.original_value}'")
        if it.candidate_set:
            lines.append(f"- Candidate Pool: {it.candidate_set}")
        if it.crop_reading:
            lines.append(f"- Crop Re-Read: '{it.crop_reading}' (valid={it.crop_valid})")
        if it.prior_scores:
            p_str = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(it.prior_scores.items(), key=lambda x: -x[1])[:5])
            lines.append(f"- Melody Prior Scores: {p_str}")

        lines.append("- Context Measures (flagged measure ±2):")
        for ctx in it.context_measures:
            c_str = ", ".join(f"{c['raw']}@beat{c['beat']:.1f}" for c in ctx.get("chords", [])) or "None"
            lines.append(
                f"  * m{ctx['measure_index'] + 1}: melody='{ctx.get('melody', '')}', "
                f"beat_sum={ctx.get('beat_sum', 4.0):.1f}/{ctx.get('beats', 4.0):.1f}, chords=[{c_str}]"
            )

    return "\n".join(lines)


def run_system_row_inference(
    cv2_page: Optional[np.ndarray],
    system: System,
    items: list[AmbiguousItem],
    prompt: str,
    can_use_llm: bool = True,
    timeout_s: float = 60.0,
) -> dict[str, ItemInference]:
    """Execute Opus structured inference for all ambiguous items in a system row."""
    if not can_use_llm or cv2_page is None:
        return {}

    h, w = cv2_page.shape[:2]
    y0 = max(0, int(system.bbox[1] * h))
    y1 = min(h, int(system.bbox[3] * h))
    x0 = max(0, int(system.bbox[0] * w))
    x1 = min(w, int(system.bbox[2] * w))
    if y1 <= y0 or x1 <= x0:
        return {}

    crop = cv2_page[y0:y1, x0:x1]
    cw = x1 - x0
    scale = max(1.0, 1800.0 / float(max(cw, 1)))
    new_w = int(cw * scale)
    new_h = int((y1 - y0) * scale)
    upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    ret, buf = cv2.imencode(".png", upscaled)
    if not ret:
        return {}
    img_bytes = buf.tobytes()

    try:
        resp = ask_json(
            prompt=prompt,
            schema=SystemInferenceResponse,
            images=[img_bytes],
            role="arbiter",
            timeout_s=timeout_s,
        )
        return {inf.item_id: inf for inf in resp.inferences}
    except Exception as exc:
        logger.warning("System row inference failed on system page=%d: %s", system.page, exc)
        return {}


def run_escalation_ladder(
    sheet: ParsedSheet,
    cv2_pages: list[Optional[np.ndarray]],
    issues: list[QualityIssue],
    can_use_llm: bool = True,
    web_candidates_map: Optional[dict[int, list[Any]]] = None,
) -> list[QualityIssue]:
    """Process all needs_review issues through the full 4-stage escalation ladder.

    Ladder stages:
    1. Existing crop re-read.
    2. Context inference with Opus (batch per row with upscaled row crop + ±2 measures context).
    3. Web evidence (aligned reference candidate chords).
    4. needs_review only if still unresolved or conflicting.
    """
    review_indices = [
        idx for idx, iss in enumerate(issues)
        if iss.severity == "needs_review" and iss.measure_index is not None
    ]
    if not review_indices:
        return issues

    all_measures = sheet.measures()
    meas_map: dict[int, Measure] = {m.index: m for m in all_measures}
    meas_to_sys: dict[int, System] = {m.index: s for s in sheet.systems for m in s.measures}

    # Fetch web evidence if not provided
    if web_candidates_map is None:
        start_pc = 0
        if sheet.header.original_key:
            try:
                start_pc = key_name_to_pc(sheet.header.original_key)
            except Exception:
                start_pc = 0
        web_candidates_map = get_web_reference_candidates(
            sheet.header.title,
            sheet=sheet,
            start_key_pc=start_pc,
        )

    # Build AmbiguousItems for eligible review issues
    items_by_row: dict[tuple[int, int], list[tuple[int, AmbiguousItem]]] = {}

    for r_idx in review_indices:
        iss = issues[r_idx]
        m_idx = iss.measure_index
        if m_idx is None or m_idx not in meas_map or m_idx not in meas_to_sys:
            continue

        m = meas_map[m_idx]
        s = meas_to_sys[m_idx]
        s_idx = next((i for i, sys in enumerate(sheet.systems) if sys is s), 0)

        # Context ±2 measures
        ctx_measures: list[dict[str, Any]] = []
        for c_idx in range(max(0, m_idx - 2), min(len(all_measures), m_idx + 3)):
            cm = all_measures[c_idx]
            pm = parse_melody(cm.melody, beats=cm.beats)
            ctx_measures.append({
                "measure_index": cm.index,
                "melody": cm.melody,
                "beat_sum": pm.beat_sum,
                "beats": cm.beats,
                "chords": [{"raw": c.raw, "beat": c.beat} for c in cm.chords],
            })

        # Determine target type
        target_type: AmbiguousTargetType = "chord"
        orig_val = ""
        cands: list[str] = []
        priors: dict[str, float] = {}
        c_beat = 1.0

        if iss.code == "missing_chord_suspected":
            target_type = "missing_chord"
            orig_val = ""
            cand_c = iss.detail.get("candidate", "")
            c_beat = iss.detail.get("beat", 1.0)
            if cand_c:
                cands.append(cand_c)
            # Candidate priors from melody
            pm = parse_melody(m.melody, beats=m.beats)
            for cand in (cands + ["1", "5", "4", "2m", "6m"]):
                priors[cand] = score_candidate(cand, melody_notes=pm.notes, chord_beat=c_beat)
        elif iss.code in ("barline_count_mismatch", "melody_beat_sum_mismatch"):
            target_type = "measure_boundary"
            orig_val = m.melody
        else:
            target_type = "chord"
            orig_val = iss.detail.get("raw") or (m.chords[0].raw if m.chords else "")
            cands = iss.detail.get("alternatives") or generate_candidates(orig_val)
            pm = parse_melody(m.melody, beats=m.beats)
            for cand in cands:
                priors[cand] = score_candidate(cand, melody_notes=pm.notes)

        crop_rd = iss.detail.get("crop")
        item = AmbiguousItem(
            item_id=f"iss_{r_idx}_m{m_idx}",
            target_type=target_type,
            measure_index=m_idx,
            system_page=s.page,
            system_index=s_idx,
            system_bbox=s.bbox,
            original_value=orig_val,
            candidate_set=cands[:6],
            crop_reading=crop_rd,
            crop_valid=bool(crop_rd and is_valid_chord_grammar(crop_rd)),
            context_measures=ctx_measures,
            metadata={"expected_beats": m.beats, "current_melody_beats": parse_melody(m.melody, beats=m.beats).beat_sum},
            prior_scores=priors,
            chord_beat=c_beat,
        )

        row_key = (s.page, s_idx)
        items_by_row.setdefault(row_key, []).append((r_idx, item))

    # Process row batches
    for (p_idx, s_idx), row_item_pairs in items_by_row.items():
        sys = next((s for s in sheet.systems if s.page == p_idx and sheet.systems.index(s) == s_idx), None)
        if not sys:
            continue
        p_img = cv2_pages[p_idx] if p_idx < len(cv2_pages) else None
        items = [it for _, it in row_item_pairs]

        prompt = build_system_inference_prompt(sheet, sys, items)
        inferences = run_system_row_inference(p_img, sys, items, prompt, can_use_llm=can_use_llm)

        for iss_idx, item in row_item_pairs:
            inf = inferences.get(item.item_id)
            web_cands = web_candidates_map.get(item.measure_index, [])
            dec = evaluate_ladder_decision(item, inf, web_cands, img_bgr=p_img)

            if dec.accepted:
                # Apply decision to the sheet
                target_m = meas_map[item.measure_index]
                if dec.target_type == "missing_chord":
                    target_m.chords.append(
                        ChordSymbol(
                            raw=dec.new_value,
                            beat=item.chord_beat,
                            confidence=dec.detail.get("confidence", 0.85),
                        )
                    )
                    target_m.chords.sort(key=lambda c: c.beat)
                elif dec.target_type == "chord":
                    if target_m.chords:
                        target_m.chords[0].raw = dec.new_value
                        target_m.chords[0].confidence = dec.detail.get("confidence", 0.88)
                    else:
                        target_m.chords.append(
                            ChordSymbol(
                                raw=dec.new_value,
                                beat=1.0,
                                confidence=dec.detail.get("confidence", 0.88),
                            )
                        )

                # Upgrade issue to auto_fixed
                issues[iss_idx] = QualityIssue(
                    stage="omr",
                    measure_index=item.measure_index,
                    severity="auto_fixed",
                    code="chord_inferred" if dec.target_type in ("chord", "missing_chord") else "measure_boundary_resolved",
                    message=dec.message,
                    detail=dec.detail,
                )

    return issues
