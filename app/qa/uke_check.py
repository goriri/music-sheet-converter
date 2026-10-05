"""Ukulele QA validator and auto-repair layer.

Invariants checked:
U1: Shape correctness (sounding pitch classes equal sounding chord tones)
U2: Playability (<= 4 fingers, span <= 3/2 frets, fingers 0 iff open, barre consistency, no muted strings in strum)
U3: Diagram (base_fret <= lowest fretted fret, highest fret - base_fret <= 4)
U4: Rhythm (sorted strokes, within [0, beats), non-overlapping, stroke at each chord beat, pluck strings)
U5: Sections (<= 2 distinct pattern_ids per section excluding last bar, pattern_id in legend)
U6: Key / Consistency (shape_key + capo == start_key, parallel lengths, capo 0..7)
U7: Difficulty (beginner <= 4 strokes, no chuck/pluck, dur >= 0.5; intermediate no plucks after beat 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Optional

from app.models import (
    Arrangement,
    Difficulty,
    Measure,
    MeasureArrangement,
    ParsedSheet,
    QualityIssue,
    ResolvedChord,
    SectionPlan,
    UkeMeasure,
    UkeShape,
    UkeStroke,
)
from app.theory.keys import canonical_key_for_pc, key_name_to_pc

logger = logging.getLogger(__name__)

# Standard ukulele tuning: G4=7, C4=0, E4=4, A4=9 (strings 4..1)
OPEN_PCS = (7, 0, 4, 9)


@dataclass
class Violation:
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


def _get_shape_sounding_pcs(shape: UkeShape, capo: int) -> set[int]:
    """Calculate pitch classes produced by a shape with a given capo."""
    sounding: set[int] = set()
    for s_idx, f in enumerate(shape.frets):
        if f >= 0:
            pc = (OPEN_PCS[s_idx] + f + capo) % 12
            sounding.add(pc)
    return sounding


def _get_required_pcs(
    chord: ResolvedChord,
    difficulty: Difficulty = "intermediate",
) -> set[int]:
    """Calculate essential pitch classes required for a chord derived from SOUNDING chord.pcs.

    Theory layer output chord.pcs is authoritative.
    - Required: root + 3rd/sus tone + 7th/6th (only if in pcs) + altered 5th (only if in pcs)
    - At beginner: a shape whose tones are subset of chord.pcs and contains root + 3rd/sus is valid.
    - At intermediate: allows dropping 9/11/13 extension when needed.
    - At advanced: root may be omitted for 9/11/13 chords.
    - Optional: perfect 5th.
    - 'add9' / '(9)' / '2' never imply a 7th.
    """
    root = chord.root_pc % 12
    chord_pcs = set(chord.pcs)
    intervals = {(pc - root) % 12 for pc in chord_pcs}

    req: set[int] = set()

    # 1. 3rd or sus tone
    third_or_sus: Optional[int] = None
    if 4 in intervals:
        third_or_sus = (root + 4) % 12
    elif 3 in intervals:
        third_or_sus = (root + 3) % 12
    elif 5 in intervals:
        third_or_sus = (root + 5) % 12
    elif 2 in intervals:
        third_or_sus = (root + 2) % 12

    if third_or_sus is not None:
        req.add(third_or_sus)
    elif 7 in intervals:
        req.add((root + 7) % 12)

    # 2. Root
    is_extension_chord = any(i in intervals for i in (1, 2, 5, 9)) and (10 in intervals or 11 in intervals)
    can_omit_root = (difficulty == "advanced" and is_extension_chord)
    if not can_omit_root and root in chord_pcs:
        req.add(root)

    # 3. Beginner simplification policy: root + 3rd/sus is sufficient!
    if difficulty == "beginner":
        return req

    # 4. 7th or 6th (ONLY if present in pcs!)
    if 11 in intervals:
        req.add((root + 11) % 12)
    elif 10 in intervals:
        req.add((root + 10) % 12)
    elif 9 in intervals and (3 in intervals or 4 in intervals or 6 in intervals):
        req.add((root + 9) % 12)

    # 5. Altered 5th (ONLY if present in pcs!)
    if 6 in intervals:
        req.add((root + 6) % 12)
    elif 8 in intervals and ("aug" in chord.quality.lower() or "7#5" in chord.quality.lower() or 4 not in intervals):
        req.add((root + 8) % 12)

    # 6. Extension / add tone (9/11/13)
    if difficulty == "advanced":
        for ext_int in (1, 2):
            if ext_int in intervals and (root + ext_int) % 12 != third_or_sus:
                req.add((root + ext_int) % 12)

    return req


def _check_u1_shape_correctness(
    chord: ResolvedChord,
    shape: UkeShape,
    capo: int,
    m_idx: int,
    difficulty: Difficulty = "intermediate",
) -> list[Violation]:
    """U1: Pitch classes from frets must equal sounding chord tones."""
    violations: list[Violation] = []
    sounding_pcs = _get_shape_sounding_pcs(shape, capo)

    # 1. Sounded pitch classes must be subset of chord tones (or slash bass)
    allowed_pcs = set(chord.pcs) | {chord.bass_pc % 12}
    non_chord = sounding_pcs - allowed_pcs
    if non_chord:
        violations.append(
            Violation(
                code="uke_shape_incorrect",
                message=f"第{m_idx + 1}小节指法图 {shape.name} 包含和弦外音: {sorted(non_chord)}",
                detail={"chord": chord.name, "shape": shape.name, "non_chord_pcs": list(non_chord)},
            )
        )

    # 2. Sounded pitch classes must contain required essential tones
    req_pcs = _get_required_pcs(chord, difficulty)
    missing = req_pcs - sounding_pcs
    if missing:
        violations.append(
            Violation(
                code="uke_shape_incorrect",
                message=f"第{m_idx + 1}小节指法图 {shape.name} 缺少必要音: {sorted(missing)}",
                detail={"chord": chord.name, "shape": shape.name, "missing_pcs": list(missing)},
            )
        )

    return violations


def _check_u2_playability(
    shape: UkeShape,
    technique: str,
    difficulty: Difficulty,
    m_idx: int,
) -> list[Violation]:
    """U2: Playability constraints (fingers, span, open consistency, barre)."""
    violations: list[Violation] = []

    # fingers 0 iff open / muted
    for s_idx, (f, fing) in enumerate(zip(shape.frets, shape.fingers)):
        if f <= 0 and fing != 0:
            violations.append(
                Violation(
                    code="uke_playability_invalid",
                    message=f"第{m_idx + 1}小节指法图 {shape.name} 空弦或静音弦标记了手指号: 弦{4-s_idx}, 手指{fing}",
                    detail={"string": 4 - s_idx, "fret": f, "finger": fing},
                )
            )
        elif f > 0 and fing == 0:
            violations.append(
                Violation(
                    code="uke_playability_invalid",
                    message=f"第{m_idx + 1}小节指法图 {shape.name} 按弦品未标记手指号: 弦{4-s_idx}, 品{f}",
                    detail={"string": 4 - s_idx, "fret": f},
                )
            )

    # Barre consistency
    pos_fret_1 = [shape.frets[i] for i in range(4) if shape.fingers[i] == 1 and shape.frets[i] > 0]
    if shape.barre is not None:
        if not pos_fret_1 or min(pos_fret_1) != shape.barre:
            violations.append(
                Violation(
                    code="uke_playability_invalid",
                    message=f"第{m_idx + 1}小节指法图 {shape.name} 大横按品位 {shape.barre} 与食指最低品不符",
                    detail={"barre": shape.barre, "fret_1": pos_fret_1},
                )
            )

    # Finger count <= 4
    used_fingers = set(f for f in shape.fingers if f > 0)
    if len(used_fingers) > 4:
        violations.append(
            Violation(
                code="uke_playability_invalid",
                message=f"第{m_idx + 1}小节指法图 {shape.name} 手指数量超过4指限制: {used_fingers}",
                detail={"fingers": shape.fingers},
            )
        )

    # Span
    fretted = [f for f in shape.frets if f > 0]
    if fretted:
        span = max(fretted) - min(fretted)
        max_allowed_span = 2 if difficulty == "beginner" else 3
        if span > max_allowed_span:
            violations.append(
                Violation(
                    code="uke_playability_invalid",
                    message=f"第{m_idx + 1}小节指法图 {shape.name} 品位跨度 {span} 超过{difficulty}难度限制 {max_allowed_span}",
                    detail={"span": span, "limit": max_allowed_span},
                )
            )

    # No muted strings for strummed shapes
    if technique == "strum" and any(f == -1 for f in shape.frets):
        violations.append(
            Violation(
                code="uke_playability_invalid",
                message=f"第{m_idx + 1}小节扫弦伴奏指法图 {shape.name} 包含静音弦",
                detail={"frets": shape.frets},
            )
        )

    return violations


def _check_u3_diagram(shape: UkeShape, m_idx: int) -> list[Violation]:
    """U3: Diagram base fret and range validity."""
    violations: list[Violation] = []
    fretted = [f for f in shape.frets if f > 0]
    if fretted:
        min_fret = min(fretted)
        max_fret = max(fretted)
        if shape.base_fret > min_fret:
            violations.append(
                Violation(
                    code="uke_diagram_invalid",
                    message=f"第{m_idx + 1}小节指法图 {shape.name} 标注起始品 {shape.base_fret} 高于最低按弦品 {min_fret}",
                    detail={"base_fret": shape.base_fret, "min_fret": min_fret},
                )
            )
        if max_fret - shape.base_fret > 4:
            violations.append(
                Violation(
                    code="uke_diagram_invalid",
                    message=f"第{m_idx + 1}小节指法图 {shape.name} 最高品位 {max_fret} 超出4品显示范围 (base={shape.base_fret})",
                    detail={"base_fret": shape.base_fret, "max_fret": max_fret},
                )
            )
    return violations


def _check_u4_rhythm(
    uke_m: UkeMeasure,
    m_arr: MeasureArrangement,
    expected_beats: float,
    m_idx: int,
) -> list[Violation]:
    """U4: Stroke sorting, timing bounds, non-overlap, and chord beat alignment."""
    violations: list[Violation] = []
    strokes = uke_m.strokes

    # 1. Sorted by onset
    for i in range(len(strokes) - 1):
        if strokes[i].onset > strokes[i + 1].onset:
            violations.append(
                Violation(
                    code="uke_rhythm_invalid",
                    message=f"第{m_idx + 1}小节扫弦节奏未按时间升序排列: {strokes[i].onset} > {strokes[i+1].onset}",
                    detail={"index": i},
                )
            )

    # 2. Bounds and non-overlapping
    for i, s in enumerate(strokes):
        if s.onset < 0.0 or s.onset >= expected_beats:
            violations.append(
                Violation(
                    code="uke_rhythm_invalid",
                    message=f"第{m_idx + 1}小节扫弦起始点 {s.onset} 超出拍数范围 [0, {expected_beats})",
                    detail={"onset": s.onset},
                )
            )
        if s.onset + s.duration > expected_beats + 1e-6:
            violations.append(
                Violation(
                    code="uke_rhythm_invalid",
                    message=f"第{m_idx + 1}小节扫弦持续时间超出小节总拍数: {s.onset + s.duration} > {expected_beats}",
                    detail={"onset": s.onset, "duration": s.duration},
                )
            )
        if i + 1 < len(strokes):
            if s.onset + s.duration > strokes[i + 1].onset + 1e-6:
                violations.append(
                    Violation(
                        code="uke_rhythm_invalid",
                        message=f"第{m_idx + 1}小节扫弦事件重叠: {s.onset}+{s.duration} > {strokes[i+1].onset}",
                        detail={"onset1": s.onset, "dur1": s.duration, "onset2": strokes[i+1].onset},
                    )
                )

        # Pluck string validations
        if s.kind == "pluck":
            if not s.strings or not set(s.strings).issubset({1, 2, 3, 4}):
                violations.append(
                    Violation(
                        code="uke_rhythm_invalid",
                        message=f"第{m_idx + 1}小节勾弦指令弦号无效: {s.strings}",
                        detail={"strings": s.strings},
                    )
                )
        else:
            if s.strings:
                violations.append(
                    Violation(
                        code="uke_rhythm_invalid",
                        message=f"第{m_idx + 1}小节非勾弦指令包含了弦号: {s.kind}, {s.strings}",
                        detail={"kind": s.kind, "strings": s.strings},
                    )
                )

    # 3. Chord onset synchronization
    if uke_m.technique not in ("hold", "rest"):
        stroke_onsets = [s.onset for s in strokes]
        for c in m_arr.chords:
            exp_onset = c.beat - 1.0
            if not any(abs(so - exp_onset) < 1e-4 for so in stroke_onsets):
                violations.append(
                    Violation(
                        code="uke_rhythm_invalid",
                        message=f"第{m_idx + 1}小节和弦 {c.name} 在其拍位 (onset={exp_onset}) 处缺少对应弹奏动作",
                        detail={"chord": c.name, "beat": c.beat, "expected_onset": exp_onset},
                    )
                )

    return violations


def _check_u5_sections(arr: Arrangement) -> list[Violation]:
    """U5: In a section instance, <= 2 distinct pattern_ids (excl. last bar); all patterns in legend."""
    violations: list[Violation] = []

    # 1. Legend coverage
    for m_idx, uke_m in enumerate(arr.uke_measures):
        if uke_m.pattern_id and uke_m.pattern_id not in arr.pattern_legend:
            violations.append(
                Violation(
                    code="uke_section_pattern_inconsistent",
                    message=f"第{m_idx + 1}小节节奏型 {uke_m.pattern_id} 未在节奏图例 pattern_legend 中登记",
                    detail={"pattern_id": uke_m.pattern_id},
                )
            )

    # 2. Distinct patterns within section instance
    if arr.sections:
        # Group measures by contiguous section instance
        curr_id = None
        instance_ms: list[int] = []
        instances: list[list[int]] = []

        for p in arr.sections:
            if p.section_id != curr_id:
                if instance_ms:
                    instances.append(instance_ms)
                curr_id = p.section_id
                instance_ms = [p.measure_index]
            else:
                instance_ms.append(p.measure_index)
        if instance_ms:
            instances.append(instance_ms)

        for inst in instances:
            # Exclude last bar
            eval_ms = inst[:-1] if len(inst) > 1 else inst
            patterns = {
                arr.uke_measures[m].pattern_id
                for m in eval_ms
                if m < len(arr.uke_measures) and arr.uke_measures[m].pattern_id
            }
            if len(patterns) > 2:
                violations.append(
                    Violation(
                        code="uke_section_pattern_inconsistent",
                        message=f"段落 {arr.sections[inst[0]].section_id} 内部使用了超过2种不同节奏型: {sorted(patterns)}",
                        detail={"section_id": arr.sections[inst[0]].section_id, "patterns": list(patterns)},
                    )
                )

    return violations


def _check_u6_key_consistency(arr: Arrangement) -> list[Violation]:
    """U6: Consistency of keys, capo, lengths, and shape dictionary."""
    violations: list[Violation] = []

    # Capo range 0..7
    if not (0 <= arr.capo <= 7):
        violations.append(
            Violation(
                code="uke_key_or_count_inconsistent",
                message=f"变调夹品位 {arr.capo} 超出有效范围 0..7",
                detail={"capo": arr.capo},
            )
        )

    # Key math: shape_key + capo == start_key (pitch class)
    if arr.shape_key is None:
        violations.append(
            Violation(
                code="uke_key_or_count_inconsistent",
                message="尤克里里伴奏未指定按弦调 shape_key",
                detail={},
            )
        )
    else:
        shape_pc = key_name_to_pc(arr.shape_key)
        start_pc = key_name_to_pc(arr.start_key)
        if (shape_pc + arr.capo) % 12 != start_pc:
            violations.append(
                Violation(
                    code="uke_key_or_count_inconsistent",
                    message=f"按弦调 {arr.shape_key} + 变调夹 {arr.capo} 与目标调 {arr.start_key} 音高不相符: (({shape_pc}+{arr.capo})%12 != {start_pc})",
                    detail={"shape_key": arr.shape_key, "capo": arr.capo, "start_key": arr.start_key},
                )
            )

    # Length consistency
    if len(arr.uke_measures) != len(arr.measures):
        violations.append(
            Violation(
                code="uke_key_or_count_inconsistent",
                message=f"尤克里里小节数 {len(arr.uke_measures)} 与总小节数 {len(arr.measures)} 不一致",
                detail={"uke_measures": len(arr.uke_measures), "measures": len(arr.measures)},
            )
        )

    # Shapes count per measure and presence in dictionary
    for i in range(min(len(arr.uke_measures), len(arr.measures))):
        u_m = arr.uke_measures[i]
        m_arr = arr.measures[i]
        if len(u_m.shapes) != len(m_arr.chords):
            violations.append(
                Violation(
                    code="uke_key_or_count_inconsistent",
                    message=f"第{i + 1}小节指法图数量 {len(u_m.shapes)} 与和弦数量 {len(m_arr.chords)} 不一致",
                    detail={"shapes": len(u_m.shapes), "chords": len(m_arr.chords)},
                )
            )
        for sh_name in u_m.shapes:
            if sh_name not in arr.uke_shapes:
                violations.append(
                    Violation(
                        code="uke_key_or_count_inconsistent",
                        message=f"第{i + 1}小节使用的指法图 {sh_name} 未在 uke_shapes 字典中提供",
                        detail={"shape_name": sh_name},
                    )
                )

    return violations


def _check_u7_difficulty(
    uke_m: UkeMeasure,
    expected_beats: float,
    difficulty: Difficulty,
    m_idx: int,
) -> list[Violation]:
    """U7: Difficulty constraints on stroke density, technique, and duration."""
    violations: list[Violation] = []
    strokes = uke_m.strokes

    if difficulty == "beginner":
        max_strokes = int(math.ceil(expected_beats))
        if len(strokes) > max_strokes:
            violations.append(
                Violation(
                    code="uke_difficulty_violation",
                    message=f"第{m_idx + 1}小节初级难度扫弦动作数量 {len(strokes)} 超过每小节 {max_strokes} 拍上限",
                    detail={"strokes_count": len(strokes), "limit": max_strokes},
                )
            )
        for s in strokes:
            if s.kind == "chuck":
                violations.append(
                    Violation(
                        code="uke_difficulty_violation",
                        message=f"第{m_idx + 1}小节初级难度禁止使用切音 (chuck)",
                        detail={"onset": s.onset},
                    )
                )
            if s.kind == "pluck":
                violations.append(
                    Violation(
                        code="uke_difficulty_violation",
                        message=f"第{m_idx + 1}小节初级难度禁止使用勾弦 (pluck)",
                        detail={"onset": s.onset},
                    )
                )
            if s.duration < 0.5 - 1e-6:
                violations.append(
                    Violation(
                        code="uke_difficulty_violation",
                        message=f"第{m_idx + 1}小节初级难度音符时值 {s.duration} 短于0.5拍 (八分音符)",
                        detail={"onset": s.onset, "duration": s.duration},
                    )
                )

    elif difficulty == "intermediate":
        for s in strokes:
            if s.kind == "pluck" and s.onset > 1e-4:
                violations.append(
                    Violation(
                        code="uke_difficulty_violation",
                        message=f"第{m_idx + 1}小节中级难度仅允许在第1拍出现勾弦/抓弦，禁止在第{s.onset + 1}拍勾弦",
                        detail={"onset": s.onset},
                    )
                )

    return violations


def validate_uke(sheet: ParsedSheet, arr: Arrangement) -> list[Violation]:
    """Validate all ukulele arrangement invariants (U1..U7).

    Returns:
        List of Violation objects (empty if 100% compliant).
    """
    violations: list[Violation] = []

    # U6: Key and consistency
    violations.extend(_check_u6_key_consistency(arr))

    # U5: Section pattern consistency
    violations.extend(_check_u5_sections(arr))

    measures = sheet.measures()
    sheet_m_by_idx = {m.index: m for m in measures}

    for i in range(min(len(arr.uke_measures), len(arr.measures))):
        uke_m = arr.uke_measures[i]
        m_arr = arr.measures[i]
        m_sheet = sheet_m_by_idx.get(m_arr.measure_index)
        expected_beats = m_sheet.beats if m_sheet else 4.0

        # U4: Rhythm
        violations.extend(_check_u4_rhythm(uke_m, m_arr, expected_beats, i))

        # U7: Difficulty
        violations.extend(_check_u7_difficulty(uke_m, expected_beats, arr.difficulty, i))

        # Shapes validations: U1, U2, U3
        for sh_idx, sh_name in enumerate(uke_m.shapes):
            if sh_name in arr.uke_shapes:
                shape = arr.uke_shapes[sh_name]

                # U1: Shape correctness
                if sh_idx < len(m_arr.chords):
                    chord = m_arr.chords[sh_idx]
                    violations.extend(_check_u1_shape_correctness(chord, shape, arr.capo, i, arr.difficulty))

                # U2: Playability
                violations.extend(_check_u2_playability(shape, uke_m.technique, arr.difficulty, i))

                # U3: Diagram
                violations.extend(_check_u3_diagram(shape, i))

    return violations


def _make_tiled_fallback_strokes(
    chords: list[ResolvedChord],
    expected_beats: float,
) -> list[UkeStroke]:
    """Generate simple down-stroke rhythm aligned with chords."""
    if not chords:
        return [UkeStroke(onset=0.0, duration=expected_beats, kind="down", strings=[])]

    strokes: list[UkeStroke] = []
    num_c = len(chords)
    for idx, c in enumerate(chords):
        onset = max(0.0, min(expected_beats - 0.25, c.beat - 1.0))
        next_onset = max(onset + 0.25, min(expected_beats, chords[idx + 1].beat - 1.0)) if idx + 1 < num_c else expected_beats
        dur = round(next_onset - onset, 4)
        strokes.append(UkeStroke(onset=onset, duration=dur, kind="down", strings=[]))
    return strokes


def check_and_repair_uke(sheet: ParsedSheet, arr: Arrangement) -> Arrangement:
    """Validate ukulele arrangement, repair any defect safely, and record QualityIssue logs.

    Repairs are recorded as QualityIssue with severity='auto_fixed' or 'info'. Never raises.
    """
    try:
        return _repair_uke_impl(sheet, arr)
    except Exception as exc:
        logger.error("check_and_repair_uke encountered unexpected error: %s; returning original", exc, exc_info=True)
        return arr


def _repair_uke_impl(sheet: ParsedSheet, arr: Arrangement) -> Arrangement:
    # Lazy import candidate_shapes
    try:
        from app.arrange.uke_shapes import candidate_shapes
    except ImportError:
        candidate_shapes = None

    measures = sheet.measures()
    sheet_m_by_idx = {m.index: m for m in measures}

    # 1. Repair U6 key and consistency
    arr.capo = max(0, min(7, arr.capo))
    start_pc = key_name_to_pc(arr.start_key)
    expected_shape_pc = (start_pc - arr.capo) % 12
    expected_shape_key = canonical_key_for_pc(expected_shape_pc)

    if arr.shape_key is None or key_name_to_pc(arr.shape_key) != expected_shape_pc:
        old_k = arr.shape_key
        arr.shape_key = expected_shape_key
        arr.issues.append(
            QualityIssue(
                stage="arrange",
                severity="auto_fixed",
                code="uke_shape_key_repaired",
                message=f"已自动修正按弦调为 {expected_shape_key} (原: {old_k})",
                detail={"old_shape_key": old_k, "new_shape_key": expected_shape_key},
            )
        )

    # Pad or trim uke_measures to match measures length
    diff_len = len(arr.measures) - len(arr.uke_measures)
    if diff_len > 0:
        for k in range(len(arr.uke_measures), len(arr.measures)):
            m_arr = arr.measures[k]
            exp_beats = sheet_m_by_idx[m_arr.measure_index].beats if m_arr.measure_index in sheet_m_by_idx else 4.0
            fallback_strokes = _make_tiled_fallback_strokes(m_arr.chords, exp_beats)
            fallback_shapes = [c.name for c in m_arr.chords]
            arr.uke_measures.append(
                UkeMeasure(
                    measure_index=m_arr.measure_index,
                    shapes=fallback_shapes,
                    technique="strum",
                    pattern_id="fallback_strum",
                    strokes=fallback_strokes,
                )
            )
        arr.issues.append(
            QualityIssue(
                stage="arrange",
                severity="auto_fixed",
                code="uke_measure_count_repaired",
                message=f"已补齐缺失的 {diff_len} 个伴奏小节",
                detail={"added": diff_len},
            )
        )
    elif diff_len < 0:
        arr.uke_measures = arr.uke_measures[: len(arr.measures)]
        arr.issues.append(
            QualityIssue(
                stage="arrange",
                severity="auto_fixed",
                code="uke_measure_count_repaired",
                message=f"已移除多余的 {-diff_len} 个伴奏小节",
                detail={"removed": -diff_len},
            )
        )

    # Align shapes count per measure with chords count
    for i in range(len(arr.measures)):
        m_arr = arr.measures[i]
        u_m = arr.uke_measures[i]
        if len(u_m.shapes) != len(m_arr.chords):
            u_m.shapes = [c.name for c in m_arr.chords]
            arr.issues.append(
                QualityIssue(
                    stage="arrange",
                    severity="auto_fixed",
                    measure_index=m_arr.measure_index,
                    code="uke_measure_shapes_repaired",
                    message=f"第{i + 1}小节指法图对应数量已自动与和弦对齐",
                    detail={},
                )
            )

    # 2. Repair U5 pattern legend and consistency
    for u_m in arr.uke_measures:
        if u_m.pattern_id and u_m.pattern_id not in arr.pattern_legend:
            arr.pattern_legend[u_m.pattern_id] = "↓  ↓↑  ↓↑"
            arr.issues.append(
                QualityIssue(
                    stage="arrange",
                    severity="auto_fixed",
                    measure_index=u_m.measure_index,
                    code="uke_pattern_legend_repaired",
                    message=f"节奏型 {u_m.pattern_id} 已自动补充登记到图例",
                    detail={"pattern_id": u_m.pattern_id},
                )
            )

    if arr.sections:
        curr_id = None
        instance_ms: list[int] = []
        instances: list[list[int]] = []
        for p in arr.sections:
            if p.section_id != curr_id:
                if instance_ms:
                    instances.append(instance_ms)
                curr_id = p.section_id
                instance_ms = [p.measure_index]
            else:
                instance_ms.append(p.measure_index)
        if instance_ms:
            instances.append(instance_ms)

        for inst in instances:
            eval_ms = inst[:-1] if len(inst) > 1 else inst
            pats = [arr.uke_measures[m].pattern_id for m in eval_ms if m < len(arr.uke_measures) and arr.uke_measures[m].pattern_id]
            unique_pats: list[str] = []
            for p in pats:
                if p not in unique_pats:
                    unique_pats.append(p)
            if len(unique_pats) > 2:
                primary_pats = unique_pats[:2]
                for m in eval_ms:
                    if arr.uke_measures[m].pattern_id not in primary_pats:
                        arr.uke_measures[m].pattern_id = primary_pats[0]
                arr.issues.append(
                    QualityIssue(
                        stage="arrange",
                        severity="auto_fixed",
                        measure_index=inst[0],
                        code="uke_section_pattern_repaired",
                        message=f"段落 {arr.sections[inst[0]].section_id} 内部节奏型过多，已规范化",
                        detail={"allowed": primary_pats},
                    )
                )

    # 3. Repair U1, U2, U3 shapes
    for i in range(len(arr.measures)):
        m_arr = arr.measures[i]
        u_m = arr.uke_measures[i]
        for sh_idx, c in enumerate(m_arr.chords):
            sh_name = u_m.shapes[sh_idx] if sh_idx < len(u_m.shapes) else c.name
            shape = arr.uke_shapes.get(sh_name)

            needs_shape_repair = False
            if shape is None:
                needs_shape_repair = True
            else:
                v1 = _check_u1_shape_correctness(c, shape, arr.capo, i, arr.difficulty)
                v2 = _check_u2_playability(shape, u_m.technique, arr.difficulty, i)
                v3 = _check_u3_diagram(shape, i)
                if v1 or v2 or v3:
                    needs_shape_repair = True

            if needs_shape_repair:
                # Find valid candidate shape
                repaired_shape = None
                if candidate_shapes is not None:
                    # Calculate shape chord root in shape key: (sounding_root - capo) % 12
                    shape_root_pc = (c.root_pc - arr.capo) % 12
                    cands = candidate_shapes(shape_root_pc, c.quality, difficulty=arr.difficulty)
                    for cand in cands:
                        if not _check_u1_shape_correctness(c, cand, arr.capo, i, arr.difficulty) and not _check_u2_playability(
                            cand, u_m.technique, arr.difficulty, i
                        ) and not _check_u3_diagram(cand, i):
                            repaired_shape = cand
                            break

                if repaired_shape is not None:
                    arr.uke_shapes[sh_name] = repaired_shape
                    arr.issues.append(
                        QualityIssue(
                            stage="arrange",
                            severity="auto_fixed",
                            measure_index=i,
                            code="uke_shape_repaired",
                            message=f"第{i + 1}小节指法图 {sh_name} 已自动修复为最佳合规按法",
                            detail={"chord": c.name, "frets": repaired_shape.frets},
                        )
                    )
                else:
                    # Log info if candidate shape generation unavailable
                    logger.warning("Could not auto-repair shape %s for chord %s", sh_name, c.name)

            # Check beginner barre info
            final_shape = arr.uke_shapes.get(sh_name)
            if final_shape and arr.difficulty == "beginner" and final_shape.barre is not None:
                arr.issues.append(
                    QualityIssue(
                        stage="arrange",
                        severity="info",
                        measure_index=i,
                        code="uke_beginner_barre",
                        message=f"第{i + 1}小节指法图 {sh_name} 包含横按，建议初学者使用替代按法或变调夹",
                        detail={"barre": final_shape.barre},
                    )
                )

    # 4. Repair U4 and U7 rhythm and difficulty
    for i in range(len(arr.measures)):
        m_arr = arr.measures[i]
        u_m = arr.uke_measures[i]
        m_sheet = sheet_m_by_idx.get(m_arr.measure_index)
        exp_beats = m_sheet.beats if m_sheet else 4.0

        rhythm_viol = _check_u4_rhythm(u_m, m_arr, exp_beats, i)
        diff_viol = _check_u7_difficulty(u_m, exp_beats, arr.difficulty, i)

        if rhythm_viol or diff_viol:
            # Replace with safe tiled rhythm
            u_m.strokes = _make_tiled_fallback_strokes(m_arr.chords, exp_beats)
            u_m.technique = "strum"
            arr.issues.append(
                QualityIssue(
                    stage="arrange",
                    severity="auto_fixed",
                    measure_index=i,
                    code="uke_rhythm_repaired",
                    message=f"第{i + 1}小节尤克里里伴奏节奏已自动重置为标准下扫伴奏",
                    detail={"original_violations": [v.code for v in (rhythm_viol + diff_viol)]},
                )
            )

    return arr
