"""Arrangement verification and auto-repair layer for piano accompaniments."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel
from app.models import (
    Arrangement,
    Difficulty,
    Event,
    Measure,
    MeasureArrangement,
    Note,
    ParsedSheet,
    QualityIssue,
    ResolvedChord,
)
from app.theory.keys import canonical_key_for_pc, key_name_to_pc, spell
from app.qa.fallback import safe_measure
from app.qa.music21_ref import cross_check_chord_pcs, get_chord_reference

try:
    from app.qa.llm import ask_json
except ImportError:
    ask_json = None

# Constraints per PLAN & prompt
RH_MIN_MIDI: int = 64  # E4
RH_MAX_MIDI: int = 81  # A5
LH_MIN_MIDI: int = 36  # C2
LH_MAX_MIDI: int = 64  # E4


class ArrangementQAError(Exception):
    """Raised when an arrangement fails validation even after safe fallback."""


@dataclass
class Violation:
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


# Schemas for optional LLM review
class FlaggedMeasure(BaseModel):
    measure_index: int
    reason: str


class LLMReviewResponse(BaseModel):
    suspicious_measures: list[FlaggedMeasure] = []


def _pitch_to_name(midi: int, tonic_pc: int = 0) -> str:
    """Format MIDI pitch as note name with octave, e.g. 60 -> C4."""
    pc = midi % 12
    octave = (midi // 12) - 1
    spelled = spell(pc, tonic_pc)
    return f"{spelled}{octave}"


def build_compact_score_text(arrangement: Arrangement) -> str:
    """Build a compact human/LLM-readable text score of the arrangement."""
    lines: list[str] = [
        f"Song Arrangement: {arrangement.start_key} | {arrangement.difficulty} | {arrangement.style}",
        "=" * 60,
    ]
    for m in arrangement.measures:
        chord_str = ", ".join(f"{c.name}@{c.beat}" for c in m.chords)
        lines.append(f"Measure {m.measure_index + 1} (Key: {m.key_name}): Chords: [{chord_str}]")

        # RH
        rh_parts: list[str] = []
        for e in m.rh:
            if not e.notes:
                rh_parts.append(f"rest({e.duration})")
            else:
                n_str = "+".join(f"{_pitch_to_name(n.midi, m.tonic_pc)}:{n.finger}" for n in e.notes)
                rh_parts.append(f"[{e.onset}..{e.onset + e.duration}] {n_str}")
        lines.append(f"  RH: {' '.join(rh_parts)}")

        # LH
        lh_parts: list[str] = []
        for e in m.lh:
            if not e.notes:
                lh_parts.append(f"rest({e.duration})")
            else:
                n_str = "+".join(f"{_pitch_to_name(n.midi, m.tonic_pc)}:{n.finger}" for n in e.notes)
                lh_parts.append(f"[{e.onset}..{e.onset + e.duration}] {n_str}")
        lines.append(f"  LH: {' '.join(lh_parts)}")

    return "\n".join(lines)


def _check_key_in_force(
    m_arr: MeasureArrangement,
    expected_tonic_pc: int,
) -> list[Violation]:
    """Validate that key in force matches expected tonic shifted by key changes."""
    violations: list[Violation] = []
    if m_arr.tonic_pc != expected_tonic_pc:
        violations.append(
            Violation(
                code="key_mismatch",
                message=f"第{m_arr.measure_index + 1}小节调性不符：实际={m_arr.tonic_pc}，期望={expected_tonic_pc}",
                detail={"actual_tonic": m_arr.tonic_pc, "expected_tonic": expected_tonic_pc},
            )
        )
    return violations


def _check_tiling_and_grid(
    events: list[Event],
    hand: str,
    measure_index: int,
    expected_beats: float,
) -> list[Violation]:
    """Validate exact tiling and 1/16 grid onsets and durations."""
    violations: list[Violation] = []
    total_dur = round(sum(e.duration for e in events), 4)
    exp_dur = round(expected_beats, 4)

    if total_dur != exp_dur:
        violations.append(
            Violation(
                code="tiling_mismatch",
                message=f"第{measure_index + 1}小节{hand}时值总和不等于小节拍数：{total_dur} != {exp_dur}",
                detail={"hand": hand, "total_dur": total_dur, "expected_dur": exp_dur},
            )
        )

    curr_time = 0.0
    for idx, e in enumerate(events):
        # 1/16 grid check (0.25 beat)
        if round(e.onset % 0.25, 4) not in (0.0, 0.25):
            violations.append(
                Violation(
                    code="grid_alignment",
                    message=f"第{measure_index + 1}小节{hand}音符起始点未对齐1/16网格：{e.onset}",
                    detail={"hand": hand, "onset": e.onset},
                )
            )
        if round(e.duration % 0.25, 4) not in (0.0, 0.25):
            violations.append(
                Violation(
                    code="grid_alignment",
                    message=f"第{measure_index + 1}小节{hand}音符时值未对齐1/16网格：{e.duration}",
                    detail={"hand": hand, "duration": e.duration},
                )
            )

        # Onset continuity
        if abs(e.onset - curr_time) > 1e-4:
            violations.append(
                Violation(
                    code="onset_discontinuity",
                    message=f"第{measure_index + 1}小节{hand}音符起始点不连续：{e.onset} != {curr_time}",
                    detail={"hand": hand, "onset": e.onset, "expected": curr_time},
                )
            )
        curr_time = round(curr_time + e.duration, 4)

    return violations


def _check_chord_voicing_presence(
    m_arr: MeasureArrangement,
    measure: Measure,
) -> list[Violation]:
    """Validate that every chord is voiced at its beat with root or bass in RH or LH."""
    violations: list[Violation] = []
    num_chords = len(m_arr.chords)
    measure_beats = measure.beats if measure else 4.0

    for i, chord in enumerate(m_arr.chords):
        c_onset = round(chord.beat - 1.0, 4)
        c_onset = max(0.0, min(measure_beats - 0.25, c_onset))
        next_onset = round(m_arr.chords[i + 1].beat - 1.0, 4) if i + 1 < num_chords else measure_beats
        next_onset = max(c_onset + 0.25, min(measure_beats, next_onset))

        # Look for attacks within [c_onset, next_onset)
        voiced = False
        valid_pcs = {chord.root_pc % 12, chord.bass_pc % 12}

        # Check all RH and LH events within chord span
        all_events = m_arr.rh + m_arr.lh
        for ev in all_events:
            if not ev.notes:
                continue
            if c_onset <= round(ev.onset, 4) < next_onset:
                ev_pcs = {n.midi % 12 for n in ev.notes}
                if any(p in valid_pcs for p in ev_pcs):
                    voiced = True
                    break

        if not voiced:
            violations.append(
                Violation(
                    code="chord_not_voiced",
                    message=f"第{m_arr.measure_index + 1}小节和弦 {chord.name} 在其拍位 {chord.beat} 未被弹奏（缺少根音或低音）",
                    detail={"chord": chord.name, "beat": chord.beat, "onset": c_onset},
                )
            )

    return violations


def _check_rh_defining_tones(
    m_arr: MeasureArrangement,
    difficulty: Difficulty,
) -> list[Violation]:
    """Validate that RH chord attacks contain required defining tones."""
    violations: list[Violation] = []
    num_chords = len(m_arr.chords)
    measure_beats = 4.0

    for i, chord in enumerate(m_arr.chords):
        c_onset = round(chord.beat - 1.0, 4)
        next_onset = round(m_arr.chords[i + 1].beat - 1.0, 4) if i + 1 < num_chords else measure_beats

        try:
            _, _, _, def_pcs = get_chord_reference(chord)
        except Exception:
            def_pcs = [(chord.root_pc + (3 if "m" in chord.quality else 4)) % 12]

        # For beginner: 7th/9th reduced to triad; colour tones: 4th for sus, b5 for m7b5/dim
        if difficulty == "beginner":
            req_defs = [def_pcs[0]] if def_pcs else []
            if chord.quality in ("m7b5", "dim"):
                b5_pc = (chord.root_pc + 6) % 12
                if b5_pc not in req_defs:
                    req_defs.append(b5_pc)
        else:
            req_defs = def_pcs

        if not req_defs:
            continue

        # Check RH attacks within the chord's span
        for ev in m_arr.rh:
            if not ev.notes:
                continue
            # A chord attack is either an event at chord onset, or any multi-note chord in the span
            ev_onset = round(ev.onset, 4)
            is_chord_attack = (c_onset <= ev_onset < next_onset) and (
                len(ev.notes) >= 2 or abs(ev_onset - c_onset) < 1e-4
            )
            if is_chord_attack:
                ev_pcs = {n.midi % 12 for n in ev.notes}
                missing = [p for p in req_defs if p not in ev_pcs]
                if missing:
                    spelled_missing = [spell(p, m_arr.tonic_pc) for p in missing]
                    violations.append(
                        Violation(
                            code="missing_defining_tone",
                            message=f"第{m_arr.measure_index + 1}小节右手和弦 {chord.name} 缺少特征音 {', '.join(spelled_missing)}",
                            detail={"chord": chord.name, "missing_pcs": missing, "onset": ev.onset},
                        )
                    )

    return violations


def _is_valid_passing_tone(
    note: Note,
    onset: float,
    duration: float,
    next_bass_pc: Optional[int],
) -> bool:
    """Check if a non-chord LH tone qualifies as a weak-subdivision passing tone."""
    # 1. Onset not on an integer beat
    if round(onset % 1.0, 4) in (0.0, 1.0):
        return False

    # 2. Duration <= 0.5 (eighth note or shorter)
    if duration > 0.5001:
        return False

    # 3. Stepwise approach into next chord's bass
    if next_bass_pc is None:
        return False

    diff = (note.midi % 12 - next_bass_pc % 12) % 12
    if diff in (1, 2, 10, 11):
        return True

    return False


def _check_lh_chord_tones(
    m_arr: MeasureArrangement,
    next_measure_chord: Optional[ResolvedChord],
) -> list[Violation]:
    """Validate that LH notes are chord tones or valid weak-subdivision passing tones."""
    violations: list[Violation] = []
    num_chords = len(m_arr.chords)
    measure_beats = 4.0

    for i, chord in enumerate(m_arr.chords):
        c_onset = round(chord.beat - 1.0, 4)
        next_onset = round(m_arr.chords[i + 1].beat - 1.0, 4) if i + 1 < num_chords else measure_beats

        if i + 1 < num_chords:
            next_bass = m_arr.chords[i + 1].bass_pc
        elif next_measure_chord:
            next_bass = next_measure_chord.bass_pc
        else:
            next_bass = chord.bass_pc

        allowed_pcs = set(chord.pcs) | {chord.bass_pc % 12}

        for ev in m_arr.lh:
            if not ev.notes:
                continue
            ev_onset = round(ev.onset, 4)
            if c_onset <= ev_onset < next_onset:
                for n in ev.notes:
                    pc = n.midi % 12
                    if pc not in allowed_pcs:
                        if not _is_valid_passing_tone(n, ev_onset, ev.duration, next_bass):
                            bad_name = spell(pc, m_arr.tonic_pc)
                            violations.append(
                                Violation(
                                    code="lh_non_chord_tone",
                                    message=f"第{m_arr.measure_index + 1}小节左手含和弦外音 {bad_name}",
                                    detail={"pitch": n.midi, "pc": pc, "onset": ev.onset, "chord": chord.name},
                                )
                            )

    return violations


def _check_fill_diatonic(
    m_arr: MeasureArrangement,
    measure: Measure,
) -> list[Violation]:
    """Validate that fill and passing figures are diatonic to the key in force."""
    violations: list[Violation] = []
    diatonic_pcs = {(m_arr.tonic_pc + s) % 12 for s in (0, 2, 4, 5, 7, 9, 11)}

    # Only single-note melodic fill runs (e.g. on last 1-2 beats of a fill measure, or 16th runs)
    for ev in m_arr.rh:
        if not ev.notes:
            continue
        is_fill_event = len(ev.notes) == 1 and (
            (measure.fill and ev.onset >= measure.beats - 2.0) or ev.duration <= 0.25
        )
        if is_fill_event:
            for n in ev.notes:
                pc = n.midi % 12
                if pc not in diatonic_pcs:
                    bad_name = spell(pc, m_arr.tonic_pc)
                    violations.append(
                        Violation(
                            code="non_diatonic_fill",
                            message=f"第{m_arr.measure_index + 1}小节加花音含调外音 {bad_name}",
                            detail={"pitch": n.midi, "pc": pc, "onset": ev.onset},
                        )
                    )

    return violations


def _check_pitch_ranges_and_spans(
    m_arr: MeasureArrangement,
    difficulty: Difficulty,
) -> list[Violation]:
    """Validate RH and LH register limits and hand spans."""
    violations: list[Violation] = []
    max_span = 12 if difficulty == "beginner" else 14

    # RH ranges: lowest note >= 64 (E4), highest note <= 81 (A5)
    for ev in m_arr.rh:
        if not ev.notes:
            continue
        for n in ev.notes:
            if n.midi < RH_MIN_MIDI:
                violations.append(
                    Violation(
                        code="pitch_out_of_range",
                        message=f"第{m_arr.measure_index + 1}小节右手音符 {n.midi} 低于最低音 E4 (64)",
                        detail={"pitch": n.midi, "hand": "RH", "limit": RH_MIN_MIDI},
                    )
                )
            if n.midi > RH_MAX_MIDI:
                violations.append(
                    Violation(
                        code="pitch_out_of_range",
                        message=f"第{m_arr.measure_index + 1}小节右手音符 {n.midi} 高于最高音 A5 (81)",
                        detail={"pitch": n.midi, "hand": "RH", "limit": RH_MAX_MIDI},
                    )
                )

        if len(ev.notes) > 1:
            span = max(n.midi for n in ev.notes) - min(n.midi for n in ev.notes)
            if span > max_span:
                violations.append(
                    Violation(
                        code="span_too_large",
                        message=f"第{m_arr.measure_index + 1}小节右手手跨度 {span} 超过难度限制 {max_span}",
                        detail={"span": span, "hand": "RH", "limit": max_span},
                    )
                )

    # LH ranges: lowest note >= 36 (C2), highest note <= 64 (E4)
    for ev in m_arr.lh:
        if not ev.notes:
            continue
        for n in ev.notes:
            if n.midi < LH_MIN_MIDI:
                violations.append(
                    Violation(
                        code="pitch_out_of_range",
                        message=f"第{m_arr.measure_index + 1}小节左手音符 {n.midi} 低于最低音 C2 (36)",
                        detail={"pitch": n.midi, "hand": "LH", "limit": LH_MIN_MIDI},
                    )
                )
            if n.midi > LH_MAX_MIDI:
                violations.append(
                    Violation(
                        code="pitch_out_of_range",
                        message=f"第{m_arr.measure_index + 1}小节左手音符 {n.midi} 高于最高音 E4 (64)",
                        detail={"pitch": n.midi, "hand": "LH", "limit": LH_MAX_MIDI},
                    )
                )

        if len(ev.notes) > 1:
            span = max(n.midi for n in ev.notes) - min(n.midi for n in ev.notes)
            if span > max_span:
                violations.append(
                    Violation(
                        code="span_too_large",
                        message=f"第{m_arr.measure_index + 1}小节左手手跨度 {span} 超过难度限制 {max_span}",
                        detail={"span": span, "hand": "LH", "limit": max_span},
                    )
                )

    return violations


def _check_fingering(
    events: list[Event],
    hand: str,
    measure_index: int,
) -> list[Violation]:
    """Validate fingers 1..5, chord monotonicity, and sequential finger continuity."""
    violations: list[Violation] = []

    # 1. Note finger valid & chord monotonicity
    for ev in events:
        if not ev.notes:
            continue
        fingers = [n.finger for n in ev.notes]
        if any(f is None or not (1 <= f <= 5) for f in fingers):
            violations.append(
                Violation(
                    code="invalid_fingering",
                    message=f"第{measure_index + 1}小节{hand}指法不在 1..5 范围内：{fingers}",
                    detail={"hand": hand, "fingers": fingers},
                )
            )

        if len(ev.notes) > 1:
            if len(fingers) != len(set(fingers)):
                violations.append(
                    Violation(
                        code="duplicate_fingering",
                        message=f"第{measure_index + 1}小节{hand}同时弹奏出现重复指法：{fingers}",
                        detail={"hand": hand, "fingers": fingers},
                    )
                )

            sorted_notes = sorted(ev.notes, key=lambda n: n.midi)
            s_fingers = [n.finger for n in sorted_notes if n.finger is not None]
            if hand == "RH":
                # Must strictly increase with pitch
                for i in range(len(s_fingers) - 1):
                    if s_fingers[i] >= s_fingers[i + 1]:
                        violations.append(
                            Violation(
                                code="fingering_not_monotonic",
                                message=f"第{measure_index + 1}小节右手和弦指法未随音高递增：{s_fingers}",
                                detail={"hand": hand, "fingers": s_fingers},
                            )
                        )
            else:
                # LH: must strictly decrease with pitch
                for i in range(len(s_fingers) - 1):
                    if s_fingers[i] <= s_fingers[i + 1]:
                        violations.append(
                            Violation(
                                code="fingering_not_monotonic",
                                message=f"第{measure_index + 1}小节左手和弦指法未随音高递减：{s_fingers}",
                                detail={"hand": hand, "fingers": s_fingers},
                            )
                        )

    # 2. Sequential fingering continuity
    sounding = [(i, ev) for i, ev in enumerate(events) if ev.notes]
    for idx in range(len(sounding) - 1):
        orig_i, ev1 = sounding[idx]
        orig_next_i, ev2 = sounding[idx + 1]

        if len(ev1.notes) != 1 or len(ev2.notes) != 1:
            continue

        n1 = ev1.notes[0]
        n2 = ev2.notes[0]

        if n1.midi == n2.midi:
            continue

        # If rest intervenes, position shift is allowed
        if orig_next_i > orig_i + 1:
            has_rest = any(len(events[k].notes) == 0 for k in range(orig_i + 1, orig_next_i))
            if has_rest:
                continue

        delta = round(ev2.onset - ev1.onset, 4)
        if delta <= 0.5:
            if n1.finger == n2.finger:
                violations.append(
                    Violation(
                        code="consecutive_same_finger",
                        message=f"第{measure_index + 1}小节{hand}连续不同音高重复使用同指（指号{n1.finger}）：{n1.midi} -> {n2.midi}",
                        detail={"hand": hand, "finger": n1.finger, "p1": n1.midi, "p2": n2.midi},
                    )
                )

    return violations


def validate_measure(
    m_arr: MeasureArrangement,
    measure: Measure,
    expected_tonic_pc: int,
    difficulty: Difficulty,
    next_measure_chord: Optional[ResolvedChord] = None,
) -> list[Violation]:
    """Run all arrangement validation checks on a single measure."""
    violations: list[Violation] = []

    # Rule 1: Key in force
    violations.extend(_check_key_in_force(m_arr, expected_tonic_pc))

    # Rule 2: Tiling and grid
    expected_beats = measure.beats if measure else 4.0
    violations.extend(_check_tiling_and_grid(m_arr.rh, "RH", m_arr.measure_index, expected_beats))
    violations.extend(_check_tiling_and_grid(m_arr.lh, "LH", m_arr.measure_index, expected_beats))

    # Rule 3: Every chord voiced at its beat
    violations.extend(_check_chord_voicing_presence(m_arr, measure))

    # Rule 4: RH defining tones
    violations.extend(_check_rh_defining_tones(m_arr, difficulty))

    # Rule 5: LH chord tones & passing tones
    violations.extend(_check_lh_chord_tones(m_arr, next_measure_chord))

    # Rule 6: Fill and passing notes diatonic
    violations.extend(_check_fill_diatonic(m_arr, measure))

    # Rule 7 & 8: Ranges and spans
    violations.extend(_check_pitch_ranges_and_spans(m_arr, difficulty))

    # Rule 9: Fingering
    violations.extend(_check_fingering(m_arr.rh, "RH", m_arr.measure_index))
    violations.extend(_check_fingering(m_arr.lh, "LH", m_arr.measure_index))

    return violations


def fails_stricter_heuristic(
    m_arr: MeasureArrangement,
    difficulty: Difficulty,
    prev_rh_voicing: Optional[tuple[int, ...]],
) -> tuple[bool, str]:
    """Stricter heuristic for auditing measures flagged by optional LLM review.

    Checks for:
      - RH hand span > 12 even in intermediate/advanced modes
      - Extreme edge register: lowest note <= 65 (F4) or top note >= 79 (G5)
      - LH leaps > 8 semitones (excluding octave 12)
      - Voice leading leap > 7 semitones from previous RH voicing
    """
    # 1. Span > 12
    for ev in m_arr.rh:
        if len(ev.notes) > 1:
            span = max(n.midi for n in ev.notes) - min(n.midi for n in ev.notes)
            if span > 12:
                return True, f"手跨度偏大 ({span} > 12)"

    # 2. Extreme register
    for ev in m_arr.rh:
        for n in ev.notes:
            if n.midi <= 65:
                return True, f"右手声部偏低 ({n.midi} <= 65)"
            if n.midi >= 79:
                return True, f"右手声部偏高 ({n.midi} >= 79)"

    # 3. LH awkward leaps
    lh_pitches = [ev.notes[0].midi for ev in m_arr.lh if len(ev.notes) == 1]
    for i in range(len(lh_pitches) - 1):
        diff = abs(lh_pitches[i + 1] - lh_pitches[i])
        if diff > 8 and diff != 12:
            return True, f"左手进行出现突兀大跳 ({diff} 半音)"

    # 4. Voice leading leap from previous voicing
    if prev_rh_voicing:
        first_rh = next((ev for ev in m_arr.rh if len(ev.notes) >= 2), None)
        if first_rh:
            top_prev = prev_rh_voicing[-1]
            top_curr = max(n.midi for n in first_rh.notes)
            if abs(top_curr - top_prev) > 7:
                return True, f"和弦连接声部跳进过大 ({abs(top_curr - top_prev)} 半音)"

    return False, ""


def check_and_repair(
    sheet: ParsedSheet,
    arrangement: Arrangement,
    *,
    use_llm: bool = False,
) -> Arrangement:
    """Validate arrangement invariants, repair failing measures via fallback, and optionally audit with LLM."""
    start_pc = key_name_to_pc(arrangement.start_key)
    current_tonic_pc = start_pc

    # Index key changes
    key_changes_by_m: dict[int, list[int]] = {}
    for kc in sheet.key_changes:
        key_changes_by_m.setdefault(kc.at_measure, []).append(kc.semitones)

    measures_by_idx = {m.index: m for m in sheet.measures()}

    # Compute expected tonics across measures
    expected_tonics: list[int] = []
    for m_arr in arrangement.measures:
        m_idx = m_arr.measure_index
        if m_idx in key_changes_by_m:
            for shift in key_changes_by_m[m_idx]:
                current_tonic_pc = (current_tonic_pc + shift) % 12
        expected_tonics.append(current_tonic_pc)

    # Cross-check chord theory vs music21 for all chords in arrangement (info log)
    for m_arr in arrangement.measures:
        for chord in m_arr.chords:
            is_match, reason = cross_check_chord_pcs(chord)
            if not is_match:
                arrangement.issues.append(
                    QualityIssue(
                        stage="arrange",
                        severity="info",
                        measure_index=m_arr.measure_index,
                        code="chord_theory_cross_check_mismatch",
                        message=f"第{m_arr.measure_index + 1}小节和弦理论对照提示：{reason}",
                        detail={"chord": chord.name, "raw": chord.raw, "reason": reason},
                    )
                )

    # Validate and repair each measure
    prev_rh_voicing: Optional[tuple[int, ...]] = None
    num_measures = len(arrangement.measures)

    for i in range(num_measures):
        m_arr = arrangement.measures[i]
        m_idx = m_arr.measure_index
        measure = measures_by_idx.get(m_idx, Measure(index=m_idx, bbox=(0, 0, 1, 1)))
        exp_tonic = expected_tonics[i]

        next_chord = None
        if i + 1 < num_measures and arrangement.measures[i + 1].chords:
            next_chord = arrangement.measures[i + 1].chords[0]

        violations = validate_measure(m_arr, measure, exp_tonic, arrangement.difficulty, next_chord)

        if violations:
            primary = violations[0]
            # Apply safe fallback
            safe_rh, safe_lh = safe_measure(
                measure=measure,
                resolved_chords=m_arr.chords,
                tonic_pc=exp_tonic,
                difficulty=arrangement.difficulty,
                prev_rh_voicing=prev_rh_voicing,
            )
            m_arr.rh = safe_rh
            m_arr.lh = safe_lh
            m_arr.tonic_pc = exp_tonic
            m_arr.key_name = canonical_key_for_pc(exp_tonic)

            # Re-validate repaired measure
            recheck_violations = validate_measure(m_arr, measure, exp_tonic, arrangement.difficulty, next_chord)
            if recheck_violations:
                recheck_errs = "; ".join(v.message for v in recheck_violations)
                raise ArrangementQAError(
                    f"Safe fallback failed re-validation on measure {m_idx + 1}: {recheck_errs}"
                )

            # Record auto-fixed QualityIssue
            arrangement.issues.append(
                QualityIssue(
                    stage="arrange",
                    severity="auto_fixed",
                    measure_index=m_idx,
                    code=primary.code,
                    message=f"第{m_idx + 1}小节检测到编曲问题（{primary.message}），已改用稳妥弹法",
                    detail={"original_violations": [v.code for v in violations]},
                )
            )

        # Track previous RH voicing for voice leading
        for ev in m_arr.rh:
            if len(ev.notes) >= 2:
                prev_rh_voicing = tuple(sorted(n.midi for n in ev.notes))

    # Optional LLM Review
    if use_llm:
        compact_text = build_compact_score_text(arrangement)
        try:
            reviewer_fn = ask_json
            if reviewer_fn is None:
                from app.qa.llm import ask_json as imported_ask_json
                reviewer_fn = imported_ask_json

            review_resp: LLMReviewResponse = reviewer_fn(
                prompt=f"请作为专业钢琴编配专家，审核以下钢琴伴奏谱并标出可疑/不自然小节：\n\n{compact_text}",
                role="reviewer",
                schema=LLMReviewResponse,
            )
            flagged_map = {item.measure_index: item for item in review_resp.suspicious_measures}

            for i, m_arr in enumerate(arrangement.measures):
                if m_arr.measure_index in flagged_map:
                    item = flagged_map[m_arr.measure_index]
                    measure = measures_by_idx.get(
                        m_arr.measure_index, Measure(index=m_arr.measure_index, bbox=(0, 0, 1, 1))
                    )
                    fails_strict, strict_reason = fails_stricter_heuristic(
                        m_arr, arrangement.difficulty, prev_rh_voicing
                    )
                    if fails_strict:
                        # Apply fallback
                        safe_rh, safe_lh = safe_measure(
                            measure=measure,
                            resolved_chords=m_arr.chords,
                            tonic_pc=expected_tonics[i],
                            difficulty=arrangement.difficulty,
                            prev_rh_voicing=prev_rh_voicing,
                        )
                        m_arr.rh = safe_rh
                        m_arr.lh = safe_lh
                        arrangement.issues.append(
                            QualityIssue(
                                stage="arrange",
                                severity="auto_fixed",
                                measure_index=m_arr.measure_index,
                                code="llm_audit_fallback",
                                message=f"第{m_arr.measure_index + 1}小节经AI审核与严格规则复核（{strict_reason}；AI理由：{item.reason}），已改用稳妥弹法",
                                detail={"llm_reason": item.reason, "strict_reason": strict_reason},
                            )
                        )
                    else:
                        # Log as informational issue
                        arrangement.issues.append(
                            QualityIssue(
                                stage="arrange",
                                severity="info",
                                measure_index=m_arr.measure_index,
                                code="llm_audit_flagged_passed",
                                message=f"第{m_arr.measure_index + 1}小节AI提示可疑（{item.reason}），但已通过严格规则检验保留原编曲",
                                detail={"llm_reason": item.reason},
                            )
                        )
        except Exception as e:
            # Never let LLM errors break the pipeline!
            arrangement.issues.append(
                QualityIssue(
                    stage="arrange",
                    severity="info",
                    measure_index=None,
                    code="llm_review_error",
                    message=f"AI辅助复核跳过：{e}",
                    detail={"error": str(e)},
                )
            )

    return arrangement
