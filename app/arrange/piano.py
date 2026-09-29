"""Piano arrangement generator for Taiwanese band charts."""
from __future__ import annotations

from typing import Optional

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
    System,
)
from app.theory.chords import clean_raw_chord, resolve_chord
from app.theory.keys import canonical_key_for_pc, key_name_to_pc
from app.theory.stacked import infer_stacked_orientation
from app.arrange.fingering import assign_fingering_for_measure
from app.arrange.styles import generate_measure_events, normalize_style
from app.arrange.voicing import select_lh_bass_notes, select_rh_voicings


def validate_sequential_fingering(events: list[Event], hand: str = "RH") -> None:
    """Validate that consecutive single notes in broken patterns/fills do not reuse fingers on different pitches without a position shift."""
    sounding = [(i, ev) for i, ev in enumerate(events) if ev.notes]
    for idx in range(len(sounding) - 1):
        orig_i, ev1 = sounding[idx]
        orig_next_i, ev2 = sounding[idx + 1]

        # Only check single-note events (broken patterns, arpeggios, fills, bass hints)
        if len(ev1.notes) != 1 or len(ev2.notes) != 1:
            continue

        n1 = ev1.notes[0]
        n2 = ev2.notes[0]

        # Repeated note on the same pitch is allowed
        if n1.midi == n2.midi:
            continue

        # If there is an explicit rest event between them, a position shift occurs
        if orig_next_i > orig_i + 1:
            has_rest = any(len(events[k].notes) == 0 for k in range(orig_i + 1, orig_next_i))
            if has_rest:
                continue

        # Check time delta for continuous notes without rest
        delta = round(ev2.onset - ev1.onset, 4)
        if delta <= 0.5:
            if n1.finger == n2.finger:
                raise ValueError(
                    f"{hand} sequential fingering violation at onset {ev2.onset}: "
                    f"finger {n1.finger} reused on different pitch ({n1.midi} -> {n2.midi}) "
                    f"with delta {delta} without position shift"
                )


def validate_arrangement(arr: Arrangement, sheet: ParsedSheet) -> None:
    """Validate arrangement invariants across all measures without raising fatal exceptions on valid input."""
    measures_by_idx = {m.index: m for m in sheet.measures()}

    for m_arr in arr.measures:
        m_spec = measures_by_idx.get(m_arr.measure_index)
        expected_beats = m_spec.beats if m_spec else 4.0

        # 1. Tiling check
        rh_dur = round(sum(e.duration for e in m_arr.rh), 4)
        lh_dur = round(sum(e.duration for e in m_arr.lh), 4)
        exp_dur = round(expected_beats, 4)

        if rh_dur != exp_dur:
            arr.issues.append(QualityIssue(
                stage="arrange",
                measure_index=m_arr.measure_index,
                severity="auto_fixed",
                code="tiling_mismatch",
                message=f"Measure {m_arr.measure_index} RH tiling mismatch: {rh_dur} != {exp_dur}",
            ))
        if lh_dur != exp_dur:
            arr.issues.append(QualityIssue(
                stage="arrange",
                measure_index=m_arr.measure_index,
                severity="auto_fixed",
                code="tiling_mismatch",
                message=f"Measure {m_arr.measure_index} LH tiling mismatch: {lh_dur} != {exp_dur}",
            ))

        # 2. Continuous onsets
        curr = 0.0
        for e in m_arr.rh:
            if abs(e.onset - curr) > 1e-4:
                arr.issues.append(QualityIssue(
                    stage="arrange",
                    measure_index=m_arr.measure_index,
                    severity="auto_fixed",
                    code="onset_discontinuity",
                    message=f"Measure {m_arr.measure_index} RH onset discontinuity at {e.onset} != {curr}",
                ))
            curr = round(curr + e.duration, 4)

        curr = 0.0
        for e in m_arr.lh:
            if abs(e.onset - curr) > 1e-4:
                arr.issues.append(QualityIssue(
                    stage="arrange",
                    measure_index=m_arr.measure_index,
                    severity="auto_fixed",
                    code="onset_discontinuity",
                    message=f"Measure {m_arr.measure_index} LH onset discontinuity at {e.onset} != {curr}",
                ))
            curr = round(curr + e.duration, 4)

        # 3. Fingering validity
        try:
            validate_sequential_fingering(m_arr.rh, hand="RH")
            validate_sequential_fingering(m_arr.lh, hand="LH")
        except ValueError as f_err:
            arr.issues.append(QualityIssue(
                stage="arrange",
                measure_index=m_arr.measure_index,
                severity="auto_fixed",
                code="sequential_fingering",
                message=str(f_err),
            ))

        for e in m_arr.rh:
            if not e.notes:
                continue
            fingers = [n.finger for n in e.notes]
            if any(f is None or not (1 <= f <= 5) for f in fingers):
                arr.issues.append(QualityIssue(
                    stage="arrange",
                    measure_index=m_arr.measure_index,
                    severity="auto_fixed",
                    code="invalid_finger",
                    message=f"Measure {m_arr.measure_index} RH invalid finger in {fingers}",
                ))
            if len(fingers) != len(set(fingers)):
                arr.issues.append(QualityIssue(
                    stage="arrange",
                    measure_index=m_arr.measure_index,
                    severity="auto_fixed",
                    code="duplicate_fingers",
                    message=f"Measure {m_arr.measure_index} RH duplicate fingers at onset {e.onset}: {fingers}",
                ))
            # Monotonicity with pitch for chords
            sorted_notes = sorted(e.notes, key=lambda n: n.midi)
            s_fingers = [n.finger for n in sorted_notes]
            for i in range(len(s_fingers) - 1):
                if s_fingers[i] >= s_fingers[i + 1]:
                    arr.issues.append(QualityIssue(
                        stage="arrange",
                        measure_index=m_arr.measure_index,
                        severity="auto_fixed",
                        code="fingering_not_monotonic",
                        message=f"Measure {m_arr.measure_index} RH fingers not increasing with pitch: {s_fingers}",
                    ))

        for e in m_arr.lh:
            if not e.notes:
                continue
            fingers = [n.finger for n in e.notes]
            if any(f is None or not (1 <= f <= 5) for f in fingers):
                arr.issues.append(QualityIssue(
                    stage="arrange",
                    measure_index=m_arr.measure_index,
                    severity="auto_fixed",
                    code="invalid_finger",
                    message=f"Measure {m_arr.measure_index} LH invalid finger in {fingers}",
                ))
            if len(fingers) != len(set(fingers)):
                arr.issues.append(QualityIssue(
                    stage="arrange",
                    measure_index=m_arr.measure_index,
                    severity="auto_fixed",
                    code="duplicate_fingers",
                    message=f"Measure {m_arr.measure_index} LH duplicate fingers at onset {e.onset}: {fingers}",
                ))
            sorted_notes = sorted(e.notes, key=lambda n: n.midi)
            s_fingers = [n.finger for n in sorted_notes]
            for i in range(len(s_fingers) - 1):
                if s_fingers[i] <= s_fingers[i + 1]:
                    arr.issues.append(QualityIssue(
                        stage="arrange",
                        measure_index=m_arr.measure_index,
                        severity="auto_fixed",
                        code="fingering_not_monotonic",
                        message=f"Measure {m_arr.measure_index} LH fingers not decreasing with pitch: {s_fingers}",
                    ))

        # 4. Difficulty constraints
        if arr.difficulty == "beginner":
            # Max 2 attacks per bar for RH
            attacks = sum(1 for e in m_arr.rh if len(e.notes) > 0)
            if attacks > 2:
                arr.issues.append(QualityIssue(
                    stage="arrange",
                    measure_index=m_arr.measure_index,
                    severity="auto_fixed",
                    code="attacks_too_dense",
                    message=f"Measure {m_arr.measure_index} beginner RH has {attacks} attacks (> 2)",
                ))
            # Span <= 12
            for e in m_arr.rh:
                if len(e.notes) > 1:
                    span = max(n.midi for n in e.notes) - min(n.midi for n in e.notes)
                    if span > 12:
                        arr.issues.append(QualityIssue(
                            stage="arrange",
                            measure_index=m_arr.measure_index,
                            severity="auto_fixed",
                            code="span_too_large",
                            message=f"Measure {m_arr.measure_index} beginner RH span {span} > 12",
                        ))


def arrange(
    sheet: ParsedSheet,
    start_key: str,
    difficulty: Difficulty = "intermediate",
) -> Arrangement:
    """Transform a ParsedSheet into a complete, playable piano Arrangement.

    Tracks tonics and key changes, carries chord symbols across empty bars,
    optimizes voice leading via Viterbi DP, generates styled accompaniment,
    and assigns physically playable fingering to all notes.
    """
    raw_style = sheet.header.style if sheet.header else ""
    raw_ts = sheet.header.time_signature if sheet.header else ""
    norm_style = normalize_style(raw_style, raw_ts)

    start_pc = key_name_to_pc(start_key)
    current_tonic_pc = start_pc
    current_key_name = canonical_key_for_pc(current_tonic_pc, start_key)

    notation = sheet.header.chord_notation if (sheet.header and sheet.header.chord_notation) else "number"
    printed_tonic_pc: Optional[int] = None
    arr_notes: list[str] = [
        f"Arrangement in {difficulty} style ({norm_style})",
        f"Start key: {start_key} (tonic pc {start_pc})",
    ]
    arr_issues: list[QualityIssue] = []

    stacked_res = infer_stacked_orientation(sheet)
    stacked_orientation = stacked_res.orientation
    has_stacked_chords = any(
        c.stacked for m in sheet.measures() for c in m.chords
    )
    if has_stacked_chords:
        if stacked_orientation == "top_is_bass":
            arr_notes.append("叠写和弦按上方为低音解读")
        else:
            arr_notes.append("叠写和弦按下方为低音解读")

    if notation == "letter":
        orig_key = sheet.header.original_key if sheet.header else None
        if orig_key:
            try:
                printed_tonic_pc = key_name_to_pc(orig_key)
            except ValueError:
                printed_tonic_pc = start_pc
                arr_notes.append(
                    f"Invalid original key {orig_key!r} for letter chart; assumed printed key equals start key (no transposition)"
                )
        else:
            printed_tonic_pc = start_pc
            arr_notes.append(
                "No original key specified for letter chart; assumed printed key equals start key (no transposition)"
            )

    # Index key changes by measure index
    key_changes_by_m: dict[int, list[int]] = {}
    for kc in sheet.key_changes:
        key_changes_by_m.setdefault(kc.at_measure, []).append(kc.semitones)

    # Map measures to systems for section labels
    system_by_measure: dict[int, System] = {}
    for sys in sheet.systems:
        for m in sys.measures:
            system_by_measure[m.index] = sys

    measures = sheet.measures()
    if not measures:
        return Arrangement(
            instrument="piano",
            difficulty=difficulty,
            start_key=start_key,
            style=norm_style,
            measures=[],
            notes=arr_notes + ["Empty sheet"],
            issues=arr_issues,
        )

    # 1. Resolve chords and track tonic per measure
    measure_resolved_chords: list[list[ResolvedChord]] = []
    measure_tonic_pcs: list[int] = []
    measure_key_names: list[str] = []
    last_chords: list[ResolvedChord] = []

    for m in measures:
        # Apply key changes at this measure
        if m.index in key_changes_by_m:
            for shift in key_changes_by_m[m.index]:
                current_tonic_pc = (current_tonic_pc + shift) % 12
                current_key_name = canonical_key_for_pc(current_tonic_pc)

        measure_tonic_pcs.append(current_tonic_pc)
        measure_key_names.append(current_key_name)

        if m.chords:
            resolved = []
            for cs in m.chords:
                prev_c = resolved[-1] if resolved else (last_chords[-1] if last_chords else None)
                try:
                    rc = resolve_chord(
                        cs.raw,
                        current_tonic_pc,
                        cs.beat,
                        current_key_name,
                        notation=notation,
                        printed_tonic_pc=printed_tonic_pc,
                        prev_chord=prev_c,
                        stacked=cs.stacked,
                        stacked_orientation=stacked_orientation,
                    )
                    resolved.append(rc)
                except ValueError:
                    cleaned = clean_raw_chord(cs.raw)
                    if cleaned.startswith("/") and prev_c is None:
                        tonic_prior = resolve_chord("1", current_tonic_pc, cs.beat, current_key_name)
                        try:
                            degraded_rc = resolve_chord(
                                cs.raw,
                                current_tonic_pc,
                                cs.beat,
                                current_key_name,
                                notation=notation,
                                printed_tonic_pc=printed_tonic_pc,
                                prev_chord=tonic_prior,
                                stacked=cs.stacked,
                                stacked_orientation=stacked_orientation,
                            )
                        except ValueError:
                            degraded_rc = tonic_prior

                        resolved.append(degraded_rc)
                        arr_issues.append(
                            QualityIssue.model_construct(
                                stage="arrange",
                                severity="warning",
                                code="bass_only_without_context",
                                message=f"第 {m.index + 1} 小节开头的低音和弦 {cs.raw!r} 缺少前置和弦，已降级为主和弦 {degraded_rc.name}",
                                measure_index=m.index,
                                detail={"raw": cs.raw, "degraded_to": degraded_rc.name},
                            )
                        )
                        arr_notes.append(
                            f"Measure {m.index}: Leading slash chord {cs.raw!r} without context degraded to tonic chord {degraded_rc.name}"
                        )
                        continue

                    # Fallback on alternatives or carried chord
                    alt_resolved = None
                    for alt in cs.alternatives:
                        try:
                            alt_resolved = resolve_chord(
                                alt,
                                current_tonic_pc,
                                cs.beat,
                                current_key_name,
                                notation=notation,
                                printed_tonic_pc=printed_tonic_pc,
                                prev_chord=prev_c,
                                stacked=cs.stacked,
                                stacked_orientation=stacked_orientation,
                            )
                            break
                        except ValueError:
                            continue
                    if alt_resolved is not None:
                        resolved.append(alt_resolved)
                        arr_notes.append(
                            f"Measure {m.index}: Unparseable chord {cs.raw!r} replaced with alternative {alt_resolved.raw!r}"
                        )
                    elif last_chords:
                        last_ch = last_chords[-1]
                        carried = ResolvedChord(
                            raw=cs.raw,
                            name=last_ch.name,
                            beat=cs.beat,
                            root_pc=last_ch.root_pc,
                            bass_pc=last_ch.bass_pc,
                            pcs=last_ch.pcs,
                            quality=last_ch.quality,
                        )
                        resolved.append(carried)
                        arr_notes.append(
                            f"Measure {m.index}: Unparseable chord {cs.raw!r} replaced with carried chord {last_ch.name}"
                        )
                    else:
                        default_rc = resolve_chord("1", current_tonic_pc, cs.beat, current_key_name)
                        resolved.append(default_rc)
                        arr_notes.append(
                            f"Measure {m.index}: Unparseable chord {cs.raw!r} replaced with default tonic chord"
                        )
            if resolved:
                last_chords = resolved
        else:
            # Carry last chord into measures with no chord box
            if last_chords:
                last_ch = last_chords[-1]
                resolved = [
                    ResolvedChord(
                        raw=last_ch.raw,
                        name=last_ch.name,
                        beat=1.0,
                        root_pc=last_ch.root_pc,
                        bass_pc=last_ch.bass_pc,
                        pcs=last_ch.pcs,
                        quality=last_ch.quality,
                    )
                ]
            else:
                # Default to tonic chord '1'
                resolved = [resolve_chord("1", current_tonic_pc, 1.0, current_key_name)]

        measure_resolved_chords.append(resolved)

    # 2. Global Viterbi voice leading over the whole chord progression
    flattened_chords: list[ResolvedChord] = []
    chord_slices: list[slice] = []
    offset = 0
    for m_chords in measure_resolved_chords:
        count = len(m_chords)
        chord_slices.append(slice(offset, offset + count))
        flattened_chords.extend(m_chords)
        offset += count

    all_voicings = select_rh_voicings(flattened_chords, difficulty)
    all_basses = select_lh_bass_notes(flattened_chords)

    # 3. Generate events and assign fingering for each measure
    measure_arrangements: list[MeasureArrangement] = []

    active_bass_hint: Optional[str] = None
    last_system_id: Optional[int] = None

    for i, m in enumerate(measures):
        c_slice = chord_slices[i]
        m_chords = measure_resolved_chords[i]
        m_voicings = all_voicings[c_slice]
        m_basses = all_basses[c_slice]
        m_tonic_pc = measure_tonic_pcs[i]
        m_key_name = measure_key_names[i]

        sys = system_by_measure.get(m.index)
        section_label = sys.section_label if sys else None

        # Reset active_bass_hint when changing systems
        sys_id = id(sys) if sys else None
        if sys_id != last_system_id:
            active_bass_hint = None
            last_system_id = sys_id

        is_inherited_bass_hint = False
        if m.bass_hint:
            m_bass_hint = m.bass_hint
            if "~" in m.bass_hint:
                active_bass_hint = m.bass_hint
            else:
                active_bass_hint = None
        else:
            if active_bass_hint:
                m_bass_hint = active_bass_hint
                is_inherited_bass_hint = True
            else:
                m_bass_hint = None

        nxt_chord = flattened_chords[c_slice.stop] if c_slice.stop < len(flattened_chords) else None
        nxt_bass = all_basses[c_slice.stop] if c_slice.stop < len(all_basses) else None

        rh_events, lh_events = generate_measure_events(
            chords=m_chords,
            rh_voicings=m_voicings,
            lh_bass_pitches=m_basses,
            difficulty=difficulty,
            style=norm_style,
            measure_beats=m.beats,
            tonic_pc=m_tonic_pc,
            key_name=m_key_name,
            section_label=section_label,
            is_stop=m.is_stop,
            fill=m.fill,
            bass_hint=m_bass_hint,
            is_inherited_bass_hint=is_inherited_bass_hint,
            next_bass_pitch=nxt_bass,
            next_bass_pc=nxt_chord.bass_pc if nxt_chord else None,
        )

        # Fingering
        primary_voicing = m_voicings[0] if m_voicings else None
        primary_bass = m_basses[0] if m_basses else None
        chord_basses = [(round(c.beat - 1.0, 4), b) for c, b in zip(m_chords, m_basses)]
        assign_fingering_for_measure(
            rh_events=rh_events,
            lh_events=lh_events,
            rh_voicing=primary_voicing,
            bass_pitch=primary_bass,
            lh_pattern_type="arpeggio_wide" if difficulty == "advanced" else "root_5_8",
            chord_basses=chord_basses,
        )

        m_arr = MeasureArrangement(
            measure_index=m.index,
            tonic_pc=m_tonic_pc,
            key_name=m_key_name,
            chords=m_chords,
            rh=rh_events,
            lh=lh_events,
        )
        measure_arrangements.append(m_arr)

    arr = Arrangement(
        instrument="piano",
        difficulty=difficulty,
        start_key=start_key,
        style=norm_style,
        measures=measure_arrangements,
        notes=arr_notes,
        issues=arr_issues,
    )

    # Invariant validation
    validate_arrangement(arr, sheet)

    return arr
