"""Ukulele arranger for Taiwanese band charts and jianpu lead sheets."""
from __future__ import annotations

from typing import Optional

from app.arrange.piano import arrange as arrange_piano
from app.arrange.uke_patterns import (
    PATTERN_REGISTRY,
    choose_section_pattern,
    generate_measure_strokes,
    get_time_signature_family,
)
from app.arrange.uke_shapes import (
    DEFAULT_ROOT_NAMES,
    candidate_shapes,
    shape_cost,
)
from app.models import (
    Arrangement,
    Difficulty,
    MeasureArrangement,
    ParsedSheet,
    QualityIssue,
    ResolvedChord,
    SectionPlan,
    UkeMeasure,
    UkeShape,
    UkeStroke,
)
from app.theory.keys import canonical_key_for_pc, key_name_to_pc, spell


def transition_cost(s1: UkeShape, s2: UkeShape) -> float:
    """Calculate voice-leading transition cost between two consecutive ukulele shapes."""
    anchors = sum(1 for i in range(4) if s1.frets[i] == s2.frets[i] and s1.frets[i] > 0)
    common_open = sum(1 for i in range(4) if s1.frets[i] == s2.frets[i] == 0)
    changed = sum(1 for i in range(4) if s1.frets[i] != s2.frets[i])

    pos1 = max(s1.frets)
    pos2 = max(s2.frets)
    jump = abs(pos1 - pos2)

    barre_diff = 1.0 if (s1.barre is None) != (s2.barre is None) else 0.0

    cost = (
        changed * 0.5
        + jump * 0.7
        + barre_diff * 1.0
        - anchors * 1.5
        - common_open * 0.4
    )
    return cost


def normalize_measure_chords_for_omr_jitter(
    chords: list[ResolvedChord],
    sheet_chords: list[Any],
    difficulty: Difficulty,
) -> list[ResolvedChord]:
    """If a measure's first chord starts at <= 0.5 beat with previous chord carried before it,
    merge (start the new chord at beat 1) for beginner/intermediate.
    """
    if difficulty not in ("beginner", "intermediate") or not chords:
        return chords

    # Case 1: First chord starts slightly late (> 1.0 and <= 1.5)
    if chords[0].beat > 1.0 and chords[0].beat <= 1.5:
        c0 = chords[0].model_copy(update={"beat": 1.0})
        return [c0] + chords[1:]

    # Case 2: chords[0] is at beat 1.0 carried from previous measure,
    # while the printed chord was placed at beat <= 1.5
    if len(chords) >= 2 and chords[1].beat <= 1.5:
        printed_starts_at_1 = any(abs(sc.beat - 1.0) < 1e-3 for sc in sheet_chords)
        if not printed_starts_at_1:
            c1 = chords[1].model_copy(update={"beat": 1.0})
            return [c1] + chords[2:]

    return chords


def get_sections(
    sheet: ParsedSheet,
    sections: Optional[list[SectionPlan]] = None,
) -> list[SectionPlan]:
    """Retrieve section plan for sheet, using provided, lazy-imported, or fallback plan."""
    measures = sheet.measures()
    if sections is not None and len(sections) == len(measures):
        return sections

    try:
        from app.arrange.sections import plan_sections
        res = plan_sections(sheet, use_llm=False)
        if res and len(res) == len(measures):
            return res
    except Exception:
        pass

    # Trivial local fallback: all measures one section, role 'unknown', energy 1
    fallback: list[SectionPlan] = []
    n = len(measures)
    for i, m in enumerate(measures):
        fallback.append(
            SectionPlan(
                measure_index=m.index,
                section_id="section_all",
                role="unknown",
                energy=1,
                is_section_start=(i == 0),
                is_section_end=(i == n - 1),
                source="heuristic",
            )
        )
    return fallback


def select_capo_and_shape_key(
    start_key: str,
    all_chords: list[ResolvedChord],
    difficulty: Difficulty,
) -> tuple[int, str]:
    """Select best capo in 0..5 minimizing count-weighted shape cost. Capo 0 wins ties."""
    start_tonic_pc = key_name_to_pc(start_key)

    best_capo = 0
    best_cost = float("inf")
    best_shape_key = start_key

    # Capo penalty per fret
    capo_penalty = 0.4

    for c in range(6):
        shape_tonic = (start_tonic_pc - c) % 12
        if c == 0:
            candidate_key = start_key
        else:
            base_key = canonical_key_for_pc(shape_tonic)
            # Preserve minor quality if song start_key was minor
            candidate_key = f"{base_key}m" if start_key.endswith("m") else base_key

        total_cost = 0.0
        for ch in all_chords:
            # Ukulele drops slash bass for chord shape
            r_shape = (ch.root_pc - c) % 12
            cands = candidate_shapes(r_shape, ch.quality, difficulty=difficulty)
            if cands:
                best_cand = cands[0]
                total_cost += shape_cost(
                    best_cand.frets,
                    best_cand.fingers,
                    best_cand.barre,
                    difficulty,
                    is_curated=True,
                )
            else:
                total_cost += 20.0

        total_cost += c * capo_penalty

        # Strictly less than ensures capo 0 wins ties
        if total_cost < best_cost:
            best_cost = total_cost
            best_capo = c
            best_shape_key = candidate_key

    return best_capo, best_shape_key


def select_chord_shapes_dp(
    chord_items: list[tuple[int, ResolvedChord, str]],  # (measure_idx, chord, section_id)
    capo: int,
    shape_key: str,
    difficulty: Difficulty,
) -> list[UkeShape]:
    """Select optimal shapes for the song chord sequence using Viterbi DP."""
    if not chord_items:
        return []

    shape_tonic_pc = key_name_to_pc(shape_key)

    # 1. Gather candidate shapes for each chord occurrence
    candidates_per_step: list[list[UkeShape]] = []
    for _, ch, _ in chord_items:
        r_shape = (ch.root_pc - capo) % 12
        # Spell root note in shape key
        spelled_root = spell(r_shape, tonic_pc=shape_tonic_pc, key_name=shape_key)
        cands = candidate_shapes(
            r_shape,
            ch.quality,
            difficulty=difficulty,
            root_name=spelled_root,
        )[:6]
        if not cands:
            # Safe fallback: open root shape
            def_root = DEFAULT_ROOT_NAMES[r_shape]
            cands = [UkeShape(name=def_root, frets=(0, 0, 0, 0), fingers=(0, 0, 0, 0), barre=None, base_fret=1)]
        candidates_per_step.append(cands)

    n_steps = len(chord_items)
    # dp[step][cand_idx] = (min_cost, prev_cand_idx)
    dp: list[list[float]] = []
    backtrack: list[list[int]] = []

    # Initialize step 0
    cands_0 = candidates_per_step[0]
    dp.append([
        shape_cost(s.frets, s.fingers, s.barre, difficulty)
        for s in cands_0
    ])
    backtrack.append([-1] * len(cands_0))

    # Run DP forward
    for t in range(1, n_steps):
        prev_cands = candidates_per_step[t - 1]
        curr_cands = candidates_per_step[t]
        _, prev_chord, prev_sec = chord_items[t - 1]
        _, curr_chord, curr_sec = chord_items[t]

        same_chord_in_section = (
            prev_sec == curr_sec
            and prev_chord.root_pc == curr_chord.root_pc
            and prev_chord.quality == curr_chord.quality
        )

        step_costs: list[float] = []
        step_back: list[int] = []

        for j, s_curr in enumerate(curr_cands):
            s_cost = shape_cost(s_curr.frets, s_curr.fingers, s_curr.barre, difficulty)
            best_trans = float("inf")
            best_prev_idx = 0

            for i, s_prev in enumerate(prev_cands):
                t_cost = transition_cost(s_prev, s_curr)
                if same_chord_in_section:
                    if s_prev.frets == s_curr.frets:
                        t_cost -= 4.0  # Reward keeping the same shape within a section
                    else:
                        t_cost += 3.0

                total_cand_cost = dp[t - 1][i] + t_cost
                if total_cand_cost < best_trans:
                    best_trans = total_cand_cost
                    best_prev_idx = i

            step_costs.append(best_trans + s_cost)
            step_back.append(best_prev_idx)

        dp.append(step_costs)
        backtrack.append(step_back)

    # Backtrack best path
    best_last_idx = int(min(range(len(dp[-1])), key=lambda i: dp[-1][i]))
    chosen_path: list[int] = [best_last_idx]
    for t in range(n_steps - 1, 0, -1):
        best_last_idx = backtrack[t][best_last_idx]
        chosen_path.append(best_last_idx)

    chosen_path.reverse()
    return [candidates_per_step[t][chosen_path[t]] for t in range(n_steps)]


def arrange_ukulele(
    sheet: ParsedSheet,
    start_key: str,
    difficulty: Difficulty,
    *,
    sections: Optional[list[SectionPlan]] = None,
) -> Arrangement:
    """Generate ukulele accompaniment arrangement for a parsed sheet.

    NEVER raises on valid ParsedSheet input.
    """
    # 1. Obtain identical sounding chord resolution from piano arrangement
    piano_arr = arrange_piano(sheet, start_key, "beginner")
    norm_style = piano_arr.style

    sheet_measures = sheet.measures()
    m_specs_by_idx = {m.index: m for m in sheet_measures}

    measures: list[MeasureArrangement] = []
    for pm in piano_arr.measures:
        m_spec = m_specs_by_idx.get(pm.measure_index)
        sheet_chords = m_spec.chords if m_spec else []
        norm_chords = normalize_measure_chords_for_omr_jitter(pm.chords, sheet_chords, difficulty)
        measures.append(
            MeasureArrangement(
                measure_index=pm.measure_index,
                tonic_pc=pm.tonic_pc,
                key_name=pm.key_name,
                chords=norm_chords,
                rh=[],
                lh=[],
            )
        )

    # 2. Sections
    resolved_sections = get_sections(sheet, sections)
    sec_by_measure = {s.measure_index: s for s in resolved_sections}

    # 3. Flatten chords to select capo and shape key
    all_sounding_chords: list[ResolvedChord] = []
    chord_items: list[tuple[int, ResolvedChord, str]] = []
    for m in measures:
        sec = sec_by_measure.get(m.measure_index)
        sec_id = sec.section_id if sec else "section_all"
        for ch in m.chords:
            all_sounding_chords.append(ch)
            chord_items.append((m.measure_index, ch, sec_id))

    capo, shape_key = select_capo_and_shape_key(
        start_key=start_key,
        all_chords=all_sounding_chords,
        difficulty=difficulty,
    )

    # 4. Global voice-leading DP to select shapes
    chosen_shapes = select_chord_shapes_dp(
        chord_items=chord_items,
        capo=capo,
        shape_key=shape_key,
        difficulty=difficulty,
    )

    # Map chosen shapes back to measures
    shape_iter = iter(chosen_shapes)
    measure_shapes: list[list[UkeShape]] = []
    for m in measures:
        m_s = [next(shape_iter) for _ in m.chords]
        measure_shapes.append(m_s)

    # 5. Pattern selection and stroke generation
    ts_family = get_time_signature_family(sheet.header.time_signature, sheet.header.style)
    tempo_bpm = sheet.header.tempo_bpm

    # Assign base pattern per section instance
    section_patterns: dict[str, str] = {}
    for plan in resolved_sections:
        if plan.section_id not in section_patterns:
            pid = choose_section_pattern(
                ts_family=ts_family,
                difficulty=difficulty,
                role=plan.role,
                energy=plan.energy,
                tempo_bpm=tempo_bpm,
            )
            section_patterns[plan.section_id] = pid

    sheet_measures = sheet.measures()
    m_specs_by_idx = {m.index: m for m in sheet_measures}

    uke_measures: list[UkeMeasure] = []
    uke_shapes: dict[str, UkeShape] = {}
    pattern_legend: dict[str, str] = {}
    issues: list[QualityIssue] = list(piano_arr.issues)

    n_measures = len(measures)
    full_bar_beats = 4.0
    if ts_family == "3/4":
        full_bar_beats = 3.0
    elif ts_family in ("2/4", "6/8"):
        full_bar_beats = 2.0

    for i, m_arr in enumerate(measures):
        m_spec = m_specs_by_idx.get(m_arr.measure_index)
        m_beats = m_spec.beats if m_spec else 4.0
        sec_plan = sec_by_measure.get(m_arr.measure_index)

        sec_id = sec_plan.section_id if sec_plan else "section_all"
        base_pid = section_patterns.get(sec_id, "uke_4_4_mid_island")

        is_last = (i == n_measures - 1)
        is_stop = m_spec.is_stop if m_spec else False
        is_fill = m_spec.fill if m_spec else False
        is_end = sec_plan.is_section_end if sec_plan else False
        is_start = sec_plan.is_section_start if sec_plan else (i == 0)

        next_role = None
        if i + 1 < n_measures:
            next_plan = sec_by_measure.get(measures[i + 1].measure_index)
            if next_plan:
                next_role = next_plan.role

        m_shapes_list = measure_shapes[i]
        shape_names = [s.name for s in m_shapes_list]

        # Register shapes
        for s in m_shapes_list:
            if s.name not in uke_shapes:
                uke_shapes[s.name] = s

        is_partial = (m_beats < full_bar_beats - 1e-4)
        is_empty_printed = (m_spec is not None and len(m_spec.chords) == 0)

        if is_partial and (is_empty_printed or not m_arr.chords):
            # Pickup / partial bar: use technique 'hold' (no new attack) at all levels,
            # or a single light ↓ on its first melody onset at advanced; never a full pattern.
            actual_pid = "uke_held"
            technique = "hold"
            if difficulty == "advanced" and m_spec and m_spec.melody:
                tokens = m_spec.melody.strip().split()
                first_beat_idx = next((idx for idx, tok in enumerate(tokens) if tok not in ("0", "-", "")), 0)
                first_onset = min(float(first_beat_idx), max(0.0, m_beats - 0.25))
                if first_onset > 0.0:
                    s1 = UkeStroke(onset=0.0, duration=first_onset, kind="rest", strings=[], accent=False)
                    s2 = UkeStroke(
                        onset=first_onset,
                        duration=round(m_beats - first_onset, 4),
                        kind="down",
                        strings=[],
                        accent=False,
                    )
                    strokes = [s1, s2]
                else:
                    strokes = [UkeStroke(onset=0.0, duration=m_beats, kind="down", strings=[], accent=False)]
            else:
                strokes = [UkeStroke(onset=0.0, duration=m_beats, kind="down", strings=[], accent=False)]
        else:
            try:
                actual_pid, technique, strokes = generate_measure_strokes(
                    pattern_id=base_pid,
                    chords=m_arr.chords,
                    measure_beats=m_beats,
                    is_stop=is_stop,
                    fill=is_fill,
                    is_last_measure=is_last,
                    is_section_end=is_end,
                    is_section_start=is_start,
                    next_role=next_role,
                    difficulty=difficulty,
                    tempo_bpm=tempo_bpm,
                )
            except Exception as e:
                # Graceful degrade: single down-stroke held
                issues.append(
                    QualityIssue(
                        stage="arrange",
                        measure_index=m_arr.measure_index,
                        severity="info",
                        code="uke_measure_fallback",
                        message=f"Ukulele measure {m_arr.measure_index} fallback to single stroke: {e}",
                    )
                )
                actual_pid = "uke_held"
                technique = "hold"
                strokes = [UkeStroke(onset=0.0, duration=m_beats, kind="down", strings=[], accent=True)]

        # Register pattern legend
        if actual_pid not in pattern_legend:
            spec = PATTERN_REGISTRY.get(actual_pid)
            pattern_legend[actual_pid] = spec.legend if spec else actual_pid

        uke_measures.append(
            UkeMeasure(
                measure_index=m_arr.measure_index,
                shapes=shape_names,
                technique=technique,
                pattern_id=actual_pid,
                strokes=strokes,
            )
        )

    # 6. Notes
    notes: list[str] = []
    if capo > 0:
        notes.append(f"变调夹夹第{capo}品，按{shape_key}调指法弹奏（实际音高 {start_key} 调）")
    else:
        notes.append(f"不使用变调夹，按{shape_key}调指法弹奏（实际音高 {start_key} 调）")
    notes.append("定弦 G C E A（高音 G）")

    return Arrangement(
        instrument="ukulele",
        difficulty=difficulty,
        start_key=start_key,
        style=norm_style,
        measures=measures,
        notes=notes,
        issues=issues,
        sections=resolved_sections,
        capo=capo,
        shape_key=shape_key,
        uke_measures=uke_measures,
        uke_shapes=uke_shapes,
        pattern_legend=pattern_legend,
    )
