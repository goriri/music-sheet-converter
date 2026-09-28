"""Property-based tests for arrangement generator and QA repair layer using Hypothesis."""
from __future__ import annotations

from hypothesis import given, settings, strategies as st
from app.arrange.piano import arrange
from app.models import (
    ChordSymbol,
    Difficulty,
    KeyChange,
    Measure,
    ParsedSheet,
    SongHeader,
    System,
)
from app.qa.arrange_check import check_and_repair, validate_measure
from app.theory.keys import key_name_to_pc

# Taiwanese number-notation chord grammar tokens
TAIWANESE_CHORDS = [
    # Basic triads & 7ths
    "1", "2m", "3m", "4", "5", "6m", "7dim",
    "1(2)", "2(2)", "4(2)", "5(2)", "6m9",
    "1M7", "4M7", "57", "67", "2m7", "3m7", "6m7",
    # Color chords & alterations
    "57sus", "1sus", "5sus", "6m7-5", "7m7-5", "1aug", "5aug",
    "b7", "6b", "3b", "7b", "1#", "4#", "5#",
    # Slash chords
    "5/7", "1/3", "4/5", "2m7/5", "5m/7b", "67/1#", "17/7b", "6m/1",
    "1/5", "4/1", "2m/4", "5/2",
]

START_KEYS = ["C", "F", "G", "Bb", "D", "Eb", "A", "F#", "Ab", "Db", "B", "E"]
LEVELS: list[Difficulty] = ["beginner", "intermediate", "advanced"]


@st.composite
def random_parsed_sheets(draw: st.DrawFn) -> ParsedSheet:
    """Generate arbitrary valid ParsedSheets covering Taiwanese chart conventions."""
    num_measures = draw(st.integers(min_value=2, max_value=4))
    measures: list[Measure] = []

    for i in range(num_measures):
        num_chords = draw(st.integers(min_value=1, max_value=2))
        beats = [1.0] if num_chords == 1 else [1.0, draw(st.sampled_from([2.5, 3.0]))]
        chords = [ChordSymbol(raw=draw(st.sampled_from(TAIWANESE_CHORDS)), beat=b) for b in beats]

        is_stop = draw(st.booleans())
        fill = draw(st.booleans())
        bass_hint = draw(st.sampled_from([None, "11 11 11 112", "11 11 11 112~"]))

        measures.append(
            Measure(
                index=i,
                bbox=(0.0, 0.0, 1.0, 1.0),
                beats=4.0,
                chords=chords,
                is_stop=is_stop,
                fill=fill,
                bass_hint=bass_hint,
            )
        )

    section_label = draw(st.sampled_from([None, "Only PN(RHY)", "+Bs in", "OG in"]))
    systems = [
        System(
            page=0,
            bbox=(0.0, 0.0, 1.0, 1.0),
            section_label=section_label,
            measures=measures,
        )
    ]

    key_changes: list[KeyChange] = []
    if num_measures >= 3 and draw(st.booleans()):
        kc_measure = draw(st.integers(min_value=1, max_value=num_measures - 1))
        shift = draw(st.sampled_from([-2, 1, 2, 3]))
        key_changes.append(
            KeyChange(at_measure=kc_measure, raw="轉調", semitones=shift)
        )

    return ParsedSheet(
        header=SongHeader(
            title="Hypothesis Song",
            style=draw(st.sampled_from(["Slow Soul", "Ballad", "Pop", "Slow Rock"])),
        ),
        pages=[],
        systems=systems,
        key_changes=key_changes,
    )


# --------------------------------------------------------------------------- Property Tests
@settings(max_examples=200, deadline=None)
@given(
    sheet=random_parsed_sheets(),
    start_key=st.sampled_from(START_KEYS),
    level=st.sampled_from(LEVELS),
)
def test_raw_arranger_output_has_zero_violations(
    sheet: ParsedSheet,
    start_key: str,
    level: Difficulty,
) -> None:
    """Raw arranger output must not raise, and validator reports 0 violations.

    NOTE: This test is EXPECTED to fail while the arranger has bugs (e.g. slash chord
    LH non-chord tones, missing 7th in beginner color chords). We do not mark xfail;
    Hypothesis discovers minimal failing examples for bug discovery.
    """
    arr = arrange(sheet, start_key, level)
    assert arr is not None

    measures_by_idx = {m.index: m for m in sheet.measures()}
    start_pc = key_name_to_pc(start_key)
    curr_pc = start_pc
    key_changes_by_m = {kc.at_measure: kc.semitones for kc in sheet.key_changes}

    all_violations = []
    for i, m_arr in enumerate(arr.measures):
        if m_arr.measure_index in key_changes_by_m:
            curr_pc = (curr_pc + key_changes_by_m[m_arr.measure_index]) % 12
        m_spec = measures_by_idx[m_arr.measure_index]
        nxt = (
            arr.measures[i + 1].chords[0]
            if i + 1 < len(arr.measures) and arr.measures[i + 1].chords
            else None
        )
        violations = validate_measure(m_arr, m_spec, curr_pc, arr.difficulty, nxt)
        if violations:
            all_violations.extend((m_arr.measure_index, v) for v in violations)

    assert len(all_violations) == 0, (
        f"Raw arranger output has {len(all_violations)} violations on {start_key} {level}: "
        f"{[f'm{m_idx+1}:{v.code}' for m_idx, v in all_violations]}"
    )


@settings(max_examples=200, deadline=None)
@given(
    sheet=random_parsed_sheets(),
    start_key=st.sampled_from(START_KEYS),
    level=st.sampled_from(LEVELS),
)
def test_check_and_repair_output_always_zero_violations(
    sheet: ParsedSheet,
    start_key: str,
    level: Difficulty,
) -> None:
    """check_and_repair must ALWAYS auto-repair any arrangement defects to exactly 0 violations."""
    raw_arr = arrange(sheet, start_key, level)
    repaired_arr = check_and_repair(sheet, raw_arr)

    measures_by_idx = {m.index: m for m in sheet.measures()}
    start_pc = key_name_to_pc(start_key)
    curr_pc = start_pc
    key_changes_by_m = {kc.at_measure: kc.semitones for kc in sheet.key_changes}

    for i, m_arr in enumerate(repaired_arr.measures):
        if m_arr.measure_index in key_changes_by_m:
            curr_pc = (curr_pc + key_changes_by_m[m_arr.measure_index]) % 12
        m_spec = measures_by_idx[m_arr.measure_index]
        nxt = (
            repaired_arr.measures[i + 1].chords[0]
            if i + 1 < len(repaired_arr.measures) and repaired_arr.measures[i + 1].chords
            else None
        )
        violations = validate_measure(m_arr, m_spec, curr_pc, repaired_arr.difficulty, nxt)
        assert len(violations) == 0, (
            f"Repaired arrangement still has violations on measure {m_arr.measure_index + 1}: "
            f"{[v.code for v in violations]}"
        )
