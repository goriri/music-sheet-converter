"""Unit tests for Ukulele QA validator and auto-repair layer (U1..U7)."""
from __future__ import annotations

import copy
import pytest

from app.models import (
    Arrangement,
    ChordSymbol,
    KeyChange,
    Measure,
    MeasureArrangement,
    PageInfo,
    ParsedSheet,
    QualityIssue,
    ResolvedChord,
    SectionPlan,
    SongHeader,
    System,
    UkeMeasure,
    UkeShape,
    UkeStroke,
)
from app.qa.uke_check import _check_u1_shape_correctness, check_and_repair_uke, validate_uke


def make_valid_base_arrangement() -> tuple[ParsedSheet, Arrangement]:
    """Create a fully compliant, minimal Ukulele arrangement (in C, Capo 0)."""
    measures = [
        Measure(
            index=i,
            bbox=(0, 0, 1, 1),
            beats=4.0,
            chords=[ChordSymbol(raw="1", beat=1.0)],
            lyrics="测试",
        )
        for i in range(4)
    ]
    sys = System(page=0, bbox=(0, 0, 1, 1), section_label="主歌", measures=measures)
    sheet = ParsedSheet(
        header=SongHeader(title="尤克里里测试", original_key="C", time_signature="4/4"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[sys],
        key_changes=[],
    )

    # C major chord: C, E, G (pcs: 0, 4, 7)
    resolved_c = ResolvedChord(
        raw="1",
        name="C",
        root_pc=0,
        bass_pc=0,
        quality="maj",
        pcs=[0, 4, 7],
        beat=1.0,
    )
    # Shape for C: 0 0 0 3 (G4, C4, E4, C5 -> 7, 0, 4, 0)
    shape_c = UkeShape(
        name="C",
        frets=[0, 0, 0, 3],
        fingers=[0, 0, 0, 3],
        base_fret=1,
        barre=None,
    )

    strokes = [
        UkeStroke(onset=0.0, duration=1.0, kind="down", strings=[]),
        UkeStroke(onset=1.0, duration=1.0, kind="down", strings=[]),
        UkeStroke(onset=2.0, duration=1.0, kind="down", strings=[]),
        UkeStroke(onset=3.0, duration=1.0, kind="down", strings=[]),
    ]

    uke_measures = [
        UkeMeasure(
            measure_index=i,
            shapes=["C"],
            technique="strum",
            pattern_id="P1",
            strokes=strokes,
        )
        for i in range(4)
    ]

    arr_measures = [
        MeasureArrangement(
            measure_index=i,
            tonic_pc=0,
            key_name="C",
            chords=[resolved_c],
            rh=[],
            lh=[],
        )
        for i in range(4)
    ]

    sections = [
        SectionPlan(
            measure_index=i,
            section_id="A1",
            role="verse",
            energy=1,
            is_section_start=(i == 0),
            is_section_end=(i == 3),
            source="heuristic",
        )
        for i in range(4)
    ]

    arr = Arrangement(
        title="尤克里里测试",
        start_key="C",
        difficulty="beginner",
        shape_key="C",
        capo=0,
        pattern_legend={"P1": "↓  ↓  ↓  ↓"},
        uke_shapes={"C": shape_c},
        sections=sections,
        measures=arr_measures,
        uke_measures=uke_measures,
        notes=[],
        issues=[],
    )

    return sheet, arr


class TestUkeInvariantsU1ToU7:
    """Test individual invariant checks U1..U7."""

    def test_valid_arrangement_has_zero_violations(self):
        sheet, arr = make_valid_base_arrangement()
        violations = validate_uke(sheet, arr)
        assert violations == []

    # ------------------ U1: Shape Correctness ------------------
    def test_u1_shape_with_non_chord_tones(self):
        sheet, arr = make_valid_base_arrangement()
        # Replace C shape with frets (2, 2, 2, 0) sounding D major (A, D, F#, A)
        arr.uke_shapes["C"] = UkeShape(
            name="C",
            frets=[2, 2, 2, 0],
            fingers=[1, 2, 3, 0],
            base_fret=1,
            barre=None,
        )
        violations = validate_uke(sheet, arr)
        u1_viols = [v for v in violations if v.code == "uke_shape_incorrect"]
        assert len(u1_viols) > 0
        assert any("外音" in v.message or "缺少必要音" in v.message for v in u1_viols)

    def test_u1_shape_missing_essential_third(self):
        sheet, arr = make_valid_base_arrangement()
        # Shape (0, -1, -1, 3) sounding G4, C5 (no E!)
        arr.uke_shapes["C"] = UkeShape(
            name="C",
            frets=[0, -1, -1, 3],
            fingers=[0, 0, 0, 3],
            base_fret=1,
            barre=None,
        )
        # Set technique to fingerstyle to allow muted strings for U2
        arr.uke_measures[0].technique = "fingerstyle"
        violations = validate_uke(sheet, arr)
        u1_viols = [v for v in violations if v.code == "uke_shape_incorrect"]
        assert len(u1_viols) > 0
        assert any("缺少必要音" in v.message for v in u1_viols)

    # ------------------ U2: Playability ------------------
    def test_u2_finger_on_open_string(self):
        sheet, arr = make_valid_base_arrangement()
        # Fret 0 but finger 1
        arr.uke_shapes["C"] = UkeShape(
            name="C",
            frets=[0, 0, 0, 3],
            fingers=[1, 0, 0, 3],
            base_fret=1,
            barre=None,
        )
        violations = validate_uke(sheet, arr)
        u2_viols = [v for v in violations if v.code == "uke_playability_invalid"]
        assert len(u2_viols) > 0
        assert any("空弦或静音弦标记了手指号" in v.message for v in u2_viols)

    def test_u2_no_finger_on_fretted_string(self):
        sheet, arr = make_valid_base_arrangement()
        # Fret 3 but finger 0
        arr.uke_shapes["C"] = UkeShape(
            name="C",
            frets=[0, 0, 0, 3],
            fingers=[0, 0, 0, 0],
            base_fret=1,
            barre=None,
        )
        violations = validate_uke(sheet, arr)
        u2_viols = [v for v in violations if v.code == "uke_playability_invalid"]
        assert len(u2_viols) > 0
        assert any("按弦品未标记手指号" in v.message for v in u2_viols)

    def test_u2_span_too_large(self):
        sheet, arr = make_valid_base_arrangement()
        # Frets 1 and 4 -> span 3 (exceeds beginner limit of 2)
        arr.uke_shapes["C"] = UkeShape(
            name="C",
            frets=[1, 0, 0, 4],
            fingers=[1, 0, 0, 4],
            base_fret=1,
            barre=None,
        )
        violations = validate_uke(sheet, arr)
        u2_viols = [v for v in violations if v.code == "uke_playability_invalid"]
        assert any("品位跨度" in v.message for v in u2_viols)

    def test_u2_muted_string_in_strum(self):
        sheet, arr = make_valid_base_arrangement()
        # Muted string on strum
        arr.uke_shapes["C"] = UkeShape(
            name="C",
            frets=[-1, 0, 0, 3],
            fingers=[0, 0, 0, 3],
            base_fret=1,
            barre=None,
        )
        violations = validate_uke(sheet, arr)
        u2_viols = [v for v in violations if v.code == "uke_playability_invalid"]
        assert any("包含静音弦" in v.message for v in u2_viols)

    # ------------------ U3: Diagram ------------------
    def test_u3_base_fret_higher_than_min_fret(self):
        sheet, arr = make_valid_base_arrangement()
        # base_fret 3 but frets include fret 1
        arr.uke_shapes["C"] = UkeShape(
            name="C",
            frets=[1, 0, 0, 3],
            fingers=[1, 0, 0, 3],
            base_fret=3,
            barre=None,
        )
        violations = validate_uke(sheet, arr)
        u3_viols = [v for v in violations if v.code == "uke_diagram_invalid"]
        assert len(u3_viols) > 0
        assert any("标注起始品" in v.message for v in u3_viols)

    def test_u3_fret_span_exceeds_4_frets(self):
        sheet, arr = make_valid_base_arrangement()
        # base_fret 1, but max_fret 6 (6 - 1 = 5 > 4)
        arr.uke_shapes["C"] = UkeShape(
            name="C",
            frets=[1, 0, 0, 6],
            fingers=[1, 0, 0, 4],
            base_fret=1,
            barre=None,
        )
        violations = validate_uke(sheet, arr)
        u3_viols = [v for v in violations if v.code == "uke_diagram_invalid"]
        assert len(u3_viols) > 0
        assert any("超出4品显示范围" in v.message for v in u3_viols)

    # ------------------ U4: Rhythm ------------------
    def test_u4_unsorted_strokes(self):
        sheet, arr = make_valid_base_arrangement()
        # Unsorted onsets: 2.0 then 1.0
        arr.uke_measures[0].strokes = [
            UkeStroke(onset=0.0, duration=1.0, kind="down", strings=[]),
            UkeStroke(onset=2.0, duration=1.0, kind="down", strings=[]),
            UkeStroke(onset=1.0, duration=1.0, kind="down", strings=[]),
            UkeStroke(onset=3.0, duration=1.0, kind="down", strings=[]),
        ]
        violations = validate_uke(sheet, arr)
        u4_viols = [v for v in violations if v.code == "uke_rhythm_invalid"]
        assert any("未按时间升序排列" in v.message for v in u4_viols)

    def test_u4_overlapping_strokes(self):
        sheet, arr = make_valid_base_arrangement()
        # Overlap: onset 0.0 with dur 2.0, next onset 1.0
        arr.uke_measures[0].strokes = [
            UkeStroke(onset=0.0, duration=2.0, kind="down", strings=[]),
            UkeStroke(onset=1.0, duration=1.0, kind="down", strings=[]),
        ]
        violations = validate_uke(sheet, arr)
        u4_viols = [v for v in violations if v.code == "uke_rhythm_invalid"]
        assert any("重叠" in v.message for v in u4_viols)

    def test_u4_chord_missing_stroke_at_onset(self):
        sheet, arr = make_valid_base_arrangement()
        # Chord is on beat 1 (onset 0.0), but strokes start at onset 1.0
        arr.uke_measures[0].strokes = [
            UkeStroke(onset=1.0, duration=1.0, kind="down", strings=[]),
            UkeStroke(onset=2.0, duration=1.0, kind="down", strings=[]),
        ]
        violations = validate_uke(sheet, arr)
        u4_viols = [v for v in violations if v.code == "uke_rhythm_invalid"]
        assert any("缺少对应弹奏动作" in v.message for v in u4_viols)

    def test_u4_pluck_with_invalid_strings(self):
        sheet, arr = make_valid_base_arrangement()
        arr.difficulty = "advanced"  # Pluck allowed in advanced
        arr.uke_measures[0].strokes = [
            UkeStroke(onset=0.0, duration=1.0, kind="pluck", strings=[5]),  # String 5 invalid!
            UkeStroke(onset=1.0, duration=3.0, kind="down", strings=[]),
        ]
        violations = validate_uke(sheet, arr)
        u4_viols = [v for v in violations if v.code == "uke_rhythm_invalid"]
        assert any("勾弦指令弦号无效" in v.message for v in u4_viols)

    # ------------------ U5: Sections ------------------
    def test_u5_pattern_not_in_legend(self):
        sheet, arr = make_valid_base_arrangement()
        arr.uke_measures[0].pattern_id = "P99"
        violations = validate_uke(sheet, arr)
        u5_viols = [v for v in violations if v.code == "uke_section_pattern_inconsistent"]
        assert any("未在节奏图例" in v.message for v in u5_viols)

    def test_u5_more_than_two_patterns_in_section(self):
        sheet, arr = make_valid_base_arrangement()
        # 4 measures in A1: assign P1, P2, P3 to m0..m2 (excluding m3)
        arr.pattern_legend["P2"] = "↓↑ ↓↑"
        arr.pattern_legend["P3"] = "↓  ↓"
        arr.uke_measures[0].pattern_id = "P1"
        arr.uke_measures[1].pattern_id = "P2"
        arr.uke_measures[2].pattern_id = "P3"
        violations = validate_uke(sheet, arr)
        u5_viols = [v for v in violations if v.code == "uke_section_pattern_inconsistent"]
        assert any("超过2种不同节奏型" in v.message for v in u5_viols)

    # ------------------ U6: Key and Consistency ------------------
    def test_u6_key_math_mismatch(self):
        sheet, arr = make_valid_base_arrangement()
        # shape_key C (0) + capo 2 != start_key E (4)
        arr.capo = 2
        arr.start_key = "E"
        arr.shape_key = "C"
        violations = validate_uke(sheet, arr)
        u6_viols = [v for v in violations if v.code == "uke_key_or_count_inconsistent"]
        assert any("音高不相符" in v.message for v in u6_viols)

    def test_u6_capo_out_of_bounds(self):
        sheet, arr = make_valid_base_arrangement()
        arr.capo = 9
        violations = validate_uke(sheet, arr)
        u6_viols = [v for v in violations if v.code == "uke_key_or_count_inconsistent"]
        assert any("变调夹品位" in v.message for v in u6_viols)

    def test_u6_length_mismatch(self):
        sheet, arr = make_valid_base_arrangement()
        arr.uke_measures = arr.uke_measures[:2]  # Only 2 uke measures vs 4 sheet measures
        violations = validate_uke(sheet, arr)
        u6_viols = [v for v in violations if v.code == "uke_key_or_count_inconsistent"]
        assert any("小节数" in v.message for v in u6_viols)

    def test_u6_shape_missing_from_dictionary(self):
        sheet, arr = make_valid_base_arrangement()
        arr.uke_measures[0].shapes = ["G"]  # G is not in uke_shapes
        violations = validate_uke(sheet, arr)
        u6_viols = [v for v in violations if v.code == "uke_key_or_count_inconsistent"]
        assert any("未在 uke_shapes 字典中提供" in v.message for v in u6_viols)

    # ------------------ U7: Difficulty ------------------
    def test_u7_beginner_prohibits_chuck(self):
        sheet, arr = make_valid_base_arrangement()
        arr.difficulty = "beginner"
        arr.uke_measures[0].strokes = [
            UkeStroke(onset=0.0, duration=1.0, kind="chuck", strings=[]),
            UkeStroke(onset=1.0, duration=3.0, kind="down", strings=[]),
        ]
        violations = validate_uke(sheet, arr)
        u7_viols = [v for v in violations if v.code == "uke_difficulty_violation"]
        assert any("禁止使用切音" in v.message for v in u7_viols)

    def test_u7_beginner_prohibits_pluck(self):
        sheet, arr = make_valid_base_arrangement()
        arr.difficulty = "beginner"
        arr.uke_measures[0].strokes = [
            UkeStroke(onset=0.0, duration=1.0, kind="pluck", strings=[1, 2]),
            UkeStroke(onset=1.0, duration=3.0, kind="down", strings=[]),
        ]
        violations = validate_uke(sheet, arr)
        u7_viols = [v for v in violations if v.code == "uke_difficulty_violation"]
        assert any("禁止使用勾弦" in v.message for v in u7_viols)

    def test_u7_beginner_stroke_count_cap(self):
        sheet, arr = make_valid_base_arrangement()
        arr.difficulty = "beginner"
        # 5 strokes in 4-beat measure
        arr.uke_measures[0].strokes = [
            UkeStroke(onset=i * 0.7, duration=0.6, kind="down", strings=[]) for i in range(5)
        ]
        violations = validate_uke(sheet, arr)
        u7_viols = [v for v in violations if v.code == "uke_difficulty_violation"]
        assert any("动作数量" in v.message for v in u7_viols)

    def test_u7_intermediate_pluck_after_beat1_prohibited(self):
        sheet, arr = make_valid_base_arrangement()
        arr.difficulty = "intermediate"
        arr.uke_measures[0].strokes = [
            UkeStroke(onset=0.0, duration=1.0, kind="down", strings=[]),
            UkeStroke(onset=1.0, duration=1.0, kind="pluck", strings=[1, 2]),  # Pluck on beat 2!
            UkeStroke(onset=2.0, duration=2.0, kind="down", strings=[]),
        ]
        violations = validate_uke(sheet, arr)
        u7_viols = [v for v in violations if v.code == "uke_difficulty_violation"]
        assert any("仅允许在第1拍出现勾弦" in v.message for v in u7_viols)


class TestUkeAutoRepair:
    """Test check_and_repair_uke fixes violations and logs QualityIssues."""

    def test_repair_fixes_corrupted_arrangement_to_zero_violations(self):
        sheet, arr = make_valid_base_arrangement()

        # Introduce multiple deliberate violations:
        # 1. Invalid key math: shape_key "C", capo 2, start_key "C"
        arr.capo = 2
        arr.start_key = "D"
        arr.shape_key = "F"  # Corrupted shape_key

        # 2. Length mismatch: truncate uke_measures
        arr.uke_measures = arr.uke_measures[:2]

        # 3. Unregistered pattern_id
        arr.uke_measures[0].pattern_id = "UNKNOWN_RHYTHM"

        # 4. Overlapping / invalid rhythm
        arr.uke_measures[0].strokes = [
            UkeStroke(onset=0.0, duration=3.0, kind="down", strings=[]),
            UkeStroke(onset=1.0, duration=1.0, kind="down", strings=[]),
        ]

        # 5. Invalid shape with wrong frets
        arr.uke_shapes["C"] = UkeShape(
            name="C",
            frets=[9, 9, 9, 9],  # Wrong frets
            fingers=[1, 1, 1, 1],
            base_fret=1,
            barre=None,
        )

        # Confirm corrupted state fails validation
        initial_violations = validate_uke(sheet, arr)
        assert len(initial_violations) > 0

        # Run auto repair
        repaired = check_and_repair_uke(sheet, arr)

        # Repaired arrangement must have zero violations
        post_violations = validate_uke(sheet, repaired)
        assert post_violations == [], f"Remaining violations after repair: {post_violations}"

        # Must record auto_fixed issues
        assert len(repaired.issues) > 0
        assert all(iss.severity in ("auto_fixed", "info") for iss in repaired.issues)

    def test_repair_never_raises_on_empty_or_broken_input(self):
        empty_sheet = ParsedSheet(
            header=SongHeader(title="空谱", original_key="C", time_signature="4/4"),
            pages=[],
            systems=[],
        )
        empty_arr = Arrangement(
            title="空编曲",
            start_key="C",
            difficulty="beginner",
            shape_key="C",
            capo=0,
            pattern_legend={},
            uke_shapes={},
            sections=[],
            measures=[],
            uke_measures=[],
            notes=[],
            issues=[],
        )
        repaired = check_and_repair_uke(empty_sheet, empty_arr)
        assert repaired is not None


class TestUkeAdd9AndSimplificationRegressions:
    """Regression tests for add9 tones and beginner simplification."""

    def test_fadd9_shape_has_no_missing_tones_at_intermediate(self):
        # Case 1: Fadd9 sounding chord [5, 9, 0, 7] with shape Fadd9 (0, 0, 1, 0), capo 0
        chord = ResolvedChord(
            raw="Fadd9",
            name="Fadd9",
            beat=1.0,
            root_pc=5,
            bass_pc=5,
            pcs=[5, 9, 0, 7],
            quality="add9",
        )
        shape = UkeShape(name="Fadd9", frets=(0, 0, 1, 0), fingers=(0, 0, 1, 0), base_fret=1, barre=None)
        viols = _check_u1_shape_correctness(chord, shape, capo=0, m_idx=0, difficulty="intermediate")
        assert viols == [], f"Unexpected violations for Fadd9: {viols}"

    def test_cadd9_for_ebadd9_has_no_missing_tones_at_intermediate(self):
        # Case 2: Ebadd9 sounding chord [3, 7, 10, 5] with shape Cadd9 (0, 2, 0, 3), capo 3
        chord = ResolvedChord(
            raw="Ebadd9",
            name="Ebadd9",
            beat=1.0,
            root_pc=3,
            bass_pc=3,
            pcs=[3, 7, 10, 5],
            quality="add9",
        )
        shape = UkeShape(name="Cadd9", frets=(0, 2, 0, 3), fingers=(0, 1, 0, 2), base_fret=1, barre=None)
        viols = _check_u1_shape_correctness(chord, shape, capo=3, m_idx=0, difficulty="intermediate")
        assert viols == [], f"Unexpected violations for Cadd9 as Ebadd9: {viols}"

    def test_c_for_ebadd9_simplified_valid_at_beginner(self):
        # Case 3: Ebadd9 sounding chord [3, 7, 10, 5] simplified to shape C (0, 0, 0, 3), capo 3 at beginner
        chord = ResolvedChord(
            raw="Ebadd9",
            name="Ebadd9",
            beat=1.0,
            root_pc=3,
            bass_pc=3,
            pcs=[3, 7, 10, 5],
            quality="add9",
        )
        shape = UkeShape(name="C", frets=(0, 0, 0, 3), fingers=(0, 0, 0, 3), base_fret=1, barre=None)
        viols = _check_u1_shape_correctness(chord, shape, capo=3, m_idx=0, difficulty="beginner")
        assert viols == [], f"Unexpected violations for beginner C shape as Ebadd9: {viols}"

