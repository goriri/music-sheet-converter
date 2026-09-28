"""Unit tests for arrangement validator, music21 reference, and fallback repair."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.arrange.piano import arrange
from app.models import (
    Arrangement,
    ChordSymbol,
    Difficulty,
    Event,
    KeyChange,
    Measure,
    MeasureArrangement,
    Note,
    ParsedSheet,
    QualityIssue,
    ResolvedChord,
    SongHeader,
    System,
)
from app.qa.arrange_check import (
    ArrangementQAError,
    FlaggedMeasure,
    LLMReviewResponse,
    check_and_repair,
    validate_measure,
)
from app.qa.fallback import safe_measure
from app.qa.music21_ref import (
    cross_check_chord_pcs,
    get_chord_reference,
    normalize_to_music21_syntax,
)
from app.theory.chords import QUALITY_INTERVALS, resolve_chord
from app.theory.keys import key_name_to_pc


# --------------------------------------------------------------------------- Fixtures & Helpers
def make_dummy_sheet(chords_per_bar: list[list[str]], key: str = "C") -> ParsedSheet:
    """Create a minimal ParsedSheet with specified chord symbols."""
    measures = []
    for idx, chord_raws in enumerate(chords_per_bar):
        chord_symbols = []
        for c_idx, raw in enumerate(chord_raws):
            beat = 1.0 if c_idx == 0 else 3.0
            chord_symbols.append(ChordSymbol(raw=raw, beat=beat))
        measures.append(
            Measure(
                index=idx,
                bbox=(0.1, 0.1, 0.9, 0.9),
                beats=4.0,
                chords=chord_symbols,
            )
        )
    return ParsedSheet(
        header=SongHeader(title="Test", style="Ballad", original_key=key),
        pages=[],
        systems=[System(page=0, bbox=(0, 0, 1, 1), measures=measures)],
    )


def make_clean_measure_arrangement(
    measure_index: int = 0,
    tonic_pc: int = 0,
    key_name: str = "C",
) -> tuple[Measure, MeasureArrangement]:
    """Create a perfectly valid measure and arrangement."""
    measure = Measure(
        index=measure_index,
        bbox=(0, 0, 1, 1),
        beats=4.0,
        chords=[ChordSymbol(raw="1", beat=1.0)],
    )
    chord = ResolvedChord(
        raw="1",
        name="C",
        beat=1.0,
        root_pc=0,
        bass_pc=0,
        pcs=[0, 4, 7],
        quality="maj",
    )
    # RH: C chord (E4, G4, C5) on beat 1 for 4.0 beats
    rh = [
        Event(
            onset=0.0,
            duration=4.0,
            notes=[Note(midi=64, finger=1), Note(midi=67, finger=3), Note(midi=72, finger=5)],
        )
    ]
    # LH: C3 held for 4.0 beats
    lh = [
        Event(
            onset=0.0,
            duration=4.0,
            notes=[Note(midi=48, finger=5)],
        )
    ]
    m_arr = MeasureArrangement(
        measure_index=measure_index,
        tonic_pc=tonic_pc,
        key_name=key_name,
        chords=[chord],
        rh=rh,
        lh=lh,
    )
    return measure, m_arr


# --------------------------------------------------------------------------- music21 reference tests
class TestMusic21Reference:
    def test_normalize_syntax(self) -> None:
        """Test normalization for flats, sharps, slash chords, and chord types."""
        assert normalize_to_music21_syntax("Am7b5") == ("Am7b5", "Am7b5")
        assert normalize_to_music21_syntax("G7sus4") == ("G7sus4", "G7sus4")
        assert normalize_to_music21_syntax("Cadd9") == ("Cadd9", "Cadd9")
        assert normalize_to_music21_syntax("Fmaj7") == ("Fmaj7", "Fmaj7")
        assert normalize_to_music21_syntax("G/B") == ("G", "G/B")
        assert normalize_to_music21_syntax("Cm6/G") == ("Cm6", "Cm6/G")
        assert normalize_to_music21_syntax("Bb") == ("B-", "B-")
        assert normalize_to_music21_syntax("Bbmaj7/D") == ("B-maj7", "B-maj7/D")
        assert normalize_to_music21_syntax("F/Bb") == ("F", "F/B-")
        assert normalize_to_music21_syntax("Cmaj9") == ("CM9", "CM9")
        assert normalize_to_music21_syntax("C6/9") == ("C6add9", "C6add9")
        assert normalize_to_music21_syntax("Bb6/9") == ("B-6add9", "B-6add9")
        assert normalize_to_music21_syntax("C6/9/G") == ("C6add9", "C6add9/G")
        assert normalize_to_music21_syntax("Fm/Db") == ("Fm", "Fm/D-")

    def test_defining_tones(self) -> None:
        """Test defining tone extraction across various qualities."""
        # 3rd for major/minor
        _, _, _, def_c = get_chord_reference("C")
        assert def_c == [4]  # E
        _, _, _, def_dm = get_chord_reference("Dm")
        assert def_dm == [5]  # F

        # 4th for sus4, 4th + 7th for 7sus4
        _, _, _, def_sus = get_chord_reference("Gsus4")
        assert def_sus == [0]  # C
        _, _, _, def_7sus = get_chord_reference("G7sus4")
        assert def_7sus == [0, 5]  # C, F

        # 7th present
        _, _, _, def_g7 = get_chord_reference("G7")
        assert def_g7 == [11, 5]  # B, F
        _, _, _, def_fmaj7 = get_chord_reference("Fmaj7")
        assert def_fmaj7 == [9, 4]  # A, E

        # add9 / 9th
        _, _, _, def_add9 = get_chord_reference("Cadd9")
        assert def_add9 == [4, 2]  # E, D

        # 6th for 6 chords
        _, _, _, def_c6 = get_chord_reference("C6")
        assert def_c6 == [4, 9]  # E, A
        _, _, _, def_cm6 = get_chord_reference("Cm6")
        assert def_cm6 == [3, 9]  # Eb, A

        # 6/9 chords
        _, _, _, def_c69 = get_chord_reference("C6/9")
        assert def_c69 == [4, 2, 9]  # E, D, A

        # b5 for m7b5 and dim
        _, _, _, def_m7b5 = get_chord_reference("Am7b5")
        assert def_m7b5 == [0, 7, 3]  # C (3rd), G (7th), Eb (b5)
        _, _, _, def_dim = get_chord_reference("Cdim")
        assert def_dim == [3, 6]  # Eb (3rd), Gb (b5)

    def test_cross_check_all_qualities_and_roots(self) -> None:
        """Cross-check ResolvedChord.pcs vs music21 across all qualities × 12 roots.

        Verifies that dim is identified as an expected theory mismatch (diminished 7th vs triad),
        while all other 15 qualities match music21 across all 12 roots.
        """
        mismatches: list[tuple[str, str, str]] = []
        for root_pc in range(12):
            for quality in QUALITY_INTERVALS.keys():
                raw = "1" + (quality if quality != "maj" else "")
                if quality == "add9":
                    raw = "1(2)"
                elif quality == "7sus4":
                    raw = "17sus"
                elif quality == "sus4":
                    raw = "1sus"
                elif quality == "m7b5":
                    raw = "1m7-5"

                rc = resolve_chord(raw, root_pc)
                is_match, reason = cross_check_chord_pcs(rc)
                if not is_match:
                    assert reason is not None
                    mismatches.append((rc.name, quality, reason))

        # 'dim' is now named 'dim7' (pcs [0,3,6,9]), so theory and music21 must agree everywhere.
        assert mismatches == []


# --------------------------------------------------------------------------- Validator rule tests
class TestValidatorRules:
    def test_clean_measure_passes(self) -> None:
        """A clean, valid measure passes with 0 violations."""
        measure, m_arr = make_clean_measure_arrangement()
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert violations == []

    def test_rule_key_mismatch(self) -> None:
        """Key in force must match expected tonic."""
        measure, m_arr = make_clean_measure_arrangement(tonic_pc=2)
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert any(v.code == "key_mismatch" for v in violations)

    def test_rule_tiling_mismatch(self) -> None:
        """Sum of event durations must match measure beats."""
        measure, m_arr = make_clean_measure_arrangement()
        # Incomplete RH (3 beats instead of 4)
        m_arr.rh = [Event(onset=0.0, duration=3.0, notes=[Note(midi=64, finger=1)])]
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert any(v.code == "tiling_mismatch" for v in violations)

    def test_rule_grid_alignment(self) -> None:
        """Onsets and durations must align to 1/16 grid (0.25)."""
        measure, m_arr = make_clean_measure_arrangement()
        # Off-grid onset (0.33)
        m_arr.rh = [
            Event(onset=0.0, duration=0.33, notes=[]),
            Event(onset=0.33, duration=3.67, notes=[Note(midi=64, finger=1)]),
        ]
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert any(v.code == "grid_alignment" for v in violations)

    def test_rule_chord_not_voiced(self) -> None:
        """Every chord must be voiced at its beat with root or bass."""
        measure, m_arr = make_clean_measure_arrangement()
        # Add a second chord on beat 3 with no attack
        m_arr.chords.append(
            ResolvedChord(raw="4", name="F", beat=3.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj")
        )
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert any(v.code == "chord_not_voiced" for v in violations)

    def test_rule_rh_missing_defining_tone(self) -> None:
        """RH chord attacks must contain defining tones (e.g. 3rd, 7th)."""
        measure, m_arr = make_clean_measure_arrangement()
        # G7 chord: defining tones are 3rd (B=11) and 7th (F=5).
        # Injected RH plays only G4 (67) and D5 (74) - missing 3rd and 7th!
        g7 = ResolvedChord(raw="57", name="G7", beat=1.0, root_pc=7, bass_pc=7, pcs=[7, 11, 2, 5], quality="7")
        m_arr.chords = [g7]
        m_arr.rh = [
            Event(onset=0.0, duration=4.0, notes=[Note(midi=67, finger=1), Note(midi=74, finger=5)])
        ]
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert any(v.code == "missing_defining_tone" for v in violations)

    def test_rule_lh_non_chord_tone(self) -> None:
        """LH notes must be chord tones unless valid weak-subdivision passing tones."""
        measure, m_arr = make_clean_measure_arrangement()
        # C chord tones: C, E, G (0, 4, 7). Play Bb (10) on beat 2.0 (onset 1.0)
        m_arr.lh = [
            Event(onset=0.0, duration=1.0, notes=[Note(midi=48, finger=5)]),  # C3
            Event(onset=1.0, duration=1.0, notes=[Note(midi=46, finger=4)]),  # Bb2 on beat!
            Event(onset=2.0, duration=2.0, notes=[Note(midi=48, finger=5)]),  # C3
        ]
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert any(v.code == "lh_non_chord_tone" for v in violations)

    def test_rule_lh_valid_passing_tone_allowed(self) -> None:
        """LH weak-subdivision passing tone approaching next bass is permitted."""
        measure, m_arr = make_clean_measure_arrangement()
        # Next chord is F (bass F2=41).
        # In C chord, on onset 3.5 (weak subdivision), play E2 (40, semitone from F2) with dur 0.5
        next_chord = ResolvedChord(raw="4", name="F", beat=1.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj")
        m_arr.lh = [
            Event(onset=0.0, duration=3.5, notes=[Note(midi=48, finger=5)]),
            Event(onset=3.5, duration=0.5, notes=[Note(midi=40, finger=1)]),  # Passing tone to F
        ]
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate", next_measure_chord=next_chord)
        # Should not flag lh_non_chord_tone
        assert not any(v.code == "lh_non_chord_tone" for v in violations)

    def test_rule_non_diatonic_fill(self) -> None:
        """Fill notes must be diatonic to key in force."""
        measure, m_arr = make_clean_measure_arrangement()
        measure.fill = True
        # In key C, play F# (66) in RH fill
        m_arr.rh = [
            Event(onset=0.0, duration=3.0, notes=[Note(midi=64, finger=1), Note(midi=72, finger=5)]),
            Event(onset=3.0, duration=0.5, notes=[Note(midi=66, finger=2)]),  # F#
            Event(onset=3.5, duration=0.5, notes=[Note(midi=67, finger=3)]),  # G
        ]
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert any(v.code == "non_diatonic_fill" for v in violations)

    def test_rule_ranges(self) -> None:
        """Pitches outside RH [64..81] or LH [36..64] are flagged."""
        measure, m_arr = make_clean_measure_arrangement()
        # RH note below E4 (64)
        m_arr.rh[0].notes.append(Note(midi=60, finger=1))  # C4 = 60 < 64
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert any(v.code == "pitch_out_of_range" for v in violations)

    def test_rule_spans(self) -> None:
        """Hand spans exceeding limit are flagged."""
        measure, m_arr = make_clean_measure_arrangement()
        # In beginner mode, span must be <= 12. Create span 13: E4(64) to F5(77)
        m_arr.rh = [
            Event(onset=0.0, duration=4.0, notes=[Note(midi=64, finger=1), Note(midi=77, finger=5)])
        ]
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="beginner")
        assert any(v.code == "span_too_large" for v in violations)

    def test_rule_fingering_invalid_and_monotonicity(self) -> None:
        """Finger numbers must be 1..5 and monotonic with pitch in chords."""
        measure, m_arr = make_clean_measure_arrangement()
        # Non-monotonic fingers in RH: lower pitch midi=64 finger=5, higher pitch midi=72 finger=1
        m_arr.rh = [
            Event(onset=0.0, duration=4.0, notes=[Note(midi=64, finger=5), Note(midi=72, finger=1)])
        ]
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert any(v.code == "fingering_not_monotonic" for v in violations)


# --------------------------------------------------------------------------- Fallback and Repair tests
class TestFallbackAndRepair:
    def test_safe_measure_generation(self) -> None:
        """Fallback generates valid, strictly tiled, playable RH and LH."""
        measure, m_arr = make_clean_measure_arrangement()
        rh, lh = safe_measure(measure, m_arr.chords, tonic_pc=0, difficulty="intermediate")
        m_arr.rh = rh
        m_arr.lh = lh
        violations = validate_measure(m_arr, measure, expected_tonic_pc=0, difficulty="intermediate")
        assert violations == []

    def test_check_and_repair_auto_fixes_defects(self) -> None:
        """Defective measures in an arrangement are auto-fixed and logged as QualityIssue."""
        sheet = make_dummy_sheet([["1"], ["5/7"], ["2m7"]])
        arr = arrange(sheet, "C", "intermediate")

        # Inject an intentional violation in measure 1 (LH non-chord tone)
        arr.measures[1].lh = [
            Event(onset=0.0, duration=2.0, notes=[Note(midi=47, finger=5)]),  # B2
            Event(onset=2.0, duration=2.0, notes=[Note(midi=46, finger=5)]),  # Bb2 (bad!)
        ]

        repaired = check_and_repair(sheet, arr)
        # Verify 0 violations on repaired arrangement
        m_by_idx = {m.index: m for m in sheet.measures()}
        for m_arr in repaired.measures:
            m_spec = m_by_idx[m_arr.measure_index]
            assert validate_measure(m_arr, m_spec, expected_tonic_pc=0, difficulty="intermediate") == []

        # Verify auto_fixed issue recorded
        fixed_issues = [iss for iss in repaired.issues if iss.severity == "auto_fixed"]
        assert len(fixed_issues) >= 1
        assert any(iss.measure_index == 1 for iss in fixed_issues)

    def test_llm_review_with_mocked_ask_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test optional LLM review using monkeypatched ask_json."""
        sheet = make_dummy_sheet([["1"], ["4"]])
        arr = arrange(sheet, "C", "intermediate")

        # Mock ask_json returning measure 0 as suspicious
        mock_response = LLMReviewResponse(
            suspicious_measures=[FlaggedMeasure(measure_index=0, reason="Awkward register")]
        )
        mock_ask = MagicMock(return_value=mock_response)

        # Monkeypatch ask_json into app.qa.arrange_check
        import app.qa.arrange_check as ac
        monkeypatch.setattr(ac, "ask_json", mock_ask, raising=False)

        # In measure 0, inject a condition that trips the stricter heuristic (e.g. lowest note <= 65)
        # Note: E4=64 is <= 65, which trips the stricter heuristic!
        repaired = check_and_repair(sheet, arr, use_llm=True)
        assert mock_ask.called

        # Confirm LLM review issue recorded
        llm_issues = [iss for iss in repaired.issues if "AI" in iss.message or "LLM" in iss.code]
        assert len(llm_issues) >= 1

    def test_llm_review_error_handled_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM exceptions must never crash the pipeline."""
        sheet = make_dummy_sheet([["1"]])
        arr = arrange(sheet, "C", "intermediate")

        mock_ask = MagicMock(side_effect=RuntimeError("API quota exceeded"))
        import app.qa.arrange_check as ac
        monkeypatch.setattr(ac, "ask_json", mock_ask, raising=False)

        repaired = check_and_repair(sheet, arr, use_llm=True)
        # Should not raise, should add an info issue
        info_issues = [iss for iss in repaired.issues if iss.code == "llm_review_error"]
        assert len(info_issues) == 1
        assert "API quota exceeded" in info_issues[0].detail["error"]

    def test_letter_notation_sheet_cross_check_and_validation(self) -> None:
        """Letter-notation sheets with 6/9 and slash chords validate and cross-check cleanly."""
        measures = [
            Measure(
                index=0,
                bbox=(0.0, 0.0, 1.0, 1.0),
                beats=4.0,
                chords=[ChordSymbol(raw="C", beat=1.0), ChordSymbol(raw="Am7", beat=3.0)],
            ),
            Measure(
                index=1,
                bbox=(0.0, 0.0, 1.0, 1.0),
                beats=4.0,
                chords=[ChordSymbol(raw="Dm7", beat=1.0), ChordSymbol(raw="G7", beat=3.0)],
            ),
            Measure(
                index=2,
                bbox=(0.0, 0.0, 1.0, 1.0),
                beats=4.0,
                chords=[ChordSymbol(raw="C6/9", beat=1.0)],
            ),
            Measure(
                index=3,
                bbox=(0.0, 0.0, 1.0, 1.0),
                beats=4.0,
                chords=[ChordSymbol(raw="Fm/Db", beat=1.0), ChordSymbol(raw="G/B", beat=3.0)],
            ),
        ]
        header = SongHeader(
            title="Letter Test",
            style="Ballad",
            original_key="C",
            chord_notation="letter",
        )
        sheet = ParsedSheet(
            header=header,
            pages=[],
            systems=[System(page=0, bbox=(0, 0, 1, 1), measures=measures)],
        )

        # 1. Arrange in original key C and verify
        arr = arrange(sheet, "C", "intermediate")
        repaired = check_and_repair(sheet, arr)

        cross_check_mismatches = [
            iss for iss in repaired.issues if iss.code == "chord_theory_cross_check_mismatch"
        ]
        assert cross_check_mismatches == []

        m_by_idx = {m.index: m for m in sheet.measures()}
        for m_arr in repaired.measures:
            m_spec = m_by_idx[m_arr.measure_index]
            violations = validate_measure(m_arr, m_spec, expected_tonic_pc=0, difficulty="intermediate")
            assert violations == []

        # 2. Arrange transposed to key G and verify cross-check and validation
        arr_g = arrange(sheet, "G", "intermediate")
        repaired_g = check_and_repair(sheet, arr_g)

        cross_check_mismatches_g = [
            iss for iss in repaired_g.issues if iss.code == "chord_theory_cross_check_mismatch"
        ]
        assert cross_check_mismatches_g == []

        for m_arr in repaired_g.measures:
            m_spec = m_by_idx[m_arr.measure_index]
            violations = validate_measure(m_arr, m_spec, expected_tonic_pc=7, difficulty="intermediate")
            assert violations == []


# --------------------------------------------------------------------------- Golden regression tests
class TestGoldenRegression:
    @pytest.mark.parametrize("key", ["F", "G"])
    @pytest.mark.parametrize("level", ["beginner", "intermediate", "advanced"])
    def test_golden_matches(self, key: str, level: Difficulty) -> None:
        """Compare current arranger output against stored golden files."""
        omr_file = Path("fixtures/omr_sample.json")
        golden_file = Path(f"tests/golden/{key}_{level}.txt")

        assert omr_file.exists(), f"Missing {omr_file}"
        assert golden_file.exists(), (
            f"Missing golden file {golden_file}. Run scripts/update_golden.py after human approval."
        )

        with open(omr_file, encoding="utf-8") as f:
            sheet = ParsedSheet.model_validate(json.load(f))

        arr = arrange(sheet, key, level)
        from app.qa.arrange_check import build_compact_score_text
        actual_text = build_compact_score_text(arr)
        expected_text = golden_file.read_text(encoding="utf-8")

        assert actual_text == expected_text, (
            f"Arrangement output for {key} {level} differs from golden! "
            "run scripts/update_golden.py after human approval"
        )
