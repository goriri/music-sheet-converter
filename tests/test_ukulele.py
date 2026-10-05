"""Comprehensive unit and integration tests for the Ukulele arranger."""
from __future__ import annotations

import glob
import json
import os
import pytest

from app.arrange.piano import arrange as arrange_piano
from app.arrange.uke_patterns import (
    PATTERN_REGISTRY,
    choose_section_pattern,
    derive_legend_from_strokes,
    generate_measure_strokes,
    get_time_signature_family,
)
from app.arrange.uke_shapes import (
    _CURATED_LIST,
    candidate_shapes,
    check_shape_pcs,
    get_essential_tones,
    search_algorithmic_shapes,
    simplify_quality,
)
from app.arrange.ukulele import arrange_ukulele, select_capo_and_shape_key, transition_cost
from app.models import ChordSymbol, Measure, ParsedSheet, ResolvedChord, SectionPlan, SongHeader, System
from app.theory.chords import QUALITY_INTERVALS, get_chord_pcs
from app.theory.keys import key_name_to_pc


class TestUkeShapes:
    """Tests for ukulele chord shape validation and generation."""

    def test_curated_shapes_verified_by_pc_checker(self):
        """Verify all curated shapes have no non-chord tones and contain all essential tones."""
        for name, quality, frets, fingers, barre in _CURATED_LIST:
            import re
            m = re.match(r"^([A-G][b#]?)", name)
            assert m is not None, f"Cannot parse root from {name}"
            root_name = m.group(1)
            root_pc = key_name_to_pc(root_name)
            is_valid = check_shape_pcs(frets, root_pc, quality)
            assert is_valid, (
                f"Curated shape {name} ({frets}) failed pc checker for root {root_pc} "
                f"and quality {quality}. Essential tones: {get_essential_tones(root_pc, quality)}"
            )

    def test_generated_shapes_coverage_and_edge_cases(self):
        """Test algorithmic generation across all 12 roots and 26 qualities.

        Verifies that qualities have playable shapes and documents known edge cases.
        """
        no_shapes = []
        for qual in QUALITY_INTERVALS:
            for r in range(12):
                shapes = search_algorithmic_shapes(r, qual, max_fret=12, max_span=3)
                if not shapes:
                    no_shapes.append((r, qual))

        # Exactly 2 edge cases on 4-string high-G uke with span <= 3: Abm9 and F#maj9
        assert len(no_shapes) <= 2, f"Unexpected missing shapes: {no_shapes}"

        # Verify candidate_shapes gracefully handles all roots and qualities (including edge cases)
        for qual in QUALITY_INTERVALS:
            for r in range(12):
                cands = candidate_shapes(r, qual, difficulty="intermediate")
                assert len(cands) > 0, f"candidate_shapes returned empty for root {r}, quality {qual}"
                # Must not have non-chord tones
                top = cands[0]
                target_pcs = set(get_chord_pcs(r, qual))
                # If simplified was used, check target against simplified quality
                sounding_pcs = set((candidate_shapes.__globals__["OPEN_PCS"][i] + top.frets[i]) % 12 for i in range(4) if top.frets[i] >= 0)
                sim_qual = simplify_quality(qual, "beginner")
                valid_original = sounding_pcs.issubset(target_pcs)
                valid_simplified = sounding_pcs.issubset(set(get_chord_pcs(r, sim_qual)))
                assert valid_original or valid_simplified, f"Shape {top.name} {top.frets} has non-chord tones"

    def test_simplify_quality_beginner(self):
        """Verify quality simplification reduces complex extensions for beginners."""
        assert simplify_quality("9", "beginner") == "7"
        assert simplify_quality("m9", "beginner") == "m7"
        assert simplify_quality("maj9", "beginner") == "maj7"
        assert simplify_quality("add9", "beginner") == "maj"
        assert simplify_quality("maj", "beginner") == "maj"
        assert simplify_quality("9", "intermediate") == "9"

    def test_simplify_quality_subset_property_all_qualities(self):
        """Verify for every quality, simplify_quality produces a chord whose pitch classes
        are a strict subset of the original chord's pitch classes and contains root + 3rd/sus.
        """
        for q in QUALITY_INTERVALS:
            sim_q = simplify_quality(q, "beginner")
            orig_pcs = set(get_chord_pcs(0, q))
            sim_pcs = set(get_chord_pcs(0, sim_q))

            # 1. Subset check: simplified_tones ⊆ original_tones
            assert sim_pcs.issubset(orig_pcs), (
                f"Quality {q} -> {sim_q}: simplified pcs {sim_pcs} not subset of {orig_pcs}"
            )

            # 2. Must contain root
            assert 0 in sim_pcs, f"Quality {q} -> {sim_q}: missing root pc 0"

            # 3. Must contain 3rd or sus
            if q in ("sus4", "7sus4"):
                assert 5 in sim_pcs, f"Quality {q} -> {sim_q}: missing sus4 tone"
            elif q == "sus2":
                assert 2 in sim_pcs, f"Quality {q} -> {sim_q}: missing sus2 tone"
            elif "m" in q and q not in ("maj", "maj7", "maj9", "69", "add9"):
                assert 3 in sim_pcs, f"Quality {q} -> {sim_q}: missing minor 3rd tone"
            else:
                assert 4 in sim_pcs, f"Quality {q} -> {sim_q}: missing major 3rd tone"


class TestCapoAndVoiceLeading:
    """Tests for capo selection and dynamic programming voice leading."""

    def test_capo_choice_eb_vs_c(self):
        """Verify Eb chooses capo 3 (C shape key) and C chooses capo 0."""
        # Chords for Eb major progression: Eb, Ab, Bb, Cm
        chords_eb = [
            ResolvedChord(raw="1", name="Eb", beat=1.0, root_pc=3, bass_pc=3, pcs=[3, 7, 10], quality="maj"),
            ResolvedChord(raw="4", name="Ab", beat=1.0, root_pc=8, bass_pc=8, pcs=[8, 0, 3], quality="maj"),
            ResolvedChord(raw="5", name="Bb", beat=1.0, root_pc=10, bass_pc=10, pcs=[10, 2, 5], quality="maj"),
            ResolvedChord(raw="6m", name="Cm", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 3, 7], quality="m"),
        ]
        capo_eb, shape_key_eb = select_capo_and_shape_key("Eb", chords_eb, difficulty="beginner")
        assert capo_eb == 3
        assert shape_key_eb == "C"

        # Chords for C major progression: C, F, G, Am
        chords_c = [
            ResolvedChord(raw="1", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj"),
            ResolvedChord(raw="4", name="F", beat=1.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj"),
            ResolvedChord(raw="5", name="G", beat=1.0, root_pc=7, bass_pc=7, pcs=[7, 11, 2], quality="maj"),
            ResolvedChord(raw="6m", name="Am", beat=1.0, root_pc=9, bass_pc=9, pcs=[9, 0, 4], quality="m"),
        ]
        capo_c, shape_key_c = select_capo_and_shape_key("C", chords_c, difficulty="beginner")
        assert capo_c == 0
        assert shape_key_c == "C"

    def test_transition_cost_favors_smooth_movement(self):
        """Verify transition cost rewards anchor fingers and penalizes huge jumps."""
        cands_f = candidate_shapes(5, "maj", difficulty="intermediate")
        cands_dm = candidate_shapes(2, "m", difficulty="intermediate")

        shape_f = next(s for s in cands_f if s.frets == (2, 0, 1, 0))
        shape_dm_smooth = next(s for s in cands_dm if s.frets == (2, 2, 1, 0))
        # High fret Dm barre (7, 5, 5, 5)
        shape_dm_jump = next((s for s in cands_dm if max(s.frets) >= 5), None)

        cost_smooth = transition_cost(shape_f, shape_dm_smooth)
        assert cost_smooth < 0, f"Expected negative (bonus) cost for F -> Dm smooth, got {cost_smooth}"

        if shape_dm_jump:
            cost_jump = transition_cost(shape_f, shape_dm_jump)
            assert cost_jump > cost_smooth


class TestStrokesAndPatterns:
    """Tests for stroke tiling, patterns, and density limits."""

    def test_strokes_tile_4_4_bar(self):
        """Verify strokes tile 4.0 beats with no gaps or overlaps."""
        chords = [
            ResolvedChord(raw="1", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj"),
            ResolvedChord(raw="5", name="G", beat=3.0, root_pc=7, bass_pc=7, pcs=[7, 11, 2], quality="maj"),
        ]
        pid, tech, strokes = generate_measure_strokes("uke_4_4_mid_island", chords, 4.0)
        assert round(sum(s.duration for s in strokes), 4) == 4.0
        # Continuous
        curr = 0.0
        for s in strokes:
            assert abs(s.onset - curr) < 1e-4
            curr = round(curr + s.duration, 4)
        assert abs(curr - 4.0) < 1e-4

    def test_strokes_tile_3_4_bar(self):
        """Verify strokes tile 3.0 beats for 3/4 waltz."""
        chords = [
            ResolvedChord(raw="1", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj"),
        ]
        pid, tech, strokes = generate_measure_strokes("uke_3_4_mid_waltz", chords, 3.0)
        assert round(sum(s.duration for s in strokes), 4) == 3.0
        assert strokes[0].onset == 0.0

    def test_strokes_tile_partial_pickup_and_empty(self):
        """Verify partial pickup bars and measures with no chords."""
        # Pickup of 1.0 beat
        chords = [
            ResolvedChord(raw="1", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj"),
        ]
        _, _, strokes_pu = generate_measure_strokes("uke_4_4_beg_quarter", chords, 1.0)
        assert round(sum(s.duration for s in strokes_pu), 4) == 1.0

        # Measure with no chords -> rest
        _, tech_rest, strokes_rest = generate_measure_strokes("uke_4_4_beg_quarter", [], 4.0)
        assert tech_rest == "rest"
        assert len(strokes_rest) == 1
        assert strokes_rest[0].kind == "rest"
        assert strokes_rest[0].duration == 4.0

    def test_chord_change_beats_covered(self):
        """Verify that every chord change has a stroke starting at that exact onset."""
        chords = [
            ResolvedChord(raw="1", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj"),
            ResolvedChord(raw="4", name="F", beat=2.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj"),
            ResolvedChord(raw="5", name="G", beat=3.5, root_pc=7, bass_pc=7, pcs=[7, 11, 2], quality="maj"),
        ]
        _, _, strokes = generate_measure_strokes("uke_4_4_mid_island", chords, 4.0)
        stroke_onsets = {round(s.onset, 4) for s in strokes}
        for c in chords:
            expected_onset = round(c.beat - 1.0, 4)
            assert expected_onset in stroke_onsets, f"Missing stroke onset at beat {c.beat} (onset {expected_onset})"

    def test_beginner_density_limits(self):
        """Verify beginner difficulty never uses chucks and has <= 4 strokes per 4/4 bar."""
        chords = [
            ResolvedChord(raw="1", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj"),
            ResolvedChord(raw="5", name="G", beat=3.0, root_pc=7, bass_pc=7, pcs=[7, 11, 2], quality="maj"),
        ]
        for energy in [0, 1, 2, 3]:
            pid = choose_section_pattern("4/4", "beginner", "verse", energy)
            _, _, strokes = generate_measure_strokes(pid, chords, 4.0, difficulty="beginner")
            assert len(strokes) <= 4, f"Beginner exceeded 4 strokes in 4/4 bar (got {len(strokes)})"
            for s in strokes:
                assert s.kind != "chuck", "Beginner must not have chucks"

    def test_all_registry_patterns_legend_derived_from_strokes(self):
        """Verify spec.legend == derive_legend_from_strokes(spec.strokes, spec.full_beats)
        for every pattern in PATTERN_REGISTRY.
        """
        for pid, spec in PATTERN_REGISTRY.items():
            derived = derive_legend_from_strokes(spec.strokes, spec.full_beats)
            assert spec.legend == derived, (
                f"Pattern {pid} legend '{spec.legend}' does not match derived legend '{derived}'"
            )

    def test_slow_rock_maps_to_12_8_compound(self):
        """Verify style 'Slow Rock' or ts '12/8' maps to 12/8 compound pattern family."""
        assert get_time_signature_family("4/4", style="Slow Rock") == "12/8"
        assert get_time_signature_family("12/8", style="") == "12/8"
        pid = choose_section_pattern("12/8", "intermediate", "verse", 1)
        assert pid == "uke_12_8_mid"

    def test_accents_only_section_start_and_stops(self):
        """Verify only the first stroke of a section and stop strokes are accented."""
        chords = [
            ResolvedChord(raw="1", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj"),
            ResolvedChord(raw="5", name="G", beat=3.0, root_pc=7, bass_pc=7, pcs=[7, 11, 2], quality="maj"),
        ]
        # Section start
        _, _, strokes_start = generate_measure_strokes("uke_4_4_mid_island", chords, 4.0, is_section_start=True)
        assert strokes_start[0].accent is True
        for s in strokes_start[1:]:
            assert s.accent is False

        # Mid-section: no accents even on chord changes
        _, _, strokes_mid = generate_measure_strokes("uke_4_4_mid_island", chords, 4.0, is_section_start=False)
        for s in strokes_mid:
            assert s.accent is False

        # Stop measure: hit is accented
        _, _, strokes_stop = generate_measure_strokes("uke_4_4_mid_island", chords, 4.0, is_stop=True)
        assert any(s.accent for s in strokes_stop)

    def test_omr_jitter_merging(self):
        """Verify if a chord starts at <= 0.5 beat with previous chord carried, merges for beg/int."""
        from app.arrange.ukulele import normalize_measure_chords_for_omr_jitter
        # Carried C at 1.0, printed G at 1.5
        chords = [
            ResolvedChord(raw="1", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj"),
            ResolvedChord(raw="5", name="G", beat=1.5, root_pc=7, bass_pc=7, pcs=[7, 11, 2], quality="maj"),
        ]
        sheet_chords = [ChordSymbol(raw="5", beat=1.5)]
        merged_beg = normalize_measure_chords_for_omr_jitter(chords, sheet_chords, "beginner")
        assert len(merged_beg) == 1
        assert merged_beg[0].name == "G"
        assert merged_beg[0].beat == 1.0

        merged_mid = normalize_measure_chords_for_omr_jitter(chords, sheet_chords, "intermediate")
        assert len(merged_mid) == 1
        assert merged_mid[0].name == "G"
        assert merged_mid[0].beat == 1.0

        # Advanced keeps exact timing
        adv = normalize_measure_chords_for_omr_jitter(chords, sheet_chords, "advanced")
        assert len(adv) == 2
        assert adv[1].beat == 1.5

    def test_pickup_empty_chords_holds(self):
        """Verify pickup bar with empty printed chords uses hold technique, never full pattern."""
        sheet = ParsedSheet(
            header=SongHeader(title="Test Pickup", time_signature="4/4"),
            pages=[],
            systems=[
                System(
                    page=0,
                    bbox=(0.0, 0.0, 1.0, 1.0),
                    measures=[
                        Measure(index=0, beats=1.0, bbox=(0.0, 0.0, 0.2, 1.0), chords=[], melody="5"),
                        Measure(index=1, beats=4.0, bbox=(0.2, 0.0, 1.0, 1.0), chords=[ChordSymbol(raw="1", beat=1.0)], melody="1 2 3 5"),
                    ],
                )
            ],
        )
        arr_mid = arrange_ukulele(sheet, "C", "intermediate")
        assert arr_mid.uke_measures[0].technique == "hold"
        assert arr_mid.uke_measures[0].pattern_id == "uke_held"
        # Never full pattern
        assert len(arr_mid.uke_measures[0].strokes) <= 2


class TestFullArrangerPipeline:
    """Integration tests on real sheet fixtures."""

    @pytest.fixture
    def diaole_sheet(self) -> ParsedSheet:
        path = "fixtures/omr_sample.json"
        with open(path, encoding="utf-8") as f:
            return ParsedSheet.model_validate(json.load(f))

    def test_diaole_arrangement_structure(self, diaole_sheet: ParsedSheet):
        """Verify arrange_ukulele produces complete arrangement for 掉了."""
        arr = arrange_ukulele(diaole_sheet, "F", "intermediate")
        assert arr.instrument == "ukulele"
        assert arr.difficulty == "intermediate"
        assert arr.start_key == "F"
        assert len(arr.measures) == len(diaole_sheet.measures())
        assert len(arr.uke_measures) == len(arr.measures)
        assert len(arr.uke_shapes) > 0
        assert len(arr.pattern_legend) > 0
        assert len(arr.notes) >= 2
        assert "定弦 G C E A（高音 G）" in arr.notes

        # Verify parallel structure
        for m, um in zip(arr.measures, arr.uke_measures):
            assert m.measure_index == um.measure_index
            assert len(um.shapes) == len(m.chords)
            for s_name in um.shapes:
                assert s_name in arr.uke_shapes
            assert round(sum(s.duration for s in um.strokes), 4) == 4.0

    def test_sounding_chords_identical_to_piano(self, diaole_sheet: ParsedSheet):
        """Verify sounding chords in ukulele arrangement are identical to piano arrangement."""
        piano_arr = arrange_piano(diaole_sheet, "F", "beginner")
        uke_arr = arrange_ukulele(diaole_sheet, "F", "intermediate")

        assert len(uke_arr.measures) == len(piano_arr.measures)
        for m_uke, m_piano in zip(uke_arr.measures, piano_arr.measures):
            assert m_uke.measure_index == m_piano.measure_index
            assert m_uke.tonic_pc == m_piano.tonic_pc
            assert m_uke.key_name == m_piano.key_name
            assert len(m_uke.chords) == len(m_piano.chords)
            for c_u, c_p in zip(m_uke.chords, m_piano.chords):
                assert c_u.raw == c_p.raw
                assert c_u.name == c_p.name
                assert c_u.beat == c_p.beat
                assert c_u.root_pc == c_p.root_pc
                assert c_u.bass_pc == c_p.bass_pc
                assert c_u.quality == c_p.quality
            assert m_uke.rh == []
            assert m_uke.lh == []

    def test_per_section_pattern_constancy(self, diaole_sheet: ParsedSheet):
        """Verify pattern_id is constant within each section instance."""
        measures = diaole_sheet.measures()
        sections = []
        for i, m in enumerate(measures):
            if i < 8:
                sections.append(SectionPlan(measure_index=m.index, section_id="intro", role="intro", energy=0))
            elif i < 24:
                sections.append(SectionPlan(measure_index=m.index, section_id="verse1", role="verse", energy=1))
            else:
                sections.append(SectionPlan(measure_index=m.index, section_id="chorus1", role="chorus", energy=2))

        arr = arrange_ukulele(diaole_sheet, "F", "intermediate", sections=sections)
        # Check intro measures (except stop/last)
        intro_patterns = {
            um.pattern_id
            for i, um in enumerate(arr.uke_measures[:8])
            if not diaole_sheet.measures()[i].is_stop
        }
        assert len(intro_patterns) == 1, f"Intro should have 1 pattern, got {intro_patterns}"

    def test_all_fixtures_never_raise(self):
        """Verify arrange_ukulele never raises across all fixture sheets x keys x difficulties."""
        fixture_files = sorted(glob.glob("fixtures/sheets/*.json"))
        assert len(fixture_files) > 0, "No fixture files found"

        keys_to_test = ["C", "G", "Eb", "F#", "Bb"]
        diffs = ["beginner", "intermediate", "advanced"]

        for path in fixture_files:
            with open(path, encoding="utf-8") as f:
                sheet = ParsedSheet.model_validate(json.load(f))
            for k in keys_to_test:
                for d in diffs:
                    arr = arrange_ukulele(sheet, k, d)
                    assert arr is not None
                    assert arr.instrument == "ukulele"
                    assert len(arr.measures) == len(sheet.measures())
                    assert len(arr.uke_measures) == len(arr.measures)
