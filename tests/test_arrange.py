"""Unit tests for piano arrangement, voice leading, fingering, styling, and invariants."""
import glob
import json
import os
import pytest

from app.models import (
    ChordSymbol,
    KeyChange,
    Measure,
    PageInfo,
    ParsedSheet,
    SongHeader,
    System,
)
from app.arrange.piano import arrange, validate_arrangement, validate_sequential_fingering


def make_sample_sheet() -> ParsedSheet:
    """Build a test fixture based on the real song '掉了' (pages 1 and 2)."""
    header = SongHeader(
        title="掉了",
        style="Slow Soul",
        time_signature="4/4",
        tempo_bpm=81.0,
        original_key="F#",
        male_key="Bb",
        female_key="F",
        raw="Slow Soul 4/4 (F# - Ab) 男調(Bb-C) 女調(F-G) 掉了",
    )

    dummy_bbox = (0.05, 0.1, 0.95, 0.2)
    m_bbox = (0.1, 0.1, 0.3, 0.2)

    systems = [
        # System 1: Intro (measures 0-3) - Only PN
        System(
            page=0,
            bbox=dummy_bbox,
            section_label="Only PN(RHY) / / / ~",
            measures=[
                Measure(index=0, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1(2)", beat=1.0)]),
                Measure(index=1, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="5/7", beat=1.0)]),
                Measure(index=2, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="2m7", beat=1.0)]),
                Measure(index=3, bbox=m_bbox, beats=4.0, chords=[]),  # Carried chord
            ],
        ),
        # System 2: Intro (measures 4-7)
        System(
            page=0,
            bbox=dummy_bbox,
            section_label="Only PN+AG(RHY)",
            measures=[
                Measure(index=4, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1(2)", beat=1.0)]),
                Measure(index=5, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="5/7", beat=1.0)]),
                Measure(index=6, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="2m7/6", beat=1.0)]),
                Measure(index=7, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="5sus", beat=1.0)]),
            ],
        ),
        # System 3: Verse (measures 8-11) - chromatic & slash chords
        System(
            page=0,
            bbox=dummy_bbox,
            section_label="Only PN(RHY) / / / ~",
            measures=[
                Measure(index=8, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1(2)", beat=1.0)]),
                Measure(index=9, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="5/7", beat=1.0)]),
                Measure(index=10, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="5m/7b", beat=1.0)]),
                Measure(
                    index=11,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="5m6/2", beat=1.0), ChordSymbol(raw="67/1#", beat=3.0)],
                ),
            ],
        ),
        # System 4: Verse continued (measures 12-15)
        System(
            page=0,
            bbox=dummy_bbox,
            section_label=None,
            measures=[
                Measure(index=12, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="2m7", beat=1.0)]),
                Measure(index=13, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1(2)", beat=1.0)]),
                Measure(
                    index=14,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="6b", beat=1.0), ChordSymbol(raw="6m7-5", beat=3.0)],
                ),
                Measure(
                    index=15,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="5sus", beat=1.0), ChordSymbol(raw="5", beat=3.0)],
                ),
            ],
        ),
        # System 5: Chorus with Bs in & bass hint (measures 16-19)
        System(
            page=0,
            bbox=dummy_bbox,
            section_label="Only PN+AG(RHY)+Bs in",
            measures=[
                Measure(
                    index=16,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="1", beat=1.0)],
                    bass_hint="11 11 11 112",
                ),
                Measure(index=17, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1/3", beat=1.0)]),
                Measure(index=18, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="6m7", beat=1.0)]),
                Measure(index=19, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="6m7/5", beat=1.0)]),
            ],
        ),
        # System 6: Chorus with fill and stop (measures 20-23)
        System(
            page=0,
            bbox=dummy_bbox,
            section_label=None,
            measures=[
                Measure(index=20, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="4", beat=1.0)]),
                Measure(index=21, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1/3", beat=1.0)]),
                Measure(
                    index=22,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="2m7", beat=1.0)],
                    is_stop=True,
                ),
                Measure(
                    index=23,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="2m7/5", beat=1.0), ChordSymbol(raw="5", beat=3.0)],
                    fill=True,
                ),
            ],
        ),
    ]

    key_changes = [
        # Modulation +2 semitones at measure 20 (F# -> Ab, or F -> G)
        KeyChange(at_measure=20, raw="轉成2調(Ab)", semitones=2),
    ]

    return ParsedSheet(
        header=header,
        pages=[PageInfo(width=1000, height=1400), PageInfo(width=1000, height=1400)],
        systems=systems,
        key_changes=key_changes,
    )


class TestArrangementInvariants:
    """Rigorous tests for piano arrangement correctness and musical invariants."""

    @pytest.mark.parametrize("diff", ["beginner", "intermediate", "advanced"])
    def test_tiling_and_continuity(self, diff):
        sheet = make_sample_sheet()
        arr = arrange(sheet, start_key="F", difficulty=diff)

        for m_arr in arr.measures:
            # Measure beats
            beats = 4.0
            # RH tiling
            rh_sum = round(sum(e.duration for e in m_arr.rh), 4)
            assert rh_sum == beats, f"Measure {m_arr.measure_index} RH duration {rh_sum} != {beats}"

            # LH tiling
            lh_sum = round(sum(e.duration for e in m_arr.lh), 4)
            assert lh_sum == beats, f"Measure {m_arr.measure_index} LH duration {lh_sum} != {beats}"

            # Onsets continuous on 16th grid
            curr = 0.0
            for e in m_arr.rh:
                assert abs(e.onset - curr) < 1e-4, f"RH onset gap in m{m_arr.measure_index}: {e.onset} != {curr}"
                # 16th grid check: multiple of 0.25
                assert round(e.duration % 0.25, 4) in (0.0, 0.25), f"Duration {e.duration} not on 16th grid"
                assert round(e.onset % 0.25, 4) in (0.0, 0.25), f"Onset {e.onset} not on 16th grid"
                curr = round(curr + e.duration, 4)

            curr = 0.0
            for e in m_arr.lh:
                assert abs(e.onset - curr) < 1e-4, f"LH onset gap in m{m_arr.measure_index}: {e.onset} != {curr}"
                assert round(e.duration % 0.25, 4) in (0.0, 0.25), f"Duration {e.duration} not on 16th grid"
                assert round(e.onset % 0.25, 4) in (0.0, 0.25), f"Onset {e.onset} not on 16th grid"
                curr = round(curr + e.duration, 4)

    @pytest.mark.parametrize("diff", ["beginner", "intermediate", "advanced"])
    def test_fingering_validity(self, diff):
        sheet = make_sample_sheet()
        arr = arrange(sheet, start_key="F", difficulty=diff)

        for m_arr in arr.measures:
            # RH fingering: every note has 1 <= finger <= 5
            for ev in m_arr.rh:
                if not ev.notes:
                    continue
                for n in ev.notes:
                    assert n.finger is not None, f"None finger in m{m_arr.measure_index}"
                    assert 1 <= n.finger <= 5, f"Finger {n.finger} out of bounds in m{m_arr.measure_index}"

                # No finger reused for simultaneous notes
                fingers = [n.finger for n in ev.notes]
                assert len(fingers) == len(set(fingers)), f"Duplicate RH fingers {fingers} at onset {ev.onset}"

                # RH finger order monotonic with pitch: p1 < p2 => f1 < f2
                sorted_notes = sorted(ev.notes, key=lambda n: n.midi)
                for i in range(len(sorted_notes) - 1):
                    assert sorted_notes[i].finger < sorted_notes[i + 1].finger, (
                        f"RH fingers not strictly increasing with pitch: "
                        f"{[n.midi for n in sorted_notes]} with fingers {[n.finger for n in sorted_notes]}"
                    )

            # LH fingering: every note has 1 <= finger <= 5
            for ev in m_arr.lh:
                if not ev.notes:
                    continue
                for n in ev.notes:
                    assert n.finger is not None, f"None finger in m{m_arr.measure_index}"
                    assert 1 <= n.finger <= 5, f"Finger {n.finger} out of bounds in m{m_arr.measure_index}"

                fingers = [n.finger for n in ev.notes]
                assert len(fingers) == len(set(fingers)), f"Duplicate LH fingers {fingers} at onset {ev.onset}"

                # LH finger order monotonic with pitch: p1 < p2 => f1 > f2
                sorted_notes = sorted(ev.notes, key=lambda n: n.midi)
                for i in range(len(sorted_notes) - 1):
                    assert sorted_notes[i].finger > sorted_notes[i + 1].finger, (
                        f"LH fingers not strictly decreasing with pitch: "
                        f"{[n.midi for n in sorted_notes]} with fingers {[n.finger for n in sorted_notes]}"
                    )

    @pytest.mark.parametrize("diff", ["beginner", "intermediate", "advanced"])
    def test_pitch_ranges(self, diff):
        sheet = make_sample_sheet()
        arr = arrange(sheet, start_key="F", difficulty=diff)

        for m_arr in arr.measures:
            for ev in m_arr.rh:
                for n in ev.notes:
                    assert 64 <= n.midi <= 81, f"RH note {n.midi} outside E4(64)..A5(81) in m{m_arr.measure_index}"

            for ev in m_arr.lh:
                for n in ev.notes:
                    assert n.midi <= 64, f"LH note {n.midi} too high (> E4) in m{m_arr.measure_index}"

    def test_beginner_attack_limit_and_span(self):
        sheet = make_sample_sheet()
        arr = arrange(sheet, start_key="F", difficulty="beginner")

        for m_arr in arr.measures:
            rh_attacks = sum(1 for e in m_arr.rh if len(e.notes) > 0)
            assert rh_attacks <= 2, f"Beginner RH attacks {rh_attacks} > 2 in m{m_arr.measure_index}"

            for ev in m_arr.rh:
                if len(ev.notes) > 1:
                    span = max(n.midi for n in ev.notes) - min(n.midi for n in ev.notes)
                    assert span <= 12, f"Beginner RH span {span} > 12 in m{m_arr.measure_index}"

    def test_carried_chords(self):
        sheet = make_sample_sheet()
        arr = arrange(sheet, start_key="F", difficulty="intermediate")

        # Measure 2 has chord 2m7 (Gm7 in key F), Measure 3 has no chord box -> should carry Gm7
        m2_arr = arr.measures[2]
        m3_arr = arr.measures[3]
        assert m2_arr.chords[0].name == "Gm7"
        assert m3_arr.chords[0].name == "Gm7"
        assert m3_arr.chords[0].raw == "2m7"

    def test_key_change_applied_at_right_measure(self):
        sheet = make_sample_sheet()
        # Start in key F# (tonic_pc = 6), key change +2 at measure 20 -> Ab (tonic_pc = 8)
        arr = arrange(sheet, start_key="F#", difficulty="intermediate")

        # Before measure 20: tonic_pc == 6 (F#)
        for m_arr in arr.measures[:20]:
            assert m_arr.tonic_pc == 6
            assert m_arr.key_name == "F#"

        # From measure 20 onwards: tonic_pc == 8 (Ab)
        for m_arr in arr.measures[20:]:
            assert m_arr.tonic_pc == 8
            assert m_arr.key_name == "Ab"

        # In measure 20, chord raw is "4". In key Ab, 4 is Db!
        m20_arr = arr.measures[20]
        assert m20_arr.chords[0].name == "Db"

    def test_average_rh_voice_leading_movement(self):
        sheet = make_sample_sheet()
        arr = arrange(sheet, start_key="F", difficulty="intermediate")

        total_movement = 0.0
        transition_count = 0

        # Collect distinct RH voicings across chord changes
        voicings = []
        for m_arr in arr.measures:
            for ev in m_arr.rh:
                if ev.notes and len(ev.notes) > 1:
                    v = tuple(sorted(n.midi for n in ev.notes))
                    if not voicings or voicings[-1] != v:
                        voicings.append(v)

        for i in range(len(voicings) - 1):
            v1, v2 = voicings[i], voicings[i + 1]
            if len(v1) == len(v2):
                dist = sum(abs(b - a) for a, b in zip(v1, v2))
            else:
                dist = sum(min(abs(b - a) for a in v1) for b in v2)
            total_movement += dist
            transition_count += 1

        assert transition_count > 0
        avg_movement = total_movement / transition_count
        # Assert average RH movement per chord change <= 4 semitones
        assert avg_movement <= 4.0, f"Average RH movement {avg_movement:.2f} > 4.0 semitones"

    def test_texture_rules_stop_fill_and_bass_hint(self):
        sheet = make_sample_sheet()
        arr = arrange(sheet, start_key="F", difficulty="advanced")

        # Measure 22 is is_stop=True: hit and hold for the chord duration
        m22 = arr.measures[22]
        assert len(m22.rh) == 1
        assert m22.rh[0].onset == 0.0 and m22.rh[0].duration == 4.0 and len(m22.rh[0].notes) > 0
        assert len(m22.lh) == 1
        assert m22.lh[0].onset == 0.0 and m22.lh[0].duration == 4.0

        # Multi-chord is_stop measure: hit and hold on every chord beat without dropping chords
        stop_sheet = make_sample_sheet()
        stop_sheet.systems[5].measures[2].chords = [
            ChordSymbol(raw="2m7", beat=1.0),
            ChordSymbol(raw="5", beat=3.0),
        ]
        stop_arr = arrange(stop_sheet, start_key="F", difficulty="intermediate")
        m22_stop = stop_arr.measures[22]
        assert len(m22_stop.rh) == 2
        assert m22_stop.rh[0].onset == 0.0 and m22_stop.rh[0].duration == 2.0 and len(m22_stop.rh[0].notes) > 0
        assert m22_stop.rh[1].onset == 2.0 and m22_stop.rh[1].duration == 2.0 and len(m22_stop.rh[1].notes) > 0
        assert len(m22_stop.lh) == 2
        assert m22_stop.lh[0].onset == 0.0 and m22_stop.lh[0].duration == 2.0
        assert m22_stop.lh[1].onset == 2.0 and m22_stop.lh[1].duration == 2.0

        # Measure 23 has fill=True: fill figure on beat 4 (16th notes)
        m23 = arr.measures[23]
        rh_durations = [e.duration for e in m23.rh]
        assert 0.25 in rh_durations, "Expected sixteenth notes for fill in advanced"

        # Measure 16 has bass_hint="11 11 11 112"
        m16 = arr.measures[16]
        lh_onsets = [e.onset for e in m16.lh]
        assert 3.75 in lh_onsets, "Expected 16th pickup at 3.75 from '112' bass hint"

    def test_chord_defining_tones_per_difficulty_level(self):
        """Assert that chord-defining colour tones (9th for add9, 7th for 7ths, 4th for sus) are present."""
        sheet = make_sample_sheet()

        # 1. Intermediate and Advanced: add9 MUST contain the 9th
        for diff in ["intermediate", "advanced"]:
            arr = arrange(sheet, start_key="F", difficulty=diff)
            # Measure 0 is 1(2) -> Fadd9 (root F=5, 9th G=7)
            m0_rh_notes = [n.midi % 12 for ev in arr.measures[0].rh for n in ev.notes]
            assert 7 in m0_rh_notes, f"9th (G/pc 7) missing from {diff} voicing of Fadd9: {m0_rh_notes}"

            # Measure 2 is 2m7 -> Gm7 (root G=7, 7th F=5)
            m2_rh_notes = [n.midi % 12 for ev in arr.measures[2].rh for n in ev.notes]
            assert 5 in m2_rh_notes, f"7th (F/pc 5) missing from {diff} voicing of Gm7: {m2_rh_notes}"

            # Measure 6 is 2m7/6 -> Gm7/D (root G=7, 3rd Bb=10, 5th D=2, 7th F=5)
            # Intermediate and advanced must strictly retain 3rd (Bb) and 7th (F)
            m6_rh_notes = [n.midi % 12 for ev in arr.measures[6].rh for n in ev.notes]
            assert 10 in m6_rh_notes, f"3rd (Bb/pc 10) missing from {diff} voicing of Gm7/D: {m6_rh_notes}"
            assert 5 in m6_rh_notes, f"7th (F/pc 5) missing from {diff} voicing of Gm7/D: {m6_rh_notes}"

            # Measure 7 is 5sus -> Csus4 (root C=0, 4th F=5)
            m7_rh_notes = [n.midi % 12 for ev in arr.measures[7].rh for n in ev.notes]
            assert 5 in m7_rh_notes, f"4th (F/pc 5) missing from {diff} voicing of Csus4: {m7_rh_notes}"

            # Measure 14 has 6m7-5 -> Dm7b5 (root D=2, b5 Ab=8, 7th C=0)
            m14_rh_notes = [n.midi % 12 for ev in arr.measures[14].rh for n in ev.notes]
            assert 8 in m14_rh_notes, f"b5 (Ab/pc 8) missing from {diff} voicing of Dm7b5: {m14_rh_notes}"
            assert 0 in m14_rh_notes, f"7th (C/pc 0) missing from {diff} voicing of Dm7b5: {m14_rh_notes}"

        # 2. Beginner: sus4 MUST keep the 4th, m7b5 MUST keep b5
        arr_b = arrange(sheet, start_key="F", difficulty="beginner")
        # Measure 7 is 5sus -> Csus4 (4th F=5 must be present!)
        m7_b_notes = [n.midi % 12 for ev in arr_b.measures[7].rh for n in ev.notes]
        assert 5 in m7_b_notes, f"4th (F/pc 5) missing from beginner Csus4: {m7_b_notes}"

        # Measure 14 has 6m7-5 -> Dm7b5 (b5 Ab=8 must be present!)
        m14_b_notes = [n.midi % 12 for ev in arr_b.measures[14].rh for n in ev.notes]
        assert 8 in m14_b_notes, f"b5 (Ab/pc 8) missing from beginner Dm7b5: {m14_b_notes}"

    def test_run_on_omr_sample_if_present(self):
        for fixture_path in ["fixtures/omr_sample.json", "fixtures/synthetic_sheet.json"]:
            if os.path.exists(fixture_path):
                with open(fixture_path, "r", encoding="utf-8") as f:
                    sheet = ParsedSheet.model_validate_json(f.read())
                arr = arrange(sheet, start_key="F#", difficulty="intermediate")
                validate_arrangement(arr, sheet)
                assert len(arr.measures) > 0

    def test_fill_notes_diatonic_to_new_key(self):
        sheet = make_sample_sheet()
        # Modulates to G at m20; m23 has fill=True in advanced
        arr = arrange(sheet, start_key="F", difficulty="advanced")
        m23 = arr.measures[23]
        # In m23, fill is on beat 4 (onsets 3.0, 3.25, 3.5, 3.75)
        fill_events = [e for e in m23.rh if e.onset >= 3.0 and e.notes]
        assert len(fill_events) == 4, f"Expected 4 fill note events, got {len(fill_events)}"
        fill_pitches = [e.notes[0].midi for e in fill_events]
        fill_fingers = [e.notes[0].finger for e in fill_events]

        # Diatonic scale in key G (pc 7): {G, A, B, C, D, E, F#}
        # Fill descends from top chord note D5 (74): [74, 72, 71, 69] -> D5, C5, B4, A4
        assert fill_pitches == [74, 72, 71, 69], f"Fill pitches {fill_pitches} != [74, 72, 71, 69]"
        # Bb4 (pitch 70) must NOT be present
        assert 70 not in fill_pitches, "Bb4 (non-diatonic to G major) found in fill"
        # Fingering must be sequential descending [4, 3, 2, 1]
        assert fill_fingers == [4, 3, 2, 1], f"Fill fingers {fill_fingers} != [4, 3, 2, 1]"

    @pytest.mark.parametrize("diff", ["beginner", "intermediate", "advanced"])
    def test_consecutive_fingering_playability(self, diff):
        sheet = make_sample_sheet()
        arr = arrange(sheet, start_key="F", difficulty=diff)
        for m_arr in arr.measures:
            validate_sequential_fingering(m_arr.rh, hand="RH")
            validate_sequential_fingering(m_arr.lh, hand="LH")

    def test_bass_hint_passing_tone_fingering(self):
        sheet = make_sample_sheet()
        # Measure 16 has bass_hint="11 11 11 112"
        arr = arrange(sheet, start_key="F", difficulty="intermediate")
        m16 = arr.measures[16]
        # At onset 3.75, digit 2 should produce a passing tone (G2, pitch 43) with finger 4
        passing_ev = [e for e in m16.lh if abs(e.onset - 3.75) < 1e-4]
        assert len(passing_ev) == 1, "Missing onset 3.75 for passing tone in m16"
        note = passing_ev[0].notes[0]
        assert note.midi == 43, f"Expected passing tone G2 (pitch 43), got {note.midi}"
        assert note.finger == 4, f"Expected finger 4 for passing tone degree 2, got {note.finger}"

    @pytest.mark.parametrize("diff", ["beginner", "intermediate", "advanced"])
    def test_lh_notes_are_chord_tones(self, diff):
        """Assert that all LH accompaniment notes are strictly chord tones (or bass pc),

        except for passing tones in bass_hint measures.
        """
        sheet = make_sample_sheet()
        arr = arrange(sheet, start_key="F", difficulty=diff)
        for m_arr in arr.measures:
            orig_m = next((m for m in sheet.measures() if m.index == m_arr.measure_index), None)
            has_bass_hint = bool(orig_m and orig_m.bass_hint)
            for ev in m_arr.lh:
                if not ev.notes:
                    continue
                active_chord = m_arr.chords[0]
                for c in m_arr.chords:
                    if (c.beat - 1.0) <= ev.onset + 1e-4:
                        active_chord = c
                chord_pcs = set(active_chord.pcs) | {active_chord.bass_pc % 12}
                for note in ev.notes:
                    pitch_pc = note.midi % 12
                    if pitch_pc not in chord_pcs:
                        assert has_bass_hint, (
                            f"Measure {m_arr.measure_index} ({active_chord.name}) LH note {note.midi} "
                            f"(pc {pitch_pc}) not in chord pcs {chord_pcs}"
                        )


class TestCrashFixAndBatchSweep:
    """Tests for crash resilience, beginner density enforcement, and 14-song batch sweep (Objective 3)."""

    def test_xindong_measure_29_beginner_density(self):
        """Heartbeat test: xindong measure 29 beginner RH attacks must be <= 2 without exceptions."""
        path = "out/batch/xindong/verified.json"
        assert os.path.exists(path), f"Missing {path}"
        with open(path, encoding="utf-8") as f:
            sheet = ParsedSheet.model_validate(json.load(f))

        arr = arrange(sheet, start_key="C", difficulty="beginner")
        m29 = next(m for m in arr.measures if m.measure_index == 29)
        attacks = sum(1 for e in m29.rh if len(e.notes) > 0)
        assert attacks <= 2, f"Measure 29 beginner RH has {attacks} attacks (> 2)"

    def test_all_14_batch_songs_sweep(self):
        """Run arrange on all 14 out/batch/*/verified.json x keys {C, G, Eb} x 3 levels and assert 0 exceptions."""
        paths = sorted(glob.glob("out/batch/*/verified.json"))
        assert len(paths) >= 14, f"Found {len(paths)} batch songs, expected at least 14"

        keys = ["C", "G", "Eb"]
        levels = ["beginner", "intermediate", "advanced"]
        total_runs = 0

        for p in paths:
            with open(p, encoding="utf-8") as f:
                sheet = ParsedSheet.model_validate(json.load(f))

            for k in keys:
                for lvl in levels:
                    arr = arrange(sheet, start_key=k, difficulty=lvl)
                    total_runs += 1
                    assert arr is not None
                    if lvl == "beginner":
                        for m in arr.measures:
                            attacks = sum(1 for e in m.rh if len(e.notes) > 0)
                            assert attacks <= 2, f"{p} key={k} m{m.measure_index} attacks={attacks} > 2"

        assert total_runs == len(paths) * 3 * 3


class TestTimeSignatures:
    """Tests for 3/4 and 12/8 / 6/8 pattern selection and generation (Objective 4)."""

    def test_xiaobaichuan_3_4_waltz(self):
        """Verify 3/4 time signature triggers waltz pattern with exact 3.0 duration tiling."""
        path = "out/batch/xiaobaichuan/verified.json"
        with open(path, encoding="utf-8") as f:
            sheet = ParsedSheet.model_validate(json.load(f))

        for lvl in ["beginner", "intermediate", "advanced"]:
            arr = arrange(sheet, start_key="Eb", difficulty=lvl)
            assert arr.style == "waltz"
            for m in arr.measures:
                rh_dur = round(sum(e.duration for e in m.rh), 4)
                lh_dur = round(sum(e.duration for e in m.lh), 4)
                assert rh_dur == 3.0, f"m{m.measure_index} RH tiling mismatch: {rh_dur} != 3.0"
                assert lh_dur == 3.0, f"m{m.measure_index} LH tiling mismatch: {lh_dur} != 3.0"

    def test_huochuai_12_8_slow_rock(self):
        """Verify 12/8 and Slow Rock selection and accompaniment generation."""
        path = "out/batch/huochuai/verified.json"
        with open(path, encoding="utf-8") as f:
            sheet = ParsedSheet.model_validate(json.load(f))

        # Test natural Slow Soul
        arr_natural = arrange(sheet, start_key="F#", difficulty="intermediate")
        assert arr_natural.style == "slow soul"

        # Test override to 12/8 and Slow Rock
        sheet.header.time_signature = "12/8"
        sheet.header.style = "Slow Rock"
        arr_sr = arrange(sheet, start_key="G", difficulty="intermediate")
        assert arr_sr.style == "slow rock"
        for m in arr_sr.measures:
            rh_dur = round(sum(e.duration for e in m.rh), 4)
            lh_dur = round(sum(e.duration for e in m.lh), 4)
            assert rh_dur == 4.0, f"m{m.measure_index} RH tiling mismatch: {rh_dur} != 4.0"
            assert lh_dur == 4.0, f"m{m.measure_index} LH tiling mismatch: {lh_dur} != 4.0"


class TestLetterChordArrangement:
    """Tests for arrangement with letter-notation chord charts (Objective 1)."""

    def test_letter_chart_arrangement_and_transposition(self):
        header = SongHeader(
            title="Letter Test",
            style="Ballad",
            original_key="C",
            chord_notation="letter",
        )
        measures = [
            Measure(
                index=0,
                bbox=(0.1, 0.1, 0.9, 0.9),
                beats=4.0,
                chords=[ChordSymbol(raw="C", beat=1.0), ChordSymbol(raw="G/B", beat=3.0)],
            ),
            Measure(
                index=1,
                bbox=(0.1, 0.1, 0.9, 0.9),
                beats=4.0,
                chords=[ChordSymbol(raw="Am7", beat=1.0), ChordSymbol(raw="F", beat=3.0)],
            ),
        ]
        sheet = ParsedSheet(
            header=header,
            pages=[],
            systems=[System(page=0, bbox=(0, 0, 1, 1), measures=measures)],
        )

        # Transpose from printed key C to target key G (+7 semitones)
        arr_g = arrange(sheet, start_key="G", difficulty="intermediate")
        assert arr_g.measures[0].chords[0].name == "G"
        assert arr_g.measures[0].chords[1].name == "D/F#"
        assert arr_g.measures[1].chords[0].name == "Em7"
        assert arr_g.measures[1].chords[1].name == "C"

    def test_letter_chart_missing_original_key_remark(self):
        header = SongHeader(
            title="Letter Test No Key",
            style="Ballad",
            original_key=None,
            chord_notation="letter",
        )
        measures = [
            Measure(
                index=0,
                bbox=(0.1, 0.1, 0.9, 0.9),
                beats=4.0,
                chords=[ChordSymbol(raw="C", beat=1.0)],
            ),
        ]
        sheet = ParsedSheet(
            header=header,
            pages=[],
            systems=[System(page=0, bbox=(0, 0, 1, 1), measures=measures)],
        )

        arr = arrange(sheet, start_key="C", difficulty="beginner")
        assert any("No original key specified" in n for n in arr.notes)

    def test_arrange_stacked_chords_tinghai_and_xiaobaichuan(self):
        """Test arrange() applies top_is_bass orientation and records remark for stacked charts."""
        for slug, key in [("tinghai", "Bb"), ("xiaobaichuan", "Eb")]:
            with open(f"fixtures/groundtruth/{slug}.json", encoding="utf-8") as f:
                d = json.load(f)
            header = SongHeader.model_validate(d.get("header", {}))
            systems = []
            idx = 0
            for r in d.get("rows", []):
                measures = []
                for m in r.get("measures", []):
                    chords = []
                    for c in m.get("chords", []):
                        raw = c["raw"]
                        is_stacked = ("/" in raw and not any(raw.endswith(e) for e in ("9", "11", "13", "11-9", "9-5")))
                        chords.append(ChordSymbol(raw=raw, beat=c.get("beat", 1.0), stacked=is_stacked))
                    beats = 3.0 if "3/4" in header.time_signature else 4.0
                    measures.append(Measure(index=idx, bbox=(0, 0, 1, 1), beats=beats, chords=chords))
                    idx += 1
                systems.append(System(page=r.get("page", 0), bbox=(0, 0, 1, 1), measures=measures))
            sheet = ParsedSheet(header=header, pages=[], systems=systems)

            arr = arrange(sheet, start_key=key, difficulty="intermediate")
            assert any("叠写和弦按上方为低音解读" in n for n in arr.notes)
            validate_arrangement(arr, sheet)

            if slug == "xiaobaichuan":
                # M1 is 7/5 -> under top_is_bass in Eb, chord 5 over bass 7 -> Bb/D
                assert arr.measures[1].chords[0].name == "Bb/D"
                # M3 is 5/1 -> under top_is_bass in Eb, chord 1 over bass 5 -> Eb/Bb
                assert arr.measures[3].chords[0].name == "Eb/Bb"
            elif slug == "tinghai":
                # M0 has 1add9 (Bbadd9) and 1/2m7-5 (Cm7b5/Bb)
                assert arr.measures[0].chords[0].name == "Bbadd9"
                assert arr.measures[0].chords[1].name == "Cm7b5/Bb"

    def test_arrange_unstacked_preserves_inline_slashes(self):
        """Unstacked inline slashes stay chord/bass unchanged and add no stacked note."""
        sheet = make_sample_sheet()
        arr = arrange(sheet, start_key="F#", difficulty="intermediate")
        assert not any("叠写和弦按" in n for n in arr.notes)
        # M1 has 5/7 -> chord 5 over bass 7 (C#/F in F#)
        assert arr.measures[1].chords[0].name in ("C#/E#", "C#/F")
        # M6 has 2m7/6 -> chord 2m7 over bass 6 (G#m7/D# in F#)
        assert arr.measures[6].chords[0].name == "G#m7/D#"


