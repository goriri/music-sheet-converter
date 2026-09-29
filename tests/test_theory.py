"""Unit tests for music theory, chord parsing, key resolution, and enharmonic spelling."""
import pytest

from app.theory.chords import parse_chord, parse_letter_chord, resolve_chord
from app.theory.keys import key_name_to_pc, spell


class TestKeys:
    """Tests for key name parsing and pitch class resolution."""

    def test_standard_keys(self):
        assert key_name_to_pc("C") == 0
        assert key_name_to_pc("C#") == 1
        assert key_name_to_pc("Db") == 1
        assert key_name_to_pc("D") == 2
        assert key_name_to_pc("Eb") == 3
        assert key_name_to_pc("E") == 4
        assert key_name_to_pc("F") == 5
        assert key_name_to_pc("F#") == 6
        assert key_name_to_pc("Gb") == 6
        assert key_name_to_pc("G") == 7
        assert key_name_to_pc("Ab") == 8
        assert key_name_to_pc("A") == 9
        assert key_name_to_pc("Bb") == 10
        assert key_name_to_pc("B") == 11

    def test_chinese_key_names(self):
        assert key_name_to_pc("降B") == 10
        assert key_name_to_pc("升F") == 6
        assert key_name_to_pc("降E") == 3
        assert key_name_to_pc("降A") == 8
        assert key_name_to_pc("升C") == 1
        assert key_name_to_pc("降D") == 1

    def test_key_name_suffixes_and_variations(self):
        assert key_name_to_pc("C調") == 0
        assert key_name_to_pc("F#大調") == 6
        assert key_name_to_pc("Bb major") == 10
        assert key_name_to_pc("  f#  ") == 6
        assert key_name_to_pc("bb") == 10

    def test_invalid_key_names(self):
        for bad in ["", "   ", "H", "XYZ", "123", None]:
            with pytest.raises(ValueError):
                key_name_to_pc(bad)  # type: ignore


class TestChordParsing:
    """Tests for parsing Taiwanese number notation."""

    def test_prompt_examples_in_c(self):
        examples = [
            ("5/7", "G/B"),
            ("5m/7b", "Gm/Bb"),
            ("67/1#", "A7/C#"),
            ("6m7-5", "Am7b5"),
            ("57sus", "G7sus4"),
            ("1(2)", "Cadd9"),
            ("4M7", "Fmaj7"),
            ("7b", "Bb"),
            ("6b", "Ab"),
        ]
        for raw, expected in examples:
            resolved = resolve_chord(raw, tonic_pc=0)
            assert resolved.name == expected, f"Failed for {raw}: got {resolved.name}, expected {expected}"

    def test_prompt_examples_in_other_keys(self):
        # Key F (tonic_pc=5)
        assert resolve_chord("5/7", tonic_pc=5).name == "C/E"
        assert resolve_chord("4", tonic_pc=5).name == "Bb"

        # Key F# (tonic_pc=6)
        assert resolve_chord("4", tonic_pc=6).name == "B"
        assert resolve_chord("7b", tonic_pc=6).name == "E"

        # Key Ab (tonic_pc=8)
        assert resolve_chord("5", tonic_pc=8).name == "Eb"

    def test_accidental_before_and_after(self):
        # Accidental before or after degree must produce identical specs
        spec1 = parse_chord("b7")
        spec2 = parse_chord("7b")
        assert spec1.degree == 7 and spec1.accidental == -1
        assert spec2.degree == 7 and spec2.accidental == -1

        spec3 = parse_chord("#1")
        spec4 = parse_chord("1#")
        assert spec3.degree == 1 and spec3.accidental == 1
        assert spec4.degree == 1 and spec4.accidental == 1

        # Slash bass accidental before or after
        s_bass1 = parse_chord("5m/7b")
        s_bass2 = parse_chord("5m/b7")
        assert s_bass1.bass_degree == 7 and s_bass1.bass_accidental == -1
        assert s_bass2.bass_degree == 7 and s_bass2.bass_accidental == -1

        s_bass3 = parse_chord("67/1#")
        s_bass4 = parse_chord("67/#1")
        assert s_bass3.bass_degree == 1 and s_bass3.bass_accidental == 1
        assert s_bass4.bass_degree == 1 and s_bass4.bass_accidental == 1

    def test_all_chord_qualities(self):
        qualities = [
            ("1", "maj", "C"),
            ("2m", "m", "Dm"),
            ("57", "7", "G7"),
            ("4M7", "maj7", "Fmaj7"),
            ("4maj7", "maj7", "Fmaj7"),
            ("2m7", "m7", "Dm7"),
            ("7m7b5", "m7b5", "Bm7b5"),
            ("7m7-5", "m7b5", "Bm7b5"),
            ("7dim", "dim", "Bdim7"),
            ("7o", "dim", "Bdim7"),
            ("7dim7", "dim", "Bdim7"),
            ("7o7", "dim", "Bdim7"),
            ("1aug", "aug", "Caug"),
            ("1+", "aug", "Caug"),
            ("5sus", "sus4", "Gsus4"),
            ("5sus4", "sus4", "Gsus4"),
            ("57sus", "7sus4", "G7sus4"),
            ("57sus4", "7sus4", "G7sus4"),
            ("16", "6", "C6"),
            ("2m6", "m6", "Dm6"),
            ("1(2)", "add9", "Cadd9"),
            ("1add2", "add9", "Cadd9"),
            ("1(9)", "add9", "Cadd9"),
            ("12", "add9", "Cadd9"),
            ("59", "9", "G9"),
            ("2m9", "m9", "Dm9"),
            ("4M9", "maj9", "Fmaj9"),
        ]
        for raw, expected_qual, expected_name in qualities:
            spec = parse_chord(raw)
            assert spec.quality == expected_qual, f"Quality mismatch for {raw}: {spec.quality} != {expected_qual}"
            res = resolve_chord(raw, tonic_pc=0)
            assert res.name == expected_name, f"Name mismatch for {raw}: {res.name} != {expected_name}"

    def test_noise_tolerance(self):
        noisy_cases = [
            (" １（２） ", "Cadd9"),
            ("５m／７♭", "Gm/Bb"),
            ("６７／１＃", "A7/C#"),
            ("6m7－5", "Am7b5"),
            ("6m7—5", "Am7b5"),
            ("5⁷sus⁴", "G7sus4"),
            ("\"5sus\"", "Gsus4"),
            ("'1(2)'", "Cadd9"),
            ("5m / 7b", "Gm/Bb"),
            ("67 / 1#", "A7/C#"),
            (" 7b ", "Bb"),
        ]
        for raw, expected in noisy_cases:
            res = resolve_chord(raw, tonic_pc=0)
            assert res.name == expected, f"Failed for noisy '{raw}': got {res.name}, expected {expected}"

    def test_resolved_chord_structure(self):
        res = resolve_chord("5m/7b", tonic_pc=0, beat=2.0)
        assert res.raw == "5m/7b"
        assert res.name == "Gm/Bb"
        assert res.beat == 2.0
        assert res.root_pc == 7  # G
        assert res.bass_pc == 10  # Bb
        assert res.pcs[0] == 7  # root first
        assert set(res.pcs) == {7, 10, 2}  # G, Bb, D
        assert res.quality == "m"

    def test_garbage_chords_raise_value_error(self):
        bad_inputs = ["", "   ", "xyz", "hello", "12345", "/5", "5/", "8m"]
        for bad in bad_inputs:
            with pytest.raises(ValueError):
                parse_chord(bad)
            with pytest.raises(ValueError):
                resolve_chord(bad, tonic_pc=0)


class TestEnharmonicSpelling:
    """Tests for enharmonic spelling helper function."""

    def test_spell_in_c(self):
        # In C: chromatic notes b3, b6, b7 flats, #1, #4, #5 sharps
        assert spell(0, tonic_pc=0) == "C"
        assert spell(1, tonic_pc=0) == "C#"
        assert spell(2, tonic_pc=0) == "D"
        assert spell(3, tonic_pc=0) == "Eb"
        assert spell(4, tonic_pc=0) == "E"
        assert spell(5, tonic_pc=0) == "F"
        assert spell(6, tonic_pc=0) == "F#"
        assert spell(7, tonic_pc=0) == "G"
        assert spell(8, tonic_pc=0) == "Ab"
        assert spell(9, tonic_pc=0) == "A"
        assert spell(10, tonic_pc=0) == "Bb"
        assert spell(11, tonic_pc=0) == "B"

    def test_spell_in_flat_key(self):
        # In F (1 flat: Bb)
        assert spell(10, tonic_pc=5) == "Bb"
        assert spell(3, tonic_pc=5) == "Eb"


class TestLetterChords:
    """Tests for letter-notation chord parsing and resolution (Objective 1)."""

    def test_letter_chord_catalog(self):
        cases = [
            ("C", 0, "C", [0, 4, 7]),
            ("C#m7", 1, "C#m7", [1, 4, 8, 11]),
            ("Bb/D", 10, "Bb/D", [10, 2, 5]),
            ("Am7b5", 9, "Am7b5", [9, 0, 3, 7]),
            ("F#dim7", 6, "F#dim7", [6, 9, 0, 3]),
            ("Gsus4", 7, "Gsus4", [7, 0, 2]),
            ("G7sus4", 7, "G7sus4", [7, 0, 2, 5]),
            ("Cadd9", 0, "Cadd9", [0, 4, 7, 2]),
            ("C(add9)", 0, "Cadd9", [0, 4, 7, 2]),
            ("Cmaj7", 0, "Cmaj7", [0, 4, 7, 11]),
            ("CM7", 0, "Cmaj7", [0, 4, 7, 11]),
            ("C6", 0, "C6", [0, 4, 7, 9]),
            ("Cm6", 0, "Cm6", [0, 3, 7, 9]),
            ("C9", 0, "C9", [0, 4, 7, 10, 2]),
            ("C7(b9)", 0, "C7b9", [0, 4, 7, 10, 1]),
            ("E7(#9)", 4, "E7#9", [4, 8, 11, 2, 7]),
            ("Dm7/G", 2, "Dm7/G", [2, 5, 9, 0]),
            ("C6/9", 0, "C6/9", [0, 4, 7, 9, 2]),
            ("C6/9/G", 0, "C6/9/G", [0, 4, 7, 9, 2]),
        ]
        for raw, tonic, exp_name, exp_pcs in cases:
            rc = resolve_chord(raw, tonic, notation="letter")
            assert rc.name == exp_name, f"Failed for {raw}: {rc.name} != {exp_name}"
            assert rc.pcs == exp_pcs, f"Pcs failed for {raw}: {rc.pcs} != {exp_pcs}"

    def test_letter_chord_transposition(self):
        # Printed in C (0), transposed to G (7)
        res_g = resolve_chord("C", 7, notation="letter", printed_tonic_pc=0)
        assert res_g.name == "G"
        assert res_g.root_pc == 7

        # Printed Bb/D in Bb (10), transposed to C (0): shift = +2
        res_c = resolve_chord("Bb/D", 0, notation="letter", printed_tonic_pc=10)
        assert res_c.name == "C/E"
        assert res_c.root_pc == 0
        assert res_c.bass_pc == 4

        # Printed Dm7/G in C (0), transposed to F (5): shift = +5
        res_f = resolve_chord("Dm7/G", 5, notation="letter", printed_tonic_pc=0)
        assert res_f.name == "Gm7/C"
        assert res_f.root_pc == 7
        assert res_f.bass_pc == 0

        # Printed C#m7 in F# (6), transposed to F (5): shift = -1
        res_cm7 = resolve_chord("C#m7", 5, notation="letter", printed_tonic_pc=6)
        assert res_cm7.name == "Cm7"
        assert res_cm7.root_pc == 0


class TestNumberChordSpellings:
    """Tests for advanced Taiwanese number chord spellings (Objective 2)."""

    def test_number_chord_spellings_catalog(self):
        cases = [
            ("5sus7", "G7sus4"),
            ("7b(6.9)", "Bb6/9"),
            ("7b(b9)", "Bb7b9"),
            ("57(b9)", "G7b9"),
            ("4m/6b", "Fm/Ab"),
            ("4/5", "F/G"),
            ("2m7/5", "Dm7/G"),
            ("6m7/5", "Am7/G"),
            ("1/5", "C/G"),
            ("1/6m", "C/A"),
            ("1(9)", "Cadd9"),
            ("1(11)", "C11"),
            ("1(13)", "C13"),
            ("1add9", "Cadd9"),
            ("169", "C6/9"),
            ("16/9", "C6/9"),
            ("16/9/5", "C6/9/G"),
            ("1m(maj7)", "CmM7"),
            ("1+", "Caug"),
            ("1aug", "Caug"),
            ("57(#5)", "G7#5"),
            ("57(#9)", "G7#9"),
            ("57(13)", "G7add13"),
            ("57/9", "G9"),
            ("57/11", "G11"),
            ("57/11-9", "G11"),
            ("17/9", "C9"),
            ("67", "A7"),
            ("6madd9", "Amadd9"),
            ("4madd9", "Fmadd9"),
            ("#4m7-5", "F#m7b5"),
            ("#4m-5", "F#m7b5"),
            ("#4/2", "F#/D"),
            ("1add9", "Cadd9"),
            ("4maj7", "Fmaj7"),
            ("⑤7/9", "G9"),
            ("⑤7/11", "G11"),
            ("⑤7/11 -9", "G11"),
            ("①7/9", "C9"),
            ("①add9", "Cadd9"),
            ("④maj7", "Fmaj7"),
            ("#④m7-5", "F#m7b5"),
            ("⑥madd9", "Amadd9"),
        ]
        for raw, exp_name in cases:
            rc = resolve_chord(raw, 0)
            assert rc.name == exp_name, f"Failed for {raw}: got {rc.name}, expected {exp_name}"

    def test_leading_slash_with_prev_chord(self):
        prev = resolve_chord("1", 0)
        rc = resolve_chord("/5", 0, beat=2.0, prev_chord=prev)
        assert rc.name == "C/G"
        assert rc.bass_pc == 7

        prev_am7 = resolve_chord("6m7", 0)
        rc2 = resolve_chord("/5", 0, beat=2.0, prev_chord=prev_am7)
        assert rc2.name == "Am7/G"
        assert rc2.bass_pc == 7

    def test_garbage_and_invalid_degree_raises_value_error(self):
        bad_cases = ["056", "0", "07", "8m", "99", "", "   "]
        for bad in bad_cases:
            with pytest.raises(ValueError):
                parse_chord(bad)
            with pytest.raises(ValueError):
                resolve_chord(bad, 0)


class TestStackedOrientation:
    """Tests for stacked fraction chord orientation inference and resolution."""

    def test_quality_asymmetry_evidence(self):
        # bottom has quality -> top is bass
        assert parse_chord("1/2m7-5").bass_degree == 1
        rc1 = resolve_chord("1/2m7-5", 0)
        assert rc1.name == "Dm7b5/C"
        assert rc1.root_pc == 2 and rc1.bass_pc == 0

        # top has quality -> bottom is bass
        rc2 = resolve_chord("2m7/5", 0)
        assert rc2.name == "Dm7/G"
        assert rc2.root_pc == 2 and rc2.bass_pc == 7

    def test_stacked_explicit_orientation(self):
        # 7/5 with top_is_bass -> chord 5 over bass 7 (G/B in C)
        rc_top = resolve_chord("7/5", 0, stacked=True, stacked_orientation="top_is_bass")
        assert rc_top.name == "G/B"
        assert rc_top.root_pc == 7 and rc_top.bass_pc == 11

        # 7/5 with bottom_is_bass -> chord 7 over bass 5 (B/G in C)
        rc_bot = resolve_chord("7/5", 0, stacked=True, stacked_orientation="bottom_is_bass")
        assert rc_bot.name == "B/G"
        assert rc_bot.root_pc == 11 and rc_bot.bass_pc == 7

    def test_infer_orientation_tinghai_and_xiaobaichuan(self):
        import json
        from app.models import ParsedSheet, SongHeader, System, Measure, ChordSymbol
        from app.theory.stacked import infer_stacked_orientation

        for slug in ["tinghai", "xiaobaichuan"]:
            with open(f"fixtures/groundtruth/{slug}.json", encoding="utf-8") as f:
                d = json.load(f)
            header = SongHeader.model_validate(d.get("header", {}))
            systems = []
            idx = 0
            for r in d.get("rows", []):
                measures = []
                for m in r.get("measures", []):
                    chords = [ChordSymbol(raw=c["raw"], beat=c.get("beat", 1.0), stacked=True) for c in m.get("chords", [])]
                    measures.append(Measure(index=idx, bbox=(0, 0, 1, 1), chords=chords))
                    idx += 1
                systems.append(System(page=r.get("page", 0), bbox=(0, 0, 1, 1), measures=measures))
            sheet = ParsedSheet(header=header, pages=[], systems=systems)
            res = infer_stacked_orientation(sheet)
            assert res.orientation == "top_is_bass"
            assert res.margin > 0.0

    def test_groundtruth_all_chords_sweep_resolves(self):
        import json
        from pathlib import Path
        gt_dir = Path("fixtures/groundtruth")
        total = 0
        unresolved = []
        for p in sorted(gt_dir.glob("*.json")):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            notation = data.get("header", {}).get("chord_notation", "number")
            rows = data.get("rows", data.get("systems", []))
            for r in rows:
                for m in r.get("measures", []):
                    prev = None
                    for c in m.get("chords", []):
                        raw = c.get("raw")
                        if not raw:
                            continue
                        total += 1
                        try:
                            rc = resolve_chord(raw, 0, notation=notation, prev_chord=prev)
                            prev = rc
                        except Exception as e:
                            unresolved.append((p.stem, raw, str(e)))
        assert unresolved == [], f"Unresolved chords: {unresolved}"
        assert total >= 500
