"""Unit tests for music theory, chord parsing, key resolution, and enharmonic spelling."""
import pytest

from app.theory.chords import parse_chord, resolve_chord
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
