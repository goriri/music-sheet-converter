"""Unit and regression tests for the OMR verification layer (app.qa).

Tests:
1. Melody parsing: tokens, onsets, accidentals, beat-sum, robustness, no exceptions.
2. Music theory priors: melody_fit, bass_hint_fit, progression_plausibility.
3. Candidate generation: OCR confusions (accidentals, m/M7, 6/b, slash 1/7, sus).
4. Decision rule & issue generation: concordance, auto-fixes, needs_review.
5. Structural checks: key changes restored from header, 2nd chord beats, melody mismatches.
6. Graceful degradation: offline mode (QA_OFFLINE=1) degrades cleanly with info issue.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from app.models import ChordSymbol, Measure, PageInfo, ParsedSheet, SongHeader, System
from app.qa.llm import LLMUnavailable, ask_json, llm_available
from app.qa.melody import ParsedMelody, parse_melody
from app.qa.omr_verify import verify_sheet
from app.qa.priors import (
    bass_hint_fit,
    generate_candidates,
    melody_fit,
    progression_plausibility,
    score_candidate,
)


# ---------------------------------------------------------------------------
# 1. Melody Parsing Tests
# ---------------------------------------------------------------------------

def test_melody_parsing_standard_beat_groups():
    res = parse_melody("2 2 23 21", beats=4.0)
    assert res is not None
    assert res.beat_sum == 4.0
    assert len(res.notes) == 6
    # Check onsets and degrees
    assert res.notes[0] == (1.0, 2, 0)
    assert res.notes[1] == (2.0, 2, 0)
    assert res.notes[2] == (3.0, 2, 0)
    assert res.notes[3] == (3.5, 3, 0)
    assert res.notes[4] == (4.0, 2, 0)
    assert res.notes[5] == (4.5, 1, 0)


def test_melody_parsing_dotted_notes():
    res = parse_melody("1. 5 55 65", beats=4.0)
    assert res is not None
    assert res.beat_sum == 4.0
    assert res.notes[0] == (1.0, 1, 0)
    assert res.notes[1] == (2.5, 5, 0)
    assert res.notes[2] == (3.0, 5, 0)
    assert res.notes[3] == (3.5, 5, 0)


def test_melody_parsing_dashes_and_rests():
    res = parse_melody("2 - 0 0", beats=4.0)
    assert res is not None
    assert res.beat_sum == 4.0
    # Rest 0 excluded by default from pitched notes
    assert len(res.notes) == 1
    assert res.notes[0] == (1.0, 2, 0)


def test_melody_parsing_high_and_low_octave():
    res = parse_melody("0 .76 .77 ii", beats=4.0)
    assert res is not None
    assert res.beat_sum == 4.0
    degrees = [n[1] for n in res.notes]
    assert 7 in degrees
    assert 1 in degrees  # 'i' parsed as degree 1


def test_melody_parsing_accidentals():
    res = parse_melody("0 b76 b77 11", beats=4.0)
    assert res is not None
    # b7 should have accidental -1
    b7_notes = [n for n in res.notes if n[1] == 7 and n[2] == -1]
    assert len(b7_notes) >= 2


def test_melody_parsing_missing_beats():
    res = parse_melody("2 2", beats=4.0)
    assert res is not None
    assert res.beat_sum == 2.0  # Detects missing 2 beats!


def test_melody_parsing_never_raises_on_garbage():
    assert parse_melody("") == ParsedMelody(notes=[], beat_sum=0.0)
    assert parse_melody(None) == ParsedMelody(notes=[], beat_sum=0.0)
    assert parse_melody("!!!???%%%") is None
    assert parse_melody("abc def") is None


# ---------------------------------------------------------------------------
# 2. Priors Tests
# ---------------------------------------------------------------------------

def test_priors_melody_fit_high():
    # 1(2) in key C: notes C(0), D(2), E(4), G(7). Melody notes D(degree 2)
    score = melody_fit("1(2)", [(1.0, 2, 0), (2.0, 2, 0)], chord_beat=1.0, chord_duration=2.0)
    assert score == 1.0


def test_priors_melody_fit_clash():
    # 1(2) in key C: notes C, D, E, G. Melody note F#(degree 4, acc 1) or Bb(deg 7, acc -1)
    score = melody_fit("1(2)", [(1.0, 4, 1), (2.0, 4, 1)], chord_beat=1.0, chord_duration=2.0)
    assert score == 0.0


def test_priors_bass_hint_fit():
    assert bass_hint_fit("1", "11 11 11 112") == 1.0
    assert bass_hint_fit("5/7", "7. 7. 7. 7") == 1.0
    assert bass_hint_fit("5m/7b", ".7.7 .7.74 .7.7 .7.7") == 1.0
    assert bass_hint_fit("5", "11 11 11 112") == 0.0
    assert bass_hint_fit("1", None) is None
    assert bass_hint_fit("1", "") is None


def test_priors_progression_plausibility():
    assert progression_plausibility("1", prev_chord="5") == 1.0
    assert progression_plausibility("5m/7b", prev_chord="5/7", next_chord="5m6/2") == 1.0
    assert progression_plausibility("2m7", prev_chord="67/1#") == 1.0
    assert progression_plausibility("6m7-5", prev_chord="6b") == 1.0
    # Implausible transition: tritone jump 1 -> 4#
    score_weird = progression_plausibility("4#", prev_chord="1")
    assert score_weird <= 0.50


def test_candidate_generation_rules():
    # 1. Accidental dropped in slash bass
    cands_5m7 = generate_candidates("5m/7")
    assert "5m/7b" in cands_5m7

    # 2. Accidental dropped in borrowed chord
    cands_6 = generate_candidates("6")
    assert "6b" in cands_6

    # 3. M7 <-> 7
    cands_47 = generate_candidates("47")
    assert "4M7" in cands_47

    # 4. Lost (2)
    cands_1 = generate_candidates("1")
    assert "1(2)" in cands_1

    # 5. 5sus <-> 57sus
    cands_5sus = generate_candidates("5sus")
    assert "57sus" in cands_5sus

    # 6. Lost 'm'
    cands_27 = generate_candidates("27")
    assert "2m7" in cands_27


# ---------------------------------------------------------------------------
# 3. OMR Verification Layer Tests (Offline / QA_OFFLINE=1)
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_sheet() -> ParsedSheet:
    sample_path = Path("fixtures/omr_sample.json")
    with open(sample_path, encoding="utf-8") as f:
        return ParsedSheet.model_validate_json(f.read())


@pytest.fixture
def dummy_images() -> list[bytes]:
    # 1x1 dummy JPEGs
    import cv2
    import numpy as np
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".jpg", img)
    return [buf.tobytes(), buf.tobytes()]


def test_verify_sheet_offline_clean(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    verified = verify_sheet(dummy_images, clean_sheet, use_llm=False)

    needs_review = [i for i in verified.issues if i.severity == "needs_review"]
    auto_fixed = [i for i in verified.issues if i.severity == "auto_fixed"]

    # Target: clean fixture <= 2 needs_review and 0 wrong auto-fixes
    assert len(needs_review) <= 2
    assert len(auto_fixed) == 0

    # Ensure valid chords were preserved
    m0_chord = verified.systems[0].measures[0].chords[0].raw
    assert m0_chord == "1(2)"


def test_verify_sheet_restores_missing_key_change(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    # Simulate corrupted sheet where key_changes was stripped
    clean_sheet.key_changes = []

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=False)
    assert len(verified.key_changes) > 0
    assert verified.key_changes[0].semitones == 2

    # Verify auto_fixed issue was emitted
    fix_issue = next((i for i in verified.issues if i.code == "key_change_restored"), None)
    assert fix_issue is not None
    assert fix_issue.severity == "auto_fixed"


def test_verify_sheet_auto_fixes_2nd_chord_beat(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    # Corrupt 2nd chord in m11 to beat 1.0
    meas = [m for s in clean_sheet.systems for m in s.measures if m.index == 11][0]
    assert len(meas.chords) == 2
    meas.chords[1].beat = 1.0

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=False)
    v_meas = [m for s in verified.systems for m in s.measures if m.index == 11][0]
    assert v_meas.chords[1].beat == 3.0

    beat_issue = next((i for i in verified.issues if i.code == "chord_beat_corrected"), None)
    assert beat_issue is not None
    assert beat_issue.severity == "auto_fixed"


def test_verify_sheet_flags_melody_beat_mismatch(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    # Truncate melody in m23 to 1 beat
    meas = [m for s in clean_sheet.systems for m in s.measures if m.index == 23][0]
    meas.melody = "2"

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=False)
    mel_issue = next((i for i in verified.issues if i.code == "melody_beat_sum_mismatch" and i.measure_index == 23), None)
    assert mel_issue is not None
    assert mel_issue.severity == "needs_review"


def test_verify_sheet_flags_unparseable_chord(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    # Add an unparseable chord 'XYZ'
    meas = [m for s in clean_sheet.systems for m in s.measures if m.index == 3][0]
    meas.chords.append(ChordSymbol(raw="XYZ", beat=1.0, bbox=(0.7, 0.1, 0.8, 0.2)))

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=False)
    bad_issue = next((i for i in verified.issues if i.code == "invalid_chord_grammar" and i.measure_index == 3), None)
    assert bad_issue is not None
    assert bad_issue.severity == "needs_review"


def test_verify_sheet_graceful_degradation_on_exception(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    # Should never raise even if images are invalid or empty
    result = verify_sheet([], clean_sheet, use_llm=True)
    assert isinstance(result, ParsedSheet)
    info_issues = [i for i in result.issues if i.severity == "info"]
    assert len(info_issues) >= 1
