"""Unit and regression tests for the OMR verification layer (app.qa).

Tests:
1. Melody parsing: tokens, onsets, accidentals, beat-sum, robustness, no exceptions.
2. Music theory priors: melody_fit, bass_hint_fit, progression_plausibility.
3. Candidate generation: OCR confusions (accidentals, m/M7, 6/b, slash 1/7, sus).
4. Header key pair parsing: various formats, semitones mod 12 smaller signed value.
5. Located key-change -> auto_fixed with image confirmation.
6. Unlocated key-change -> needs_review, never invent a location.
7. Missing chord inspection: no chord insertion without arbiter confirmation (confidence >= 0.8).
8. Decision rule & issue generation: concordance, beat corrections, melody mismatches.
9. Graceful degradation: offline mode (QA_OFFLINE=1) degrades cleanly with info issue.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import pytest

from app.models import ChordSymbol, Measure, PageInfo, ParsedSheet, QualityIssue, SongHeader, System
from app.qa.llm import LLMUnavailable, ask_json, llm_available
from app.qa.melody import ParsedMelody, parse_melody
from app.qa.omr_verify import (
    ArbiterBatchResponse,
    CandidateInsertionBatchResponse,
    CandidateInsertionDecision,
    DetectedMeasureChordBox,
    LocatedKeyAnnotation,
    MeasureCropChordResponse,
    MissingChordArbiterConfirmation,
    PageCropReadings,
    PageKeyLocateResponse,
    PageRowTranscriptionResponse,
    RowTranscribedChord,
    SystemRowChords,
    align_measure_chords,
    map_transcribed_chords_to_measures,
    parse_header_key_changes,
    resolve_chord_beat_geo,
    verify_sheet,
)
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
    res = parse_melody("2 2 12 21", beats=4.0)
    assert res is not None
    assert res.beat_sum == 4.0
    assert len(res.notes) == 6
    assert res.notes[0] == (1.0, 2, 0)
    assert res.notes[1] == (2.0, 2, 0)
    assert res.notes[2] == (3.0, 1, 0)
    assert res.notes[3] == (3.5, 2, 0)
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
    assert len(res.notes) == 1
    assert res.notes[0] == (1.0, 2, 0)


def test_melody_parsing_high_and_low_octave():
    res = parse_melody("0 .76 .77 ii", beats=4.0)
    assert res is not None
    assert res.beat_sum == 4.0
    degrees = [n[1] for n in res.notes]
    assert 7 in degrees
    assert 1 in degrees


def test_melody_parsing_accidentals():
    res = parse_melody("0 b76 b77 11", beats=4.0)
    assert res is not None
    b7_notes = [n for n in res.notes if n[1] == 7 and n[2] == -1]
    assert len(b7_notes) >= 2


def test_melody_parsing_missing_beats():
    res = parse_melody("2 2", beats=4.0)
    assert res is not None
    assert res.beat_sum == 2.0


def test_melody_parsing_never_raises_on_garbage():
    assert parse_melody("") == ParsedMelody(notes=[], beat_sum=0.0)
    assert parse_melody(None) == ParsedMelody(notes=[], beat_sum=0.0)
    assert parse_melody("!!!???%%%") is None
    assert parse_melody("abc def") is None


# ---------------------------------------------------------------------------
# 2. Priors Tests
# ---------------------------------------------------------------------------

def test_priors_melody_fit_high():
    score = melody_fit("1(2)", [(1.0, 2, 0), (2.0, 2, 0)], chord_beat=1.0, chord_duration=2.0)
    assert score == 1.0


def test_priors_melody_fit_clash():
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
    score_weird = progression_plausibility("4#", prev_chord="1")
    assert score_weird <= 0.5


def test_candidate_generation_rules():
    cands_5m7 = generate_candidates("5m/7")
    assert "5m/7b" in cands_5m7

    cands_6 = generate_candidates("6")
    assert "6b" in cands_6

    cands_47 = generate_candidates("47")
    assert "4M7" in cands_47

    cands_1 = generate_candidates("1")
    assert "1(2)" in cands_1

    cands_5sus = generate_candidates("5sus")
    assert "57sus" in cands_5sus

    cands_27 = generate_candidates("27")
    assert "2m7" in cands_27


# ---------------------------------------------------------------------------
# 3. Header Key Pair Parsing Tests
# ---------------------------------------------------------------------------

def test_parse_header_key_changes_various_formats():
    assert parse_header_key_changes("(F# - Ab)") == 2
    assert parse_header_key_changes("(F#-Ab)") == 2
    assert parse_header_key_changes("F#→Ab") == 2
    assert parse_header_key_changes("F#->Ab") == 2
    assert parse_header_key_changes("男調(Bb-C) 女調(F-G)") == 2
    assert parse_header_key_changes("(C - D)") == 2
    assert parse_header_key_changes("(G - E)") == -3
    assert parse_header_key_changes("(A - F#)") == -3
    assert parse_header_key_changes("[降B - C]") == 2
    assert parse_header_key_changes("Slow Soul 4/4") == 0
    assert parse_header_key_changes("") == 0


# ---------------------------------------------------------------------------
# 4. OMR Verification Layer Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_sheet() -> ParsedSheet:
    sample_path = Path("fixtures/omr_sample.json")
    with open(sample_path, encoding="utf-8") as f:
        return ParsedSheet.model_validate_json(f.read())


@pytest.fixture
def dummy_images() -> list[bytes]:
    import cv2
    import numpy as np
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".jpg", img)
    return [buf.tobytes(), buf.tobytes()]


def test_verify_sheet_offline_clean(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    verified = verify_sheet(dummy_images, clean_sheet, use_llm=False)

    needs_review = [i for i in verified.issues if i.severity == "needs_review"]
    auto_fixed = [i for i in verified.issues if i.severity == "auto_fixed"]

    assert len(needs_review) <= 2
    assert len(auto_fixed) == 0

    first_chord = verified.systems[0].measures[0].chords[0].raw
    assert first_chord == "1(2)"


def _mock_ask_json_located(prompt: str, schema: Any, images=None, role: str = "reader", timeout_s: float = 60.0):
    if schema == PageKeyLocateResponse:
        return PageKeyLocateResponse(
            annotations=[
                LocatedKeyAnnotation(
                    system_index=0,
                    measure_index=0,
                    approx_x=0.15,
                    verbatim_text="轉2調",
                    semitones=2,
                )
            ]
        )
    elif schema == PageRowTranscriptionResponse:
        return PageRowTranscriptionResponse(rows=[])
    elif schema == CandidateInsertionBatchResponse:
        return CandidateInsertionBatchResponse(decisions=[])
    elif schema == PageCropReadings:
        return PageCropReadings(crops=[])
    elif schema == ArbiterBatchResponse:
        return ArbiterBatchResponse(decisions=[])
    raise RuntimeError(f"Unexpected schema: {schema}")


def _mock_ask_json_unlocated(prompt: str, schema: Any, images=None, role: str = "reader", timeout_s: float = 60.0):
    if schema == PageKeyLocateResponse:
        return PageKeyLocateResponse(annotations=[])
    elif schema == PageRowTranscriptionResponse:
        return PageRowTranscriptionResponse(rows=[])
    elif schema == CandidateInsertionBatchResponse:
        return CandidateInsertionBatchResponse(decisions=[])
    elif schema == PageCropReadings:
        return PageCropReadings(crops=[])
    elif schema == ArbiterBatchResponse:
        return ArbiterBatchResponse(decisions=[])
    raise RuntimeError(f"Unexpected schema: {schema}")


def _mock_ask_json_missing_unconfirmed(prompt: str, schema: Any, images=None, role: str = "reader", timeout_s: float = 60.0):
    if schema == MeasureCropChordResponse:
        return MeasureCropChordResponse(chords=[DetectedMeasureChordBox(chord="5", beat=3.0, confidence=0.8)])
    elif schema == MissingChordArbiterConfirmation:
        return MissingChordArbiterConfirmation(confirmed=False, confidence=0.4, reason="Ink not visible")
    elif schema == PageRowTranscriptionResponse:
        return PageRowTranscriptionResponse(rows=[])
    elif schema == CandidateInsertionBatchResponse:
        return CandidateInsertionBatchResponse(decisions=[])
    elif schema == PageCropReadings:
        return PageCropReadings(crops=[])
    elif schema == ArbiterBatchResponse:
        return ArbiterBatchResponse(decisions=[])
    raise RuntimeError(f"Unexpected schema {schema}")


def _mock_ask_json_missing_confirmed(prompt: str, schema: Any, images=None, role: str = "reader", timeout_s: float = 60.0):
    if schema == MeasureCropChordResponse:
        return MeasureCropChordResponse(chords=[DetectedMeasureChordBox(chord="5", beat=3.0, confidence=0.8)])
    elif schema == MissingChordArbiterConfirmation:
        return MissingChordArbiterConfirmation(confirmed=True, confirmed_chord="5", confidence=0.9, reason="Box clearly visible")
    elif schema == PageRowTranscriptionResponse:
        return PageRowTranscriptionResponse(rows=[])
    elif schema == CandidateInsertionBatchResponse:
        return CandidateInsertionBatchResponse(decisions=[])
    elif schema == PageCropReadings:
        return PageCropReadings(crops=[])
    elif schema == ArbiterBatchResponse:
        return ArbiterBatchResponse(decisions=[])
    raise RuntimeError(f"Unexpected schema {schema}")


def _mock_ask_json_row_candidate_confirmed(prompt: str, schema: Any, images=None, role: str = "reader", timeout_s: float = 60.0):
    if schema == PageRowTranscriptionResponse:
        return PageRowTranscriptionResponse(
            rows=[
                SystemRowChords(
                    row_index=0,
                    chords=[
                        RowTranscribedChord(measure_in_row=1, chord="1(2)", approx_x=0.15, beat=1.0),
                        RowTranscribedChord(measure_in_row=2, chord="5/7", approx_x=0.35, beat=1.0),
                        RowTranscribedChord(measure_in_row=3, chord="2m7", approx_x=0.60, beat=1.0),
                        RowTranscribedChord(measure_in_row=3, chord="5", approx_x=0.75, beat=3.0),
                    ],
                )
            ]
        )
    elif schema == CandidateInsertionBatchResponse:
        return CandidateInsertionBatchResponse(
            decisions=[
                CandidateInsertionDecision(
                    item_id=1,
                    confirmed=True,
                    confirmed_chord="5",
                    confidence=0.92,
                    reason="Clear 5 chord box visible in measure 3",
                )
            ]
        )
    elif schema == PageCropReadings:
        return PageCropReadings(crops=[])
    elif schema == ArbiterBatchResponse:
        return ArbiterBatchResponse(decisions=[])
    raise RuntimeError(f"Unexpected schema: {schema}")


def _mock_ask_json_row_candidate_unconfirmed(prompt: str, schema: Any, images=None, role: str = "reader", timeout_s: float = 60.0):
    if schema == PageRowTranscriptionResponse:
        return PageRowTranscriptionResponse(
            rows=[
                SystemRowChords(
                    row_index=0,
                    chords=[
                        RowTranscribedChord(measure_in_row=1, chord="1(2)", approx_x=0.15, beat=1.0),
                        RowTranscribedChord(measure_in_row=2, chord="5/7", approx_x=0.35, beat=1.0),
                        RowTranscribedChord(measure_in_row=3, chord="2m7", approx_x=0.60, beat=1.0),
                        RowTranscribedChord(measure_in_row=3, chord="5", approx_x=0.75, beat=3.0),
                    ],
                )
            ]
        )
    elif schema == CandidateInsertionBatchResponse:
        return CandidateInsertionBatchResponse(
            decisions=[
                CandidateInsertionDecision(
                    item_id=1,
                    confirmed=True,
                    confirmed_chord=None,
                    confidence=0.35,
                    reason="Low confidence possible chord box",
                )
            ]
        )
    elif schema == PageCropReadings:
        return PageCropReadings(crops=[])
    elif schema == ArbiterBatchResponse:
        return ArbiterBatchResponse(decisions=[])
    raise RuntimeError(f"Unexpected schema: {schema}")


def _mock_ask_json_row_candidate_rejected(prompt: str, schema: Any, images=None, role: str = "reader", timeout_s: float = 60.0):
    if schema == PageRowTranscriptionResponse:
        return PageRowTranscriptionResponse(
            rows=[
                SystemRowChords(
                    row_index=0,
                    chords=[
                        RowTranscribedChord(measure_in_row=1, chord="1(2)", approx_x=0.15, beat=1.0),
                        RowTranscribedChord(measure_in_row=2, chord="5/7", approx_x=0.35, beat=1.0),
                        RowTranscribedChord(measure_in_row=3, chord="2m7", approx_x=0.60, beat=1.0),
                        RowTranscribedChord(measure_in_row=3, chord="5", approx_x=0.75, beat=3.0),
                    ],
                )
            ]
        )
    elif schema == CandidateInsertionBatchResponse:
        return CandidateInsertionBatchResponse(
            decisions=[
                CandidateInsertionDecision(
                    item_id=1,
                    confirmed=False,
                    confirmed_chord=None,
                    confidence=0.10,
                    reason="No box printed at beat 3; ink is a volta ending bracket",
                )
            ]
        )
    elif schema == PageCropReadings:
        return PageCropReadings(crops=[])
    elif schema == ArbiterBatchResponse:
        return ArbiterBatchResponse(decisions=[])
    raise RuntimeError(f"Unexpected schema: {schema}")


def test_verify_sheet_located_key_change_auto_fixed(monkeypatch, clean_sheet, dummy_images):
    clean_sheet.key_changes = []
    clean_sheet.header.raw = "4/4 (C - D)"
    clean_sheet.header.original_key = "C"

    first_meas = clean_sheet.systems[0].measures[0]
    expected_m_idx = first_meas.index

    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_json_located)
    monkeypatch.setattr("app.qa.omr_verify.llm_available", lambda: True)

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=True)
    assert len(verified.key_changes) == 1
    assert verified.key_changes[0].semitones == 2
    assert verified.key_changes[0].at_measure == expected_m_idx

    fix_issue = next((i for i in verified.issues if i.code == "key_change_restored"), None)
    assert fix_issue is not None
    assert fix_issue.severity == "auto_fixed"


def test_verify_sheet_unlocated_key_change_needs_review(monkeypatch, clean_sheet, dummy_images):
    clean_sheet.key_changes = []
    clean_sheet.header.raw = "4/4 (C - D)"
    clean_sheet.header.original_key = "C"

    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_json_unlocated)
    monkeypatch.setattr("app.qa.omr_verify.llm_available", lambda: True)

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=True)
    assert len(verified.key_changes) == 0

    unloc_issue = next((i for i in verified.issues if i.code == "key_change_unlocated"), None)
    assert unloc_issue is not None
    assert unloc_issue.severity == "needs_review"
    assert "谱头显示有转调但未能定位，请在核对表中设置转调小节" in unloc_issue.message


def test_no_chord_insertion_without_arbiter_confirmation(monkeypatch, clean_sheet, dummy_images):
    two_chord_meas = next(
        m for s in clean_sheet.systems for m in s.measures if m.fill and len(m.chords) >= 2 and m.chords[1].beat == 3.0
    )
    del two_chord_meas.chords[1]
    assert len(two_chord_meas.chords) == 1
    target_idx = two_chord_meas.index

    # Case A: Arbiter does NOT confirm (confidence < 0.8) -> NO insertion, needs_review
    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_json_missing_unconfirmed)
    monkeypatch.setattr("app.qa.omr_verify.llm_available", lambda: True)

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=True)
    v_meas = next(m for s in verified.systems for m in s.measures if m.index == target_idx)
    assert len(v_meas.chords) == 1  # Not inserted!

    review_issue = next(
        (i for i in verified.issues if i.code == "missing_chord_suspected" and i.measure_index == target_idx),
        None,
    )
    assert review_issue is not None
    assert review_issue.severity == "needs_review"

    # Case B: Arbiter DOES confirm (confidence >= 0.8) -> Auto-fixed and inserted!
    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_json_missing_confirmed)

    verified2 = verify_sheet(dummy_images, clean_sheet, use_llm=True)
    v_meas2 = next(m for s in verified2.systems for m in s.measures if m.index == target_idx)
    assert len(v_meas2.chords) == 2
    assert v_meas2.chords[1].raw == "5"

    restored_issue = next(
        (i for i in verified2.issues if i.code == "chord_restored" and i.measure_index == target_idx),
        None,
    )
    assert restored_issue is not None
    assert restored_issue.severity == "auto_fixed"


def test_verify_sheet_auto_fixes_2nd_chord_beat(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    target_meas = next(
        m for s in clean_sheet.systems for m in s.measures if len(m.chords) >= 2 and m.chords[1].beat == 3.0
    )
    target_idx = target_meas.index
    target_meas.chords[1].beat = 1.0

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=False)
    v_meas = next(m for s in verified.systems for m in s.measures if m.index == target_idx)
    assert v_meas.chords[1].beat == 3.0

    beat_issue = next(
        (i for i in verified.issues if i.code == "chord_beat_corrected" and i.measure_index == target_idx),
        None,
    )
    assert beat_issue is not None
    assert beat_issue.severity == "auto_fixed"


def test_verify_sheet_flags_melody_beat_mismatch(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    target_meas = clean_sheet.systems[0].measures[0]
    target_idx = target_meas.index
    target_meas.melody = "2"

    # Single chord: severity is 'info' (melody text underlines absent)
    verified = verify_sheet(dummy_images, clean_sheet, use_llm=False)
    mel_issue = next(
        (i for i in verified.issues if i.code == "melody_beat_sum_mismatch" and i.measure_index == target_idx),
        None,
    )
    assert mel_issue is not None
    assert mel_issue.severity == "info"

    # >= 2 chords with beat disagreement: severity escalates to 'needs_review'
    target_meas.chords.append(ChordSymbol(raw="5", beat=1.0))
    verified2 = verify_sheet(dummy_images, clean_sheet, use_llm=False)
    mel_issue2 = next(
        (i for i in verified2.issues if i.code == "melody_beat_sum_mismatch" and i.measure_index == target_idx),
        None,
    )
    assert mel_issue2 is not None
    assert mel_issue2.severity == "needs_review"



def test_verify_sheet_flags_unparseable_chord(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    target_meas = clean_sheet.systems[0].measures[0]
    target_idx = target_meas.index
    target_meas.chords.append(ChordSymbol(raw="XYZ", beat=1.0, bbox=(0.7, 0.1, 0.8, 0.2)))

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=False)
    bad_issue = next(
        (i for i in verified.issues if i.code == "invalid_chord_grammar" and i.measure_index == target_idx),
        None,
    )
    assert bad_issue is not None
    assert bad_issue.severity == "needs_review"


def test_verify_sheet_graceful_degradation_on_exception(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setenv("QA_OFFLINE", "1")
    result = verify_sheet([], clean_sheet, use_llm=True)
    assert isinstance(result, ParsedSheet)
    info_issues = [i for i in result.issues if i.severity == "info"]
    assert len(info_issues) >= 1


def test_align_measure_chords_exact_match():
    sheet_chords = [ChordSymbol(raw="1(2)", beat=1.0), ChordSymbol(raw="5", beat=3.0)]
    row_chords = [RowTranscribedChord(chord="1(2)", beat=1.0), RowTranscribedChord(chord="5", beat=3.0)]
    unmatched_sheet, candidates = align_measure_chords(sheet_chords, row_chords)
    assert len(unmatched_sheet) == 0
    assert len(candidates) == 0


def test_align_measure_chords_detects_candidate_insertion():
    sheet_chords = [ChordSymbol(raw="1(2)", beat=1.0)]
    row_chords = [RowTranscribedChord(chord="1(2)", beat=1.0), RowTranscribedChord(chord="5", beat=3.0)]
    unmatched_sheet, candidates = align_measure_chords(sheet_chords, row_chords)
    assert len(unmatched_sheet) == 0
    assert len(candidates) == 1
    assert candidates[0].chord == "5"
    assert candidates[0].beat == 3.0


def test_align_measure_chords_detects_unmatched_sheet_chord():
    sheet_chords = [ChordSymbol(raw="1(2)", beat=1.0), ChordSymbol(raw="4", beat=3.0)]
    row_chords = [RowTranscribedChord(chord="1(2)", beat=1.0)]
    unmatched_sheet, candidates = align_measure_chords(sheet_chords, row_chords)
    assert len(unmatched_sheet) == 1
    assert unmatched_sheet[0].raw == "4"
    assert len(candidates) == 0


def test_verify_sheet_candidate_insertion_auto_fixed_with_arbiter(monkeypatch, clean_sheet, dummy_images):
    m2 = clean_sheet.systems[0].measures[2]
    target_idx = m2.index
    orig_count = len(m2.chords)

    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_json_row_candidate_confirmed)
    monkeypatch.setattr("app.qa.omr_verify.llm_available", lambda: True)

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=True)
    v_m2 = next(m for s in verified.systems for m in s.measures if m.index == target_idx)
    assert len(v_m2.chords) == orig_count + 1
    assert any(c.raw == "5" and c.beat == 3.0 for c in v_m2.chords)

    restored_issue = next(
        (i for i in verified.issues if i.code == "chord_restored" and i.measure_index == target_idx),
        None,
    )
    assert restored_issue is not None
    assert restored_issue.severity == "auto_fixed"


def test_verify_sheet_candidate_insertion_unconfirmed_needs_review(monkeypatch, clean_sheet, dummy_images):
    m2 = clean_sheet.systems[0].measures[2]
    target_idx = m2.index
    orig_count = len(m2.chords)

    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_json_row_candidate_unconfirmed)
    monkeypatch.setattr("app.qa.omr_verify.llm_available", lambda: True)

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=True)
    v_m2 = next(m for s in verified.systems for m in s.measures if m.index == target_idx)
    assert len(v_m2.chords) == orig_count

    review_issue = next(
        (i for i in verified.issues if i.code == "missing_chord_suspected" and i.measure_index == target_idx),
        None,
    )
    assert review_issue is not None
    assert review_issue.severity == "needs_review"


def test_verify_sheet_candidate_insertion_rejected_discarded(monkeypatch, clean_sheet, dummy_images):
    m2 = clean_sheet.systems[0].measures[2]
    target_idx = m2.index
    orig_count = len(m2.chords)

    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_json_row_candidate_rejected)
    monkeypatch.setattr("app.qa.omr_verify.llm_available", lambda: True)

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=True)
    v_m2 = next(m for s in verified.systems for m in s.measures if m.index == target_idx)
    assert len(v_m2.chords) == orig_count

    # The rejected candidate must NOT produce a missing_chord_suspected issue
    review_issue = next(
        (i for i in verified.issues if i.code == "missing_chord_suspected" and i.measure_index == target_idx),
        None,
    )
    assert review_issue is None


def _mock_ask_json_barline(prompt: str, schema: Any, images=None, role: str = "reader", timeout_s: float = 60.0):
    if schema == PageRowTranscriptionResponse:
        return PageRowTranscriptionResponse(
            rows=[
                SystemRowChords(row_index=9, measure_count=4, chords=[])
            ]
        )
    elif schema == PageCropReadings:
        return PageCropReadings()
    elif schema == ArbiterBatchResponse:
        return ArbiterBatchResponse()
    elif schema == CandidateInsertionBatchResponse:
        return CandidateInsertionBatchResponse()
    return None


def test_verify_sheet_barline_mismatch_downgraded_with_row_transcription(monkeypatch, clean_sheet, dummy_images):
    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_json_barline)
    monkeypatch.setattr("app.qa.omr_verify.llm_available", lambda: True)

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=True)
    consistent_issue = next((i for i in verified.issues if i.code == "barline_count_consistent"), None)
    mismatch_issue = next((i for i in verified.issues if i.code == "barline_count_mismatch"), None)

    assert consistent_issue is not None
    assert consistent_issue.severity == "info"
    assert mismatch_issue is None


def test_resolve_chord_beat_geo_within_threshold():
    m = Measure(index=0, beats=4, bbox=[0.2, 0.1, 0.6, 0.2])
    c1 = ChordSymbol(raw="C", beat=1.0, bbox=[0.201, 0.11, 0.25, 0.19])
    c2 = ChordSymbol(raw="F", beat=3.0, bbox=[0.405, 0.11, 0.45, 0.19])

    resolved_b1, is_ambig1 = resolve_chord_beat_geo(c1, m, [1.0, 2.0])
    assert is_ambig1 is False
    assert resolved_b1 == 1.0

    resolved_b2, is_ambig2 = resolve_chord_beat_geo(c2, m, [2.0, 3.0])
    assert is_ambig2 is False
    assert resolved_b2 == 3.0

    resolved_b_no_competing, is_ambig_no_competing = resolve_chord_beat_geo(c2, m, [])
    assert is_ambig_no_competing is False
    assert resolved_b_no_competing == 3.0


def test_resolve_chord_beat_geo_ambiguous():
    m = Measure(index=0, beats=4, bbox=[0.2, 0.1, 0.6, 0.2])
    c = ChordSymbol(raw="Am", beat=1.0, bbox=[0.30, 0.11, 0.35, 0.19])

    resolved_b, is_ambig = resolve_chord_beat_geo(c, m, [1.0, 4.0])
    assert is_ambig is True
    assert resolved_b == 2.0

    c_nobbox = ChordSymbol(raw="G", beat=1.0, bbox=None)
    resolved_none, is_ambig_nobbox = resolve_chord_beat_geo(c_nobbox, m, [1.0])
    assert is_ambig_nobbox is True
    assert resolved_none is None


def _mock_ask_json_geo_resolved(prompt: str, schema: Any, images=None, role: str = "reader", timeout_s: float = 60.0):
    if schema == PageRowTranscriptionResponse:
        return PageRowTranscriptionResponse(
            rows=[
                SystemRowChords(row_index=0, measure_count=4, chords=[]),
                SystemRowChords(
                    row_index=1,
                    measure_count=4,
                    chords=[
                        RowTranscribedChord(measure_in_row=2, chord="17/7b", beat=1.0, approx_x=0.35),
                        RowTranscribedChord(measure_in_row=2, chord="4M7", beat=2.0, approx_x=0.55),
                    ],
                ),
            ]
        )
    elif schema == PageCropReadings:
        return PageCropReadings()
    elif schema == ArbiterBatchResponse:
        return ArbiterBatchResponse()
    elif schema == CandidateInsertionBatchResponse:
        return CandidateInsertionBatchResponse()
    return None


def test_verify_sheet_melody_mismatch_resolved_geometrically(monkeypatch, clean_sheet, dummy_images):
    m45 = next(m for s in clean_sheet.systems for m in s.measures if m.index == 45)
    m45.melody = "1 2 3"

    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_json_geo_resolved)
    monkeypatch.setattr("app.qa.omr_verify.llm_available", lambda: True)

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=True)

    geo_resolved_issue = next(
        (i for i in verified.issues if i.code == "chord_beat_resolved_geo" and i.measure_index == 45),
        None,
    )
    assert geo_resolved_issue is not None
    assert geo_resolved_issue.severity == "info"

    review_issue_m45 = next(
        (i for i in verified.issues if i.severity == "needs_review" and i.measure_index == 45),
        None,
    )
    assert review_issue_m45 is None


def test_verify_sheet_melody_mismatch_unresolved_geometrically_needs_review(monkeypatch, clean_sheet, dummy_images):
    m45 = next(m for s in clean_sheet.systems for m in s.measures if m.index == 45)
    m45.melody = "1 2 3"
    m45.chords[1].bbox = None

    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_json_geo_resolved)
    monkeypatch.setattr("app.qa.omr_verify.llm_available", lambda: True)

    verified = verify_sheet(dummy_images, clean_sheet, use_llm=True)

    review_issue_m45 = next(
        (i for i in verified.issues if i.code == "melody_beat_sum_mismatch" and i.measure_index == 45),
        None,
    )
    assert review_issue_m45 is not None
    assert review_issue_m45.severity == "needs_review"


# ---------------------------------------------------------------------------
# 10. v2 Classical CV Layout Verification Tests
# ---------------------------------------------------------------------------

def _mock_ask_no_row_transcription(prompt: str, schema: Any, images=None, role: str = "reader", timeout_s: float = 60.0):
    if schema == PageRowTranscriptionResponse:
        raise AssertionError("_transcribe_page_row should NOT be called for v2 geometry!")
    elif schema == PageCropReadings:
        return PageCropReadings()
    elif schema == ArbiterBatchResponse:
        return ArbiterBatchResponse()
    elif schema == CandidateInsertionBatchResponse:
        return CandidateInsertionBatchResponse()
    elif schema == MeasureCropChordResponse:
        return MeasureCropChordResponse(chords=[])
    raise RuntimeError(f"Unexpected schema: {schema}")


def test_v2_geometry_verification_pipeline(monkeypatch, clean_sheet, dummy_images):
    """Verify that v2 geometry sheets skip row transcription and suppress false fill alerts."""
    v2_sheet = clean_sheet.model_copy(deep=True)
    v2_sheet.warnings = ["[v2_geometry] Classical CV layout authoritative boundaries"]
    v2_sheet.layout_confidence = 0.95
    v2_sheet.layout_source = "cv"

    monkeypatch.setattr("app.qa.omr_verify.ask_json", _mock_ask_no_row_transcription)
    monkeypatch.setattr("app.qa.omr_verify.llm_available", lambda: True)

    verified = verify_sheet(dummy_images, v2_sheet, use_llm=True)

    # 1. Barline checks emitted barline_count_consistent info issues
    barline_issues = [i for i in verified.issues if i.code == "barline_count_consistent"]
    assert len(barline_issues) > 0
    assert all(i.severity == "info" for i in barline_issues)

    # 2. No needs_review barline issues
    barline_mismatches = [i for i in verified.issues if i.code == "barline_count_mismatch" and i.severity == "needs_review"]
    assert len(barline_mismatches) == 0

    # 3. Fill measures did not trigger false missing chord needs_review issues
    fill_needs_review = [i for i in verified.issues if i.code == "missing_chord_suspected" and i.severity == "needs_review"]
    assert len(fill_needs_review) == 0


def test_v2_dropped_barline_emits_needs_review(clean_sheet, dummy_images):
    """Verify that a missed barline (merged measures ~2x beats) raises barline_count_mismatch needs_review."""
    sheet = clean_sheet.model_copy(deep=True)
    sheet.layout_source = "cv"
    sheet.layout_confidence = 0.95
    sheet.warnings = ["[v2_geometry] Classical CV layout authoritative boundaries"]

    # Simulate merged measure with ~8 beats in 4/4
    target_m = sheet.systems[0].measures[1]
    target_m.melody = "1 - - - 1 - - -"
    target_m.beats = 4.0

    verified = verify_sheet(dummy_images, sheet, use_llm=False)
    mismatch_issues = [i for i in verified.issues if i.code == "barline_count_mismatch"]
    assert len(mismatch_issues) > 0
    assert any(i.severity == "needs_review" for i in mismatch_issues)
    assert mismatch_issues[0].detail["system"] == 0


def test_v2_boxed_chart_measure_without_box_emits_info_carry_over(clean_sheet, dummy_images):
    """Verify that in a boxed chart, absence of a chord box emits info chord_carry_over."""
    sheet = clean_sheet.model_copy(deep=True)
    sheet.layout_source = "cv"
    sheet.layout_confidence = 0.95
    sheet.warnings = ["[v2_geometry] Classical CV layout authoritative boundaries"]

    # Ensure system measures have chord boxes to establish a boxed chart
    for s in sheet.systems:
        for m in s.measures:
            for c in m.chords:
                if not c.bbox:
                    c.bbox = (0.1, 0.2, 0.2, 0.25)

    # Set up cadence in measure 2 (c0=4 cadence to 1 in measure 3) with no chord at beat 3
    m2 = sheet.systems[0].measures[2]
    m2.chords = [ChordSymbol(raw="4", beat=1.0, bbox=(0.5, 0.2, 0.6, 0.25))]
    m2.melody = "5 5 2 3"
    m3 = sheet.systems[0].measures[3]
    m3.chords = [ChordSymbol(raw="1", beat=1.0, bbox=(0.7, 0.2, 0.8, 0.25))]

    verified = verify_sheet(dummy_images, sheet, use_llm=False)
    carry_over_issues = [i for i in verified.issues if i.code == "chord_carry_over" and i.measure_index == m2.index]
    assert len(carry_over_issues) > 0
    assert carry_over_issues[0].severity == "info"

    # Make sure no needs_review was emitted for m2
    m2_needs_review = [i for i in verified.issues if i.measure_index == m2.index and i.severity == "needs_review"]
    assert len(m2_needs_review) == 0


def test_v2_llm_unavailable_missing_chord_suspected_emitted(clean_sheet, dummy_images):
    """Verify that when LLM is unavailable and chart is unboxed/low-confidence, missing_chord_suspected is emitted."""
    sheet = clean_sheet.model_copy(deep=True)
    sheet.layout_source = "llm"  # Unboxed / fallback whole-page layout
    sheet.layout_confidence = 0.5
    sheet.warnings = []

    # Strip bboxes so it is an unboxed chart
    for s in sheet.systems:
        for m in s.measures:
            for c in m.chords:
                c.bbox = None

    # Set up cadence measure without beat 3 chord
    m2 = sheet.systems[0].measures[2]
    m2.chords = [ChordSymbol(raw="4", beat=1.0, bbox=None)]
    m2.melody = "5 5 2 3"
    m3 = sheet.systems[0].measures[3]
    m3.chords = [ChordSymbol(raw="1", beat=1.0, bbox=None)]

    verified = verify_sheet(dummy_images, sheet, use_llm=False)
    suspected_issues = [
        i for i in verified.issues
        if i.code == "missing_chord_suspected" and i.measure_index == m2.index
    ]
    assert len(suspected_issues) > 0
    assert any(i.severity == "needs_review" for i in suspected_issues)


def test_v2_chord_only_system_bypasses_melody_beat_sum(clean_sheet, dummy_images):
    """Verify that chord_only systems bypass melody beat-sum checks and do not emit anomalies."""
    sheet = clean_sheet.model_copy(deep=True)
    sheet.layout_source = "cv"
    sheet.layout_confidence = 0.95
    sheet.warnings = [
        "[v2_geometry] Classical CV layout authoritative boundaries",
        "[v2_chord_only] page=0 system=0",
    ]

    # Chord-only system: melody is empty
    for m in sheet.systems[0].measures:
        m.melody = ""

    verified = verify_sheet(dummy_images, sheet, use_llm=False)
    s0_mismatches = [
        i for i in verified.issues
        if i.code == "barline_count_mismatch" and i.detail.get("system") == 0
    ]
    assert len(s0_mismatches) == 0




